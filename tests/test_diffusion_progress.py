from copy import deepcopy

import numpy as np
import pytest
import torch

from rilo.learning.bc_train import train_bc
from rilo.learning.checkpoints import load_dp, save_dp
from rilo.learning.data import ActionWindowDataset, snapshot_project
from rilo.learning.diffusion import DPConfig, DiffusionPolicy, FrozenDPController
from rilo.learning.progress import ProgressConfig, ProgressPredictor, train_progress
from test_learning_data import make_learning_fixture


@pytest.fixture(autouse=True)
def tiny_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return DPConfig(image_size=16, backbone="tiny", imagenet_init=False, feature_dim=16,
                    keypoints=4, unet_dims=(8,16,32), horizon=4, action_steps=2,
                    diffusion_steps=8, inference_steps=2, steps=2, batch_size=2,
                    eval_every=1, validation_batches=1, min_steps=0, device="cpu")


def test_diffusion_gradient_bounds_and_checkpoint_roundtrip(tmp_path):
    root, _, project, _ = make_learning_fixture(tmp_path)
    snapshot = snapshot_project(root, project["id"], image_size=16)
    config = tiny_config()
    model = DiffusionPolicy(snapshot.spec, config)
    window = ActionWindowDataset(snapshot.episodes, snapshot.spec, obs_steps=2, horizon=4)[0]
    batch = {key: value.unsqueeze(0) for key, value in window.items()}
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.denoiser.parameters())
    predicted = model.predict_chunk(batch["images"],batch["states"], generator=torch.Generator().manual_seed(1))
    assert predicted.shape == (1,4,2)
    assert torch.all(predicted>=torch.tensor(snapshot.spec.action_low)-1e-6)
    assert torch.all(predicted<=torch.tensor(snapshot.spec.action_high)+1e-6)
    path = save_dp(model, tmp_path/"test.pt", {"dataset":snapshot.fingerprint})
    restored, metadata = load_dp(path)
    assert metadata["dataset"] == snapshot.fingerprint
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_chunk_reconstruction_nonconsuming_peek_and_frozen_base(tmp_path):
    root, _, project, _ = make_learning_fixture(tmp_path)
    snapshot = snapshot_project(root, project["id"], image_size=16)
    model = DiffusionPolicy(snapshot.spec,tiny_config())
    controller = FrozenDPController.from_model(model,seed=9)
    episode = snapshot.episodes[0]
    calls = []
    original = model.predict_chunk
    def count_calls(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    model.predict_chunk = count_calls
    feature, first = controller.observe(episode.observation(0))
    same = episode.observation(0)
    same.id = "new-measurement-same-tick"
    feature2, first2 = controller.peek(same)
    np.testing.assert_array_equal(first,first2)
    assert len(calls)==1
    controller.advance()
    assert controller.observe(episode.observation(1))[0][-1] == .5
    assert len(calls)==1
    reconstructed = controller.reconstruct_episode(episode)
    assert reconstructed["features"].shape == (5,17)
    assert reconstructed["base_actions"].shape == (5,2)
    repeated = controller.reconstruct_episode(episode)
    np.testing.assert_array_equal(reconstructed["base_actions"],repeated["base_actions"])
    assert not any(p.requires_grad for p in controller.model.parameters())


def test_explicit_fixture_bc_and_progress_smoke_no_download(tmp_path):
    root, _, project, _ = make_learning_fixture(tmp_path)
    snapshot = snapshot_project(root,project["id"],image_size=16)
    events=[]; checks=[]
    bc = train_bc(snapshot,tmp_path/"bc",tiny_config(),callback=events.append,control=lambda:checks.append(True))
    assert bc["metrics"]["steps_completed"]==2
    assert bc["dataset"]["episode_count"]==2  # Failed example stays out of BC.
    assert set(bc["split"]["train"]).isdisjoint(bc["split"]["validation"])
    progress = train_progress(snapshot,bc["checkpoint"],tmp_path/"progress",
        ProgressConfig(steps=2,batch_size=2,history_steps=2,hidden_dim=8,eval_every=1,device="cpu"))
    predictor = ProgressPredictor(progress["checkpoint"])
    value = predictor.observe(snapshot.episodes[0].observation(0))
    assert 0 <= value <= 1
    assert progress["dataset"]["episode_count"]==2
    assert events and checks
    assert all(not p.requires_grad for p in predictor.encoder.parameters())


def test_control_cancels_before_training_step(tmp_path):
    root, _, project, _=make_learning_fixture(tmp_path)
    snapshot=snapshot_project(root,project["id"],image_size=16)
    def stop(): raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        train_bc(snapshot,tmp_path/"cancelled",tiny_config(),control=stop)
    assert not (tmp_path/"cancelled/policy.pt").exists()


def test_overlapping_observations_share_one_chunk_without_queue_race(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    root, _, project, _ = make_learning_fixture(tmp_path)
    snapshot = snapshot_project(root, project["id"], image_size=16)
    model = DiffusionPolicy(snapshot.spec,tiny_config())
    controller = FrozenDPController.from_model(model,seed=3)
    episode = snapshot.episodes[0]
    entered, release = threading.Event(), threading.Event()
    original = model.encoder.forward
    calls=[]
    def slow_first(*args, **kwargs):
        calls.append(True)
        if len(calls)==1:
            entered.set()
            assert release.wait(3)
        return original(*args, **kwargs)
    model.encoder.forward=slow_first
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(controller.observe,episode.observation(0))
        assert entered.wait(3)
        fresh=episode.observation(0)
        fresh.id="fresh-same-tick"
        second=pool.submit(controller.observe,fresh)
        try:
            time.sleep(.02)
            assert len(calls)==1  # The cancelled worker must finish before re-entry.
        finally:
            release.set()
        first_result,second_result=first.result(),second.result()
    np.testing.assert_array_equal(first_result[1],second_result[1])
    assert len(controller.queue)==2
