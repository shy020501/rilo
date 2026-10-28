from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rilo.learning.replay import MixedReplay, NStepAccumulator, ReplayBuffer, Transition
from rilo.learning.residual import ResidualTD3, TD3Config
from rilo.learning.rl_train import ResidualTrainer


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**overrides):
    return TD3Config(**{**dict(hidden_dim=16, hidden_layers=1, num_critics=2, target_subset=2,
                              batch_size=4, critic_warmup_updates=0, learning_starts=1,
                              buffer_capacity=32), **overrides})


def transition(t=0, *, reward=0., terminal=False, truncated=False, episode="a"):
    return Transition(np.array([t, 0.]), np.array([.2]), np.array([-.3]), reward,
                      np.array([t+1, 0.]), np.array([.4]), terminal, truncated, episode)


def batch():
    return {"features": np.zeros((4, 2), np.float32),
            "base_actions": np.full((4, 1), .2, np.float32),
            "actions": np.full((4, 1), -.7, np.float32),
            "next_features": np.ones((4, 2), np.float32),
            "next_base_actions": np.full((4, 1), .3, np.float32),
            "rewards": np.array([[0.], [1.], [0.], [1.]], np.float32),
            "discounts": np.array([[.99], [0.], [.99], [0.]], np.float32)}


def test_zero_residual_preserves_base_and_noise_remains_inside_bound():
    config = tiny_config(residual_scale=.1, exploration_std=1.)
    agent = ResidualTD3(2, [-2., 0.], [4., 10.], config)
    base = np.array([1., 9.9], np.float32)
    np.testing.assert_array_equal(agent.act(np.zeros(2), base), base)
    actions = agent.act(np.zeros((100, 2)), np.tile(base, (100, 1)), explore=True)
    assert np.all(actions >= [-2., 0.]) and np.all(actions <= [4., 10.])
    assert np.all(np.abs(actions-base) <= np.array([.3, .5]) + 1e-6)
    agent_zero = ResidualTD3(2, [-1.], [1.], tiny_config(residual_scale=0., exploration_std=1.))
    np.testing.assert_array_equal(agent_zero.act(np.zeros(2), np.array([.5]), explore=True), [.5])


def test_n_step_returns_terminate_and_never_cross_episodes():
    nstep = NStepAccumulator(3, .5)
    assert nstep.append(transition(0, reward=1.)) == []
    assert nstep.append(transition(1, reward=2.)) == []
    with pytest.raises(ValueError, match="previous episode"):
        nstep.append(transition(9, episode="other"))
    rows = nstep.append(transition(2, reward=3., terminal=True))
    assert len(rows) == 3
    np.testing.assert_allclose([row["rewards"].item() for row in rows], [2.75, 3.5, 3.])
    assert all(row["discounts"].item() == 0 for row in rows)
    assert all(row["next_features"][0] == 3 for row in rows)
    assert not nstep.pending
    assert nstep.append(transition(10, episode="other")) == []


@pytest.mark.parametrize("bootstrap,expected", [(True, .25), (False, 0.)])
def test_time_limit_uses_last_task_state_not_reset_observation(bootstrap, expected):
    accumulator = NStepAccumulator(3, .5, bootstrap_on_truncation=bootstrap)
    accumulator.append(transition(0))
    rows = accumulator.append(transition(1, truncated=True))
    assert rows[0]["discounts"].item() == expected
    assert rows[0]["next_features"][0] == 2
    assert rows[0]["n_steps"].item() == 2


def test_mixed_replay_is_exact_half_with_separate_full_and_base_actions():
    buffers = [ReplayBuffer(8), ReplayBuffer(8)]
    for i, buffer in enumerate(buffers):
        nstep = NStepAccumulator(1)
        row = nstep.append(transition(i, terminal=True))[0]
        buffer.add(row)
    sampler = MixedReplay(*buffers, offline_fraction=.5, seed=10)
    result = sampler.sample(8)
    assert result["offline"].sum() == 4
    np.testing.assert_allclose(result["actions"], -.3)
    np.testing.assert_allclose(result["base_actions"], .2)
    empty = MixedReplay(buffers[0], ReplayBuffer(8))
    with pytest.raises(ValueError, match="not ready"):
        empty.sample(8)


class ConstantQ(nn.Module):
    def forward(self, features, action):
        return torch.tensor([2., 8.], device=features.device).expand(len(features), 2)


def test_bellman_target_uses_twin_min_and_terminal_discount_and_smoothing_bounds():
    agent = ResidualTD3(2, [-1.], [1.], tiny_config(target_noise=1., target_noise_clip=1.))
    agent.target_critic = ConstantQ()
    seen = []
    agent.target_critic.register_forward_pre_hook(lambda module, args: seen.append(args[1].detach().clone()))
    value = batch()
    result = agent.bellman_target(value)
    np.testing.assert_allclose(result.numpy().ravel(), [1.98, 1., 1.98, 1.], atol=1e-6)
    assert torch.all((seen[0] - .3).abs() <= .100001)


def test_critic_uses_executed_full_action_actor_delayed_and_no_base_gradient():
    agent = ResidualTD3(2, [-1.], [1.], tiny_config())
    value = batch()
    features = torch.zeros((4, 2), requires_grad=True)
    bases = torch.full((4, 1), .2, requires_grad=True)
    value["features"], value["base_actions"] = features, bases
    seen = []
    agent.critic.register_forward_pre_hook(lambda module, args: seen.append(args[1].detach().clone()))
    original_actor = deepcopy(agent.actor.state_dict())
    first = agent.update(value)
    np.testing.assert_allclose(seen[0].numpy(), value["actions"])
    assert first["actor_loss"] is None and first["actor_updates"] == 0
    assert all(torch.equal(original_actor[key], tensor) for key, tensor in agent.actor.state_dict().items())
    second = agent.update(value)
    assert second["actor_loss"] is not None and second["actor_updates"] == 1
    assert any(not torch.equal(original_actor[key], tensor) for key, tensor in agent.actor.state_dict().items())
    assert features.grad is None and bases.grad is None
    assert all(parameter.grad is None for parameter in agent.target_actor.parameters())
    assert np.isfinite(second["critic_loss"])


def episode(success=True, name="episode-0001"):
    return SimpleNamespace(run_id="run", episode_id=name, actions=np.full((3, 1), -.4, np.float32),
                           success=success, terminated=np.array([False, False, success]),
                           truncated=np.array([False, False, not success]))


class FrozenController:
    checkpoint_fingerprint = "frozen-dp-sha"
    def __init__(self):
        self.reconstructed = []
        self.reset_calls = 0

    def reconstruct_episode(self, ep):
        self.reconstructed.append(ep.episode_id)
        return {"features": np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], np.float32),
                "base_actions": np.array([[.1], [.2], [.3], [.4]], np.float32)}

    def reset_episode(self):
        self.reset_calls += 1


def test_offline_reconstruction_includes_failures_and_original_full_actions():
    trainer = ResidualTrainer(2, [-1.], [1.], tiny_config())
    dp = FrozenController()
    snapshot = SimpleNamespace(episodes=[episode(True), episode(False, "episode-0002")], fingerprint="data-sha")
    stats = trainer.add_offline(snapshot, dp)
    assert stats["offline_success_episodes"] == stats["offline_failure_episodes"] == 1
    assert stats["offline_transitions"] == 6
    assert dp.reconstructed == ["episode-0001", "episode-0002"]
    np.testing.assert_allclose(trainer.replay.offline.arrays["actions"][:6], -.4)
    np.testing.assert_allclose(trainer.replay.offline.arrays["base_actions"][:3].ravel(), [.1, .2, .3])
    np.testing.assert_allclose(trainer.replay.offline.arrays["next_base_actions"][:3], .4)
    assert trainer.replay.offline.arrays["rewards"][3:6].sum() == 0.
    assert trainer.replay.offline.arrays["discounts"][:3].sum() == 0.
    assert np.all(trainer.replay.offline.arrays["discounts"][3:6] > 0.)
    # Absence of failed examples is explicitly allowed.
    trainer_success = ResidualTrainer(2, [-1.], [1.], tiny_config())
    assert trainer_success.add_offline(SimpleNamespace(episodes=[episode()], fingerprint="success"), dp)["offline_failure_episodes"] == 0


def test_online_utd_resume_replay_optimizer_rng_and_base_identity(tmp_path):
    trainer = ResidualTrainer(2, [-1.], [1.], tiny_config(utd_ratio=2))
    trainer.add_offline(SimpleNamespace(episodes=[episode()], fingerprint="data"), FrozenController())
    assert trainer.update()["updates_performed"] == 0
    for t in range(3):
        trainer.observe(np.array([t, 0.]), [.2], [-.1], float(t == 2), [t+1, 0.], [.3],
                        terminated=t == 2, episode_id="online-one")
    metrics = trainer.update()
    assert metrics["updates_performed"] == 2
    assert metrics["sampled_offline_fraction"] == .5
    path = tmp_path / "residual.pt"
    trainer.save_checkpoint(path, {"purpose": "synthetic smoke"})
    resumed = ResidualTrainer.load_checkpoint(path, expected_base_fingerprint="frozen-dp-sha")
    assert len(resumed.replay.offline) == len(resumed.replay.online) == 3
    assert resumed.online_steps == 3
    next_original, next_resumed = trainer.update(), resumed.update()
    assert next_original == next_resumed
    for name, tensor in trainer.agent.actor.state_dict().items():
        torch.testing.assert_close(tensor, resumed.agent.actor.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="different frozen BC"):
        ResidualTrainer.load_checkpoint(path, expected_base_fingerprint="another-checkpoint")


def test_manual_episode_end_flushes_pending_without_using_reset_state():
    trainer = ResidualTrainer(2, [-1.], [1.], tiny_config())
    trainer.observe([0., 0.], [.2], [.3], 0, [1., 0.], [.4], episode_id="one")
    assert len(trainer.replay.online) == 0
    assert trainer.end_episode() == 1
    row = trainer.replay.online.sample(1, np.random.default_rng(0))
    np.testing.assert_allclose(row["next_features"], [[1., 0.]])
    np.testing.assert_allclose(row["discounts"], [[trainer.config.gamma]])
    trainer.observe([0., 0.], [.2], [.3], 0, [1., 0.], [.4], episode_id="two")


def test_invalid_residual_config_or_applied_action_rejected():
    with pytest.raises(ValueError):
        tiny_config(target_subset=3)
    trainer = ResidualTrainer(2, [-1.], [1.], tiny_config())
    with pytest.raises(ValueError, match="actual applied"):
        trainer.observe([0., 0.], [0.], [2.], 0., [0., 0.], [0.])


def test_real_frozen_diffusion_chunk_cache_survives_residual_smoke_update():
    from rilo.contracts import Observation
    from rilo.learning.data import ObservationSpec
    from rilo.learning.diffusion import DPConfig, DiffusionPolicy, FrozenDPController

    spec = ObservationSpec(cameras=("camera",), state_fields=("position",), state_sizes=(1,),
                           action_names=("dx",), action_low=(-1.,), action_high=(1.,), image_size=16)
    config = DPConfig(image_size=16, feature_dim=16, keypoints=4, horizon=4, action_steps=2,
                       unet_dims=(8, 16, 32), diffusion_steps=4, inference_steps=2,
                       backbone="tiny", imagenet_init=False)
    controller = FrozenDPController.from_model(DiffusionPolicy(spec, config))
    frozen_state = {key: value.clone() for key, value in controller.model.state_dict().items()}

    class SyntheticEpisode:
        run_id, episode_id, success = "synthetic", "episode-0001", True
        actions = np.array([[.1], [.2], [.3]], np.float32)
        terminated = np.array([False, False, True])
        truncated = np.array([False, False, False])
        def observation(self, index):
            return Observation(str(index), float(index), "fixture", {"position": [float(index)]},
                               {"camera": np.zeros((16, 16, 3), np.uint8)})

    trainer = ResidualTrainer(controller.feature_dim, [-1.], [1.], tiny_config(utd_ratio=2))
    trainer.add_offline(SimpleNamespace(episodes=[SyntheticEpisode()], fingerprint="synthetic-data"), controller)
    ep = SyntheticEpisode()
    for step in range(3):
        features, base = controller.observe(ep.observation(step))
        same_features, same_base = controller.observe(ep.observation(step))
        np.testing.assert_array_equal(base, same_base)
        np.testing.assert_array_equal(features, same_features)
        full = trainer.act(features, base, explore=False)
        controller.advance()
        following_features, following_base = controller.observe(ep.observation(step + 1))
        trainer.observe(features, base, full, float(step == 2), following_features, following_base,
                        terminated=step == 2, episode_id="synthetic-online")
    assert trainer.update()["critic_updates"] == 2
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in controller.model.parameters())
    for name, value in controller.model.state_dict().items():
        torch.testing.assert_close(frozen_state[name], value, rtol=0, atol=0)
