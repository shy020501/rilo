import json
from pathlib import Path

import numpy as np
import pytest

from rilo.contracts import ActionSpec, Observation
from rilo.learning.data import ActionWindowDataset, load_episode, snapshot_project, split_episodes
from rilo.projects import ProjectCreate, ProjectStore
from rilo.recording import Recorder


def make_learning_fixture(tmp_path, *, count=3):
    root = tmp_path/"runs"
    store = ProjectStore(root)
    project = store.create(ProjectCreate(name="Learning fixture", backend="demo", suite="demo"))
    spec = ActionSpec(names=("x", "grip"), low=(-1., -.5), high=(1., .5), description="Fixture", control_hz=20)
    episodes = []
    for number in range(count):
        recorder = Recorder(root, {"backend": "demo", "policy": "scripted", "suite": "demo", "task_id": 0},
                            {"action_spec": spec.to_dict()})
        success = number != count-1
        def observation(tick, stage="task"):
            return Observation(id=f"{number}-{stage}-{tick}", timestamp=float(tick), task="fixture", state={
                "position": [tick*.1, number*.01], "gripper": [tick*.05],
                "camera_calibration": {"ignored": 123}, "success_label": success},
                images={"front": np.full((16, 16, 3), 20+tick*10, dtype=np.uint8)},
                success=tick==4 and stage=="task" and success, terminated=tick==4 and success,
                info={"stage": stage})
        initial = observation(0)
        recorder.begin_episode(1, initial, 0)
        current = initial
        for tick in range(4):
            # Actual actions intentionally differ from requested proposals.
            next_observation = observation(tick+1)
            recorder.write({"kind": "transition", "stage": "task", "observation_id": current.id,
                            "requested_action": [.9, .4], "applied_action": [.1*(tick+1), -.1],
                            "next_observation": recorder.save_observation(next_observation)})
            current = next_observation
        summary = {"id": "episode-0001", "index": 1, "outcome": "success" if success else "step_budget",
                   "reset_outcome": "success", "steps": 4, "reset_steps": 1}
        recorder.write({"kind": "stage_end", "stage": "task", "outcome": summary["outcome"]})
        reset = observation(0, "reset")
        recorder.write({"kind": "transition", "stage": "reset", "observation_id": current.id,
                        "applied_action": [-.9, .4], "next_observation": recorder.save_observation(reset)})
        recorder.finish_episode(summary)
        recorder.close()
        store.link_run(project["id"], recorder.run_id)
        episodes.append(recorder.run_id)
    return root, store, project, episodes


def test_task_windows_actual_actions_padding_and_episode_split(tmp_path):
    root, store, project, _ = make_learning_fixture(tmp_path)
    snapshot = snapshot_project(root, project["id"], image_size=16)
    assert len(snapshot.episodes) == 3
    assert all(len(ep.actions) == 4 for ep in snapshot.episodes)
    assert snapshot.spec.state_fields == ("gripper", "position")
    assert snapshot.spec.state_dim == 3
    episode = snapshot.episodes[0]
    np.testing.assert_allclose(episode.actions[0], [.1, -.1])
    windows = ActionWindowDataset([episode], snapshot.spec, obs_steps=2, horizon=8)
    last = windows[3]
    assert tuple(last["images"].shape) == (2, 1, 3, 16, 16)
    assert last["mask"].tolist() == [True, False, False, False, False, False, False, False]
    assert np.allclose(last["actions"][0], [.4, -.1])
    train, val = split_episodes(snapshot.episodes, seed=8)
    assert {ep.id for ep in train}.isdisjoint(ep.id for ep in val)
    assert snapshot.fingerprint == snapshot_project(root, project["id"], image_size=16).fingerprint
    successful = snapshot.successful()
    assert len(successful.episodes) == 2
    assert all(ep.terminated[-1] and not ep.truncated[-1] for ep in successful.episodes)
    failure = next(ep for ep in snapshot.episodes if not ep.success)
    assert failure.truncated[-1] and not failure.terminated[-1]
    assert not failure.rewards.any()


def test_archive_delete_are_excluded_and_provenance_detects_image_changes(tmp_path):
    root, store, project, runs = make_learning_fixture(tmp_path)
    original = snapshot_project(root, project["id"], image_size=16)
    store.archive_episode(project["id"], runs[0], "episode-0001", archived=True)
    visible = snapshot_project(root, project["id"], image_size=16)
    assert len(visible.episodes) == 2
    assert original.fingerprint != visible.fingerprint
    store.delete_episode(project["id"], runs[1], "episode-0001")
    remaining = snapshot_project(root, project["id"], image_size=16)
    assert len(remaining.episodes) == 1
    frame = next(remaining.episodes[0].path.glob("frames/front/*.jpg"))
    from PIL import Image
    Image.new("RGB", (16,16), "red").save(frame)
    assert snapshot_project(root, project["id"], image_size=16).fingerprint != remaining.fingerprint
    with pytest.raises(ValueError, match="At least two"):
        split_episodes(remaining.episodes)


def test_representation_missing_state_or_bad_alignment_fails(tmp_path):
    root, store, project, runs = make_learning_fixture(tmp_path)
    transitions = root/runs[0]/"episode-0001/transitions.jsonl"
    rows = [json.loads(line) for line in transitions.read_text().splitlines()]
    next(row for row in rows if row.get("kind")=="transition")["observation_id"] = "wrong"
    transitions.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="alignment"):
        snapshot_project(root, project["id"], image_size=16)


@pytest.mark.parametrize("physical_terminal,time_limit,expected_terminal", [
    (False, False, False),  # GPT failure verdict: preserve bootstrap.
    (True, False, True),   # Adapter verified physical failure.
    (True, True, False),   # Adapter time limit is still a truncation.
])
def test_failure_verdict_does_not_fabricate_physical_terminal(
        tmp_path, physical_terminal, time_limit, expected_terminal):
    root, _, _, runs = make_learning_fixture(tmp_path, count=1)
    folder = root/runs[0]/"episode-0001"
    path = folder/"transitions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    final = [row for row in rows if row.get("kind") == "transition" and row.get("stage") == "task"][-1]
    final["next_observation"]["terminated"] = physical_terminal
    final["next_observation"]["info"]["time_limit"] = time_limit
    next(row for row in rows if row.get("kind") == "stage_end")["outcome"] = "failure"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    episode = load_episode(folder, {"outcome":"failure"}, json.loads((folder.parent/"manifest.json").read_text()))
    assert bool(episode.terminated[-1]) is expected_terminal
    assert bool(episode.truncated[-1]) is not expected_terminal
    assert not episode.terminated[:-1].any() and not episode.truncated[:-1].any()
    assert not episode.rewards.any()


@pytest.mark.parametrize("stage,source_policy,source_model", [
    ("collect", "openai", "fixture-teacher"),
    ("bc_eval", "diffusion_policy", "bc-123"),
    ("rl", "diffusion_policy_residual", "residual-456"),
    ("rl_eval", "diffusion_policy_residual", "residual-456"),
])
def test_provenance_distinguishes_learned_task_producer_from_gpt_teacher(
        tmp_path, stage, source_policy, source_model):
    root, _, project, runs = make_learning_fixture(tmp_path, count=1)
    path = root/runs[0]/"manifest.json"
    manifest = json.loads(path.read_text())
    learning = {"pipeline_stage":stage, "job_id":"job-789", "base_model_id":"bc-123",
                "base_checkpoint_sha256":"a"*64, "residual_model_id":"residual-456",
                "residual_checkpoint_sha256":"b"*64}
    manifest["config"].update(policy="openai", model="fixture-teacher", reasoning_effort="medium", learning=learning)
    path.write_text(json.dumps(manifest))
    snapshot = snapshot_project(root, project["id"], image_size=16)
    provenance = snapshot.manifest()["episodes"][0]
    assert provenance["source_policy"] == source_policy
    assert provenance["source_model"] == source_model
    assert provenance["learning"] == learning
    if stage != "collect":
        assert provenance["teacher"] == {"policy":"openai", "model":"fixture-teacher", "reasoning_effort":"medium"}
    else:
        assert "teacher" not in provenance
    # Producer metadata is part of the immutable dataset fingerprint, even if
    # the image/action receipts themselves have not changed.
    learning["base_checkpoint_sha256"] = "c"*64
    path.write_text(json.dumps(manifest))
    changed = snapshot_project(root, project["id"], image_size=16)
    assert changed.fingerprint != snapshot.fingerprint
    assert provenance["learning"]["base_checkpoint_sha256"] == "a"*64
