"""Residual stop/resume/config integration with tiny temporary checkpoints only."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from rilo.config import Settings
from rilo.learning.residual import TD3Config
from rilo.learning.rl_train import ResidualTrainer
from rilo.pipeline import PipelineEngine, PipelineStart, PipelineStopped
from rilo.projects import ProjectCreate, ProjectStore
from test_learning_runtime import SyntheticEpisode, controller


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class QuietRunner:
    def __init__(self):
        self.phase = "completed"
        self._task = None
        self.stats = {"episodes": 1, "successes": 1, "reset_successes": 1}
        self.recorder = SimpleNamespace(run_id="synthetic-run")
        self.error = None
        self._stop_lock = asyncio.Lock()
        self.stop_entered = asyncio.Event()

    def _safe_error(self, error):
        return str(error)

    async def pause(self):
        self.phase = "paused"

    async def stop(self):
        async with self._stop_lock:
            self.stop_entered.set()
            if self._task is not None:
                await self._task
            self.phase = "stopped"


def engine_fixture(tmp_path):
    base = controller(tmp_path)
    settings = Settings(backend="demo", suite="demo", policy="scripted", data_dir=tmp_path / "runs")
    projects = ProjectStore(settings.data_dir)
    project = projects.create(ProjectCreate(name="Residual state fixture", backend="demo", suite="demo", task_id=0))
    runner = QuietRunner()
    captured = []
    async def prepare(project_id, episodes, policy, metadata, budget):
        captured.append((policy, metadata))
        return runner
    engine = PipelineEngine(settings, projects, lambda: runner, prepare)
    config = TD3Config(hidden_dim=16, hidden_layers=1, num_critics=2, target_subset=2,
                       batch_size=4, learning_starts=1, buffer_capacity=16, critic_warmup_updates=0)
    engine.configure(project["id"], {"device": "cpu", "rl_episodes_per_round": 1,
                                      "residual": config.model_dump(mode="json")})
    engine._state["status"].update(stage="rl", phase="running", job_id="synthetic-job")
    engine._register("bc", "fixture-bc", {"checkpoint": str(tmp_path / "fixture-dp.pt"),
                                          "config": {"image_size": 16}})
    engine._register("progress", "fixture-progress", {"checkpoint": "unused-progress-fixture.pt"}, "fixture-bc")
    trainer = ResidualTrainer(base.feature_dim, [-1.], [1.], config)
    snapshot = SimpleNamespace(episodes=[SyntheticEpisode()], fingerprint="fixture-data")
    trainer.add_offline(snapshot, base)
    engine._trainer = trainer
    engine._saved_rl_steps = 0
    engine._snapshot = lambda image_size: snapshot
    return engine, trainer, runner, captured, base


def add_online(trainer, number=0):
    features = [0.] * trainer.agent.feature_dim
    trainer.observe(features, [.2], [.3], 1., features, [.4], terminated=True,
                    episode_id=f"synthetic-{number}")


@pytest.mark.parametrize("reason", ["stop", "error"])
async def test_partial_checkpoint_preserves_replay_and_metadata_without_duplicate_save(tmp_path, monkeypatch, reason):
    engine, trainer, runner, _, _ = engine_fixture(tmp_path)
    add_online(trainer)
    async def interrupted_stage(stage):
        if reason == "stop":
            raise PipelineStopped("Synthetic stop")
        raise RuntimeError("Synthetic failure")
    monkeypatch.setattr(engine, "_stage", interrupted_stage)
    request = PipelineStart(mode="manual", stage="rl")
    await engine._execute(request)
    assert engine._state["status"]["phase"] == ("stopped" if reason == "stop" else "error")
    models = [model for model in engine._state["models"] if model["kind"] == "residual"]
    assert len(models) == 1
    checkpoint = models[0]["checkpoint"]
    restored = ResidualTrainer.load_checkpoint(checkpoint, expected_base_fingerprint=trainer.base_fingerprint)
    assert restored.online_steps == len(restored.replay.online) == 1
    assert len(restored.replay.offline) == 3
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["metadata"]["rollout"]["partial"]
    assert payload["metadata"]["rollout"]["end_reason"] == engine._state["status"]["phase"]
    assert engine._saved_rl_steps == 1
    await engine._execute(request)  # No additional observations; no duplicate model.
    assert len([model for model in engine._state["models"] if model["kind"] == "residual"]) == 1


async def test_stop_waits_for_inflight_update_before_partial_checkpoint(tmp_path, monkeypatch):
    engine, trainer, runner, _, _ = engine_fixture(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    add_online(trainer)
    async def inflight_update():
        entered.set()
        await release.wait()
        # Represents the last update/transition committing after Stop arrives.
        add_online(trainer, 1)
    async def running_stage(stage):
        runner.phase = "executing"
        runner._task = asyncio.create_task(inflight_update())
        await entered.wait()
        while not engine._stop.is_set():
            await asyncio.sleep(.001)
        raise PipelineStopped("Synthetic stop during update")
    monkeypatch.setattr(engine, "_stage", running_stage)
    engine._task = asyncio.create_task(engine._execute(PipelineStart(mode="manual", stage="rl")))
    await entered.wait()
    stop = asyncio.create_task(engine.stop(engine._owner))
    try:
        await runner.stop_entered.wait()
        await asyncio.sleep(.03)
        assert not [model for model in engine._state["models"] if model["kind"] == "residual"]
        assert not list(engine._path(engine._owner).glob("models/residual-*/residual.pt"))
    finally:
        release.set()
        await stop
        await engine._task
    model = engine._latest(engine._state, "residual")
    restored = ResidualTrainer.load_checkpoint(model["checkpoint"])
    assert restored.online_steps == len(restored.replay.online) == 2


@pytest.mark.parametrize("changed", [False, True])
async def test_config_match_resumes_and_change_starts_fresh_preserving_previous_checkpoint(tmp_path, monkeypatch, changed):
    engine, trainer, runner, captured, base = engine_fixture(tmp_path)
    add_online(trainer)
    await engine._save_residual({"fixture": True})
    previous = deepcopy(engine._latest(engine._state, "residual"))
    from pathlib import Path
    previous_bytes = Path(previous["checkpoint"]).read_bytes()
    if changed:
        engine._state["config"]["residual"]["residual_scale"] = .2
    engine._trainer = None
    engine._saved_rl_steps = 0
    monkeypatch.setattr("rilo.learning.progress.ProgressPredictor", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr("rilo.pipeline.make_policy", lambda settings: SimpleNamespace())
    await engine._rollout_stage("rl")
    active = engine._trainer
    assert active.base_fingerprint == base.checkpoint_fingerprint
    assert active.online_steps == (0 if changed else 1)
    assert len(active.replay.online) == (0 if changed else 1)
    assert len(active.replay.offline) == 3
    assert active.config.residual_scale == (.2 if changed else .1)
    assert captured[0][1]["residual_model_id"] == (None if changed else previous["id"])
    assert Path(previous["checkpoint"]).read_bytes() == previous_bytes
    assert len([model for model in engine._state["models"] if model["kind"] == "residual"]) == 2
