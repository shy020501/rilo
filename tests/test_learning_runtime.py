"""Synthetic integration checks; no benchmark, real dataset, training job, or API."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rilo.config import Settings
from rilo.contracts import ActionCommand, ActionSpec, Decision, Observation
from rilo.learning.checkpoints import save_dp
from rilo.learning.data import ObservationSpec
from rilo.learning.diffusion import DPConfig, DiffusionPolicy, FrozenDPController
from rilo.learning.residual import TD3Config
from rilo.learning.rl_train import ResidualTrainer
from rilo.learning.runtime import LearnedTaskPolicy
from rilo.runner import RunConfig, Runner


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(index, position=0., *, stage="task", success=False):
    return Observation(str(index), float(index), "Synthetic runtime fixture",
                       {"position": [float(position)]},
                       {"camera": np.zeros((16, 16, 3), np.uint8)},
                       success=success, terminated=success, info={"stage": stage})


def controller(tmp_path):
    spec = ObservationSpec(cameras=("camera",), state_fields=("position",), state_sizes=(1,),
                           action_names=("dx",), action_low=(-1.,), action_high=(1.,), image_size=16)
    config = DPConfig(image_size=16, feature_dim=16, keypoints=4, horizon=4, action_steps=2,
                      unet_dims=(8, 16, 32), diffusion_steps=4, inference_steps=2,
                      backbone="tiny", imagenet_init=False)
    # Random initialized fixture checkpoint: nothing is fitted or downloaded.
    path = save_dp(DiffusionPolicy(spec, config), tmp_path / "fixture-dp.pt")
    return FrozenDPController(path, seed=3)


class SyntheticEpisode:
    run_id, episode_id, success = "fixture", "episode-0001", True
    actions = np.array([[.1], [.2], [.3]], np.float32)
    terminated = np.array([False, False, True])
    truncated = np.zeros(3, dtype=bool)

    def observation(self, index):
        return observation(index, position=index / 10.)


class ThreeTickAdapter:
    """Logic fixture with real position integration and an artificial 3-tick goal.

    Reset succeeds only when bounded policy actions physically return position
    to zero. The task predicate tests orchestration, not learned performance.
    """
    action_spec = ActionSpec(names=("dx",), low=(-1.,), high=(1.,),
                             description="Fixture displacement", control_hz=1000)

    def __init__(self):
        self.position = 0.
        self.stage = "task"
        self.task_ticks = self.sequence = self.reset_calls = 0
        self.executed = []

    def reset(self, *, seed, task_id):
        self.reset_calls += 1
        self.position = 0.
        self.task_ticks = 0
        self.stage = "task"
        return self.observe()

    def begin_reset(self):
        self.stage = "reset"
        return self.observe()

    def begin_task(self):
        assert abs(self.position) < 1e-6
        self.stage = "task"
        self.task_ticks = 0
        return self.observe()

    def describe(self):
        return {"stage": self.stage, "simulation": True, "test_fixture": True}

    def observe(self):
        self.sequence += 1
        success = self.task_ticks == 3 if self.stage == "task" else abs(self.position) < 1e-6
        return observation(self.sequence, self.position, stage=self.stage, success=success)

    def step(self, values):
        applied = self.action_spec.validate(values)
        self.position += applied[0]
        self.task_ticks += int(self.stage == "task")
        self.executed.append((self.stage, applied))
        result = self.observe()
        result.info["applied_action"] = applied
        return result

    def pause(self):
        pass

    def close(self):
        pass


class PhysicalResetPolicy:
    def __init__(self):
        self.calls = []

    async def decide(self, obs, spec, context, history):
        self.calls.append(obs.info["stage"])
        assert obs.info["stage"] == "reset"
        return Decision(observation_id=obs.id, status="continue",
                        actions=[ActionCommand(values=[float(np.clip(-obs.state["position"][0], -1., 1.))])])

    async def close(self):
        pass


async def completed(runner):
    async def poll():
        while runner.phase not in ("completed", "error"):
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), timeout=10)
    assert runner.phase == "completed", runner.error


async def test_real_dp_runner_residual_training_and_frozen_evaluation_with_physical_resets(tmp_path):
    base = controller(tmp_path)
    original_base = deepcopy(base.model.state_dict())
    config = TD3Config(hidden_dim=16, hidden_layers=1, num_critics=2, target_subset=2,
                       batch_size=4, critic_warmup_updates=0, learning_starts=1,
                       buffer_capacity=32, utd_ratio=2)
    trainer = ResidualTrainer(base.feature_dim, [-1.], [1.], config, device="auto")
    trainer.add_offline(SimpleNamespace(episodes=[SyntheticEpisode()], fingerprint="fixture"), base)
    advances = []
    original_advance = base.advance
    def count_advance():
        advances.append(True)
        original_advance()
    base.advance = count_advance
    reset = PhysicalResetPolicy()
    policy = LearnedTaskPolicy(base, reset, residual=trainer, training=True)
    adapter = ThreeTickAdapter()
    runner = Runner(Settings(backend="demo", policy="scripted", data_dir=tmp_path / "records"),
                    adapter_factory=lambda: adapter)
    run_config = RunConfig(reset_mode="agent", episodes=2, max_steps=4, max_decisions=4)
    try:
        await runner.initialize(run_config, policy_override=policy)
        await runner.start()
        await completed(runner)
        assert runner.stats["successes"] == runner.stats["reset_successes"] == 2
        assert adapter.reset_calls == 1 and runner.stats["hard_resets"] == 0
        assert reset.calls and set(reset.calls) == {"reset"}
        assert len(advances) == trainer.online_steps == len(trainer.replay.online) == 6
        assert trainer.episodes == 2 and not trainer.accumulator.pending
        assert trainer.agent.updates > 0 and trainer.agent.actor_updates > 0
        assert trainer.last_metrics["sampled_offline_fraction"] == .5
        task_actions = np.asarray([a for stage, a in adapter.executed if stage == "task"])
        np.testing.assert_allclose(trainer.replay.online.arrays["actions"][:6], task_actions)
        # All n-step windows end at each episode's terminal task state, not reset.
        assert trainer.replay.online.arrays["discounts"][:6].sum() == 0.
        for key, value in base.model.state_dict().items():
            torch.testing.assert_close(value, original_base[key], rtol=0, atol=0)
        assert not any(parameter.requires_grad for parameter in base.model.parameters())
        records = [json.loads(line) for path in (tmp_path / "records").glob("*/episode-*/transitions.jsonl")
                   for line in path.read_text().splitlines()]
        assert sum(r["kind"] == "transition" and r["stage"] == "task" for r in records) == 6
        assert any(r["kind"] == "transition" and r["stage"] == "reset" for r in records)
        assert all(r["stage"] == "task" for r in records if r["kind"] == "policy_transition")

        actor_before, critic_before = deepcopy(trainer.agent.actor.state_dict()), deepcopy(trainer.agent.critic.state_dict())
        updates_before = trainer.agent.updates
        evaluation = LearnedTaskPolicy(base, PhysicalResetPolicy(), residual=trainer, training=False)
        await runner.initialize(run_config, preserve_environment=True, policy_override=evaluation)
        await runner.start()
        await completed(runner)
        assert adapter.reset_calls == 1
        assert trainer.online_steps == 6 and len(trainer.replay.online) == 6
        assert trainer.agent.updates == updates_before and len(advances) == 12
        for module, previous in ((trainer.agent.actor, actor_before), (trainer.agent.critic, critic_before)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, previous[key], rtol=0, atol=0)
        assert runner.stats["api_requests"] == 0
    finally:
        await runner.close()


async def test_progress_judge_gate_interval_and_unverified_success_does_not_label_task(tmp_path):
    base = controller(tmp_path)
    class Progress:
        def reset_episode(self):
            self.values = iter([.2, .9, .95, .95, .95])
        def observe(self, obs):
            return next(self.values)
    class Judge:
        def __init__(self):
            self.calls = []
        async def decide(self, obs, *args):
            self.calls.append(obs.id)
            return Decision(observation_id=obs.id, status="success", note="Synthetic claim")
        async def close(self):
            pass
    judge = Judge()
    reset = PhysicalResetPolicy()
    policy = LearnedTaskPolicy(base, reset, progress=Progress(), judge=judge,
                               progress_threshold=.8, judge_interval=2)
    policy.on_episode_start("one")
    for tick in range(5):
        before, after = observation(tick), observation(tick + 1)
        decision = await policy.decide(before, ThreeTickAdapter.action_spec, {}, [])
        assert decision.status == "continue" and decision.actions[0].repeat == 1
        await policy.on_transition(before, decision.actions[0].values, after, stage="task", truncated=False)
    assert judge.calls == ["1", "3"]
    policy.on_stage_start("reset")
    await policy.decide(observation(6, .1, stage="reset"), ThreeTickAdapter.action_spec, {}, [])
    assert reset.calls == ["reset"] and judge.calls == ["1", "3"]


@pytest.mark.parametrize("outcome", ["step_budget", "failure"])
async def test_external_episode_boundary_bootstraps_last_task_observation(tmp_path, outcome):
    base = controller(tmp_path)
    trainer = ResidualTrainer(base.feature_dim, [-1.], [1.],
                              TD3Config(hidden_dim=16, hidden_layers=1, batch_size=4,
                                        learning_starts=4, buffer_capacity=16, n_step=3))
    policy = LearnedTaskPolicy(base, PhysicalResetPolicy(), residual=trainer, training=True)
    policy.on_episode_start("limited")
    before, after = observation(0, .2), observation(1, .4)
    decision = await policy.decide(before, ThreeTickAdapter.action_spec, {}, [])
    await policy.on_transition(before, decision.actions[0].values, after, stage="task",
                               truncated=outcome == "step_budget")
    # The failure branch models a GPT judgment before the following action.
    policy.on_stage_end("task", outcome)
    assert trainer.episodes == len(trainer.replay.online) == 1
    np.testing.assert_allclose(trainer.replay.online.arrays["discounts"][0], [trainer.config.gamma])
    expected_features, expected_base = base.observe(after)
    np.testing.assert_allclose(trainer.replay.online.arrays["next_features"][0], expected_features)
    np.testing.assert_allclose(trainer.replay.online.arrays["next_base_actions"][0], expected_base)
    await policy.on_transition(after, [-.4], observation(2, 0., stage="reset", success=True),
                               stage="reset", truncated=False)
    assert trainer.online_steps == 1


async def test_one_touch_http_pipeline_all_six_stages_on_tiny_synthetic_world(tmp_path, monkeypatch):
    """One update per neural stage; success is fixture logic, not model quality."""
    from httpx import ASGITransport, AsyncClient
    from rilo.learning.data import snapshot_project
    from rilo.server import create_app

    class SyntheticTeacher(PhysicalResetPolicy):
        async def decide(self, obs, spec, context, history):
            if obs.info["stage"] == "reset":
                return await super().decide(obs, spec, context, history)
            return Decision(observation_id=obs.id, status="continue",
                            actions=[ActionCommand(values=[.1])])

    adapter = ThreeTickAdapter()
    settings = Settings(backend="demo", suite="demo", policy="scripted", task_id=0,
                        data_dir=tmp_path / "runs")
    runner = Runner(settings, adapter_factory=lambda: adapter, policy_factory=SyntheticTeacher)
    # Patch only the teacher transport, leaving engine, learners, checkpoints,
    # datasets, runner and HTTP ownership routes fully connected.
    monkeypatch.setattr("rilo.pipeline.make_policy", lambda settings: SyntheticTeacher())
    app = create_app(settings, runner=runner)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/api/projects", json={"name": "Synthetic six-stage smoke",
            "backend": "demo", "suite": "demo", "task_id": 0,
            "run_config": {"max_steps": 4, "max_decisions": 4, "max_reset_steps": 6}})
        assert created.status_code == 201, created.text
        project = created.json()["project"]
        project_id = project["id"]
        prefix = f"/api/projects/{project_id}/pipeline"
        configured = await client.post(prefix + "/config", json={
            "collection_episodes": 2, "evaluation_episodes": 2, "rl_episodes_per_round": 1,
            "max_bc_cycles": 1, "max_rl_rounds": 1, "bc_success_target": 100,
            "rl_success_target": 100, "device": "cpu",
            "bc": {"image_size": 16, "feature_dim": 16, "keypoints": 4, "horizon": 4,
                "action_steps": 2, "unet_dims": [8, 16, 32], "diffusion_steps": 4,
                "inference_steps": 2, "backbone": "tiny", "imagenet_init": False,
                "steps": 1, "batch_size": 2, "eval_every": 1, "validation_batches": 1,
                "min_steps": 0},
            "progress": {"steps": 1, "batch_size": 2, "history_steps": 2, "hidden_dim": 8,
                "eval_every": 1},
            "residual": {"hidden_dim": 16, "hidden_layers": 1, "num_critics": 2,
                "target_subset": 2, "batch_size": 4, "critic_warmup_updates": 0,
                "learning_starts": 1, "buffer_capacity": 32, "utd_ratio": 1}})
        assert configured.status_code == 200, configured.text
        assert not app.state.pipeline.active and adapter.reset_calls == 0
        started = await client.post(prefix + "/start", json={"mode": "auto"})
        assert started.status_code == 200, started.text
        try:
            await asyncio.wait_for(app.state.pipeline._task, timeout=30)
            state = (await client.get(prefix)).json()
            assert state["status"]["phase"] == "completed", state["status"]
            assert [item["kind"] for item in state["models"]] == ["bc", "progress", "residual"]
            assert set(state["status"]["stage_metrics"]) == {"collect", "bc", "bc_eval", "progress", "rl", "rl_eval"}
            assert len(state["evaluations"]) == 2
            assert adapter.reset_calls == 1  # Initial setup only across every stage.
            dataset = snapshot_project(settings.data_dir, project_id, image_size=16)
            assert len(dataset.episodes) == 7
            assert all(len(ep.actions) == 3 for ep in dataset.episodes)
            trainer = app.state.pipeline._trainer
            assert trainer.online_steps == 3 and len(trainer.replay.online) == 3
            assert len(trainer.replay.offline) == 12  # Collect + BC eval, excluding all reset ticks.
            assert trainer.agent.updates > 0
            assert trainer.last_metrics["sampled_offline_fraction"] == .5
            for model in state["models"]:
                assert (await client.get(model["checkpoint_url"])).status_code == 200
            assert runner.stats["api_requests"] == 0
        finally:
            await app.state.pipeline.close()
            await runner.close()
