"""Collect and restore cycles must preserve physics and account for API work."""
import asyncio
import json
import time

import numpy as np
import pytest

from rilo.config import Settings
from rilo.contracts import ActionCommand, ActionSpec, Decision, Observation
from rilo.runner import RunConfig, Runner


async def until(predicate, timeout=3):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(poll(), timeout)


class ReversibleAdapter:
    """A real stateful test world: only step can move it between 0 and 1."""
    action_spec = ActionSpec(names=("dx",), low=(-1.,), high=(1.,),
                             description="One dimensional displacement", control_hz=1000)

    def __init__(self):
        self.position = 0.
        self.stage = "task"
        self.sequence = 0
        self.reset_calls = 0
        self.boundaries = []
        self.executed = []

    def reset(self, *, seed, task_id):
        self.reset_calls += 1
        self.position = 0.
        self.stage = "task"
        return self.observe()

    def begin_reset(self):
        self.boundaries.append(("reset", self.position))
        self.stage = "reset"
        return self.observe()

    def begin_task(self):
        self.boundaries.append(("task", self.position))
        self.stage = "task"
        return self.observe()

    def describe(self):
        return {"stage": self.stage, "simulation": True}

    def observe(self):
        self.sequence += 1
        target = 1. if self.stage == "task" else 0.
        success = self.position == target
        return Observation(id=str(self.sequence), timestamp=time.time(),
                           task="Move to one" if self.stage == "task" else "Return to zero",
                           state={"position": self.position},
                           images={"overview": np.zeros((8, 8, 3), dtype=np.uint8)},
                           success=success, terminated=success, info={"stage": self.stage})

    def step(self, values):
        self.executed.append((self.stage, tuple(values)))
        self.position += self.action_spec.validate(values)[0]
        return self.observe()

    def pause(self):
        pass

    def close(self):
        pass


class RoundTripPolicy:
    def __init__(self, reset_status="continue", delay_reset=False):
        self.reset_status = reset_status
        self.delay_reset = delay_reset
        self.reset_entered = asyncio.Event()
        self.release_reset = asyncio.Event()
        self.reset_cancelled = asyncio.Event()
        self.calls = []

    async def decide(self, observation, spec, context, history):
        resetting = observation.info["stage"] == "reset"
        self.calls.append((observation.info["stage"], observation.id))
        if resetting and self.delay_reset:
            self.reset_entered.set()
            try:
                await self.release_reset.wait()
            except asyncio.CancelledError:
                self.reset_cancelled.set()
                raise
        status = self.reset_status if resetting else "continue"
        return Decision(observation_id=observation.id, status=status,
                        actions=[ActionCommand(values=[-1. if resetting else 1.])] if status == "continue" else [],
                        note="Restore the state" if resetting else "Reach the goal")

    async def close(self):
        pass


def make_runner(tmp_path, *, adapter=None, policy=None):
    adapter, policy = adapter or ReversibleAdapter(), policy or RoundTripPolicy()
    settings = Settings(backend="demo", policy="scripted", data_dir=tmp_path)
    return Runner(settings, adapter_factory=lambda: adapter, policy_factory=lambda: policy), adapter, policy


async def test_agent_cycles_restore_with_actions_including_final_episode(tmp_path):
    runner, adapter, policy = make_runner(tmp_path)
    try:
        await runner.initialize(RunConfig(reset_mode="agent", episodes=2, seed=13))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert adapter.reset_calls == 1
        assert adapter.position == 0
        assert adapter.boundaries == [("reset", 1.), ("task", 0.), ("reset", 1.)]
        assert adapter.executed == [("task", (1.,)), ("reset", (-1.,))] * 2
        assert [stage for stage, _ in policy.calls] == ["task", "reset", "task", "reset"]
        assert runner.stats["successes"] == runner.stats["reset_successes"] == 2
        assert runner.stats["reset_attempts"] == 2
        assert runner.stats["hard_resets"] == 0
        assert runner.episode["steps"] == runner.episode["reset_steps"] == 1
        summaries = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*/episode-*/summary.json"))]
        assert len(summaries) == 2
        assert all(item["outcome"] == item["reset_outcome"] == "success" for item in summaries)
        records = [json.loads(line) for path in sorted(tmp_path.glob("*/episode-*/transitions.jsonl"))
                   for line in path.read_text().splitlines()]
        assert sum(r["kind"] == "environment_reset" for r in records) == 1
        assert sum(r["kind"] == "agent_reset_continuation" for r in records) == 1
        assert [r["seed"] for r in records if r["kind"] in ("environment_reset", "agent_reset_continuation")] == [13, 13]
        assert [r["stage"] for r in records if r["kind"] == "transition"] == ["task", "reset", "task", "reset"]
    finally:
        await runner.close()


@pytest.mark.parametrize("reset_status", ["failure", "help", "success"])
async def test_failed_or_unverified_reset_stops_without_hard_reset(tmp_path, reset_status):
    runner, adapter, _ = make_runner(tmp_path, policy=RoundTripPolicy(reset_status=reset_status))
    try:
        await runner.initialize(RunConfig(reset_mode="agent", episodes=3))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert adapter.reset_calls == 1
        assert adapter.position == 1
        assert adapter.boundaries == [("reset", 1.)]
        assert runner.stats["successes"] == 1
        assert runner.stats["reset_successes"] == runner.stats["hard_resets"] == 0
        assert runner.episode["reset_outcome"] == ("unverified_success" if reset_status == "success" else reset_status)
    finally:
        await runner.close()


async def test_pause_during_reset_preserves_stage_and_resumes_fresh_observation(tmp_path):
    policy = RoundTripPolicy(delay_reset=True)
    runner, adapter, _ = make_runner(tmp_path, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="agent"))
        await runner.start()
        await asyncio.wait_for(policy.reset_entered.wait(), 2)
        await runner.pause()
        await asyncio.wait_for(policy.reset_cancelled.wait(), 2)
        assert runner.phase == "paused" and runner.stage == "reset"
        assert runner.episode["outcome"] == "success"
        assert runner.episode["reset_outcome"] is None
        assert adapter.position == 1 and adapter.reset_calls == 1
        interrupted_observation = policy.calls[-1][1]
        policy.release_reset.set()
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert policy.calls[-1][1] != interrupted_observation
        assert adapter.position == 0 and adapter.reset_calls == 1
        assert runner.stats["reset_successes"] == 1
    finally:
        policy.release_reset.set()
        await runner.close()


async def test_single_decision_pauses_before_reset_and_final_reset_completes(tmp_path):
    runner, adapter, policy = make_runner(tmp_path)
    try:
        await runner.initialize(RunConfig(reset_mode="agent"))
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        assert runner.stage == "task" and adapter.position == 1
        assert adapter.boundaries == [] and len(policy.calls) == 1
        assert runner.episode["outcome"] == "success"
        await runner.start(single=True)
        await until(lambda: runner.phase == "completed")
        assert runner.stage == "reset" and adapter.position == 0
        assert adapter.reset_calls == 1 and len(policy.calls) == 2
        assert runner.episode["reset_outcome"] == "success"
    finally:
        await runner.close()


async def test_reset_step_budget_stops_in_place_without_environment_reset(tmp_path):
    class SmallResetPolicy(RoundTripPolicy):
        async def decide(self, observation, *args):
            decision = await super().decide(observation, *args)
            if observation.info["stage"] == "reset":
                decision.actions[0].values = [-0.1]
            return decision
    runner, adapter, _ = make_runner(tmp_path, policy=SmallResetPolicy())
    try:
        await runner.initialize(RunConfig(reset_mode="agent", episodes=3, max_reset_steps=1))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.episode["reset_outcome"] == "step_budget"
        assert runner.episode["reset_steps"] == 1
        assert adapter.position == pytest.approx(0.9)
        assert adapter.reset_calls == 1
        assert runner.stats["hard_resets"] == runner.stats["reset_successes"] == 0
    finally:
        await runner.close()


class MeteredPolicy(RoundTripPolicy):
    def __init__(self, invalid=None):
        super().__init__()
        self.usage_callback = None
        self.invalid = invalid

    async def decide(self, observation, *args):
        self.usage_callback({
            "attempt_id": f"call-{len(self.calls)}", "response_id": "mock-response",
            "tokens": {"input_tokens": 100, "output_tokens": 40, "cached_input_tokens": 50,
                       "reasoning_tokens": 20, "total_tokens": 140},
            "estimated_cost_usd": 0.125, "cost_unknown": False,
        })
        if self.invalid == "parse":
            raise ValueError("Malformed policy response")
        decision = await super().decide(observation, *args)
        decision.usage = {"input_tokens": 100, "output_tokens": 40}
        if self.invalid == "stale":
            decision.observation_id = "stale"
        return decision


async def test_usage_callback_is_counted_once_and_attributed_to_each_stage(tmp_path):
    runner, _, _ = make_runner(tmp_path, policy=MeteredPolicy())
    try:
        await runner.initialize(RunConfig(reset_mode="agent"))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.stats["api_requests"] == 2
        assert runner.stats["input_tokens"] == 200
        assert runner.stats["output_tokens"] == 80
        assert runner.stats["cached_input_tokens"] == 100
        assert runner.stats["reasoning_tokens"] == 40
        assert runner.stats["total_tokens"] == 280
        assert runner.stats["estimated_cost_usd"] == pytest.approx(0.25)
        for stage in ("task", "reset"):
            assert runner.episode["costs"][stage]["api_requests"] == 1
            assert runner.episode["costs"][stage]["estimated_cost_usd"] == pytest.approx(0.125)
        calls = [json.loads(line) for line in next(tmp_path.glob("*/api_calls.jsonl")).read_text().splitlines()]
        assert [call["stage"] for call in calls] == ["task", "reset"]
        assert all(call["episode_id"] == "episode-0001" for call in calls)
    finally:
        await runner.close()


@pytest.mark.parametrize("invalid", ["parse", "stale"])
async def test_invalid_response_usage_is_saved_even_without_action(tmp_path, invalid):
    runner, adapter, _ = make_runner(tmp_path, policy=MeteredPolicy(invalid=invalid))
    try:
        await runner.initialize(RunConfig(reset_mode="agent"))
        await runner.start()
        await until(lambda: runner.phase == "error")
        assert adapter.executed == []
        assert runner.stats["api_requests"] == 1
        assert runner.stats["input_tokens"] == 100
        assert runner.stats["estimated_cost_usd"] == pytest.approx(0.125)
        calls = [json.loads(line) for line in next(tmp_path.glob("*/api_calls.jsonl")).read_text().splitlines()]
        assert len(calls) == 1 and calls[0]["stage"] == "task"
        summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
        assert summary["stats"]["estimated_cost_usd"] == pytest.approx(0.125)
    finally:
        await runner.close()
