import asyncio
import json
import threading
import time

import pytest

from rilo.adapters.demo import DemoAdapter
from rilo.config import Settings
from rilo.contracts import ActionCommand, Decision
from rilo.runner import Runner, RunConfig, StateConflict


async def until(predicate, timeout=3):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(poll(), timeout)


class CountingAdapter(DemoAdapter):
    def __init__(self, slow=False):
        super().__init__(64)
        self.calls = []
        self.tick_started = threading.Event()
        self.slow = slow

    def reset(self, **kwargs):
        self.calls.append(("reset", threading.get_ident()))
        return super().reset(**kwargs)

    def step(self, values):
        self.calls.append(("step", threading.get_ident()))
        self.tick_started.set()
        if self.slow:
            time.sleep(0.12)
        obs = super().step(values)
        obs.success = obs.terminated = False
        return obs

    def pause(self):
        self.calls.append(("pause", threading.get_ident()))

    def close(self):
        self.calls.append(("close", threading.get_ident()))


class FixedPolicy:
    def __init__(self, repeat=10, status="continue", stale=False):
        self.repeat = repeat
        self.status = status
        self.stale = stale
        self.calls = 0

    async def decide(self, observation, spec, context, history):
        self.calls += 1
        return Decision(observation_id="old" if self.stale else observation.id, status=self.status,
                        actions=[ActionCommand(values=[0.1, 0.1], repeat=self.repeat)] if self.status == "continue" else [],
                        note="Test decision")

    async def close(self):
        pass


class DelayedPolicy(FixedPolicy):
    def __init__(self, ignore_cancel=False):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.ignore_cancel = ignore_cancel

    async def decide(self, observation, spec, context, history):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if not self.ignore_cancel:
                raise
            await self.release.wait()
        return await super().decide(observation, spec, context, history)


def make_runner(tmp_path, adapter=None, policy=None):
    adapter = adapter or CountingAdapter()
    policy = policy or FixedPolicy()
    settings = Settings(backend="demo", policy="scripted", data_dir=tmp_path)
    return Runner(settings, adapter_factory=lambda: adapter, policy_factory=lambda: policy)


async def test_pause_interrupts_inference_and_resume_uses_new_call(tmp_path):
    policy = DelayedPolicy()
    runner = make_runner(tmp_path, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await policy.entered.wait()
        await runner.pause()
        await policy.cancelled.wait()
        assert runner.phase == "paused"
        assert runner.episode["steps"] == 0
        policy.release.set()
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused" and runner.episode["steps"] > 0)
        assert runner.episode["steps"] == 10
    finally:
        await runner.close()


async def test_pause_discards_late_response_even_if_policy_ignores_cancel(tmp_path):
    policy = DelayedPolicy(ignore_cancel=True)
    runner = make_runner(tmp_path, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await policy.entered.wait()
        await runner.pause()
        policy.release.set()
        await asyncio.sleep(0.05)
        assert runner.phase == "paused"
        assert runner.episode["steps"] == 0
    finally:
        policy.release.set()
        await runner.close()


async def test_pause_during_step_waits_for_one_tick_and_drops_suffix(tmp_path):
    adapter = CountingAdapter(slow=True)
    runner = make_runner(tmp_path, adapter=adapter)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await until(adapter.tick_started.is_set)
        await runner.pause()
        steps = runner.episode["steps"]
        await asyncio.sleep(0.2)
        assert runner.phase == "paused"
        assert steps == runner.episode["steps"] == 1
        assert runner.latest_receipt["executed_steps"] == 1
        assert runner.latest_receipt["interrupted"] is True
        assert len({thread for _, thread in adapter.calls}) == 1
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused" and runner.episode["steps"] > steps)
        assert runner.episode["steps"] == 11
    finally:
        await runner.close()


async def test_step_budget_records_every_executed_transition(tmp_path):
    runner = make_runner(tmp_path)
    try:
        await runner.initialize(RunConfig(reset_mode="environment", max_steps=3))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.episode["steps"] == 3
        assert runner.episode["outcome"] == "step_budget"
        path = next(tmp_path.glob("*/episode-*/transitions.jsonl"))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        transitions = [r for r in records if r["kind"] == "transition"]
        assert len(transitions) == 3
        for record in transitions:
            frame = next(iter(record["next_observation"]["images"].values()))
            assert (path.parent / frame).is_file()
        assert records[-1]["kind"] == "episode_end"
    finally:
        await runner.close()


async def test_model_success_claim_is_not_a_success_label(tmp_path):
    runner = make_runner(tmp_path, policy=FixedPolicy(status="success"))
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.episode["outcome"] == "unverified_success"
        assert runner.stats["successes"] == 0
        assert runner.episode["steps"] == 0
    finally:
        await runner.close()


async def test_stale_action_never_reaches_adapter(tmp_path):
    runner = make_runner(tmp_path, policy=FixedPolicy(stale=True))
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await until(lambda: runner.phase == "error")
        assert runner.episode["steps"] == 0
        assert "stale" in runner.error
    finally:
        await runner.close()


async def test_batch_resets_between_bounded_episodes(tmp_path):
    runner = make_runner(tmp_path)
    try:
        await runner.initialize(RunConfig(reset_mode="environment", max_steps=1, episodes=2))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.stats["episodes"] == 2
        assert runner.stats["total_steps"] == 2
        assert len(list(tmp_path.glob("*/episode-*/summary.json"))) == 2
    finally:
        await runner.close()


async def test_stop_while_paused_prevents_restart_without_reset(tmp_path):
    runner = make_runner(tmp_path)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        await runner.stop()
        assert runner.phase == "stopped"
        with pytest.raises(StateConflict):
            await runner.start()
        await runner.initialize(RunConfig(reset_mode="environment"))
        assert runner.phase == "ready"
    finally:
        await runner.close()


async def test_single_terminal_decision_does_not_automatically_start_next_episode(tmp_path):
    policy = FixedPolicy(status="failure")
    runner = make_runner(tmp_path, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="environment", episodes=2))
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        assert policy.calls == runner.stats["episodes"] == 1
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert policy.calls == runner.stats["episodes"] == 2
    finally:
        await runner.close()


async def test_pause_on_terminal_tick_can_resume_remaining_batch(tmp_path):
    class TerminalAdapter(CountingAdapter):
        def step(self, values):
            obs = super().step(values)
            obs.success = obs.terminated = True
            return obs
    adapter = TerminalAdapter(slow=True)
    runner = make_runner(tmp_path, adapter=adapter)
    try:
        await runner.initialize(RunConfig(reset_mode="environment", episodes=2))
        await runner.start()
        await until(adapter.tick_started.is_set)
        await runner.pause()
        assert runner.phase == "paused"
        assert runner.episode["outcome"] == "success"
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert runner.stats["episodes"] == runner.stats["successes"] == 2
    finally:
        await runner.close()


async def test_stop_does_not_wait_for_uncancellable_policy(tmp_path):
    class StubbornPolicy(DelayedPolicy):
        async def decide(self, *args):
            self.entered.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass
            raise RuntimeError("late provider error must be ignored")
    policy = StubbornPolicy()
    adapter = CountingAdapter()
    runner = make_runner(tmp_path, adapter=adapter, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        await runner.start()
        await asyncio.wait_for(policy.entered.wait(), 2)
        await asyncio.wait_for(runner.stop(), 0.5)
        assert runner.phase == "stopped"
        assert any(name == "pause" for name, _ in adapter.calls)
        policy.release.set()
        await asyncio.sleep(0.02)
        assert runner.phase == "stopped"
        assert runner.error is None
        assert runner.episode["steps"] == 0
    finally:
        policy.release.set()
        await runner.close()


async def test_resume_refreshes_observation_before_request(tmp_path):
    class ObservingAdapter(CountingAdapter):
        def observe(self):
            self.calls.append(("observe", threading.get_ident()))
            return super().observe()
    class TrackingPolicy(FixedPolicy):
        def __init__(self):
            super().__init__(repeat=1)
            self.seen = []
        async def decide(self, observation, *args):
            self.seen.append(observation.id)
            return await super().decide(observation, *args)
    adapter, policy = ObservingAdapter(), TrackingPolicy()
    runner = make_runner(tmp_path, adapter=adapter, policy=policy)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"))
        initial_id = runner.observation.id
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        paused_id = runner.observation.id
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused" and policy.calls == 2)
        assert policy.seen[0] != initial_id
        assert policy.seen[1] != paused_id
        assert len([name for name, _ in adapter.calls if name == "observe"]) == 2
    finally:
        await runner.close()
