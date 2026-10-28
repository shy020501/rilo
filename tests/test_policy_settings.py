"""Model selection must change the request factory without resetting physics."""
import asyncio
import json

import pytest
from pydantic import ValidationError

from rilo.config import Settings
from rilo.policy_settings import PolicyConfig, policy_catalog
from rilo.runner import RunConfig, Runner
from test_collection import ReversibleAdapter, RoundTripPolicy
from test_runner import CountingAdapter, FixedPolicy, until


def selection(model="gpt-6-astra", effort="medium"):
    return PolicyConfig(model=model, reasoning_effort=effort)


def test_catalog_choices_are_valid_and_unsupported_values_are_rejected():
    assert len(policy_catalog()) >= 2
    for item in policy_catalog():
        for effort in item["reasoning_efforts"]:
            assert selection(item["model"], effort).model == item["model"]
    for model, effort in (("unknown", "medium"), ("gpt-6-astra", "none"),
                          ("gpt-6.1-sol", "ultra")):
        with pytest.raises(ValidationError):
            selection(model, effort)
    with pytest.raises(ValidationError):
        PolicyConfig(model="gpt-6-astra", reasoning_effort="medium", api_key="not-accepted")


async def test_model_snapshot_reaches_factory_and_is_retained_on_resume(tmp_path, monkeypatch):
    created = []

    def factory(settings):
        policy = FixedPolicy(repeat=1)
        created.append((settings, policy))
        return policy

    monkeypatch.setattr("rilo.runner.make_policy", factory)
    defaults = Settings(backend="demo", policy="openai", model="gpt-6-astra",
                        reasoning_effort="xhigh", api_key="test-secret", data_dir=tmp_path)
    runner = Runner(defaults, adapter_factory=CountingAdapter)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"),
                                policy_config=selection("gpt-6.1-sol", "low"))
        run = runner.recorder.path
        assert runner.settings is defaults
        assert runner.status()["model"] == "gpt-6.1-sol"
        assert runner.status()["reasoning_effort"] == "low"
        assert created == []  # Preparing/configuring never calls GPT.
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused" and runner.episode["steps"] == 2)
        assert len(created) == 1
        assert created[0][0].model == "gpt-6.1-sol"
        assert created[0][0].reasoning_effort == "low"
        assert created[0][0].api_key == "test-secret"
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["config"]["model"] == "gpt-6.1-sol"
        assert manifest["config"]["reasoning_effort"] == "low"
        assert "test-secret" not in (run / "manifest.json").read_text()
        await runner.initialize(RunConfig(reset_mode="environment"))
        assert runner.policy is None
        assert runner.policy_config == {"model": "gpt-6-astra", "reasoning_effort": "xhigh"}
        await runner.start(single=True)
        await until(lambda: runner.phase == "paused")
        assert len(created) == 2
        assert created[1][0] is defaults
    finally:
        await runner.close()


async def test_verified_reset_continuation_changes_model_without_environment_reset(tmp_path, monkeypatch):
    created = []

    def factory(settings):
        created.append((settings.model, settings.reasoning_effort))
        return RoundTripPolicy()

    monkeypatch.setattr("rilo.runner.make_policy", factory)
    adapter = ReversibleAdapter()
    runner = Runner(Settings(backend="demo", policy="openai", data_dir=tmp_path),
                    adapter_factory=lambda: adapter)
    try:
        await runner.initialize(RunConfig(), policy_config=selection(effort="xhigh"))
        old_run = runner.recorder.path
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert adapter.position == 0 and adapter.reset_calls == 1
        await runner.initialize(RunConfig(), preserve_environment=True,
                                policy_config=selection("gpt-6.1-sol", "medium"))
        assert runner.phase == "ready"
        assert adapter.position == 0 and adapter.reset_calls == 1
        await runner.start()
        await until(lambda: runner.phase == "completed")
        assert created == [("gpt-6-astra", "xhigh"), ("gpt-6.1-sol", "medium")]
        assert json.loads((old_run / "manifest.json").read_text())["config"]["reasoning_effort"] == "xhigh"
        assert json.loads((runner.recorder.path / "manifest.json").read_text())["config"]["model"] == "gpt-6.1-sol"
    finally:
        await runner.close()


async def test_model_change_preserves_late_usage_and_defers_old_client_close(tmp_path, monkeypatch):
    class LateMeteredPolicy(FixedPolicy):
        usage_callback = None

        def __init__(self):
            super().__init__(repeat=1)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        async def decide(self, *args):
            callback = self.usage_callback
            self.entered.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass
            callback({"attempt_id": "late-original-model", "model": "gpt-6-astra",
                      "reasoning_effort": "xhigh", "tokens": {"input_tokens": 12, "output_tokens": 3,
                      "total_tokens": 15}, "estimated_cost_usd": .1, "cost_unknown": False})
            return await super().decide(*args)

        async def close(self):
            self.closed = True

    old_policy = LateMeteredPolicy()
    monkeypatch.setattr("rilo.runner.make_policy", lambda settings: old_policy if settings.model == "gpt-6-astra" else FixedPolicy())
    runner = Runner(Settings(backend="demo", policy="openai", data_dir=tmp_path),
                    adapter_factory=CountingAdapter)
    try:
        await runner.initialize(RunConfig(reset_mode="environment"), policy_config=selection(effort="xhigh"))
        old_path = runner.recorder.path
        await runner.start()
        await asyncio.wait_for(old_policy.entered.wait(), 2)
        await runner.pause()
        await asyncio.wait_for(runner.initialize(RunConfig(reset_mode="environment"),
                                                policy_config=selection("gpt-6.1-sol", "low")), .5)
        assert old_policy.closed is False
        assert runner.stats["api_requests"] == 0
        old_policy.release.set()
        await until(lambda: old_policy.closed)
        old_event = json.loads((old_path / "api_calls.jsonl").read_text())
        assert old_event["run_id"] == old_path.name
        assert old_event["episode_id"] == "episode-0001"
        assert old_event["model"] == "gpt-6-astra"
        assert not (runner.recorder.path / "api_calls.jsonl").exists()
        assert runner.stats["api_requests"] == runner.episode["steps"] == 0
        assert runner.phase == "ready"
    finally:
        old_policy.release.set()
        await runner.close()


async def test_orchestration_events_keep_episode_boundaries_and_record_decisions(tmp_path):
    runner = Runner(Settings(backend="demo", policy="scripted", data_dir=tmp_path),
                    adapter_factory=CountingAdapter, policy_factory=lambda: FixedPolicy(repeat=1))
    try:
        await runner.initialize(RunConfig(reset_mode="environment", max_steps=3, episodes=2))
        await runner.start()
        await until(lambda: runner.phase == "completed")
        events = list(runner.events)
        assert [e["episode_id"] for e in events if e["kind"] == "episode_start"] == ["episode-0001", "episode-0002"]
        assert len([e for e in events if e["kind"] == "episode"]) == 2
        assert not any(e["kind"] in ("policy", "decision") for e in events)
        records = [json.loads(line) for path in runner.recorder.path.glob("episode-*/transitions.jsonl")
                   for line in path.read_text().splitlines()]
        assert len([record for record in records if record["kind"] == "decision"]) == 6
    finally:
        await runner.close()
