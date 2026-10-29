"""Contract tests for the remote policy; never make paid or live API requests."""
from __future__ import annotations

import asyncio
import base64
import io
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import httpx
from openai import BadRequestError
from PIL import Image
import pytest

from rilo.contracts import ActionSpec, Observation
from rilo.policies import openai_policy as module
from rilo.policies.openai_policy import OpenAIPolicy, PolicyError


@pytest.fixture
def observation():
    return Observation(
        id="episode-3:12", timestamp=123.0, task="Place the object in the tray",
        state={"position": [0.1, 0.2], "gripper": 0.0},
        images={"overhead": np.full((16, 24, 3), [190, 30, 10], dtype=np.uint8),
                "wrist": np.full((10, 10, 3), [10, 30, 190], dtype=np.uint8)},
    )


@pytest.fixture
def spec():
    # A deliberately non-LIBERO action space verifies that the policy is generic.
    return ActionSpec(names=("x", "grip"), low=(-0.5, -1.0), high=(0.5, 1.0),
                      description="x is metres in the base frame; grip -1 opens, +1 closes",
                      control_hz=20, max_repeat=3, max_chunk_steps=5)


def payload(**changes):
    value = {"observation_id": "episode-3:12", "status": "continue",
             "actions": [{"values": [0.1, -1.0], "repeat": 2}], "note": "Approach the tray.",
             "next_reasoning_effort": None}
    value.update(changes)
    return value


def response(arguments=None, **changes):
    value = dict(status="completed", id="resp-test", usage=SimpleNamespace(input_tokens=50, output_tokens=20, total_tokens=70),
                 output=[SimpleNamespace(type="function_call", name="robot_action",
                                         arguments=arguments if arguments is not None else json.dumps(payload()))])
    value.update(changes)
    return SimpleNamespace(**value)


@pytest.fixture
def client(monkeypatch):
    fake = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response())), close=AsyncMock())
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: fake)
    return fake


def make_policy(**kwargs):
    return OpenAIPolicy("test-model", api_key="test-secret-key", **kwargs)


async def test_request_has_labeled_real_images_state_and_environment_contract(client, observation, spec):
    policy = make_policy(max_history=2)
    history = [{"step": 1}, {"step": 2, "executed_steps": 1}, {"step": 3, "interrupted": True}]
    result = await policy.decide(observation, spec, {"robot": "test-arm"}, history)
    request = client.responses.create.call_args.kwargs
    assert request["store"] is False
    assert request["parallel_tool_calls"] is False
    assert request["tool_choice"] == {"type": "function", "name": "robot_action"}
    assert "test-arm" in request["instructions"]
    assert "base frame" in request["instructions"]
    content = request["input"][0]["content"]
    assert '"observation_id": "episode-3:12"' in content[0]["text"]
    assert '"step": 1' not in content[0]["text"]
    assert '"step": 2' in content[0]["text"]
    assert '"interrupted": true' in content[0]["text"]
    frames = [part for part in content if part["type"] == "input_image"]
    assert len(frames) == 2
    image = Image.open(io.BytesIO(base64.b64decode(frames[0]["image_url"].split(",", 1)[1])))
    assert image.size == (24, 16)
    assert image.getpixel((0, 0))[0] > image.getpixel((0, 0))[2]
    assert "overhead" in content[1]["text"]
    assert "wrist" in content[3]["text"]
    assert result.actions[0].values == [0.1, -1.0]
    assert result.usage == {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
    assert result.response_id == "resp-test"


def test_schema_matches_adapter_and_zero_history_is_empty(client, observation, spec):
    request = make_policy(max_history=0, reasoning_effort=None).build_request(observation, spec, {}, [{"secret_old_step": 1}])
    tool = request["tools"][0]
    assert tool["strict"] is True
    schema = tool["parameters"]
    assert set(schema["required"]) == set(schema["properties"])
    values = schema["properties"]["actions"]["items"]["properties"]["values"]
    assert values["minItems"] == values["maxItems"] == 2
    assert "reasoning" not in request
    assert "secret_old_step" not in str(request)


@pytest.mark.parametrize("changes", [
    {"observation_id": "old-frame"},
    {"actions": []},
    {"actions": [{"values": [0.1], "repeat": 1}]},
    {"actions": [{"values": [0.6, 0.0], "repeat": 1}]},
    {"actions": [{"values": [True, 0.0], "repeat": 1}]},
    {"actions": [{"values": ["0.1", 0.0], "repeat": 1}]},
    {"actions": [{"values": [0.1, 0.0], "repeat": True}]},
    {"actions": [{"values": [0.1, 0.0], "repeat": 4}]},
    {"actions": [{"values": [0.1, 0.0], "repeat": 3}] * 2},
    {"actions": [{"values": [0.1, 0.0], "repeat": 1, "extra": 2}]},
    {"status": "success"},
    {"status": "invalid"},
    {"usage": {"input_tokens": 999}},
])
async def test_rejects_unsafe_or_invalid_action_without_execution(client, observation, spec, changes):
    client.responses.create.return_value = response(json.dumps(payload(**changes)))
    with pytest.raises(PolicyError):
        await make_policy().decide(observation, spec, {}, [])


@pytest.mark.parametrize("arguments", [
    "not json", "[]", "{}",
    '{"observation_id":"episode-3:12","status":"continue","actions":[{"values":[NaN,0],"repeat":1}],"note":""}',
    '{"observation_id":"episode-3:12","status":"continue","status":"success","actions":[],"note":""}',
])
async def test_rejects_malformed_nonfinite_or_ambiguous_json(client, observation, spec, arguments):
    client.responses.create.return_value = response(arguments)
    with pytest.raises(PolicyError):
        await make_policy().decide(observation, spec, {}, [])


@pytest.mark.parametrize("result", [
    response(status="incomplete"),
    response(output=[]),
    response(output=[SimpleNamespace(type="function_call", name="shell", arguments="{}")]),
    response(output=response().output * 2),
    response(output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="No")])]),
])
async def test_no_action_accepted_for_incomplete_refusal_or_extra_calls(client, observation, spec, result):
    client.responses.create.return_value = result
    with pytest.raises(PolicyError):
        await make_policy().decide(observation, spec, {}, [])


@pytest.mark.parametrize("status", ["success", "failure", "help"])
async def test_terminal_claims_return_without_motion(client, observation, spec, status):
    client.responses.create.return_value = response(json.dumps(payload(status=status, actions=[])))
    result = await make_policy().decide(observation, spec, {}, [])
    assert result.status == status and result.actions == []
    # A success claim does not silently change the independent environment result.
    assert observation.success is False


async def test_cancellation_propagates_and_no_retry_occurs(client, observation, spec):
    entered = asyncio.Event()

    async def waiting(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    client.responses.create.side_effect = waiting
    pending = asyncio.create_task(make_policy().decide(observation, spec, {}, []))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert client.responses.create.await_count == 1


async def test_sdk_errors_do_not_expose_secrets_or_request_content(client, observation, spec):
    client.responses.create.side_effect = RuntimeError("Authorization: test-secret-key; full request body")
    with pytest.raises(PolicyError) as error:
        await make_policy().decide(observation, spec, {}, [])
    assert "test-secret-key" not in str(error.value)
    assert "request body" not in str(error.value)


def test_missing_key_fails_before_constructing_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    constructor = AsyncMock()
    monkeypatch.setattr(module, "AsyncOpenAI", constructor)
    with pytest.raises(PolicyError, match="OPENAI_API_KEY"):
        OpenAIPolicy("test-model")
    constructor.assert_not_called()


async def test_invalid_observation_never_calls_api(client, observation, spec):
    observation.state["position"] = [float("nan")]
    with pytest.raises(PolicyError):
        await make_policy().decide(observation, spec, {}, [])
    client.responses.create.assert_not_called()


async def test_close_releases_http_client(client):
    await make_policy().close()
    client.close.assert_awaited_once()


async def test_usage_callback_meters_rejected_incomplete_response(client, observation, spec):
    events = []
    raw = {"input_tokens": 1000, "output_tokens": 300,
           "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 200},
           "output_tokens_details": {"reasoning_tokens": 250}, "total_tokens": 1300}
    client.responses.create.return_value = response(status="incomplete", model="gpt-6-astra", usage=raw,
                                                    service_tier="default")
    with pytest.raises(PolicyError, match="incomplete"):
        await OpenAIPolicy("gpt-6-astra", api_key="test-secret-key", usage_callback=events.append).decide(observation, spec, {}, [])
    assert len(events) == 1
    event = events[0]
    assert event["usage"] == raw
    assert event["outcome"] == "invalid_response"
    assert event["estimated_cost_usd"] == pytest.approx(0.0246)
    assert event["tokens"]["reasoning_tokens"] == 250
    assert event["elapsed_seconds"] >= 0
    assert client.responses.create.call_args.kwargs["service_tier"] == "default"
    assert "test-secret-key" not in json.dumps(event)


async def test_cancellation_and_network_errors_have_unknown_cost(client, observation, spec):
    for error, expected in [(asyncio.CancelledError(), "cancelled"), (RuntimeError("test-secret-key"), "error")]:
        events = []
        client.responses.create.side_effect = error
        with pytest.raises((asyncio.CancelledError, PolicyError)):
            await make_policy(usage_callback=events.append).decide(observation, spec, {}, [])
        assert len(events) == 1
        assert events[0]["outcome"] == expected
        assert events[0]["estimated_cost_usd"] is None
        assert events[0]["usage"] is None
        assert events[0]["cost_unknown"] is True
        assert "test-secret-key" not in json.dumps(events)


async def test_callback_is_bound_to_original_request_even_if_owner_changes(client, observation, spec):
    started, release = asyncio.Event(), asyncio.Event()
    old_events, new_events = [], []
    async def waiting(**kwargs):
        started.set()
        await release.wait()
        return response()
    client.responses.create.side_effect = waiting
    policy = make_policy(usage_callback=old_events.append)
    task = asyncio.create_task(policy.decide(observation, spec, {}, []))
    await started.wait()
    policy.usage_callback = new_events.append
    release.set()
    await task
    assert len(old_events) == 1
    assert new_events == []


async def test_http_error_retains_only_safe_diagnostic_fields(client, observation, spec):
    events = []
    client.responses.create.side_effect = BadRequestError(
        "message includes test-secret-key and full request", response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
        body={"code": "unsupported_parameter", "param": "reasoning.effort", "message": "test-secret-key"},
    )
    with pytest.raises(PolicyError) as error:
        await make_policy(usage_callback=events.append).decide(observation, spec, {}, [])
    assert events[0]["error_code"] == "unsupported_parameter"
    assert events[0]["error_param"] == "reasoning.effort"
    assert events[0]["http_status"] == 400
    assert "unsupported_parameter" in str(error.value)
    assert "test-secret-key" not in str(error.value) + json.dumps(events)


def test_history_compaction_preserves_current_calibration_and_does_not_mutate_recording(client, observation, spec):
    observation.state["camera_calibration"] = {"wrist": {"camera_to_world": [[0.123456789]]}}
    history = [{"state_before": observation.state, "state_after": observation.state,
                "actions": [{"values": [0.123456789], "repeat": 2}], "executed_steps": 1}]
    request = make_policy().build_request(observation, spec, {}, history)
    text = request["input"][0]["content"][0]["text"]
    old, current = text.split("CURRENT observation:")
    assert "camera_calibration" not in old
    assert "camera_calibration" in current
    assert "0.12346" in text
    assert observation.state["camera_calibration"]["wrist"]["camera_to_world"] == [[0.123456789]]


async def test_meter_persistence_failure_stops_before_returning_action(client, observation, spec):
    def broken(event):
        raise OSError("disk failure test-secret-key")
    policy = make_policy(usage_callback=broken)
    with pytest.raises(PolicyError, match="record API usage"):
        await policy.decide(observation, spec, {}, [])
    assert policy.last_usage_event["outcome"] == "accepted"


async def test_prompt_override_is_hot_reloaded_but_inflight_snapshot_is_immutable(client, observation, spec, tmp_path):
    path = tmp_path / "control.md"
    path.write_text("Use the first approach.", encoding="utf-8")
    events = []
    policy = make_policy(prompt_file=path, usage_callback=events.append)
    started, release = asyncio.Event(), asyncio.Event()
    async def waiting(**kwargs):
        started.set()
        await release.wait()
        return response()
    client.responses.create.side_effect = waiting
    first = asyncio.create_task(policy.decide(observation, spec, {"stage": "task"}, []))
    await started.wait()
    path.write_text("Use the revised approach.", encoding="utf-8")
    release.set()
    await first
    original_request = client.responses.create.call_args.kwargs
    assert "Use the first approach." in original_request["instructions"]
    assert "Use the revised approach." not in original_request["instructions"]
    assert events[0]["prompt_template"] == module._INSTRUCTIONS + module._FIXED_INSTRUCTIONS + "\nOperator control guidance:\nUse the first approach."
    assert events[0]["prompt_sha256"] == hashlib.sha256(events[0]["prompt_template"].encode()).hexdigest()
    client.responses.create.side_effect = None
    await policy.decide(observation, spec, {"stage": "reset"}, [])
    assert "Use the revised approach." in client.responses.create.call_args.kwargs["instructions"]
    assert events[1]["prompt_sha256"] != events[0]["prompt_sha256"]
    assert "Environment description:" not in events[1]["prompt_template"]


@pytest.mark.parametrize("contents", [None, b"\xff", b"test-secret-key"])
async def test_missing_invalid_or_credential_prompt_stops_before_api(client, observation, spec, tmp_path, contents):
    path = tmp_path / "control.md"
    if contents is not None:
        path.write_bytes(contents)
    events = []
    with pytest.raises(PolicyError) as error:
        await make_policy(prompt_file=path, usage_callback=events.append).decide(observation, spec, {}, [])
    assert "test-secret-key" not in str(error.value)
    client.responses.create.assert_not_called()
    assert events == []


def test_prompt_file_can_be_configured_with_environment(client, observation, spec, monkeypatch, tmp_path):
    path = tmp_path / "guidance.md"
    path.write_text("Keep the reset path clear.", encoding="utf-8")
    monkeypatch.setenv("RILO_PROMPT_FILE", str(path))
    assert "Keep the reset path clear." in make_policy().build_request(observation, spec, {}, [])["instructions"]


async def test_auto_escalates_without_motion_then_returns_to_medium(client, observation, spec):
    events = []
    policy = make_policy(reasoning_effort="auto", usage_callback=events.append)
    client.responses.create.side_effect = [
        response(json.dumps(payload(status="reconsider", actions=[], next_reasoning_effort="xhigh",
            note="The next operation requires precise contact alignment."))),
        response(json.dumps(payload(next_reasoning_effort="medium",
            note="Measured contact is stable; the following motion has adequate clearance."))),
        response(json.dumps(payload(next_reasoning_effort="medium"))),
    ]
    first = await policy.decide(observation, spec, {}, [])
    assert first.status == "reconsider" and first.actions == []
    assert first.reasoning_effort == "medium"
    assert policy.current_reasoning_effort == "medium"  # Runner must first accept it.
    policy.accept_decision(first)
    assert policy.current_reasoning_effort == "xhigh"
    second = await policy.decide(observation, spec, {}, [])
    assert second.actions and second.reasoning_effort == "xhigh"
    assert policy.current_reasoning_effort == "xhigh"
    policy.accept_decision(second)
    third = await policy.decide(observation, spec, {}, [])
    assert third.reasoning_effort == "medium"
    assert [call.kwargs["reasoning"]["effort"] for call in client.responses.create.call_args_list] == ["medium", "xhigh", "medium"]
    assert [event["reasoning_effort"] for event in events] == ["medium", "xhigh", "medium"]
    assert all(event["reasoning_mode"] == "auto" for event in events)
    assert all(event["outcome"] == "accepted" for event in events)


@pytest.mark.parametrize("effort,changes", [
    ("auto", {"next_reasoning_effort": None}),
    ("auto", {"next_reasoning_effort": "high"}),
    ("auto", {"next_reasoning_effort": "xhigh"}),  # Must not move before escalation.
    ("auto", {"status": "reconsider", "actions": [], "next_reasoning_effort": "medium"}),
    ("auto", {"status": "reconsider", "next_reasoning_effort": "xhigh"}),
    ("medium", {"status": "reconsider", "actions": [], "next_reasoning_effort": "xhigh"}),
    ("medium", {"next_reasoning_effort": "medium"}),
    ("medium", {"failure_assessment": {"result": "none", "evidence": ""}}),  # Removed protocol field.
    ("medium", {"reasoning_effort": "xhigh"}),  # Actual effort is assigned locally.
])
async def test_invalid_adaptive_arguments_are_rejected(client, observation, spec, effort, changes):
    policy = make_policy(reasoning_effort=effort)
    client.responses.create.return_value = response(json.dumps(payload(**changes)))
    with pytest.raises(PolicyError):
        await policy.decide(observation, spec, {}, [])
    assert policy.current_reasoning_effort == "medium"


async def test_unaccepted_late_response_cannot_change_auto_effort(client, observation, spec):
    started, release = asyncio.Event(), asyncio.Event()
    events = []
    async def late(**kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return response(json.dumps(payload(status="reconsider", actions=[], next_reasoning_effort="xhigh")))
    client.responses.create.side_effect = late
    policy = make_policy(reasoning_effort="auto", usage_callback=events.append)
    task = asyncio.create_task(policy.decide(observation, spec, {}, []))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    discarded = await task
    assert discarded.next_reasoning_effort == "xhigh"
    assert policy.current_reasoning_effort == "medium"
    assert events[0]["reasoning_effort"] == "medium"
    assert events[0]["next_reasoning_effort"] == "xhigh"


async def test_auto_resets_at_episode_and_stage_boundaries(client, observation, spec):
    policy = make_policy(reasoning_effort="auto")
    client.responses.create.return_value = response(json.dumps(payload(status="reconsider", actions=[], next_reasoning_effort="xhigh")))
    for reset in (lambda: policy.on_episode_start("episode-2"), lambda: policy.on_stage_start("reset")):
        decision = await policy.decide(observation, spec, {}, [])
        policy.accept_decision(decision)
        assert policy.current_reasoning_effort == "xhigh"
        reset()
        assert policy.current_reasoning_effort == "medium"


async def test_xhigh_cannot_request_another_reconsideration(client, observation, spec):
    policy = make_policy(reasoning_effort="auto")
    client.responses.create.return_value = response(json.dumps(payload(status="reconsider", actions=[], next_reasoning_effort="xhigh")))
    policy.accept_decision(await policy.decide(observation, spec, {}, []))
    with pytest.raises(PolicyError, match="medium-to-xhigh"):
        await policy.decide(observation, spec, {}, [])


def test_adaptive_prompt_is_task_independent_without_per_decision_classification(client, observation, spec):
    policy = make_policy(reasoning_effort="auto")
    request = policy.build_request(observation, spec, {}, [])
    prompt = policy._prompt_template()
    assert "precision" in prompt and "stable" in prompt and "uncertain" in prompt
    assert "failure_assessment" not in str(request)
    assert "hidden chain-of-thought" in prompt
    assert observation.task not in prompt
    assert "drawer" not in prompt and "handle" not in prompt
    assert request["reasoning"]["effort"] == "medium"
    assert '"current_reasoning_effort": "medium"' in request["input"][0]["content"][0]["text"]


def test_removed_classification_is_not_reintroduced_by_old_receipt_history(client, observation, spec):
    history = [{"executed_steps": 1, "note": "Motion receipt", "failure_assessment": {"result": "failure", "evidence": "Old label"}}]
    request = make_policy().build_request(observation, spec, {}, history)
    assert "failure_assessment" not in str(request) and "Old label" not in str(request)
    assert '"executed_steps": 1' in request["input"][0]["content"][0]["text"]
    assert history[0]["failure_assessment"]["result"] == "failure"
