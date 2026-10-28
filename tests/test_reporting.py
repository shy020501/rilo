"""Cost reporting regressions use local synthetic ledgers, never API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rilo.reporting import markdown_report, report_costs


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def append_events(path: Path, *events: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "measured-run"
    write_json(path / "manifest.json", {
        "run_id": path.name, "created_at": 1,
        "config": {"model": "gpt-6-astra", "policy": "openai", "reasoning_effort": "xhigh",
                   "reset_mode": "agent", "source_sha256": "initial-source", "max_output_tokens": 16384,
                   "backend": "libero", "suite": "libero_goal", "task_id": 0},
        "environment": {"task": "Open the drawer"},
    })
    return path


def call(attempt: str, episode: int, stage: str, cost: float | None, **changes) -> dict:
    return {
        "kind": "api_usage", "attempt_id": attempt, "episode_id": f"episode-{episode:04d}",
        "stage": stage, "outcome": "accepted", "model": "gpt-6-astra", "service_tier": "default",
        "reasoning_effort": "xhigh", "prompt_sha256": "prompt-a", "elapsed_seconds": 2,
        "estimated_cost_usd": cost, "cost_unknown": cost is None,
        "tokens": None if cost is None else {"input_tokens": 100, "output_tokens": 50,
                                               "reasoning_tokens": 40, "total_tokens": 150},
        **changes,
    }


def finish(run: Path, episode: int, *, task: str = "success", reset: str = "success") -> None:
    folder = run / f"episode-{episode:04d}"
    summary = {"id": folder.name, "outcome": task, "reset_outcome": reset,
               "finished_at": 5, "reset_finished_at": 10,
               "costs": {"task": {"estimated_cost_usd": 9999}}}
    write_json(folder / "summary.json", summary)
    append_events(folder / "transitions.jsonl",
                  {"kind": "stage_end", "stage": "task", "outcome": task},
                  {"kind": "stage_end", "stage": "reset", "outcome": reset},
                  {"kind": "episode_end", **summary})


def test_verified_task_with_pending_reset_does_not_make_a_cheap_success_denominator(run):
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2))
    append_events(run / "episode-0001" / "transitions.jsonl",
                  {"kind": "stage_end", "stage": "task", "outcome": "success"})
    aggregate = report_costs([run])["aggregate"]
    assert aggregate["total"]["known_cost_usd"] == pytest.approx(0.2)
    assert aggregate["verified_task_successes_pending_cycle"] == 1
    assert aggregate["successful_demos"] == aggregate["successful_complete_cycles"] == 0
    assert aggregate["amortized_per_successful_demo"] is None
    assert aggregate["amortized_per_successful_complete_cycle"] is None


@pytest.mark.parametrize("reset_outcome", ["failure", "stopped", "error", "step_budget"])
def test_successful_task_with_finalized_unsuccessful_reset_is_demo_but_not_cycle(run, reset_outcome):
    finish(run, 1, reset=reset_outcome)
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2), call("reset", 1, "reset", 0.3))
    aggregate = report_costs([run])["aggregate"]
    assert aggregate["successful_demos"] == 1
    assert aggregate["successful_resets"] == aggregate["successful_complete_cycles"] == 0
    assert aggregate["reset_outcomes"] == {reset_outcome: 1}
    assert aggregate["successful_task_mean"]["known_cost_usd"] == pytest.approx(0.2)
    assert aggregate["amortized_per_successful_demo"]["known_cost_usd"] == pytest.approx(0.5)
    assert aggregate["amortized_per_successful_complete_cycle"] is None


def test_all_failed_attempts_and_unknown_usage_are_in_amortized_cost(run):
    finish(run, 1)
    finish(run, 2, task="failure", reset="stopped")
    append_events(run / "api_calls.jsonl", call("good-task", 1, "task", 0.2), call("good-reset", 1, "reset", 0.1),
                  call("invalid-response", 2, "task", 0.4, outcome="invalid_response"),
                  call("canceled-reset", 2, "reset", None, outcome="cancelled"))
    aggregate = report_costs([run])["aggregate"]
    assert aggregate["successful_demos"] == aggregate["successful_complete_cycles"] == 1
    assert aggregate["finalized_episodes"] == 2
    total = aggregate["total"]
    # Stale episode cost snapshots contain 9999 and must never drive totals.
    assert total["known_cost_usd"] == pytest.approx(0.7)
    assert total["api_requests"] == 4
    assert total["unknown_cost_requests"] == total["unknown_usage_requests"] == 1
    assert total["tokens"]["total_tokens"] == 450
    assert total["tokens"]["output_tokens"] == 150
    assert total["tokens"]["reasoning_tokens"] == 120
    assert total["request_outcomes"]["invalid_response"] == 1
    assert aggregate["successful_task_mean"]["known_cost_usd"] == pytest.approx(0.2)
    assert aggregate["successful_reset_mean"]["known_cost_usd"] == pytest.approx(0.1)
    assert aggregate["amortized_per_successful_complete_cycle"]["known_cost_usd"] == pytest.approx(0.7)
    assert aggregate["amortized_per_successful_complete_cycle"]["cost_estimate_complete"] is False


def test_duplicate_attempt_and_overlapping_directory_inputs_are_not_double_counted(run):
    finish(run, 1)
    request = call("one-request", 1, "task", 0.2)
    append_events(run / "api_calls.jsonl", request, request)
    report = report_costs([run.parent, run])
    assert report["metered_runs"] == 1
    assert report["aggregate"]["total"]["api_requests"] == 1
    assert report["aggregate"]["total"]["known_cost_usd"] == pytest.approx(0.2)
    assert any("duplicate API attempt" in warning for warning in report["warnings"])


def test_retry_invalidates_old_summary_until_new_finalization_and_counts_one_demo(run):
    finish(run, 1, reset="error")
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2), call("first-reset", 1, "reset", 0.1))
    transitions = run / "episode-0001" / "transitions.jsonl"
    append_events(transitions, {"kind": "stage_retry", "stage": "reset", "recorded_at": 11})
    pending = report_costs([run])["aggregate"]
    assert pending["successful_demos"] == pending["finalized_episodes"] == 0
    assert pending["reset_outcomes"] == {"pending": 1}
    assert pending["verified_task_successes_pending_cycle"] == 1
    assert pending["retry_count"] == 1

    finish(run, 1, reset="success")
    append_events(run / "api_calls.jsonl", call("second-reset", 1, "reset", 0.4))
    completed = report_costs([run])["aggregate"]
    assert completed["successful_demos"] == completed["successful_complete_cycles"] == 1
    assert completed["retry_count"] == 1
    assert completed["retries_by_stage"] == {"task": 0, "reset": 1}
    assert completed["reset_outcomes"] == {"success": 1}
    assert completed["amortized_per_successful_demo"]["known_cost_usd"] == pytest.approx(0.7)


def test_late_usage_recomputed_from_ledger_changes_cost_without_adding_success(run):
    finish(run, 1)
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2))
    before = report_costs([run])["aggregate"]
    append_events(run / "api_calls.jsonl", call("late-reset", 1, "reset", 0.3))
    after = report_costs([run])["aggregate"]
    assert before["successful_demos"] == after["successful_demos"] == 1
    assert before["amortized_per_successful_demo"]["known_cost_usd"] == pytest.approx(0.2)
    assert after["amortized_per_successful_demo"]["known_cost_usd"] == pytest.approx(0.5)


def test_mixed_prompt_cost_groups_do_not_attribute_cycle_success_to_one_prompt(run):
    finish(run, 1)
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2),
                  call("reset", 1, "reset", 0.1, prompt_sha256="prompt-b"))
    report = report_costs([run])
    assert report["configuration_groups"][0]["profile"]["mixed_prompt_configuration"] is True
    assert len(report["request_configuration_groups"]) == 2
    assert {group["profile"]["prompt_sha256"] for group in report["request_configuration_groups"]} == {"prompt-a", "prompt-b"}
    assert all("successful_demos" not in group for group in report["request_configuration_groups"])
    assert report["aggregate"]["successful_complete_cycles"] == 1


def test_unmetered_previews_and_partial_last_line_do_not_inflate_counts(run):
    preview = run.parent / "unmetered-preview"
    write_json(preview / "manifest.json", {"run_id": preview.name, "config": {"policy": "scripted"}})
    finish(preview, 1)
    append_events(run / "api_calls.jsonl", call("task", 1, "task", 0.2))
    with (run / "api_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind":"api_usage",')
    report = report_costs([run.parent])
    assert report["metered_runs"] == 1
    assert report["aggregate"]["successful_demos"] == 0
    assert report["aggregate"]["total"]["api_requests"] == 1
    assert report["skipped_unmetered_runs"] == [str(preview)]
    assert any("incomplete/invalid" in warning for warning in report["warnings"])
    assert "N/A" in markdown_report(report)
