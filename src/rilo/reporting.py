"""Recompute experiment cost from the append-only request ledger.

Episode cost snapshots can miss late responses. This report always reads the
request ledger instead, and uses finalized episode artifacts for denominators.
No credentials, raw model responses, or private environment files are read.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable


TOKEN_KEYS = ("input_tokens", "ordinary_input_tokens", "cached_input_tokens", "cache_write_tokens",
              "cache_write_30m_tokens", "cache_write_12h_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def _json_object(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        return value
    except (ValueError, OSError):
        warnings.append(f"Unreadable JSON object: {path}")
        return None


def _events(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        warnings.append(f"Unreadable ledger: {path}")
        return []
    result = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Expected object")
        except ValueError:
            warnings.append(f"Skipped incomplete/invalid record: {path}:{index}")
            continue
        result.append(event)
    return result


def _totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"api_requests": len(events), "known_cost_usd": 0.0, "unknown_cost_requests": 0,
              "unknown_usage_requests": 0, "tokens": {key: 0 for key in TOKEN_KEYS},
              "elapsed_api_seconds": 0.0, "request_outcomes": {}, "actual_models": [],
              "service_tiers": [], "pricing_sources": []}
    models, tiers, sources, outcomes = set(), set(), set(), Counter()
    for event in events:
        tokens = event.get("tokens")
        if not isinstance(tokens, dict):
            result["unknown_usage_requests"] += 1
        else:
            for key in TOKEN_KEYS:
                value = tokens.get(key, 0)
                if type(value) is int and value >= 0:
                    result["tokens"][key] += value
        cost = event.get("estimated_cost_usd")
        if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 and not event.get("cost_unknown", False):
            result["known_cost_usd"] += cost
        else:
            result["unknown_cost_requests"] += 1
        elapsed = event.get("elapsed_seconds")
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
            result["elapsed_api_seconds"] += elapsed
        if isinstance(event.get("model"), str):
            models.add(event["model"])
        if isinstance(event.get("service_tier"), str):
            tiers.add(event["service_tier"])
        if isinstance(event.get("pricing"), dict) and isinstance(event["pricing"].get("source"), str):
            sources.add(event["pricing"]["source"])
        outcomes[str(event.get("outcome", "unknown"))] += 1
    result.update(actual_models=sorted(models), service_tiers=sorted(tiers), pricing_sources=sorted(sources),
                  request_outcomes=dict(outcomes), cost_estimate_complete=result["unknown_cost_requests"] == 0,
                  usage_complete=result["unknown_usage_requests"] == 0)
    return result


def _mean(totals: dict[str, Any], denominator: int) -> dict[str, Any] | None:
    if denominator == 0:
        return None
    return {"sample_count": denominator,
            "api_requests": totals["api_requests"] / denominator,
            "known_cost_usd": totals["known_cost_usd"] / denominator,
            "cost_estimate_complete": totals["cost_estimate_complete"],
            "usage_complete": totals["usage_complete"],
            "tokens": {key: value / denominator for key, value in totals["tokens"].items()}}


def _summary(episodes: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    finalized = [episode for episode in episodes if episode["finalized"]]
    demos = [episode for episode in finalized if episode["outcome"] == "success"]
    resets = [episode for episode in finalized if episode["reset_outcome"] == "success"]
    cycles = [episode for episode in demos if episode["reset_outcome"] == "success"]
    demo_keys = {(episode["run_id"], episode["episode_id"]) for episode in demos}
    reset_keys = {(episode["run_id"], episode["episode_id"]) for episode in resets}
    successful_task_events = [event for event in events if (event["run_id"], event.get("episode_id")) in demo_keys and event.get("stage") == "task"]
    successful_reset_events = [event for event in events if (event["run_id"], event.get("episode_id")) in reset_keys and event.get("stage") == "reset"]
    total = _totals(events)
    stages = {stage: _totals([event for event in events if event.get("stage", "other") == stage]) for stage in ("task", "reset")}
    stages["other"] = _totals([event for event in events if event.get("stage") not in ("task", "reset")])
    return {
        "episodes_observed": len(episodes), "finalized_episodes": len(finalized),
        "successful_demos": len(demos), "successful_resets": len(resets), "successful_complete_cycles": len(cycles),
        "verified_task_successes_pending_cycle": sum(episode["outcome"] == "success" and not episode["finalized"] for episode in episodes),
        "retry_count": sum(episode["retry_count"] for episode in episodes),
        "retries_by_stage": {stage: sum(episode["retries_by_stage"].get(stage, 0) for episode in episodes) for stage in ("task", "reset")},
        "task_outcomes": dict(Counter(str(episode["outcome"] or "pending") for episode in episodes)),
        "reset_outcomes": dict(Counter(str(episode["reset_outcome"] or "pending") for episode in episodes)),
        "total": total, "stages": stages,
        "successful_task_mean": _mean(_totals(successful_task_events), len(demos)),
        "successful_reset_mean": _mean(_totals(successful_reset_events), len(resets)),
        "amortized_per_successful_demo": _mean(total, len(demos)),
        "amortized_per_successful_complete_cycle": _mean(total, len(cycles)),
    }


def _run(path: Path, warnings: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _json_object(path / "manifest.json", warnings) or {}
    config = manifest.get("config") or {}
    environment = manifest.get("environment") or {}
    run_id = str(manifest.get("run_id", path.name))
    events, seen = [], set()
    for event in _events(path / "api_calls.jsonl", warnings):
        if event.get("kind") != "api_usage":
            continue
        attempt_id = event.get("attempt_id")
        if isinstance(attempt_id, str):
            if attempt_id in seen:
                warnings.append(f"Ignored duplicate API attempt {attempt_id} in run {run_id}")
                continue
            seen.add(attempt_id)
        events.append({**event, "run_id": run_id})

    episodes = []
    for folder in sorted(path.glob("episode-*")):
        if not folder.is_dir():
            continue
        transitions = _events(folder / "transitions.jsonl", warnings)
        completed = [event for event in transitions if event.get("kind") == "episode_end"]
        episode_summary = _json_object(folder / "summary.json", warnings) or (completed[-1] if completed else {})
        stage_ends, retries = {}, Counter()
        last_retry_index, last_end_index = -1, -1
        for index, event in enumerate(transitions):
            if event.get("kind") == "stage_end":
                stage_ends[event.get("stage")] = event.get("outcome")
            elif event.get("kind") == "stage_retry":
                stage_ends[event.get("stage")] = None
                retries[str(event.get("stage", "other"))] += 1
                last_retry_index = index
            elif event.get("kind") == "episode_end":
                last_end_index = index
        outcome = stage_ends.get("task", episode_summary.get("outcome"))
        reset_outcome = stage_ends.get("reset", episode_summary.get("reset_outcome"))
        # stage_end can prove forward success while reset remains in progress.
        # Denominators use only a finalized episode, preventing partial runs
        # from looking artificially cheap before their reset cost arrives.
        finalized = bool(episode_summary.get("outcome") and episode_summary.get("finished_at"))
        if last_retry_index > last_end_index:
            finalized = False
        if config.get("reset_mode") == "agent":
            finalized = finalized and bool(episode_summary.get("reset_outcome") and episode_summary.get("reset_finished_at"))
        episodes.append({"run_id": run_id, "episode_id": folder.name,
                         "outcome": outcome, "reset_outcome": reset_outcome, "finalized": finalized,
                         "retry_count": sum(retries.values()), "retries_by_stage": dict(retries),
                         "task": episode_summary.get("task"),
                         "environment_resets": sum(event.get("kind") == "environment_reset" for event in transitions),
                         "agent_reset_continuation": any(event.get("kind") == "agent_reset_continuation" for event in transitions),
                         "steps": sum(event.get("kind") == "transition" and event.get("stage", "task") == "task" for event in transitions),
                         "reset_steps": sum(event.get("kind") == "transition" and event.get("stage") == "reset" for event in transitions)})

    current = _json_object(path / "summary.json", warnings) or {}
    profile = {key: config.get(key) for key in ("backend", "suite", "task_id", "policy", "model", "reasoning_effort",
                                               "max_output_tokens", "reset_mode", "source_sha256")}
    profile["actual_models"] = sorted({event["model"] for event in events if isinstance(event.get("model"), str)})
    profile["prompt_sha256s"] = sorted({event["prompt_sha256"] for event in events if isinstance(event.get("prompt_sha256"), str)})
    profile["actual_reasoning_efforts"] = sorted({event["reasoning_effort"] for event in events if isinstance(event.get("reasoning_effort"), str)})
    profile["mixed_prompt_configuration"] = len(profile["prompt_sha256s"]) > 1 or len(profile["actual_reasoning_efforts"]) > 1
    disk_hash = current.get("source_sha256")
    run = {"run_id": run_id, "path": str(path.resolve()), "created_at": manifest.get("created_at"),
           "profile": profile, "config": config, "task": environment.get("task"),
           "last_phase": current.get("phase"), "last_stage": current.get("stage"),
           "source_changed_on_disk": bool(disk_hash and profile.get("source_sha256") and disk_hash != profile["source_sha256"]),
           "episodes": episodes, **_summary(episodes, events)}
    return run, episodes, events


def report_costs(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Accept run directories or parent directories; skip unmetered previews."""
    warnings, candidates = [], set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if (path / "manifest.json").exists():
            candidates.add(path)
        elif path.is_dir():
            candidates.update(item.parent for item in path.glob("*/manifest.json"))
        else:
            warnings.append(f"Run directory does not exist: {path}")
    run_paths = sorted(path for path in candidates if (path / "api_calls.jsonl").exists())
    skipped = [str(path) for path in sorted(candidates) if path not in run_paths]
    runs, episodes, events = [], [], []
    groups: dict[str, dict[str, Any]] = {}
    for path in run_paths:
        run, run_episodes, run_events = _run(path, warnings)
        runs.append(run)
        episodes.extend(run_episodes)
        events.extend(run_events)
        key = json.dumps(run["profile"], sort_keys=True)
        group = groups.setdefault(key, {"profile": run["profile"], "run_ids": [], "episodes": [], "events": []})
        group["run_ids"].append(run["run_id"])
        group["episodes"].extend(run_episodes)
        group["events"].extend(run_events)
    configurations = [{"profile": group["profile"], "run_ids": group["run_ids"],
                       **_summary(group["episodes"], group["events"])} for group in groups.values()]
    request_groups: dict[str, dict[str, Any]] = {}
    run_profiles = {run["run_id"]: run["profile"] for run in runs}
    for event in events:
        run_profile = run_profiles[event["run_id"]]
        profile = {key: event.get(key, run_profile.get(key)) for key in
                   ("model", "reasoning_effort", "max_output_tokens", "prompt_sha256", "source_sha256", "service_tier")}
        key = json.dumps(profile, sort_keys=True)
        group = request_groups.setdefault(key, {"profile": profile, "events": []})
        group["events"].append(event)
    request_configurations = [{"profile": group["profile"], "total": _totals(group["events"]),
                              "stages": {stage: _totals([event for event in group["events"] if event.get("stage") == stage])
                                         for stage in ("task", "reset")}} for group in request_groups.values()]
    return {"schema_version": 1, "generated_at": time.time(), "metered_runs": len(runs),
            "skipped_unmetered_runs": skipped, "warnings": warnings,
            "definitions": {
                "successful_demo": "Finalized episode with independently verified forward-task success. In agent mode the reset attempt must also finish before counting.",
                "successful_complete_cycle": "Finalized episode with both verified task success and verified GPT physical reset success.",
                "amortized": "All selected API attempts, including failed/incomplete/canceled attempts and resets, divided by the stated success count.",
                "known_cost_usd": "API usage multiplied by recorded list-price rates; partial if any request has unknown cost. Not an invoice.",
                "unknown_requests": "No cost or tokens are invented for missing usage. A late response can change the ledger and a later report.",
                "configuration_groups": "Group by requested and actual model, effort, output cap, task/suite, reset mode, and initial source SHA. Do not mix tuning and final validation silently.",
                "request_configuration_groups": "Per-call model/effort/prompt/source configuration cost totals. Success denominators are omitted because a cycle can span configurations.",
            }, "aggregate": _summary(episodes, events), "configuration_groups": configurations,
            "request_configuration_groups": request_configurations, "runs": runs}


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    total = aggregate["total"]
    lines = ["# RILO API cost measurement", "",
             f"Metered runs: {report['metered_runs']}; finalized episodes: {aggregate['finalized_episodes']}; "
             f"successful demos: {aggregate['successful_demos']}; successful task + GPT-reset cycles: {aggregate['successful_complete_cycles']}; retries: {aggregate['retry_count']}.", "",
             "| Stage | API calls | Input tokens | Cached input | Cache writes | Output tokens | Reasoning tokens¹ | Known estimated USD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in [("Task", aggregate["stages"]["task"]), ("GPT reset", aggregate["stages"]["reset"]), ("Other", aggregate["stages"]["other"]), ("Total", total)]:
        tokens = values["tokens"]
        lines.append(f"| {name} | {values['api_requests']} | {tokens['input_tokens']:,} | {tokens['cached_input_tokens']:,} | "
                     f"{tokens['cache_write_tokens']:,} | {tokens['output_tokens']:,} | {tokens['reasoning_tokens']:,} | ${values['known_cost_usd']:.6f} |")
    lines.extend(["", "¹ Reasoning tokens are included in output tokens and are not charged a second time.", "",
                  "| Average | Denominator | Total tokens | Known estimated USD |", "|---|---:|---:|---:|"])
    for label, key in [("Successful forward task only", "successful_task_mean"), ("Successful GPT reset only", "successful_reset_mean"),
                       ("All attempts + resets per successful demo", "amortized_per_successful_demo"),
                       ("All attempts + resets per successful complete cycle", "amortized_per_successful_complete_cycle")]:
        value = aggregate[key]
        if value is None:
            lines.append(f"| {label} | 0 | N/A | N/A |")
        else:
            lines.append(f"| {label} | {value['sample_count']} | {value['tokens']['total_tokens']:,.1f} | ${value['known_cost_usd']:.6f} |")
    lines.extend(["", f"Unknown-cost requests: {total['unknown_cost_requests']}; unknown-usage requests: {total['unknown_usage_requests']}; "
                  f"verified task successes still awaiting final reset outcome: {aggregate['verified_task_successes_pending_cycle']}.",
                  "These are list-price estimates from response usage, not invoice charges. Unknown requests are excluded from known sums, so incomplete estimates are explicitly partial.", "",
                  "## Configuration groups", "",
                  "| Model | Effort | Suite / task | Source SHA | Runs | Success demos / cycles | Known estimated USD |",
                  "|---|---|---|---|---:|---:|---:|"])
    for group in report["configuration_groups"]:
        profile = group["profile"]
        lines.append(f"| {profile.get('model')} | {profile.get('reasoning_effort')} | {profile.get('suite')} / {profile.get('task_id')} | "
                     f"{str(profile.get('source_sha256') or 'unknown')[:12]} | {len(group['run_ids'])} | {group['successful_demos']} / {group['successful_complete_cycles']} | ${group['total']['known_cost_usd']:.6f} |")
    if total["pricing_sources"]:
        lines.extend(["", "Pricing sources: " + ", ".join(f"[official model pricing]({source})" for source in total["pricing_sources"]) + "."])
    if report["warnings"]:
        lines.extend(["", "Data warnings:", "", *[f"- {warning}" for warning in report["warnings"]]])
    return "\n".join(lines) + "\n"
