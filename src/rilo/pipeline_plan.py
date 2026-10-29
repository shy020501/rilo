"""Read-only evidence and provenance checks for resumable pipeline jobs.

File digests and trajectory decoding are cached by file identity, size and
nanosecond timestamps. A fresh explicit job revalidates the evidence. Merely
opening the workspace never creates a job or changes its collection goal.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading


LEARNED_STAGES = frozenset(("bc_eval", "rl", "rl_eval"))


def evaluation_reset_mode(project, config):
    selected = config.get("evaluation_reset_mode", "inherit")
    mode = (project.get("run_config") or {}).get("reset_mode", "agent") if selected == "inherit" else selected
    if mode not in ("agent", "environment", "free_space"):
        raise ValueError("Unsupported evaluation reset mode")
    return mode


def evaluation_signature(project, config, stage):
    run = deepcopy(project.get("run_config") or {})
    for key in ("episodes", "budget_usd"):
        run.pop(key, None)
    run["reset_mode"] = evaluation_reset_mode(project, config)
    signature = {"version": 1, "stage": stage, "backend": project.get("backend"),
        "task_id": project.get("task_id"), "run_config": run,
        "episodes": config["evaluation_episodes"],
        "policy_config": deepcopy(project.get("policy_config") or {})}
    if stage == "rl_eval":
        signature.update(progress_gate_threshold=config["progress_gate_threshold"],
                         gpt_judge_interval=config["gpt_judge_interval"])
    return signature


def evaluation_key(evaluation):
    return hashlib.sha256(json.dumps(evaluation, sort_keys=True).encode()).hexdigest()


def _stamp(path):
    try:
        stat = path.stat()
        return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return str(path), None


class PipelinePlanner:
    def __init__(self, projects, artifact_root):
        self.projects, self.artifact_root = projects, Path(artifact_root)
        self._lock = threading.RLock()
        self._episodes = {}
        self._hashes = {}

    def evidence(self, project_id, image_size, *, force=False):
        """Return individually valid active trajectories, plus a compatible snapshot."""
        from PIL import Image
        import numpy as np
        from rilo.learning.data import load_episode, _infer_spec, _safe_child, state_vector, make_snapshot
        with self._lock:
            details = self.projects.details(self.projects.get(project_id))
            episodes, specs, errors = [], [], []
            for summary in sorted(details["episodes"], key=lambda ep: (ep["run_id"], ep["id"])):
                if not summary.get("outcome") or summary.get("archived") or summary.get("delete_pending"):
                    continue
                path = self.projects._episode_path(summary["run_id"], summary["id"])
                # A live task result still waiting for reset is not a finalized demonstration.
                if not (path / "summary.json").is_file() or not (path / "transitions.jsonl").is_file():
                    continue
                manifest_path = path.parent / "manifest.json"
                files = sorted(p for p in path.rglob("*") if p.is_file())
                signature = (summary.get("outcome"), image_size, _stamp(manifest_path),
                             tuple(_stamp(p) for p in files))
                key = (project_id, summary["run_id"], summary["id"], image_size)
                cached = None if force else self._episodes.get(key)
                if cached is None or cached[0] != signature:
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        episode = load_episode(path, summary, manifest)
                        spec = _infer_spec(episode, manifest, image_size, None, None)
                        for observation in episode.observations:
                            state_vector(observation["state"], spec)
                            if any(name not in observation["images"] for name in spec.cameras):
                                raise ValueError("A required camera is missing within an episode")
                        for relative in {p for obs in episode.observations for p in obs["images"].values()}:
                            with Image.open(_safe_child(path, relative)) as image:
                                image.verify()
                        if episode.actions.shape[1] != spec.action_dim:
                            raise ValueError("Action dimension does not match the adapter specification")
                        if (np.any(episode.actions < np.asarray(spec.action_low) - 1e-6)
                                or np.any(episode.actions > np.asarray(spec.action_high) + 1e-6)):
                            raise ValueError("Recorded executed action is outside adapter bounds")
                        cached = signature, episode, spec, None
                    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                        cached = signature, None, None, str(exc)
                    self._episodes[key] = cached
                if cached[1] is None:
                    errors.append({"run_id": summary["run_id"], "episode_id": summary["id"], "reason": cached[3]})
                else:
                    episodes.append(cached[1]); specs.append(cached[2])
            incompatible = bool(specs and any(spec != specs[0] for spec in specs[1:]))
            snapshot = make_snapshot(episodes, specs[0], project_id) if episodes and not incompatible else None
            return {"episodes": episodes, "snapshot": snapshot, "errors": errors,
                "successes": sum(ep.success for ep in episodes),
                "reason": "데이터의 카메라·상태·동작 형식이 서로 다릅니다." if incompatible else
                          "학습 가능한 완료 task episode가 없습니다." if not episodes else ""}

    def snapshot(self, project_id, image_size):
        evidence = self.evidence(project_id, image_size)
        if evidence["snapshot"] is None:
            raise ValueError(evidence["reason"])
        return evidence["snapshot"]

    def _checkpoint_valid(self, project_id, model, *, force=False, allow_partial=False):
        if (model.get("status") != "ready" or not allow_partial and (
                model.get("metrics", {}).get("partial") or model.get("partial"))):
            return False
        try:
            path = Path(model["checkpoint"])
            root = (self.artifact_root / project_id).resolve()
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                return False
            signature = _stamp(path)
            cached = None if force else self._hashes.get(str(path))
            if cached is None or cached[0] != signature:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                cached = signature, digest
                self._hashes[str(path)] = cached
            return bool(model.get("checkpoint_sha256") == cached[1])
        except (OSError, ValueError, KeyError, TypeError):
            return False

    @staticmethod
    def _dataset_valid(dataset, snapshot, *, successful_only=False):
        if snapshot is None or not isinstance(dataset, dict):
            return False
        from rilo.learning.data import make_snapshot
        selected = snapshot.successful() if successful_only else snapshot
        if dataset.get("spec") != selected.spec.model_dump(mode="json"):
            return False
        current = {(ep.provenance["run_id"], ep.provenance["episode_id"]): ep for ep in selected.episodes}
        members, keys = [], set()
        for provenance in dataset.get("episodes", []):
            key = provenance.get("run_id"), provenance.get("episode_id")
            episode = current.get(key)
            if episode is None or episode.provenance != provenance or key in keys:
                return False
            members.append(episode); keys.add(key)
        if not members or make_snapshot(members, selected.spec, snapshot.project_id).fingerprint != dataset.get("fingerprint"):
            return False
        # Downstream rollouts do not invalidate their own upstream checkpoint.
        # Newly collected teacher trajectories do require a fresh training job.
        return all(key in keys or ep.provenance.get("learning", {}).get("pipeline_stage") in LEARNED_STAGES
                   for key, ep in current.items())

    def plan(self, project_id, state, *, force=False):
        with self._lock:
            cfg = state["config"]
            project = self.projects.get(project_id)
            evidence = self.evidence(project_id, cfg.get("bc", {}).get("image_size", 128), force=force)
            snapshot = evidence["snapshot"]
            models = list(reversed(state.get("models", [])))
            def model(kind, config, base=None, progress=None, *, allow_partial=False):
                legacy_algorithm = {"bc": "dp", "residual": "residual_td3", "progress": "progress"}[kind]
                selected_algorithm = cfg.get("bc_algorithm", "dp") if kind == "bc" else (
                    cfg.get("rl_algorithm", "residual_td3") if kind == "residual" else "progress")
                for item in models:
                    recorded_config = item.get("config")
                    if kind == "bc" and isinstance(recorded_config, dict):
                        # Checkpoint frequency affects storage, not the learned policy.
                        matches_config = ({key: value for key, value in recorded_config.items() if key != "checkpoint_every"}
                                          == {key: value for key, value in config.items() if key != "checkpoint_every"})
                    else:
                        matches_config = recorded_config == config
                    if item.get("kind") != kind or not matches_config:
                        continue
                    if item.get("algorithm", legacy_algorithm) != selected_algorithm:
                        continue
                    if not self._checkpoint_valid(project_id, item, force=force, allow_partial=allow_partial):
                        continue
                    provenance = item.get("provenance") or {}
                    if base and (item.get("base_model_id") != base["id"]
                            or provenance.get("base_checkpoint_sha256") != base["checkpoint_sha256"]):
                        continue
                    if progress and (provenance.get("progress_model_id") != progress["id"]
                            or provenance.get("progress_checkpoint_sha256") != progress["checkpoint_sha256"]
                            or provenance.get("progress_gate_threshold") != cfg["progress_gate_threshold"]
                            or provenance.get("gpt_judge_interval") != cfg["gpt_judge_interval"]):
                        continue
                    dataset = provenance.get("dataset") if kind == "residual" else item.get("dataset")
                    if not self._dataset_valid(dataset, snapshot,
                            successful_only=kind == "progress" or kind == "bc" and config.get("successful_only", True)):
                        continue
                    if kind == "residual" and item.get("dataset", {}).get("dataset_fingerprint") != dataset.get("fingerprint"):
                        continue
                    return item
                return None
            bc = model("bc", cfg.get("bc", {}))
            progress = model("progress", cfg.get("progress", {}), bc) if bc else None
            residual = model("residual", cfg.get("residual", {}), bc, progress) if bc and progress else None
            # A verified partial checkpoint can continue training, but cannot
            # establish completed training or be selected for evaluation.
            residual_resume = model("residual", cfg.get("residual", {}), bc, progress,
                                    allow_partial=True) if bc and progress else None
            def evaluation(stage):
                if bc is None or stage == "rl_eval" and (residual is None or progress is None):
                    return None
                signature = evaluation_signature(project, cfg, stage)
                for value in reversed(state.get("evaluations", [])):
                    prov = value.get("provenance") or {}
                    if (value.get("kind") != ("bc" if stage == "bc_eval" else "residual")
                            or value.get("model_id") != bc["id"]
                            or prov.get("base_checkpoint_sha256") != bc["checkpoint_sha256"]
                            or prov.get("evaluation") != signature
                            or value.get("episodes") != cfg["evaluation_episodes"]
                            or value.get("partial") or not value.get("run_id")):
                        continue
                    if stage == "rl_eval" and (value.get("residual_model_id") != residual["id"]
                            or prov.get("residual_checkpoint_sha256") != residual["checkpoint_sha256"]
                            or prov.get("progress_model_id") != progress["id"]
                            or prov.get("progress_checkpoint_sha256") != progress["checkpoint_sha256"]):
                        continue
                    try:
                        rate = float(value["success_rate"])
                        if not 0 <= rate <= 100:
                            continue
                    except (KeyError, TypeError, ValueError):
                        continue
                    return value
                return None
            bc_eval, rl_eval = evaluation("bc_eval"), evaluation("rl_eval")
            bc_pass = bool(bc_eval and bc_eval["success_rate"] >= cfg["bc_success_target"])
            rl_pass = bool(rl_eval and rl_eval["success_rate"] >= cfg["rl_success_target"])
            saved = state.get("auto_progress") or {}
            target = cfg["collection_episodes"]
            if saved.get("configured_collection_episodes") == target:
                target = max(target, int(saved.get("collection_goal", target)))
            extension = None
            if (bc_eval and not bc_pass and saved.get("extended_for_evaluation") != evaluation_key(bc_eval)
                    and evaluation_key(bc_eval) not in saved.get("goal_override_evaluations", [])):
                baseline = bc_eval.get("collection_successes", evidence["successes"])
                target = max(target, baseline + cfg["collection_episodes"])
                extension = {"collection_goal": target, "extended_for_evaluation": evaluation_key(bc_eval)}
            remaining = max(0, target - evidence["successes"])
            def completed(item, reason, **extra):
                return {"complete": bool(item), "model_id": item.get("id") if item else None,
                        "reason": "" if item else reason, **extra}
            stages = {
                "bc": completed(bc, "현재 데이터와 설정에 맞는 BC checkpoint가 필요합니다."),
                "bc_eval": completed(bc if bc_pass else None, "현재 BC의 목표 성공률을 충족한 평가가 필요합니다.",
                    evaluated=bool(bc_eval), success_rate=bc_eval.get("success_rate") if bc_eval else None),
                "progress": completed(progress if bc_pass else None, "목표를 통과한 BC에 맞는 progress 모델이 필요합니다."),
                "rl": completed(residual if bc_pass else None, "현재 BC·progress·설정에 맞는 완료 RL checkpoint가 필요합니다."),
                "rl_eval": completed(residual if bc_pass and rl_pass else None, "현재 residual의 목표 성공률을 충족한 평가가 필요합니다.",
                    evaluated=bool(rl_eval), success_rate=rl_eval.get("success_rate") if rl_eval else None)}
            if remaining:
                next_stage, reason = "collect", f"성공 task episode {remaining}개를 추가로 수집합니다."
            elif not bc:
                next_stage, reason = "bc", stages["bc"]["reason"]
            elif not bc_pass:
                next_stage, reason = ("bc_eval", stages["bc_eval"]["reason"]) if bc_eval is None else (
                    "collect", "BC 재학습에 사용할 새 성공 데이터를 수집해야 합니다.")
            elif not progress:
                next_stage, reason = "progress", stages["progress"]["reason"]
            elif not residual or rl_eval and not rl_pass:
                next_stage, reason = "rl", "Residual 학습과 평가를 이어갑니다."
            elif not rl_pass:
                next_stage, reason = "rl_eval", stages["rl_eval"]["reason"]
            else:
                next_stage, reason = None, "현재 데이터·설정의 모든 단계가 완료되었습니다."
            return {"completion": {"collect": {"completed": evidence["successes"], "target": target,
                    "remaining": remaining, "ready": remaining == 0,
                    "invalid_episodes": len(evidence["errors"]), "trainable_successes": evidence["successes"] if snapshot else 0,
                    "reason": evidence["reason"]}, "stages": stages},
                "auto_plan": {"next_stage": next_stage, "remaining_episodes": remaining,
                    "complete": next_stage is None, "reason": reason},
                "models": {"bc": bc, "progress": progress, "residual": residual,
                           "residual_resume": residual_resume},
                "evaluations": {"bc_eval": bc_eval, "rl_eval": rl_eval}, "extension": extension}
