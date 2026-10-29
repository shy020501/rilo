"""Import published LIBERO demonstrations without simulation, GPT, or fake resets.

The pinned LIBERO ``scripts/create_dataset.py`` saves *post-action* RGB and
proprioception: obs[i] follows actions[i]. The first action has no saved input
observation. We retain obs[0], then actions[1:] and obs[1:], including the real
published terminal frame. No final frame, velocity, or control-site quaternion
is synthesized. Recorded fields are the common measured LIBERO learning schema.

Staging never touches a project registry. Commit requires the caller's idle
guard and uses the supplied ProjectStore lock. The separate import file lock
serializes import processes, but is not a cross-process lock on unrelated web
settings writes; callers must coordinate project ownership during commit.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Callable

import numpy as np

from rilo.adapters.libero import LiberoAdapter
from rilo.contracts import Observation
from rilo.recording import Recorder, dump_json


FORMAT = "rilo.libero_import.v1"
CAMERAS = {"agentview": "agentview_rgb", "robot0_eye_in_hand": "eye_in_hand_rgb"}
STATE_FIELDS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos")
MISSING_FIELDS = ("robot0_joint_vel", "robot0_gripper_qvel", "robot0_grip_site_quat")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_attribute(group, name):
    try:
        value = json.loads(_text(group.attrs[name]))
    except (KeyError, ValueError, UnicodeError) as exc:
        raise ValueError(f"LIBERO data attribute {name!r} must contain JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LIBERO {name!r} must be an object")
    return value


def _member(group, name):
    import h5py
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"Required local HDF5 member is missing or linked externally: {name}")
    return group[name]


def _array(group, name, shape):
    dataset = _member(group, name)
    if tuple(dataset.shape) != shape or dataset.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have numeric shape {shape}, got {dataset.shape}")
    value = np.asarray(dataset, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite LIBERO {name}")
    return value


def _quaternion(axis_angle):
    angle = float(np.linalg.norm(axis_angle))
    scale = math.sin(angle / 2) / angle if angle > 1e-12 else .5
    return np.r_[axis_angle * scale, math.cos(angle / 2)].tolist()


def _source_id(suite, task_name, demo_id):
    value = ["external_dataset", "LIBERO", suite, task_name, demo_id]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _identity(project):
    return {key: project[key] for key in ("id", "backend", "suite", "task_id")}


def stage_import(source_path, staging_root, project, *, task_name,
                 published_successful=False, source_url=None, source_revision=None,
                 demo_ids=None, image_convention=None):
    """Create a complete private run; return a JSON-serializable descriptor.

    ``published_successful`` is an explicit caller attestation that this is the
    published successful-demo dataset. Terminal rewards/dones must also agree.
    Staging should reside on the destination filesystem for atomic publication.
    All selected demos are validated before any run can be committed.
    """
    import h5py
    if published_successful is not True:
        raise ValueError("Only explicitly identified published successful demonstrations may be imported")
    if project.get("backend") != "libero":
        raise ValueError("LIBERO demonstrations require a LIBERO project")
    if not isinstance(task_name, str) or not task_name or Path(task_name).name != task_name:
        raise ValueError("A canonical LIBERO task name is required")
    source_path = Path(source_path).expanduser().resolve(strict=True)
    source_digest = file_sha256(source_path)
    staging_root = Path(staging_root).expanduser()
    if staging_root.is_symlink():
        raise ValueError("The private staging directory must not be a symlink")
    staging_root.mkdir(parents=True, exist_ok=True)
    recorder = None
    try:
        with h5py.File(source_path, "r") as source:
            data = _member(source, "data")
            bddl = Path(_text(data.attrs.get("bddl_file_name", "")))
            if bddl.stem != task_name or bddl.parent.name != project["suite"]:
                raise ValueError("Dataset BDDL task/suite does not match the requested project task")
            env = _json_attribute(data, "env_args")
            kwargs = env.get("env_kwargs", {})
            if not isinstance(kwargs, dict) or not isinstance(kwargs.get("controller_configs", {}), dict):
                raise ValueError("Invalid source environment/controller configuration")
            controller = kwargs.get("controller_configs", {}).get("type", kwargs.get("controller", "OSC_POSE"))
            if controller != "OSC_POSE" or kwargs.get("robots", ["Panda"]) not in (["Panda"], "Panda"):
                raise ValueError("Only LIBERO Panda OSC_POSE demonstrations are supported")
            controls = kwargs.get("controller_configs", {})
            expected = {"input_min": -1, "input_max": 1,
                        "output_min": [-.05] * 3 + [-.5] * 3, "output_max": [.05] * 3 + [.5] * 3}
            if (controls.get("control_delta", True) is not True or
                    any(key in controls and not np.allclose(controls[key], value) for key, value in expected.items())):
                raise ValueError("Source OSC_POSE action scaling differs from the live LIBERO adapter")
            hz = float(kwargs.get("control_freq", 20))
            if not math.isfinite(hz) or hz <= 0:
                raise ValueError("Invalid source control frequency")
            convention = image_convention or _text(data.attrs.get("macros_image_convention", ""))
            if convention not in ("opengl", "opencv"):
                raise ValueError("Unknown source image convention; explicitly supply opengl or opencv")
            language = _json_attribute(data, "problem_info").get("language_instruction")
            if isinstance(language, list) and all(isinstance(word, str) for word in language):
                language = " ".join(language)
            if not isinstance(language, str) or not language.strip():
                raise ValueError("Dataset language_instruction is missing")
            available = sorted((name for name in data if re.fullmatch(r"demo_\d+", name)),
                               key=lambda name: int(name.split("_")[-1]))
            selected = available if demo_ids is None else list(demo_ids)
            if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
                raise ValueError("Select at least one unique demo_id present in the dataset")
            spec = LiberoAdapter(project["suite"], control_hz=hz).action_spec
            common = {"kind": "external_dataset", "dataset": "LIBERO", "suite": project["suite"],
                "task_name": task_name, "task": language, "task_id": project["task_id"],
                "source_file": str(source_path), "file_sha256": source_digest,
                "source_url": source_url, "source_revision": source_revision,
                "success_evidence": "published successful demonstrations with terminal reward/done",
                "observation_alignment": "post_action; obs[i-1] -> actions[i] -> obs[i]",
                "omitted_action_indices": [0], "image_convention": convention,
                "measured_state_fields": list(STATE_FIELDS), "unavailable_state_fields": list(MISSING_FIELDS)}
            config = {"backend": "libero", "suite": project["suite"], "task_id": project["task_id"],
                "policy": "external_dataset", "model": None, "source": common,
                "import": {"format": FORMAT, "target": _identity(project)}, "episodes": len(selected)}
            environment = {"name": "libero", "suite": project["suite"], "task_id": project["task_id"],
                "task": language, "robot": "Panda", "controller": "OSC_POSE", "simulation": True,
                "camera_names": list(CAMERAS), "action_spec": spec.to_dict(),
                "observation_notes": "Upright RGB; exact published measured fields; EEF quaternion xyzw. No reset recorded."}
            recorder = Recorder(staging_root, config, environment)
            imported = []
            for index, demo_id in enumerate(selected, 1):
                demo = _member(data, demo_id)
                if "next_obs" in demo:
                    raise ValueError("next_obs layout is not the supported official post-action LIBERO format")
                action_data = _member(demo, "actions")
                count = action_data.shape[0] if len(action_data.shape) == 2 else 0
                if count < 2:
                    raise ValueError(f"{demo_id} needs at least two post-action observations")
                if int(demo.attrs.get("num_samples", count)) != count:
                    raise ValueError(f"{demo_id} num_samples does not match actions")
                actions = _array(demo, "actions", (count, 7))
                if np.any(actions < -1) or np.any(actions > 1):
                    raise ValueError(f"{demo_id} actions exceed OSC_POSE normalized bounds")
                rewards = _array(demo, "rewards", (count,))
                dones = _array(demo, "dones", (count,))
                if rewards[-1] <= 0 or dones[-1] != 1 or np.any(dones[:-1] != 0):
                    raise ValueError(f"{demo_id} lacks unambiguous published terminal success evidence")
                obs = _member(demo, "obs")
                joints = _array(obs, "joint_states", (count, 7))
                grippers = _array(obs, "gripper_states", (count, 2))
                eef = _array(obs, "ee_states", (count, 6))
                images = {}
                for camera, key in CAMERAS.items():
                    dataset = _member(obs, key)
                    if (len(dataset.shape) != 4 or dataset.shape[0] != count or dataset.shape[-1] != 3
                            or min(dataset.shape[1:3]) < 1 or dataset.dtype != np.dtype("uint8")):
                        raise ValueError(f"{demo_id}/{key} must contain {count} HWC uint8 RGB frames")
                    images[camera] = dataset
                source_info = {**common, "demo_id": demo_id,
                    "source_id": _source_id(project["suite"], task_name, demo_id),
                    "original_actions": count, "imported_actions": count - 1}
                def observation(frame):
                    final = frame == count - 1
                    return Observation(id=f"{demo_id}:source-observation:{frame}", timestamp=(frame + 1) / hz,
                        task=language, state={"robot0_eef_pos": eef[frame, :3].tolist(),
                            "robot0_eef_quat": _quaternion(eef[frame, 3:]),
                            "robot0_joint_pos": joints[frame].tolist(),
                            "robot0_gripper_qpos": grippers[frame].tolist()},
                        images={name: np.ascontiguousarray(value[frame][::-1] if convention == "opengl" else value[frame])
                                for name, value in images.items()}, success=final, terminated=final,
                        info={"stage": "task", "source_observation_index": frame,
                              "published_success": final, "source_reward": float(rewards[frame]),
                              "episode_control_time": frame / hz})
                current = observation(0)
                # 'observation' is understood by the learning reader and video
                # renderer without falsely claiming a simulated/human reset.
                recorder.begin_episode(index, current, None, start_kind="observation")
                recorder.write({"kind": "import_source", "stage": "task", "source": source_info})
                for frame in range(1, count):
                    following = observation(frame)
                    recorder.write({"kind": "transition", "stage": "task", "step": frame,
                        "observation_id": current.id, "applied_action": actions[frame].tolist(),
                        "source_action_index": frame, "next_observation": recorder.save_observation(following)})
                    current = following
                recorder.write({"kind": "stage_end", "stage": "task", "outcome": "success", "source": "published_dataset"})
                summary = {"id": f"episode-{index:04d}", "index": index, "task": language,
                    "outcome": "success", "reset_outcome": None, "steps": count - 1, "reset_steps": 0,
                    "source": source_info, "quality": {"task": {"label": "unknown", "source": "external_dataset"},
                                                          "reset": {"label": "unknown", "source": "not_recorded"}}}
                recorder.finish_episode(summary)
                imported.append({"demo_id": demo_id, "source_id": source_info["source_id"],
                                 "file_sha256": source_digest, "episode_id": summary["id"]})
            manifest_path = recorder.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["source"]["imported_demos"] = imported
            manifest_path.write_text(dump_json(manifest), encoding="utf-8")
            recorder.write_summary({"phase": "imported", "episodes": recorder.summaries,
                                    "source": common, "imported_at": time.time()})
            recorder.close()
        if file_sha256(source_path) != source_digest:
            raise ValueError("Source HDF5 changed during import")
        return {"format": FORMAT, "staging_run": str(recorder.path.resolve()), "run_id": recorder.run_id,
                "project_id": project["id"], "demo_ids": selected, "episodes": len(selected),
                "transitions": sum(item["steps"] for item in recorder.summaries), "source_sha256": source_digest}
    except BaseException:
        if recorder is not None:
            recorder.close()
            shutil.rmtree(recorder.path, ignore_errors=True)
        raise


def _existing_sources(projects, project):
    records = {}
    for run_id in project.get("run_ids", []):
        manifest_path = projects.data_dir / run_id / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        for value in manifest.get("config", {}).get("source", {}).get("imported_demos", []):
            records[value["source_id"]] = {**value, "run_id": run_id}
    return records


@contextmanager
def _import_lock(path):
    import fcntl
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def commit_import(staged, projects, project_id, *, ensure_idle: Callable[[], None]):
    """Publish a prepared run and attach it once, after the caller's idle check.

    ``ensure_idle`` must raise if the project is currently collecting/training.
    It is checked while holding the supplied store's lock, immediately before
    publication. Web callers should additionally hold their ownership lock.
    """
    if not callable(ensure_idle):
        raise ValueError("An explicit project idle guard is required")
    path = Path(staged["staging_run"])
    if staged.get("format") != FORMAT or path.is_symlink() or not path.is_dir():
        raise ValueError("A complete private LIBERO staging run is required")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = manifest.get("config", {}).get("import", {}).get("target")
    if manifest.get("run_id") != staged.get("run_id") or path.name != staged["run_id"]:
        raise ValueError("Staged run identity does not match its manifest")
    incoming = manifest["config"]["source"]["imported_demos"]
    common = manifest["config"]["source"]
    if (not incoming or len({item["episode_id"] for item in incoming}) != len(incoming)
            or len({item["source_id"] for item in incoming}) != len(incoming)):
        raise ValueError("Staged import must have unique nonempty demo/episode identities")
    for item in incoming:
        if (not re.fullmatch(r"episode-\d{4,}", item["episode_id"])
                or not re.fullmatch(r"demo_\d+", item["demo_id"])
                or item["source_id"] != _source_id(common["suite"], common["task_name"], item["demo_id"])
                or item["file_sha256"] != common["file_sha256"]):
            raise ValueError("Invalid staged episode/source identity")
    projects.data_dir.mkdir(parents=True, exist_ok=True)
    destination = projects.data_dir / staged["run_id"]
    if path.stat().st_dev != projects.data_dir.stat().st_dev:
        raise ValueError("Stage on the destination filesystem to publish atomically")
    if path.resolve().is_relative_to(projects.data_dir.resolve()):
        raise ValueError("Import staging must be outside the visible run directory")
    with _import_lock(projects.data_dir.parent / ".libero-import.lock"), projects._lock:
        if ensure_idle() is False:
            raise RuntimeError("Project must be idle before importing demonstrations")
        # get()/list() also migrate historical run numbering. Import failures
        # must not incidentally alter unrelated project metadata.
        project = next((p for p in projects._load()["projects"] if p["id"] == project_id), None)
        if project is None:
            raise KeyError(project_id)
        if target != _identity(project) or staged.get("project_id") != project_id:
            raise ValueError("Project task changed or staged import belongs to a different project")
        existing = _existing_sources(projects, project)
        duplicates = []
        for item in incoming:
            previous = existing.get(item["source_id"])
            if previous:
                if previous["file_sha256"] != item["file_sha256"]:
                    raise ValueError(f"Conflicting source content for {item['demo_id']}; existing import is preserved")
                duplicates.append(item)
        if len(duplicates) == len(incoming):
            shutil.rmtree(path)
            return {"run_id": None, "imported": 0, "skipped": len(duplicates),
                    "existing_run_ids": sorted({existing[item["source_id"]]["run_id"] for item in duplicates})}
        duplicate_ids = {item["episode_id"] for item in duplicates}
        if duplicates:
            for episode_id in duplicate_ids:
                shutil.rmtree(path / episode_id)
            incoming = [item for item in incoming if item["episode_id"] not in duplicate_ids]
            manifest["config"]["source"]["imported_demos"] = incoming
            manifest["config"]["episodes"] = len(incoming)
            manifest_path.write_text(dump_json(manifest), encoding="utf-8")
            summary_path = path / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["episodes"] = [ep for ep in summary["episodes"] if ep["id"] not in duplicate_ids]
            summary_path.write_text(dump_json(summary), encoding="utf-8")
        # Recheck completed evidence immediately before making it visible.
        for item in incoming:
            episode = path / item["episode_id"]
            summary = json.loads((episode / "summary.json").read_text())
            if (episode.is_symlink() or summary.get("source", {}).get("source_id") != item["source_id"]
                    or summary.get("outcome") != "success" or not (episode / "transitions.jsonl").is_file()):
                raise ValueError("Staged episode evidence is incomplete or mismatched")
            if any(child.is_symlink() for child in episode.rglob("*")):
                raise ValueError("Staged episode evidence must not contain symlinks")
            from rilo.learning.data import load_episode
            loaded = load_episode(episode, summary, manifest)
            if len(loaded.actions) != summary["steps"] or loaded.actions.shape[1] != 7:
                raise ValueError("Staged transition counts or action schema do not match the summary")
        if destination.exists() or destination.is_symlink():
            raise ValueError("A run with this identity already exists")
        # Evidence validation can take time on a large dataset. Recheck the
        # external owner's state at the publication boundary as well.
        if ensure_idle() is False:
            raise RuntimeError("Project became active before import publication")
        path.rename(destination)
        try:
            projects.link_run(project_id, staged["run_id"])
        except BaseException:
            # Do not remove a successfully linked run if a custom store throws
            # after persistence; otherwise restore the private staging for retry.
            saved = next(p for p in projects._load()["projects"] if p["id"] == project_id)
            if staged["run_id"] not in saved.get("run_ids", []):
                destination.rename(path)
            raise
        return {"run_id": staged["run_id"], "imported": len(incoming), "skipped": len(duplicates),
                "source_sha256": staged["source_sha256"]}
