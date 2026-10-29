"""LIBERO / robosuite adapter, isolated from the policy and experiment runner.

Runtime target: LIBERO's original environment API and robosuite 1.4.1.
Heavy imports occur only on first use, so the console and other adapters do not
require MuJoCo, torch, or LIBERO. All methods must run on the runner's single
environment thread; MuJoCo rendering contexts are thread-affine.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence
import uuid

import numpy as np

from rilo.contracts import ActionSpec, Observation


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
CAMERAS = ("agentview", "robot0_eye_in_hand")
STATE_KEYS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos",
    "robot0_joint_vel", "robot0_gripper_qpos", "robot0_gripper_qvel",
)
RESET_DRAWER_TASKS = {
    "open_the_middle_drawer_of_the_cabinet": "middle",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet": "top",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet": "top",
}
FREE_SPACE_RADIUS_M = 0.20
FREE_SPACE_ENV_CLEARANCE_M = 0.02
FREE_SPACE_SELF_CLEARANCE_M = 0.005
FREE_SPACE_JOINT_SPEED_RAD_S = 0.05
FREE_SPACE_GRIPPER_SPEED_M_S = 0.01


def prepare_libero_config(config_dir: Path | None = None) -> Path:
    """Avoid LIBERO's import-time interactive prompt without editing ~/.libero.

    LIBERO reads JSON-compatible YAML. Discover its package directory without
    importing libero.libero, whose import would trigger the interactive prompt.
    A user-supplied LIBERO_CONFIG_PATH is honored and never overwritten.
    """
    selected = os.environ.get("LIBERO_CONFIG_PATH")
    directory = Path(selected).expanduser() if selected else (
        config_dir or Path.cwd() / "artifacts" / "libero-config"
    )
    directory = directory.resolve()
    config_file = directory / "config.yaml"
    if config_file.is_file():
        os.environ["LIBERO_CONFIG_PATH"] = str(directory)
        return directory

    spec = importlib.util.find_spec("libero")
    roots = [] if spec is None else list(spec.submodule_search_locations or [])
    benchmark_root = next(
        (candidate for root in roots for candidate in (Path(root) / "libero", Path(root))
         if (candidate / "bddl_files").is_dir() and (candidate / "assets").is_dir()),
        None,
    )
    if benchmark_root is None:
        raise RuntimeError(
            "LIBERO is not installed or its assets are missing. Install the libero extra "
            "and run `python scripts/setup_libero.py`; see README."
        )
    directory.mkdir(parents=True, exist_ok=True)
    dataset_dir = directory / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    paths = {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
        "datasets": str(dataset_dir),
    }
    # Exclusive creation also avoids overwriting a config made by another process.
    try:
        with config_file.open("x", encoding="utf-8") as handle:
            json.dump(paths, handle, indent=2)
    except FileExistsError:
        pass
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    return directory


class LiberoAdapter:
    def __init__(self, suite: str = "libero_spatial", image_size: int = 256,
                 control_hz: int = 20, *, config_dir: str | Path | None = None,
                 settle_steps: int = 10, horizon: int = 1000) -> None:
        if suite not in SUITES:
            raise ValueError(f"Unknown LIBERO suite {suite!r}; choose one of {SUITES}")
        if not 64 <= image_size <= 1024:
            raise ValueError("image_size must be between 64 and 1024")
        if control_hz < 1 or settle_steps < 0 or horizon < 1:
            raise ValueError("Invalid environment timing")
        self.suite_name = suite
        self.image_size = image_size
        self.control_hz = control_hz
        self.config_dir = Path(config_dir) if config_dir else None
        self.settle_steps = settle_steps
        self.horizon = horizon
        self._suite: Any = None
        self._env: Any = None
        self._env_factory: Any = None
        self._get_path: Any = None
        self._randomization_error: type[Exception] = RuntimeError
        self._flip_images = True
        self._task: Any = None
        self._task_id: int | None = None
        self._episode_id = ""
        self._step_count = 0
        self._observation_sequence = 0
        self._init_state_id: int | None = None
        self._terminal = False
        self._stage = "task"
        self._reset_mode = "agent"
        self._reset_reference: dict[str, Any] | None = None
        self._free_space_pairs: dict[str, list[tuple[int, int]]] | None = None
        self._initial_gripper_open: bool | None = None
        self._spec = ActionSpec(
            names=("dx", "dy", "dz", "drx", "dry", "drz", "gripper"),
            low=(-1.0,) * 7, high=(1.0,) * 7, control_hz=float(control_hz),
            max_repeat=10, max_chunk_steps=30,
            description=(
                "LIBERO Panda OSC_POSE, normalized relative actions in [-1,1]. "
                "The first three values request a world-frame EEF translation, scaled "
                "by 0.05 metres per control tick. The next three are a world-frame "
                "axis-angle rotation VECTOR increment, scaled by 0.5 radians, "
                "NOT Euler angles. World +z is up; "
                "use camera calibration to relate world x/y to image directions. "
                "The last value controls the gripper: negative opens, positive closes, "
                "zero retains its current command. Position feedback is metres; "
                "EEF quaternion feedback uses [x,y,z,w]. A target increment is "
                "recomputed from measured pose on EVERY repeated tick, so repeat=10 "
                "applies the increment ten times, not one movement held for ten ticks. "
                "Use small values and short repeats near objects. Targets are not "
                "guaranteed actual displacement. Simulation time stops between steps."
            ),
        )

    @property
    def action_spec(self) -> ActionSpec:
        return self._spec

    def _load_runtime(self) -> None:
        if self._suite is not None:
            return
        prepare_libero_config(self.config_dir)
        try:
            import robosuite
            from robosuite import macros
            from robosuite.utils.errors import RandomizationError
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise RuntimeError(
                f"LIBERO runtime dependency is missing ({exc.name or 'unknown module'}). Install `.[libero]` and "
                "run `python scripts/setup_libero.py`."
            ) from exc
        if not str(robosuite.__version__).startswith("1.4."):
            raise RuntimeError("This adapter targets robosuite 1.4.1; install the pinned libero extra.")
        self._flip_images = macros.IMAGE_CONVENTION == "opengl"
        self._suite = benchmark.get_benchmark_dict()[self.suite_name]()
        self._env_factory = OffScreenRenderEnv
        self._get_path = get_libero_path
        self._randomization_error = RandomizationError

    def list_tasks(self) -> list[dict[str, Any]]:
        self._load_runtime()
        return [{"id": index, "name": self._suite.get_task(index).name,
                 "language": self._suite.get_task(index).language}
                for index in range(self._suite.n_tasks)]

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": "libero", "suite": self.suite_name, "robot": "Panda",
            "controller": "OSC_POSE", "simulation": True,
            "camera_names": list(CAMERAS), "image_size": self.image_size,
            "action_spec": self._spec.to_dict(),
            "gripper_control": {"action_index": 6, "close_sign": 1, "deadband": 0.1,
                                "initial_open": self._initial_gripper_open},
            "observation_notes": (
                "Images are upright RGB. EEF position is in the simulation world frame; "
                "all quaternions are xyzw. robot0_eef_pos is the gripper grip-site position; "
                "robot0_eef_quat is the robot hand-body orientation, while "
                "robot0_grip_site_quat is the actual OSC control-site orientation. "
                "Camera transforms map OpenCV camera coordinates "
                "(+x right,+y down,+z forward) into world coordinates. Only measured "
                "robot proprioception is exposed, not object ground-truth poses. "
                "Task success uses LIBERO's predicate; physical reset success uses the "
                "independent reset verifier, never the model's claim. Preserve the other "
                "tabletop objects while approaching and manipulating the task object; "
                "plan a clear path using the images and inspect contacts after short actions."
            ),
        }
        if self._task is not None:
            result["task_id"] = self._task_id
            result["task"] = self._reset_instruction() if self._stage == "reset" else self._task.language
        result["agent_reset"] = {
            "supported": self._reset_reference is not None,
            "stage": self._stage,
        }
        free_reference = (self._reset_reference or {}).get("free_space", {})
        result["free_space_reset"] = {
            "supported": bool(free_reference.get("verified")),
            "simulation_only": True,
            "reason": free_reference.get("reason", "This task has no verified free-space reset reference."),
            "position_radius_m": FREE_SPACE_RADIUS_M,
            "environment_clearance_m": FREE_SPACE_ENV_CLEARANCE_M,
            "self_clearance_m": FREE_SPACE_SELF_CLEARANCE_M,
            "max_joint_speed_rad_s": FREE_SPACE_JOINT_SPEED_RAD_S,
            "max_gripper_speed_m_s": FREE_SPACE_GRIPPER_SPEED_M_S,
            "verification": (
                "Restore all drawers and preserve other objects, open the gripper, and stop "
                "within the reference EEF region. Check collision-geometry distance for all "
                "moving robot links against environment geometry and nonadjacent robot links. "
                "Fixed mounting geometry and same/adjacent welded-link articulation pairs "
                "are excluded. This certifies only the current simulation endpoint, not the "
                "swept path or real-hardware return safety. No initial orientation match is required."
            ),
        }
        if self._reset_reference is not None:
            result["free_space_reset"]["target_position_m"] = self._reset_reference["robot"]["robot0_eef_pos"]
        if self._reset_reference is not None:
            result["agent_reset"].update({
                "drawer_level": self._reset_reference["drawer_level"],
                "instruction": self._reset_instruction(),
                "robot_target": self._reset_reference["robot"],
                "position_tolerance_m": 0.07,
                "orientation_tolerance_rad": 0.35,
                "reference_image": "reset_reference_agentview",
                "reference_image_notes": (
                    "During reset, reset_reference_agentview is the saved initial scene, "
                    "NOT a current camera frame. Match the current agentview to this "
                    "reference when returning displaced objects. reset_object_checks "
                    "identifies which objects still fail the return-position check."
                ),
                "verification": (
                    "Independent verifier checks that all cabinet drawers are within 2 cm of "
                    "their initial closed positions, the gripper is open, the EEF has returned "
                    "near its initial pose, and other objects have not moved more than 4 cm. "
                    "This is a task-ready approximate reset, not a bit-identical simulator reset. "
                    "No object or drawer ground-truth coordinates are exposed to the policy."
                ),
            })
        if self._env is not None:
            result["cameras"] = self._camera_calibration()
            result.update(self._robot_geometry_calibration())
        return result

    def _robot_geometry_calibration(self) -> dict[str, Any]:
        """Static workspace calibration and robot geometry, never object poses."""
        result: dict[str, Any] = {}
        base = getattr(self._env, "env", self._env)
        arena = getattr(getattr(base, "model", None), "mujoco_arena", None)
        tabletop = getattr(arena, "table_top_abs", None)
        if tabletop is not None:
            result["workspace"] = {
                "table_surface_z_m": float(np.asarray(tabletop)[2]),
                "source": "calibrated static arena tabletop, world coordinates",
                "note": "Table height is not the top height of objects resting on it.",
            }
        robots = getattr(self._env, "robots", None)
        if not robots:
            return result
        robot, sim = robots[0], self._env.sim
        hand_name = robot.robot_model.eef_name
        grip_name = robot.gripper.important_sites["grip_site"]
        hand_rotation = np.asarray(sim.data.get_body_xmat(hand_name))
        grip_rotation = np.asarray(sim.data.get_site_xmat(grip_name))
        grip_position = np.asarray(sim.data.get_site_xpos(grip_name))
        pad_centers, pad_minima, pad_maxima = [], [], []
        for side in ("left_fingerpad", "right_fingerpad"):
            for name in robot.gripper.important_geoms[side]:
                geom_id = sim.model.geom_name2id(name)
                local_position = grip_rotation.T @ (sim.data.get_geom_xpos(name) - grip_position)
                half_extents = np.abs(grip_rotation.T @ sim.data.get_geom_xmat(name)) @ sim.model.geom_size[geom_id]
                pad_centers.append(float(local_position[2]))
                pad_minima.append(float(local_position[2] - half_extents[2]))
                pad_maxima.append(float(local_position[2] + half_extents[2]))
        result["gripper_geometry"] = {
            "hand_to_grip_site_translation_m": np.round(
                hand_rotation.T @ (grip_position - sim.data.get_body_xpos(hand_name)), 6).tolist(),
            "hand_to_grip_site_rotation_matrix": np.round(hand_rotation.T @ grip_rotation, 6).tolist(),
            "fingerpad_center_axial_offset_from_grip_site_m": round(float(np.mean(pad_centers)), 6),
            "fingerpad_axial_range_from_grip_site_m": [round(min(pad_minima), 6), round(max(pad_maxima), 6)],
            "note": (
                "Offsets are along the grip-site local +z axis, toward the fingertips. "
                "The grip site is already near the finger pads, not at the palm or wrist; "
                "do not subtract a whole finger length from robot0_eef_pos. Compare "
                "the measured grip-site height, these pad offsets, and calibrated table "
                "height before assuming contact. These bounds describe the pads only, "
                "not the entire gripper collision envelope."
            ),
        }
        return result

    def _initial_states(self, task_id: int) -> Any:
        import torch
        task = self._suite.get_task(task_id)
        path = Path(self._get_path("init_states")) / task.problem_folder / task.init_states_file
        if not path.is_file():
            raise RuntimeError(f"Missing LIBERO initial states: {path}. Run the setup script.")
        # The pinned upstream repository supplies NumPy-pickled benchmark states.
        # PyTorch 2.6 changed weights_only's default. Load only this trusted asset,
        # never an arbitrary filename supplied by the web client.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def reset(self, *, seed: int, task_id: int) -> Observation:
        self._load_runtime()
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < self._suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {self._suite.n_tasks - 1}]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self._env is None or self._task_id != task_id:
            self.close()
            self._task = self._suite.get_task(task_id)
            self._task_id = task_id
            bddl = (Path(self._get_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file)
            try:
                self._env = self._env_factory(
                    bddl_file_name=str(bddl), robots=["Panda"], controller="OSC_POSE",
                    camera_names=list(CAMERAS), camera_heights=self.image_size,
                    camera_widths=self.image_size, control_freq=self.control_hz,
                    # The adapter enforces a per-phase horizon. The underlying
                    # simulation must keep running across task -> physical reset
                    # -> task without rewinding its clock or controller state.
                    horizon=self.horizon + self.settle_steps, ignore_done=True,
                )
            except Exception as exc:
                self._env = None
                raise RuntimeError(
                    f"Could not create LIBERO ({type(exc).__name__}: {exc}). "
                    "Check task assets and renderer. Set MUJOCO_GL=egl for a working "
                    "GPU, or MUJOCO_GL=osmesa for a machine with libOSMesa, before starting RILO."
                ) from exc
        self._env.seed(seed)
        # The upstream wrapper retries placement forever. Bound the retries here.
        base = getattr(self._env, "env", self._env)
        for attempt in range(10):
            try:
                raw = base.reset()
                break
            except self._randomization_error:
                if attempt == 9:
                    raise RuntimeError("LIBERO reset failed after 10 placement attempts")
        states = self._initial_states(task_id)
        if len(states) == 0:
            raise RuntimeError("LIBERO initial-state file is empty")
        self._init_state_id = seed % len(states)
        raw = self._env.set_init_state(states[self._init_state_id])
        self._episode_id = uuid.uuid4().hex
        self._step_count = 0
        self._observation_sequence = 0
        self._terminal = False
        self._stage = "task"
        self._reset_mode = "agent"
        self._reset_reference = None
        self._free_space_pairs = None
        # Settling is part of reset, never attributed to GPT's trajectory.
        settled = 0
        for _ in range(self.settle_steps):
            raw, _, done, _ = self._env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
            settled += 1
            if done:
                break
        self._initial_gripper_open = self._gripper_is_open(raw)
        self._capture_reset_reference(raw)
        return self._observation(raw, {"reset_settle_steps": settled})

    def _capture_reset_reference(self, raw: dict[str, Any]) -> None:
        """Privileged state is used only by the independent reset verifier."""
        drawer_level = RESET_DRAWER_TASKS.get(self._task.name)
        if drawer_level is None:
            return
        base = getattr(self._env, "env", self._env)
        cabinet_joints = tuple(name for name in self._env.sim.model.joint_names
                               if name.startswith("wooden_cabinet_1_")
                               and name.endswith(("top_level", "middle_level", "bottom_level")))
        if len(cabinet_joints) != 3:
            raise RuntimeError("Cannot identify the three cabinet joints for reset verification")
        robot_keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        if any(key not in raw for key in robot_keys):
            raise RuntimeError("Reset verification requires measured EEF pose and gripper state")
        joints = {name: float(self._env.sim.data.get_joint_qpos(name)) for name in cabinet_joints}
        if any(abs(value) > 0.025 for value in joints.values()):
            raise RuntimeError("This physical-reset task requires an initially closed cabinet")
        objects = {}
        for name in getattr(base, "objects_dict", {}):
            body_id = base.obj_body_id[name]
            objects[name] = np.asarray(self._env.sim.data.body_xpos[body_id]).copy()
        self._reset_reference = {
            "drawer_level": drawer_level,
            "joints": joints,
            "robot": {key: np.asarray(raw[key]).tolist() for key in robot_keys},
            "objects": objects,
            "agentview": np.ascontiguousarray(
                raw["agentview_image"][::-1] if self._flip_images else raw["agentview_image"],
                dtype=np.uint8).copy(),
        }
        self._capture_free_space_reference(raw)

    @staticmethod
    def _gripper_is_open(raw: dict[str, Any]) -> bool:
        grip = np.asarray(raw.get("robot0_gripper_qpos", []), dtype=float)
        return bool(grip.shape == (2,) and np.all(np.isfinite(grip)) and np.sum(np.abs(grip)) >= 0.06)

    def _capture_free_space_reference(self, raw: dict[str, Any]) -> None:
        reference = self._reset_reference
        reference["free_space"] = {"verified": False, "reason": "Initial clearance or velocity could not be verified."}
        self._free_space_pairs = None
        try:
            base = getattr(self._env, "env", self._env)
            if not hasattr(base, "objects_dict") or not hasattr(base, "obj_body_id"):
                raise ValueError("Missing scene-object inventory for independent restoration checks")
            self._free_space_pairs = self._collision_pairs()
            clearance = self._free_space_clearance()
            verified = (all(clearance.values()) and self._robot_is_stationary(raw)
                        and self._gripper_is_open(raw))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
            return
        reference["free_space"] = {"verified": bool(verified), "reason": (
            "Initial simulation endpoint independently verified." if verified else
            "Initial clearance, open gripper or low velocity did not meet the verifier limits.")}

    def _collision_pairs(self) -> dict[str, list[tuple[int, int]]]:
        """Model-owned collision bodies; never infer ownership from GPT output."""
        model = self._env.sim.model
        robots = self._env.robots
        if len(robots) != 1:
            raise ValueError("Free-space verification requires one known robot")
        root = int(model.body_name2id(robots[0].robot_model.root_body))
        parents = np.asarray(model.body_parentid, dtype=int)
        welds = np.asarray(model.body_weldid, dtype=int)
        bodies = np.asarray(model.geom_bodyid, dtype=int)
        types = np.asarray(model.geom_type, dtype=int)
        contact = (np.asarray(model.geom_contype) != 0) | (np.asarray(model.geom_conaffinity) != 0)
        if root <= 0 or root >= len(parents) or len(welds) != len(parents):
            raise ValueError("Invalid robot topology")
        if not (len(bodies) == len(types) == len(contact) == int(model.ngeom)):
            raise ValueError("Incomplete collision geometry")
        if (np.any(parents < 0) or np.any(parents >= len(parents)) or
                np.any(welds < 0) or np.any(welds >= len(parents)) or
                np.any(bodies < 0) or np.any(bodies >= len(parents))):
            raise ValueError("Invalid collision body indices")

        def is_robot(body):
            seen = set()
            while body:
                if body == root:
                    return True
                if body in seen:
                    raise ValueError("Cyclic body topology")
                seen.add(body)
                body = int(parents[body])
            return False

        robot_bodies = {body for body in range(len(parents)) if is_robot(body)}
        robot = [i for i in range(len(bodies)) if contact[i] and bodies[i] in robot_bodies]
        moving = [i for i in robot if welds[bodies[i]] != welds[root]]
        environment = [i for i in range(len(bodies)) if contact[i] and bodies[i] not in robot_bodies]
        if not moving or not environment:
            raise ValueError("Missing robot or environment collision geometry")
        # MuJoCo 3.2 enum: plane, sphere, capsule, ellipsoid, cylinder, box, mesh.
        # Heightfields and signed-distance fields require a separately verified path.
        if (any(types[i] not in (0, 2, 3, 4, 5, 6, 7) for i in robot + environment)
                or any(types[i] == 0 for i in robot)):
            raise ValueError("Unsupported collision geometry type")
        pairs = {"environment_clear": [(a, b) for a in moving for b in environment], "robot_self_clear": []}
        for i, first in enumerate(robot):
            for second in robot[i + 1:]:
                if first not in moving and second not in moving:
                    continue
                a, b = int(welds[bodies[first]]), int(welds[bodies[second]])
                # Native MuJoCo excludes same/adjacent welded links: touching
                # articulation housings are not a free-space collision failure.
                if a == b or welds[parents[a]] == b or welds[parents[b]] == a:
                    continue
                pairs["robot_self_clear"].append((first, second))
        return pairs

    def _geom_distance(self, first: int, second: int, limit: float) -> float:
        import mujoco
        sim = self._env.sim
        return float(mujoco.mj_geomDistance(sim.model._model, sim.data._data,
                                           first, second, limit, None))

    def _free_space_clearance(self) -> dict[str, bool]:
        checks = {"environment_clear": False, "robot_self_clear": False}
        if self._free_space_pairs is None:
            return checks
        try:
            model, data = self._env.sim.model, self._env.sim.data
            positions = np.asarray(data.geom_xpos, dtype=float)
            rotations = np.asarray(data.geom_xmat, dtype=float)
            radii = np.asarray(model.geom_rbound, dtype=float)
            types = np.asarray(model.geom_type, dtype=int)
            if (positions.shape != (model.ngeom, 3) or rotations.shape not in ((model.ngeom, 9), (model.ngeom, 3, 3))
                    or radii.shape != (model.ngeom,) or np.any(radii < 0)
                    or not all(np.all(np.isfinite(x)) for x in (positions, rotations, radii))):
                return checks
            for key, pairs in self._free_space_pairs.items():
                margin = FREE_SPACE_ENV_CLEARANCE_M if key == "environment_clear" else FREE_SPACE_SELF_CLEARANCE_M
                clear = True
                for first, second in pairs:
                    # Bounding spheres provide a conservative broad phase only;
                    # planes have unbounded extent and always use exact distance.
                    if types[first] != 0 and types[second] != 0 and (
                            np.linalg.norm(positions[first] - positions[second]) > radii[first] + radii[second] + margin):
                        continue
                    distance = self._geom_distance(first, second, margin * 2)
                    if not np.isfinite(distance) or distance < margin:
                        clear = False
                        break
                checks[key] = clear
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError, ImportError):
            return {"environment_clear": False, "robot_self_clear": False}
        return checks

    @staticmethod
    def _robot_is_stationary(raw: dict[str, Any]) -> bool:
        arm = np.asarray(raw.get("robot0_joint_vel", []), dtype=float)
        grip = np.asarray(raw.get("robot0_gripper_qvel", []), dtype=float)
        return bool(arm.shape == (7,) and grip.shape == (2,) and
                    np.all(np.isfinite(arm)) and np.all(np.isfinite(grip)) and
                    np.max(np.abs(arm)) <= FREE_SPACE_JOINT_SPEED_RAD_S and
                    np.max(np.abs(grip)) <= FREE_SPACE_GRIPPER_SPEED_M_S)

    def _free_space_reset_checks(self, raw: dict[str, Any]) -> dict[str, bool]:
        full = self._reset_checks(raw)
        reference = self._reset_reference or {}
        target = reference.get("robot", {}).get("robot0_eef_pos", [])
        position = np.asarray(raw.get("robot0_eef_pos", []), dtype=float)
        return {
            "reference_verified": bool(reference.get("free_space", {}).get("verified")),
            "drawers_restored": full.get("drawers_restored", False),
            "other_objects_preserved": full.get("other_objects_preserved", False),
            "gripper_open": self._gripper_is_open(raw) and full.get("gripper_open", False),
            "robot_in_retreat_region": bool(position.shape == (3,) and np.all(np.isfinite(position))
                and len(target) == 3 and np.linalg.norm(position - target) <= FREE_SPACE_RADIUS_M),
            "robot_stationary": self._robot_is_stationary(raw),
            **self._free_space_clearance(),
        }

    def _reset_instruction(self) -> str:
        level = self._reset_reference["drawer_level"] if self._reset_reference is not None else "middle"
        if self._reset_mode == "free_space":
            return (
                f"Restore this scene after the {level}-drawer task. Physically close all drawers "
                "and restore any displaced tabletop objects using the saved initial scene image. "
                "Release the handle and open the gripper fully. Retreat through a visibly clear "
                "path into environment.free_space_reset.target_position_m's documented region. "
                "You do not need to reproduce the initial EEF orientation. Keep every moving "
                "robot link clear of objects and stop motion; use short actions and inspect the "
                "independent reset_checks. A simulator return is permitted only after the "
                "independent endpoint verifier succeeds; your success claim cannot authorize it. "
                "Endpoint clearance does not establish that the preceding path was collision-free."
            )
        return (
            f"Restore this scene for another {level}-drawer opening demonstration. "
            f"Physically push the {level} drawer fully closed; leave the other drawers "
            "closed. Restore any displaced tabletop objects by comparing "
            "the current agentview with the saved reset_reference_agentview image; "
            "reset_object_checks lists objects that still need restoration. Open the "
            "gripper to release the handle, retreat clear of the cabinet, and return "
            "the EEF near the initial robot pose in environment.agent_reset.robot_target. "
            "Use the same numerical action API; no simulator reset or state teleport is available. "
            "The reset verifier must confirm completion before another task begins."
        )

    def _reset_checks(self, raw: dict[str, Any]) -> dict[str, bool]:
        reference = self._reset_reference
        if reference is None:
            return {}
        target = reference["robot"]
        actual_quat = np.asarray(raw["robot0_eef_quat"], dtype=float)
        target_quat = np.asarray(target["robot0_eef_quat"], dtype=float)
        denominator = float(np.linalg.norm(actual_quat) * np.linalg.norm(target_quat))
        similarity = 0. if denominator == 0 else abs(float(actual_quat @ target_quat)) / denominator
        angle = 2 * np.arccos(np.clip(similarity, 0., 1.))
        grip = np.asarray(raw["robot0_gripper_qpos"], dtype=float)
        initial_grip = np.asarray(target["robot0_gripper_qpos"], dtype=float)
        return {
            "drawers_restored": all(abs(float(self._env.sim.data.get_joint_qpos(name)) - initial) <= 0.02
                                    for name, initial in reference["joints"].items()),
            "gripper_open": bool(np.linalg.norm(grip - initial_grip) <= 0.012),
            "robot_retreated": bool(np.linalg.norm(np.asarray(raw["robot0_eef_pos"])
                                                   - target["robot0_eef_pos"]) <= 0.07),
            "orientation_restored": bool(angle <= 0.35),
            "other_objects_preserved": all(self._reset_object_checks().values()),
        }

    def _reset_object_checks(self) -> dict[str, bool]:
        if self._reset_reference is None:
            return {}
        base = getattr(self._env, "env", self._env)
        return {
            name: bool(np.linalg.norm(self._env.sim.data.body_xpos[base.obj_body_id[name]] - initial) <= 0.04)
            for name, initial in self._reset_reference["objects"].items()
        }

    def _begin_stage(self, stage: str) -> Observation:
        if self._env is None or not self._episode_id:
            raise RuntimeError("Initialize the environment before changing its objective")
        if self._reset_reference is None:
            raise RuntimeError("Agent reset is not supported for this LIBERO task")
        # Deliberately do not reset / set_state / set_init_state / step here.
        self._stage = stage
        self._episode_id = uuid.uuid4().hex
        self._step_count = 0
        self._observation_sequence = 0
        self._terminal = False
        return self.observe()

    def begin_reset(self) -> Observation:
        """Switch to a physical reset objective without changing the world."""
        self._reset_mode = "agent"
        return self._begin_stage("reset")

    def begin_free_space_reset(self) -> Observation:
        """Simulation-only release/retreat, verified without exact pose matching."""
        if not (self._reset_reference or {}).get("free_space", {}).get("verified"):
            raise RuntimeError("Free-space reset is not supported: no independently verified reference")
        self._reset_mode = "free_space"
        return self._begin_stage("reset")

    def begin_task(self) -> Observation:
        """Start the next task from the independently verified restored world."""
        if self._reset_mode == "free_space":
            raise RuntimeError("Free-space reset requires the verified simulator return before another task")
        if self._stage != "reset":
            raise RuntimeError("The next task requires a completed physical reset")
        observation = self.observe()
        if not observation.success:
            raise RuntimeError("The physical reset has not passed its independent verifier")
        return self._begin_stage("task")

    def observe(self) -> Observation:
        """Refresh sensors and renders without advancing simulation physics."""
        if self._env is None or not self._episode_id:
            raise RuntimeError("Reset the environment before observing")
        self._env.sim.forward()
        raw = self._env.env._get_observations(force_update=True)
        return self._observation(raw, {"observation_only": True})

    def step(self, values: Sequence[float]) -> Observation:
        if self._env is None or not self._episode_id:
            raise RuntimeError("Reset the environment before stepping")
        if self._terminal:
            raise RuntimeError("The episode has ended; reset before stepping again")
        applied = self._spec.validate(values)
        raw, reward, done, _ = self._env.step(np.asarray(applied, dtype=np.float64))
        self._step_count += 1
        return self._observation(raw, {"reward": float(reward), "applied_action": applied}, done=bool(done))

    def _observation(self, raw: dict[str, Any], info: dict[str, Any], *, done: bool = False) -> Observation:
        state: dict[str, Any] = {}
        for name in STATE_KEYS:
            if name in raw:
                value = np.asarray(raw[name])
                if not np.all(np.isfinite(value)):
                    raise RuntimeError(f"Non-finite robot observation: {name}")
                state[name] = value.tolist()
        images: dict[str, np.ndarray] = {}
        for camera in CAMERAS:
            key = f"{camera}_image"
            if key not in raw:
                raise RuntimeError(f"Required camera observation is missing: {key}")
            frame = np.asarray(raw[key])
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Invalid RGB image from {camera}: {frame.shape}")
            images[camera] = np.ascontiguousarray(frame[::-1] if self._flip_images else frame, dtype=np.uint8)
        if self._stage == "reset" and self._reset_reference is not None:
            images["reset_reference_agentview"] = self._reset_reference["agentview"].copy()
        task_success = bool(self._env.check_success())
        reset_checks = (self._free_space_reset_checks(raw) if self._reset_mode == "free_space"
                        else self._reset_checks(raw))
        reset_success = bool(reset_checks) and all(reset_checks.values())
        success = reset_success if self._stage == "reset" else task_success
        # LIBERO's BDDL step overrides ``done`` with its original task predicate
        # even with ignore_done=True. An open drawer therefore keeps returning
        # done during the physical close/reset phase; that is not a failure or
        # completion of the reset objective. Retain unexpected non-goal endings.
        unexpected_done = done and not task_success
        self._terminal = self._terminal or success or unexpected_done or self._step_count >= self.horizon
        self._observation_sequence += 1
        # These are sensor calibration / kinematic camera poses, not task state.
        state["camera_calibration"] = self._camera_calibration()
        robots = getattr(self._env, "robots", None)
        if robots:
            from robosuite.utils.transform_utils import mat2quat
            grip_name = robots[0].gripper.important_sites["grip_site"]
            state["robot0_grip_site_quat"] = mat2quat(
                self._env.sim.data.get_site_xmat(grip_name)).tolist()
        return Observation(
            id=f"{self._episode_id}:{self._step_count}:{self._observation_sequence}", timestamp=time.time(),
            task=self._reset_instruction() if self._stage == "reset" else self._task.language,
            state=state, images=images, success=success,
            terminated=self._terminal,
            info={**info, "suite": self.suite_name, "task_id": self._task_id,
                  "stage": self._stage, "task_success": task_success,
                  "upstream_done": done,
                  "reset_success": reset_success,
                  "reset_checks": reset_checks if self._stage == "reset" else {},
                  "reset_object_checks": self._reset_object_checks() if self._stage == "reset" else {},
                  "initial_state_id": self._init_state_id, "control_step": self._step_count,
                  "simulation_time": float(getattr(self._env.sim.data, "time", self._step_count / self.control_hz)),
                  "episode_control_time": self._step_count / self.control_hz,
                  "time_limit": self._step_count >= self.horizon},
        )

    def _camera_calibration(self) -> dict[str, Any]:
        sim = self._env.sim
        result = {}
        for camera in CAMERAS:
            camera_id = sim.model.camera_name2id(camera)
            focal = 0.5 * self.image_size / np.tan(float(sim.model.cam_fovy[camera_id]) * np.pi / 360)
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3) @ np.diag([1., -1., -1.])
            pose[:3, 3] = sim.data.cam_xpos[camera_id]
            result[camera] = {
                "intrinsics": [[float(focal), 0., self.image_size / 2],
                               [0., float(focal), self.image_size / 2], [0., 0., 1.]],
                "camera_to_world": pose.tolist(),
                "width": self.image_size, "height": self.image_size,
            }
        return result

    def pause(self) -> None:
        """No background physics thread: not calling step freezes simulation."""

    def close(self) -> None:
        if self._env is not None:
            environment, self._env = self._env, None
            self._episode_id = ""
            self._reset_reference = None
            self._free_space_pairs = None
            self._initial_gripper_open = None
            environment.close()
