"""Contract tests use a fake runtime; real smoke checks are opt-in/manual."""
from types import SimpleNamespace

import numpy as np
import pytest

from rilo.adapters.libero import CAMERAS, LiberoAdapter, prepare_libero_config


class FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.actions = []
        self.closed = False
        self.success = False
        self.done = False
        self.eef_position = np.array([0., 0., 0.5])
        self.refreshes = 0
        self.resets = 0
        self.state_sets = 0
        self.gripper_position = np.array([0.04, -0.04])
        self.eef_quaternion = np.array([0., 0., 0., 1.])
        self.drawer_positions = {f"wooden_cabinet_1_{level}_level": 0.
                                 for level in ("top", "middle", "bottom")}
        self.env = self
        self.sim = SimpleNamespace(
            model=SimpleNamespace(camera_name2id=CAMERAS.index, cam_fovy=[45., 45.],
                                  joint_names=list(self.drawer_positions)),
            data=SimpleNamespace(cam_xpos=np.zeros((2, 3)),
                                 cam_xmat=np.stack([np.eye(3).flatten()] * 2),
                                 get_joint_qpos=self.drawer_positions.__getitem__, time=1.0),
            forward=lambda: None,
        )

    def observation(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[0] = [255, 0, 0]
        frame[-1] = [0, 255, 0]
        return {
            "agentview_image": frame,
            "robot0_eye_in_hand_image": frame.copy(),
            "robot0_eef_pos": self.eef_position.copy(),
            "robot0_eef_quat": self.eef_quaternion.copy(),
            "robot0_gripper_qpos": self.gripper_position.copy(),
            "robot0_joint_pos": np.zeros(7),
            "object-state": np.array([999.]),
            "bowl_pos": np.array([99., 99., 99.]),
        }

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.resets += 1
        return self.observation()

    def _get_observations(self, force_update=False):
        assert force_update
        self.refreshes += 1
        return self.observation()

    def set_init_state(self, state):
        self.state_sets += 1
        self.selected_state = state
        return self.observation()

    def step(self, values):
        self.actions.append(list(values))
        # Reward deliberately disagrees with success predicate.
        return self.observation(), 1.0, self.done, {"privileged": "do not expose"}

    def check_success(self):
        return self.success

    def close(self):
        self.closed = True


@pytest.fixture
def adapter(monkeypatch):
    adapter = LiberoAdapter(image_size=64, settle_steps=0)
    task = SimpleNamespace(name="test", language="Put the bowl on the plate",
                           problem_folder="libero_spatial", bddl_file="task.bddl")
    adapter._suite = SimpleNamespace(n_tasks=2, get_task=lambda i: task)
    adapter._get_path = lambda name: "/unused"
    adapter._env_factory = FakeEnv
    monkeypatch.setattr(adapter, "_initial_states", lambda i: [np.array([1]), np.array([2])])
    return adapter


def test_reset_uses_seeded_initial_state_and_normalizes_observation(adapter):
    observation = adapter.reset(seed=3, task_id=0)
    assert adapter._env.seed_value == 3
    assert adapter._env.selected_state.tolist() == [2]
    assert observation.info["initial_state_id"] == 1
    assert observation.state["robot0_eef_quat"] == [0., 0., 0., 1.]
    assert "object-state" not in observation.state
    assert "bowl_pos" not in observation.state
    assert observation.images["agentview"][0, 0].tolist() == [0, 255, 0]
    assert observation.images["agentview"].flags.c_contiguous
    assert observation.state["camera_calibration"]["agentview"]["camera_to_world"][2][2] == -1


def test_step_is_exactly_one_tick_and_success_uses_environment_predicate(adapter):
    start = adapter.reset(seed=0, task_id=0)
    result = adapter.step([0.] * 7)
    assert len(adapter._env.actions) == 1
    assert result.id != start.id
    assert result.info["control_step"] == 1
    assert result.info["reward"] == 1.0
    assert not result.success
    adapter._env.success = True
    result = adapter.step([0.] * 7)
    assert result.success and result.terminated
    with pytest.raises(RuntimeError, match="episode has ended"):
        adapter.step([0.] * 7)


def test_pause_does_not_step_and_bad_actions_never_reach_environment(adapter):
    adapter.reset(seed=0, task_id=0)
    adapter.pause()
    for values in ([0.] * 6, [float("nan")] + [0.] * 6, [1.1] + [0.] * 6):
        with pytest.raises(ValueError):
            adapter.step(values)
    assert adapter._env.actions == []


def test_task_switch_closes_previous_environment(adapter):
    adapter.reset(seed=0, task_id=0)
    old_env = adapter._env
    adapter.reset(seed=1, task_id=1)
    assert old_env.closed
    assert adapter._env is not old_env
    assert adapter.list_tasks()[0]["language"] == "Put the bowl on the plate"


def test_horizon_ends_episode_independently_of_success(adapter):
    adapter.horizon = 1
    adapter.reset(seed=0, task_id=0)
    result = adapter.step([0.] * 7)
    assert result.terminated and not result.success
    assert result.info["time_limit"]


def test_existing_config_is_preserved(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("existing: configuration\n")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path))
    assert prepare_libero_config() == tmp_path
    assert config.read_text() == "existing: configuration\n"


def test_robosuite_opencv_images_are_not_flipped_twice(adapter):
    adapter._flip_images = False
    result = adapter.reset(seed=0, task_id=0)
    assert result.images["agentview"][0, 0].tolist() == [255, 0, 0]


def test_invalid_task_and_seed_fail_before_constructing_environment(adapter):
    with pytest.raises(ValueError, match="task_id"):
        adapter.reset(seed=0, task_id=2)
    with pytest.raises(ValueError, match="seed"):
        adapter.reset(seed=-1, task_id=0)
    assert adapter._env is None


def test_observe_refreshes_state_without_advancing_physics(adapter):
    first = adapter.reset(seed=0, task_id=0)
    adapter._env.eef_position = np.array([0.1, 0.2, 0.3])
    second = adapter.observe()
    third = adapter.observe()
    assert second.state["robot0_eef_pos"] == [0.1, 0.2, 0.3]
    assert len({first.id, second.id, third.id}) == 3
    assert second.info["control_step"] == third.info["control_step"] == 0
    assert adapter._env.actions == []
    assert adapter._env.refreshes == 2


def test_observe_keeps_termination_latched(adapter):
    adapter.reset(seed=0, task_id=0)
    adapter._env.done = True
    assert adapter.step([0.] * 7).terminated
    adapter._env.done = False
    assert adapter.observe().terminated
    with pytest.raises(RuntimeError, match="episode has ended"):
        adapter.step([0.] * 7)


def _use_drawer_task(adapter):
    adapter._suite.get_task(0).name = "open_the_middle_drawer_of_the_cabinet"
    adapter._suite.get_task(0).language = "Open the middle drawer of the cabinet"


def test_physical_reset_switches_objective_without_rewinding_world(adapter):
    _use_drawer_task(adapter)
    initial = adapter.reset(seed=0, task_id=0)
    env = adapter._env
    assert env.kwargs["ignore_done"] is True
    env.drawer_positions["wooden_cabinet_1_middle_level"] = -0.15
    env.success = True
    assert adapter.step([0.] * 7).success
    counts = (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    restoring = adapter.begin_reset()
    assert adapter._env is env
    assert counts == (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    assert restoring.info["stage"] == "reset"
    assert restoring.info["task_success"] is True
    assert not restoring.success and not restoring.terminated
    assert np.array_equal(restoring.images["reset_reference_agentview"], initial.images["agentview"])
    assert restoring.info["control_step"] == 0
    assert "Physically push" in restoring.task
    adapter.step([0.] * 7)  # Task success is no longer the active end condition.
    with pytest.raises(RuntimeError, match="not passed"):
        adapter.begin_task()

    env.drawer_positions["wooden_cabinet_1_middle_level"] = 0.
    env.success = False
    env.gripper_position[:] = 0.
    assert not adapter.observe().success
    env.gripper_position[:] = [0.04, -0.04]
    env.eef_position[0] = 0.2
    assert not adapter.observe().success
    env.eef_position[:] = initial.state["robot0_eef_pos"]
    env.eef_quaternion[:] = [1., 0., 0., 0.]
    assert not adapter.observe().success
    env.eef_quaternion[:] = initial.state["robot0_eef_quat"]
    assert adapter.observe().success
    counts = (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    next_task = adapter.begin_task()
    assert counts == (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    assert next_task.info["stage"] == "task"
    assert not next_task.success and not next_task.terminated
    assert next_task.info["control_step"] == 0
    assert next_task.id != restoring.id
    assert "reset_reference_agentview" not in next_task.images
    assert "drawer_positions" not in next_task.state
    assert "joints" not in adapter.describe()["agent_reset"]


def test_reset_verifier_rejects_changed_other_objects(adapter):
    _use_drawer_task(adapter)
    adapter.reset(seed=0, task_id=0)
    env = adapter._env
    env.objects_dict = {"bowl": object()}
    env.obj_body_id = {"bowl": 0}
    env.sim.data.body_xpos = np.array([[0., 0., 0.8]])
    adapter._capture_reset_reference(env.observation())
    env.sim.data.body_xpos[0, 2] = 0.
    reset_obs = adapter.begin_reset()
    assert not reset_obs.success
    assert reset_obs.info["reset_checks"]["other_objects_preserved"] is False
    assert reset_obs.info["reset_object_checks"] == {"bowl": False}
    assert "objects" not in adapter.describe()["agent_reset"]


def test_reset_ignores_libero_done_while_original_task_stays_successful(adapter):
    _use_drawer_task(adapter)
    adapter.reset(seed=0, task_id=0)
    env = adapter._env
    env.drawer_positions["wooden_cabinet_1_middle_level"] = -0.15
    env.success = env.done = True
    assert adapter.step([0.] * 7).success
    assert not adapter.begin_reset().terminated
    # Upstream BDDL step returns done=True for every tick the drawer remains
    # open, despite ignore_done=True. Closing needs multiple action ticks.
    for _ in range(3):
        obs = adapter.step([0.] * 7)
        assert obs.info["upstream_done"] is True
        assert obs.info["task_success"] is True
        assert not obs.success and not obs.terminated
    env.drawer_positions["wooden_cabinet_1_middle_level"] = -0.08
    env.success = env.done = False
    assert not adapter.step([0.] * 7).terminated
    env.done = True  # An unexpected ending unrelated to the task is retained.
    assert adapter.step([0.] * 7).terminated


def test_unsupported_task_cannot_use_agent_reset(adapter):
    adapter.reset(seed=0, task_id=0)
    assert not adapter.describe()["agent_reset"]["supported"]
    with pytest.raises(RuntimeError, match="not supported"):
        adapter.begin_reset()


def test_next_task_requires_explicit_completed_reset_stage(adapter):
    _use_drawer_task(adapter)
    adapter.reset(seed=0, task_id=0)
    with pytest.raises(RuntimeError, match="requires a completed"):
        adapter.begin_task()


@pytest.mark.parametrize("scene", ["KITCHEN_SCENE1", "KITCHEN_SCENE2"])
def test_top_drawer_tasks_use_top_reset_instruction_and_preserve_physics(adapter, scene):
    task = adapter._suite.get_task(0)
    task.name = f"{scene}_open_the_top_drawer_of_the_cabinet"
    task.language = "Open the top drawer of the cabinet"
    adapter.reset(seed=0, task_id=0)
    env = adapter._env
    env.drawer_positions["wooden_cabinet_1_top_level"] = -0.15
    env.success = env.done = True
    assert adapter.step([0.] * 7).success
    counts = (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    obs = adapter.begin_reset()
    assert counts == (env.resets, env.state_sets, len(env.actions), env.sim.data.time)
    assert adapter.describe()["agent_reset"]["drawer_level"] == "top"
    assert "top drawer fully closed" in obs.task
    assert "middle-drawer" not in obs.task
    assert not adapter.step([0.] * 7).terminated
    env.drawer_positions["wooden_cabinet_1_top_level"] = 0.
    env.success = env.done = False
    assert adapter.observe().success
    assert adapter.begin_task().task == "Open the top drawer of the cabinet"


def test_geometry_calibration_distinguishes_hand_grip_site_and_table(adapter):
    adapter.reset(seed=0, task_id=0)
    env = adapter._env
    env.model = SimpleNamespace(mujoco_arena=SimpleNamespace(table_top_abs=np.array([0., 0., 0.9])))
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    site = np.array([0., 0., 0.097])
    env.robots = [SimpleNamespace(
        robot_model=SimpleNamespace(eef_name="hand"),
        gripper=SimpleNamespace(important_sites={"grip_site": "grip"},
            important_geoms={"left_fingerpad": ["left_pad"], "right_fingerpad": ["right_pad"]}))]
    env.sim.data.get_body_xmat = lambda name: np.eye(3)
    env.sim.data.get_body_xpos = lambda name: np.zeros(3)
    env.sim.data.get_site_xmat = lambda name: rotation
    env.sim.data.get_site_xpos = lambda name: site
    centers = {"left_pad": site + rotation @ [0., 0.04, -0.0036],
               "right_pad": site + rotation @ [0., -0.04, -0.0036]}
    env.sim.data.get_geom_xpos = centers.__getitem__
    env.sim.data.get_geom_xmat = lambda name: rotation
    env.sim.model.geom_name2id = ["left_pad", "right_pad"].index
    env.sim.model.geom_size = np.array([[0.008, 0.004, 0.008]] * 2)
    description = adapter.describe()
    assert description["workspace"]["table_surface_z_m"] == 0.9
    geometry = description["gripper_geometry"]
    assert geometry["hand_to_grip_site_translation_m"] == [0., 0., 0.097]
    assert geometry["hand_to_grip_site_rotation_matrix"] == rotation.tolist()
    assert geometry["fingerpad_center_axial_offset_from_grip_site_m"] == -0.0036
    assert geometry["fingerpad_axial_range_from_grip_site_m"] == [-0.0116, 0.0044]
    assert "body_xpos" not in description
