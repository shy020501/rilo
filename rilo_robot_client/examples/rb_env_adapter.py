"""Explicit integration template; importing this module never initializes ROS.

Copy and fill build_bridge_and_decoder with your existing executor hooks. The
unknown run_episode/ChunkInterpreter methods are deliberately not guessed here.
"""


def create_adapter(*, config, env_config):
    # Fail BEFORE SDK initialization until the local integration is explicitly supplied.
    hook_module = config.get("bridge_factory")
    if not hook_module or ":" not in hook_module:
        raise ValueError("Set adapter_options.bridge_factory to your explicit module:factory bridge implementation")
    import importlib
    module, name = hook_module.split(":", 1)
    make_bridge = getattr(importlib.import_module(module), name)
    # Establish ROS before rb_env imports that may themselves load ROS modules.
    from Real_Robo.ros_environment import ensure_ros_environment
    ensure_ros_environment()
    # These imports and constructor names come from the operator's existing code.
    from rb_env.envs.builder_integration import (
        build_environment, load_config, resolve_action_chunk_mode,
        resolve_policy_sides, resolve_send_chunk_length,
    )
    from rb_env.envs.chunk_interpreter import ChunkInterpreter
    cfg = load_config(env_config)
    environment = build_environment(cfg)
    # make_bridge owns the previously established relative-progress settings,
    # measured observation mapping, queue replacement and proven hold/reset hooks.
    # It may construct ChunkInterpreter with the values used by your run_episode:
    # mode=resolve_action_chunk_mode(cfg), policy_sides=resolve_policy_sides(cfg),
    # relative_progress=..., relative_gripper_progress=...,
    # relative_gripper_action_mode='absolute', relative_action_frame_yaw_deg=0,
    # disable_policy_rotation=False,
    # action_period_s=cfg['train'].get('policy_action_period_s', 0.1).
    # No constructor defaults for unknown robot semantics are invented here.
    return make_bridge(environment=environment, environment_config=cfg,
        interpreter_class=ChunkInterpreter, mode=resolve_action_chunk_mode(cfg),
        policy_sides=resolve_policy_sides(cfg), send_chunk_length=resolve_send_chunk_length(cfg),
        client_options=config)
