"""Policy inference adapters; no adapter here owns or moves a robot.

DP predicts a full chunk from actual control-tick observation history. Residual
correction is a separate one-tick operation, never a prediction of future feedback.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import threading

import numpy as np

from rilo.contracts import ActionSpec, Decision, Observation


class PolicyContractError(ValueError):
    pass


def _normalization(spec):
    low, high = np.asarray(spec.low, dtype=np.float64), np.asarray(spec.high, dtype=np.float64)
    return {"center": ((low + high) / 2).tolist(), "scale": ((high - low) / 2).tolist(),
            "formula": "environment = normalized * scale + center"}


class PolicyAdapter:
    def __init__(self, kind, action_spec, environment):
        self.action_spec, self.environment = action_spec, deepcopy(environment)
        self.phase, self.episode_id = "task", ""
        self._generation, self._closed = 0, False
        self._operation_lock = asyncio.Lock()
        self._worker_lock = threading.RLock()
        self.metadata = {"kind": kind, "model_id": None, "fingerprint": None,
            "observation_steps": 1, "action_horizon": 1, "execution_steps": 1,
            "action_dim": len(action_spec.names), "actions": "environment",
            "representation": "action_spec_vector", "action_spec": action_spec.to_dict(),
            "observation_spec": deepcopy(environment.get("observation_spec")), "simulation_only": False,
            "online_training": False, "checkpoint_save_supported": False}

    def _check(self, context=None, generation=None):
        if self._closed or generation is not None and generation != self._generation:
            raise asyncio.CancelledError("Policy episode or phase changed")
        if context and callable(context.get("is_current")) and not context["is_current"]():
            raise asyncio.CancelledError("Policy request is no longer current")

    def _observations(self, observations, *, count=1):
        if not isinstance(observations, (list, tuple)) or len(observations) < count:
            raise PolicyContractError(f"Supply at least {count} actual control-tick observations")
        selected = list(observations[-count:])
        if any(not isinstance(item, Observation) for item in selected):
            raise PolicyContractError("Policy inputs must be raw RILO Observations")
        if len({item.id for item in selected}) != len(selected):
            raise PolicyContractError("Observation history must contain distinct control ticks")
        if any(not np.isfinite(item.timestamp) for item in selected):
            raise PolicyContractError("Observation timestamps must be finite")
        if any(left.timestamp > right.timestamp for left, right in zip(selected, selected[1:])):
            raise PolicyContractError("Observation history must be chronological")
        ticks = [item.info.get("control_step") for item in selected]
        if all(isinstance(tick, int) and not isinstance(tick, bool) for tick in ticks):
            if any(right != left + 1 for left, right in zip(ticks, ticks[1:])):
                raise PolicyContractError("Observation history must contain consecutive actual control ticks")
        for observation in selected:
            if not observation.id or not observation.images:
                raise PolicyContractError("An observation ID and camera images are required")
            for frame in observation.images.values():
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise PolicyContractError("Camera images must be raw upright HWC RGB uint8, not normalized tensors")
        return selected

    @staticmethod
    def _terminal(observation):
        if not (observation.success or observation.terminated):
            return None
        decision = Decision(observation_id=observation.id,
            status="success" if observation.success else "failure", note="Environment-reported terminal observation")
        return {"actions": None, "decision": decision.model_dump(),
                "outcome": "success" if observation.success else "time_limit" if observation.info.get("time_limit") else "failure"}

    async def begin_episode(self, episode_id):
        self._check()
        if not isinstance(episode_id, str) or not episode_id:
            raise PolicyContractError("An episode ID is required")
        self._generation += 1
        self.episode_id, self.phase = episode_id, "task"

    async def begin_phase(self, stage):
        self._check()
        if stage not in ("task", "reset"):
            raise PolicyContractError("Policy phase must be task or reset")
        self._generation += 1
        self.phase = stage

    async def on_transition(self, before, action, after, metadata):
        return None

    async def close(self):
        self._closed = True
        self._generation += 1


class DummyPolicyAdapter(PolicyAdapter):
    def __init__(self, action_spec, environment):
        if environment.get("simulation") is not True:
            raise PolicyContractError("The dummy policy is simulation-only")
        super().__init__("dummy", action_spec, environment)
        self.metadata.update(simulation_only=True, actions="normalized", action_horizon=16,
                             action_normalization=_normalization(action_spec))

    async def infer(self, observations, context):
        self._check(context)
        latest = self._observations(observations)[-1]
        vector = np.asarray(self.environment.get("dummy_actions", {}).get(self.phase,
            [0.] * len(self.action_spec.names)), dtype=np.float32)
        if vector.shape != (len(self.action_spec.names),) or not np.isfinite(vector).all() or np.any(np.abs(vector) > 1):
            raise PolicyContractError("Dummy fixture actions must be finite normalized action vectors")
        return self._terminal(latest) or {"actions": np.tile(vector, (16, 1)),
                                         "decision": None, "actions_encoding": "normalized"}


class DPPolicyAdapter(PolicyAdapter):
    def __init__(self, action_spec, environment, checkpoint, *, device, seed, model_id=None):
        from rilo.learning.checkpoints import fingerprint, load_dp
        from rilo.learning.data import ObservationSpec
        import torch
        super().__init__("dp", action_spec, environment)
        self.model, self.checkpoint_metadata = load_dp(checkpoint, device=device)
        self.model.eval().requires_grad_(False)
        self.spec, self.config = self.model.spec, self.model.config
        self.seed = int(seed)
        self.device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if (tuple(action_spec.names) != self.spec.action_names
                or not np.allclose(action_spec.low, self.spec.action_low, rtol=0, atol=1e-6)
                or not np.allclose(action_spec.high, self.spec.action_high, rtol=0, atol=1e-6)):
            raise PolicyContractError("Client action names or bounds do not match the DP checkpoint")
        declared = environment.get("observation_spec")
        if declared is not None and ObservationSpec.model_validate(declared) != self.spec:
            raise PolicyContractError("Registered observation schema does not match the DP checkpoint")
        self.metadata.update(model_id=model_id or Path(checkpoint).parent.name, fingerprint=fingerprint(checkpoint),
            observation_steps=self.config.obs_steps, action_horizon=self.config.horizon,
            execution_steps=self.config.action_steps, actions="normalized", observation_spec=self.spec.model_dump(mode="json"),
            action_normalization=_normalization(action_spec), history="actual_control_ticks")

    def _arrays(self, observations):
        from rilo.learning.data import observation_arrays
        import torch
        arrays = [observation_arrays(item, self.spec) for item in observations]
        return (torch.from_numpy(np.stack([item[0] for item in arrays])).to(self.device).unsqueeze(0),
                torch.from_numpy(np.stack([item[1] for item in arrays])).to(self.device).unsqueeze(0))

    def _predict(self, observations):
        import torch
        with self._worker_lock, torch.inference_mode():
            images, states = self._arrays(observations)
            predicted = self.model.predict_chunk(images, states, generator=self.generator)[0].cpu().numpy()
            normalizer = self.metadata["action_normalization"]
            normalized = (predicted - normalizer["center"]) / normalizer["scale"]
            if normalized.shape != (self.config.horizon, self.spec.action_dim) or not np.isfinite(normalized).all():
                raise PolicyContractError("DP produced an invalid action chunk")
            if np.any(np.abs(normalized) > 1.00001):
                raise PolicyContractError("DP produced actions outside the checkpoint bounds")
            return np.clip(normalized, -1, 1).astype(np.float32)

    async def begin_episode(self, episode_id):
        await super().begin_episode(episode_id)
        def reset():
            with self._worker_lock:
                self.generator.manual_seed(self.seed)
        await asyncio.to_thread(reset)

    async def infer(self, observations, context):
        generation = self._generation
        self._check(context, generation)
        if self.phase != "task":
            raise PolicyContractError("A learned task policy cannot control GPT reset")
        selected = self._observations(observations, count=self.config.obs_steps)
        terminal = self._terminal(selected[-1])
        if terminal:
            return terminal
        async with self._operation_lock:
            self._check(context, generation)
            actions = await asyncio.to_thread(self._predict, selected)
            self._check(context, generation)
        return {"actions": actions, "decision": None, "actions_encoding": "normalized"}


class GPTPolicyAdapter(PolicyAdapter):
    def __init__(self, settings, action_spec, environment):
        from rilo.policies.openai_policy import OpenAIPolicy
        if not settings.api_key:
            raise PolicyContractError("OPENAI_API_KEY is required for the GPT policy")
        super().__init__("gpt", action_spec, environment)
        self.policy = OpenAIPolicy(model=settings.model, api_key=settings.api_key, base_url=settings.base_url,
            reasoning_effort=settings.reasoning_effort, timeout=settings.api_timeout,
            max_output_tokens=settings.max_output_tokens, prompt_file=settings.prompt_file)
        self.metadata.update(model_id=settings.model, reasoning_effort=settings.reasoning_effort,
                             action_horizon=action_spec.max_chunk_steps, execution_steps=action_spec.max_chunk_steps)

    @property
    def usage_callback(self):
        return self.policy.usage_callback

    @usage_callback.setter
    def usage_callback(self, callback):
        self.policy.usage_callback = callback

    async def begin_episode(self, episode_id):
        await super().begin_episode(episode_id)
        self.policy.on_episode_start(episode_id)

    async def begin_phase(self, stage):
        await super().begin_phase(stage)
        self.policy.on_stage_start(stage)

    def _validate(self, decision, observation):
        if decision.observation_id != observation.id:
            raise PolicyContractError("GPT returned a stale observation ID")
        if decision.status == "continue" and not decision.actions:
            raise PolicyContractError("Continuing GPT decisions require actions")
        if decision.status != "continue" and decision.actions:
            raise PolicyContractError("Non-continuing GPT decisions cannot contain movement")
        count = 0
        for action in decision.actions:
            self.action_spec.validate(action.values)
            if action.repeat > self.action_spec.max_repeat:
                raise PolicyContractError("GPT repeat exceeds the client action contract")
            count += action.repeat
        if count > self.action_spec.max_chunk_steps:
            raise PolicyContractError("GPT chunk exceeds the client action contract")
        if (self.policy.reasoning_effort == "auto" and decision.reasoning_effort == "medium"
                and decision.next_reasoning_effort == "xhigh" and decision.status != "reconsider"):
            raise PolicyContractError("Auto escalation must reconsider before movement")

    async def infer(self, observations, context):
        generation = self._generation
        self._check(context, generation)
        observation = self._observations(observations)[-1]
        terminal = self._terminal(observation)
        if terminal:
            return terminal
        public_context = {key: value for key, value in context.items() if key != "history" and not callable(value)}
        public_context = {**self.environment, **public_context, "stage": self.phase,
                          "action_spec": self.action_spec.to_dict()}
        history = list(context.get("history", []))
        async with self._operation_lock:
            for attempt in range(2):
                self._check(context, generation)
                can_request = context.get("can_request")
                if callable(can_request) and not can_request():
                    return {"actions": None, "actions_encoding": "environment", "outcome": "budget_limit",
                        "decision": Decision(observation_id=observation.id, status="failure",
                            actions=[], note="Run API request or cost limit reached").model_dump()}
                decision = await self.policy.decide(observation, self.action_spec, public_context, history)
                self._check(context, generation)
                self._validate(decision, observation)
                if decision.status == "reconsider":
                    if (attempt or self.policy.reasoning_effort != "auto" or decision.reasoning_effort != "medium"
                            or decision.next_reasoning_effort != "xhigh"):
                        raise PolicyContractError("Only one no-motion medium-to-xhigh reconsideration is allowed")
                    self.policy.accept_decision(decision)
                    history.append({"stage": self.phase, "observation_id": observation.id,
                                    "executed_steps": 0, "note": decision.note, "reasoning_effort": "medium",
                                    "next_reasoning_effort": "xhigh"})
                    continue
                self.policy.accept_decision(decision)
                vectors = [action.values for action in decision.actions for _ in range(action.repeat)]
                return {"actions": np.asarray(vectors, dtype=np.float32) if vectors else None,
                    "decision": decision.model_dump(), "actions_encoding": "environment",
                    "outcome": "unverified_success" if decision.status == "success" else decision.status}
        raise PolicyContractError("GPT reconsideration did not produce a decision")

    async def close(self):
        await super().close()
        await self.policy.close()


class ResidualPolicyAdapter(DPPolicyAdapter):
    def __init__(self, action_spec, environment, checkpoint, *, residual_checkpoint, device, seed,
                 training, model_id=None, residual_model_id=None):
        from rilo.learning.checkpoints import fingerprint
        from rilo.learning.residual import ResidualTD3
        from rilo.learning.rl_train import ResidualTrainer
        import torch
        super().__init__(action_spec, environment, checkpoint, device=device, seed=seed, model_id=model_id)
        self.training = bool(training)
        self.trainer = None
        if residual_checkpoint is None:
            if not training:
                raise PolicyContractError("Residual inference requires a registered residual checkpoint")
            self.trainer = ResidualTrainer(self.config.feature_dim + 1, self.spec.action_low, self.spec.action_high,
                                          device=device)
            self.trainer.base_fingerprint = self.metadata["fingerprint"]
            self.agent = self.trainer.agent
        else:
            payload = torch.load(residual_checkpoint, map_location="cpu", weights_only=True)
            expected_base = payload.get("base_fingerprint") or payload.get("metadata", {}).get("base_checkpoint_sha256")
            if expected_base != self.metadata["fingerprint"]:
                raise PolicyContractError("Residual checkpoint is not bound to this DP checkpoint")
            if payload.get("kind") == "residual_training":
                if training:
                    self.trainer = ResidualTrainer.load_checkpoint(residual_checkpoint, device=device,
                        expected_base_fingerprint=self.metadata["fingerprint"])
                    self.trainer.end_episode(terminated=False)
                    self.agent = self.trainer.agent
                else:
                    state = payload["agent"]
                    self.agent = ResidualTD3(state["feature_dim"], state["action_low"], state["action_high"],
                                             state["config"], device=device)
                    self.agent.load_state_dict(state)
            elif payload.get("kind") == "residual_td3":
                self.agent = ResidualTD3.load(residual_checkpoint, device=device)
                if training:
                    self.trainer = ResidualTrainer(self.agent.feature_dim, self.spec.action_low, self.spec.action_high,
                                                  self.agent.config, device=device)
                    self.trainer.agent = self.agent
                    self.trainer.base_fingerprint = self.metadata["fingerprint"]
            else:
                raise PolicyContractError("Unsupported residual checkpoint")
        if (self.agent.feature_dim != self.config.feature_dim + 1 or self.agent.action_dim != self.spec.action_dim
                or not np.allclose(self.agent.low.cpu(), self.spec.action_low, rtol=0, atol=1e-6)
                or not np.allclose(self.agent.high.cpu(), self.spec.action_high, rtol=0, atol=1e-6)):
            raise PolicyContractError("Residual feature or action schema differs from the DP checkpoint")
        self._seen_transitions = set()
        self.metadata.update(kind="residual", residual_model_id=residual_model_id,
            residual_fingerprint=fingerprint(residual_checkpoint) if residual_checkpoint else None,
            online_training=self.training, checkpoint_save_supported=self.training,
            operation_actions={"chunk": "normalized", "correct": "environment"}, correction_horizon=1,
            residual_scale=self.agent.config.residual_scale,
            transition_reward="environment_verified_success", progress_model=False, automatic_gpt_judge=False)

    def _phase(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.config.action_steps:
            raise PolicyContractError("chunk_phase must identify a tick within execution_steps")
        return value

    def _features(self, observation, phase):
        import torch
        with torch.inference_mode():
            images, states = self._arrays([observation])
            feature = self.model.encoder(images, states)[0, 0]
            return torch.cat((feature, feature.new_tensor([phase / self.config.action_steps]))).cpu().numpy()

    async def begin_episode(self, episode_id):
        await super().begin_episode(episode_id)
        self._seen_transitions.clear()
        if self.trainer is not None:
            await asyncio.to_thread(self._flush)

    def _flush(self):
        with self._worker_lock:
            self.trainer.end_episode(terminated=False)

    async def begin_phase(self, stage):
        await super().begin_phase(stage)
        if stage == "reset" and self.trainer is not None:
            await asyncio.to_thread(self._flush)

    async def infer(self, observations, context):
        operation = context.get("operation", "chunk")
        if operation == "chunk":
            return await super().infer(observations, context)
        if operation != "correct":
            raise PolicyContractError("Residual operation must be chunk or correct")
        generation = self._generation
        self._check(context, generation)
        if self.phase != "task":
            raise PolicyContractError("Residual correction is disabled during reset")
        observation = self._observations(observations)[-1]
        terminal = self._terminal(observation)
        if terminal:
            return terminal
        phase = self._phase(context.get("chunk_phase"))
        base = np.asarray(self.action_spec.validate(context.get("base_action", [])), dtype=np.float32)
        def correct():
            with self._worker_lock:
                features = self._features(observation, phase)
                return self.agent.act(features, base, explore=self.training)
        async with self._operation_lock:
            self._check(context, generation)
            action = await asyncio.to_thread(correct)
            self._check(context, generation)
        action = self.action_spec.validate(action)
        return {"actions": np.asarray([action], dtype=np.float32), "decision": None,
                "actions_encoding": "environment", "base_action": base.tolist(), "chunk_phase": phase}

    async def on_transition(self, before, action, after, metadata):
        generation, episode_id = self._generation, self.episode_id
        self._check(metadata, generation)
        if self.phase != "task" or metadata.get("phase", "task") != "task":
            return {"ignored": "reset"}
        if self.trainer is None:
            return {"training": False}
        transition_id = metadata.get("transition_id")
        if (metadata.get("executed") is not True or not isinstance(transition_id, str) or not transition_id
                or metadata.get("episode_id") != self.episode_id):
            raise PolicyContractError("Training requires an actual executed transition in the current episode")
        if transition_id in self._seen_transitions:
            return {"duplicate": True, "training": True}
        self._observations([before, after], count=2)
        base = self.action_spec.validate(metadata.get("base_action", []))
        phase = self._phase(metadata.get("chunk_phase"))
        applied = self.action_spec.validate(action)
        truncated = bool(metadata.get("truncated", after.info.get("time_limit", False))) and not after.success
        terminated = bool(after.success or after.terminated and not truncated)
        if metadata.get("bootstrap_unused") is True:
            if not terminated or truncated:
                raise PolicyContractError("Only a genuine terminal transition may omit bootstrap inputs")
            # This zero is a masked replay placeholder, never a proposed or
            # executed environment action (it need not lie within action bounds).
            next_base = np.zeros(len(self.action_spec.names), dtype=np.float32)
            next_phase = 0
        else:
            next_base = self.action_spec.validate(metadata.get("next_base_action", []))
            next_phase = self._phase(metadata.get("next_chunk_phase"))
        def record():
            with self._worker_lock:
                self._check(metadata, generation)
                features, next_features = self._features(before, phase), self._features(after, next_phase)
                self._check(metadata, generation)
                self.trainer.observe(features, base, applied, float(after.success), next_features,
                    next_base, terminated=terminated, truncated=truncated, episode_id=episode_id)
                self._seen_transitions.add(transition_id)
                return self.trainer.update()
        async with self._operation_lock:
            if transition_id in self._seen_transitions:
                return {"duplicate": True, "training": True}
            result = await asyncio.to_thread(record)
        return {"training": True, "transition_id": transition_id, "metrics": result}

    async def save_checkpoint(self, path):
        if self.trainer is None:
            raise PolicyContractError("Only enabled online training can save a checkpoint")
        def save():
            with self._worker_lock:
                self.trainer.save_checkpoint(path, metadata={"base_checkpoint_sha256": self.metadata["fingerprint"],
                    "execution_mode": "remote_per_tick_residual", "episode_id": self.episode_id})
        await asyncio.to_thread(save)

    async def close(self):
        await super().close()
        if self.trainer is not None:
            await asyncio.to_thread(self._flush)


async def create_policy(kind, *, settings, action_spec: ActionSpec, environment, checkpoint=None,
                        residual_checkpoint=None, device="cuda:0", seed=0, training=False,
                        model_id=None, residual_model_id=None):
    """Checkpoint Paths must already be authorized by the server model registry."""
    if training and kind != "residual":
        raise PolicyContractError("Online training is supported only for the residual adapter")
    if kind == "dummy":
        return DummyPolicyAdapter(action_spec, environment)
    if kind == "gpt":
        return GPTPolicyAdapter(settings, action_spec, environment)
    if kind not in ("dp", "residual"):
        raise PolicyContractError("Unknown remote policy kind")
    if checkpoint is None:
        raise PolicyContractError("A registered DP checkpoint is required")
    if kind == "dp":
        return await asyncio.to_thread(DPPolicyAdapter, action_spec, environment, checkpoint,
                                       device=device, seed=seed, model_id=model_id)
    return await asyncio.to_thread(ResidualPolicyAdapter, action_spec, environment, checkpoint,
        residual_checkpoint=residual_checkpoint, device=device, seed=seed, training=training,
        model_id=model_id, residual_model_id=residual_model_id)
