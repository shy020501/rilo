"""Lease-bound central run commands with local motion and durable evidence gates."""
from __future__ import annotations

import asyncio
import base64
import time
import uuid

from .decoder import verify_contract
from .history import ObservationHistory
from .http_client import RemoteError
from .wire import pack_message, unpack_message


class RunController:
    def __init__(self, client, journal, *, bridge=None, decoder=None, on_ready=None):
        self.client, self.journal, self.bridge, self.decoder = client, journal, bridge, decoder
        self.config, self.on_ready = client.config, on_ready
        self.history = ObservationHistory(self.config.history_capacity)
        self.session_id = None
        self.epoch, self.after_command_id = 0, 0
        self.phase, self.stage = "idle", "task"
        self.run = self.episode_id = None
        self.episode_index = self.steps = self.calls = 0
        self._generation, self._lease_until = 0, 0.
        self._closed = asyncio.Event()
        self._enabled = asyncio.Event()
        self._motion = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._tasks = []
        self.error = None
        self._prefetched = None
        self._dp_prefetch = None

    def status(self):
        return {"phase": self.phase, "stage": self.stage, "run_id": self.run.get("run_id") if self.run else None,
                "episode_id": self.episode_id, "epoch": self.epoch, "control_tick": self.history.latest.tick if len(self.history) else None,
                "error": self.error, "pending_receipts": len(self.journal.pending())}

    async def _hold(self, reason, *, phase=None):
        self._enabled.clear()
        self._generation += 1
        self._prefetched = None
        self._discard_dp_prefetch()
        if phase:
            self.phase = phase
        if self.bridge:
            async with self._motion:
                await self.bridge.clear_pending(reason)
                await self.bridge.hold(reason)

    def _discard_dp_prefetch(self):
        if self._dp_prefetch:
            task, self._dp_prefetch = self._dp_prefetch, None
            task.cancel()
            task.add_done_callback(lambda value: value.exception() if not value.cancelled() else None)

    def _live(self, generation=None):
        return (not self._closed.is_set() and self._enabled.is_set() and time.monotonic() < self._lease_until
                and (generation is None or generation == self._generation))

    def _meta(self, **extra):
        task = self.run.get("task", "") if self.run else self.bridge.describe().get("task", "") if self.bridge else ""
        return {"session_id": self.session_id, "epoch": self.epoch, "request_id": uuid.uuid4().hex,
                "task": task, **extra}

    @property
    def contract(self):
        return self.run["reset_contract"] if self.stage in ("reset", "arm_reset", "ending") else self.run["contract"]

    def _adopt_epoch(self, response):
        epoch = response.get("epoch", self.epoch)
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Invalid server epoch")
        self.epoch = max(self.epoch, epoch)

    async def serve(self):
        try:
            await self._serve()
        finally:
            self._closed.set()
            try:
                await self._hold("client shutdown", phase="stopped")
            finally:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
                if self.bridge:
                    await self.bridge.close()

    async def _serve(self):
        environment = self.bridge.describe() if self.bridge else {"name": "rilo-probe", "probe": True}
        if self.bridge and environment.get("action_spec", {}).get("control_hz") != self.config.control_hz:
            raise ValueError("Client control_hz must equal the adapter's measured control-loop contract")
        capabilities = environment.get("capabilities", {}) if self.bridge else {}
        if not isinstance(capabilities, dict) or any(type(value) is not bool for value in capabilities.values()):
            raise ValueError("Adapter capabilities must contain explicit booleans")
        capabilities = {name: capabilities.get(name, False) for name in (
            "execute", "hold", "arm_reset", "safe_reset", "residual_step", "residual_training", "simulation")}
        if self.bridge and environment.get("simulation", False) is not capabilities["simulation"]:
            raise ValueError("Environment simulation flag must match its explicit capability")
        registration = asyncio.create_task(self.client.register(self.config.client_id, environment, capabilities))
        closed = asyncio.create_task(self._closed.wait())
        try:
            done, _ = await asyncio.wait((registration, closed), return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                registration.cancel()
                await asyncio.gather(registration, return_exceptions=True)
                return
            response = await registration
        finally:
            closed.cancel()
            if not registration.done():
                registration.cancel()
            await asyncio.gather(closed, registration, return_exceptions=True)
        self.session_id = response["session_id"]
        self._adopt_epoch(response)
        self._lease_until = time.monotonic() + float(response["lease_seconds"])
        self.after_command_id = self.journal.get(f"ack:{self.session_id}", 0)
        if self.bridge:
            await self._hold("registered; waiting for explicit Run command")
            self.history.reset(await self.bridge.observe())
        if self.journal.uncertain_executions():
            self.error = "An earlier motion has no completion receipt; operator reconciliation is required"
            self.phase = "uncertain"
        if self.on_ready:
            self.on_ready()
        self._tasks = [asyncio.create_task(self._poll_loop()), asyncio.create_task(self._watchdog()),
                       asyncio.create_task(self._work_loop())]
        await self._closed.wait()

    async def shutdown(self):
        await self._hold("control channel closed", phase="stopped")
        self._closed.set()

    async def _watchdog(self):
        while not self._closed.is_set():
            if time.monotonic() >= self._lease_until and self.phase not in ("lease_expired", "uncertain", "stopped"):
                await self._hold("execution lease expired", phase="lease_expired")
            await asyncio.sleep(.05)

    async def _poll_loop(self):
        while not self._closed.is_set():
            try:
                response = await self.client.poll(self.session_id, self.after_command_id, self.status())
                if response["session_id"] != self.session_id:
                    raise ValueError("Poll returned a different session")
                self._lease_until = time.monotonic() + float(response["lease_seconds"])
                for command in response.get("commands", []):
                    await self._command(command)
                # Poll may follow a completed phase RPC; never move an epoch backwards.
                if response.get("epoch", self.epoch) > self.epoch and self._enabled.is_set():
                    await self._hold("server epoch changed without an applied run command", phase="paused")
                self._adopt_epoch(response)
            except (RemoteError, ValueError) as exc:
                self.error = str(exc)
                if self._enabled.is_set():
                    await self._hold("control connection interrupted", phase="lease_expired")
            await asyncio.sleep(.5)

    async def _command(self, command):
        key = f"command:{self.session_id}:{command['id']}"
        previous = self.journal.get(key)
        result = previous or {"status": "applied"}
        if previous is None:
            try:
                if type(command.get("epoch")) is not int or command["epoch"] < self.epoch:
                    raise ValueError("Command belongs to an old epoch")
                kind = command["type"]
                if kind in ("start", "resume") and (not self.bridge or self.journal.uncertain_executions()):
                    raise ValueError("This client cannot execute; a real bridge and reconciled journal are required")
                await self._hold(f"command {kind}")
                self.epoch = command["epoch"]
                if kind == "start":
                    if self.run and self.phase not in ("completed", "stopped", "idle"):
                        raise ValueError("Stop the previous run before starting another")
                    run = command["run"]
                    environment = self.bridge.describe()
                    required = environment.get("capabilities", {})
                    if not all(required.get(name) for name in ("execute", "hold", "arm_reset", "safe_reset")):
                        raise ValueError("Execution requires explicit hold, safe_reset, and arm_reset hooks")
                    verify_contract(run["contract"], environment)
                    verify_contract(run["reset_contract"], environment)
                    self.run, self.episode_id = run, None
                    self.episode_index, self.steps, self.calls = 1, 0, 0
                    self.stage, self.error = "task", None
                    await self.bridge.set_phase("task")
                    self.history.reset(await self.bridge.observe())
                    self.phase = "preparing"
                elif kind == "pause":
                    self.phase = "paused"
                elif kind == "resume":
                    if not self.run or self.phase in ("completed", "stopped", "uncertain"):
                        raise ValueError("No paused run is available to resume")
                    self.phase, self.error = "preparing", None
                elif kind == "stop":
                    self.phase = "stopped"
                    await self._release_pending_learning()
                    await self.flush()  # Stop ACK finalizes the server episode; preserve actual evidence first.
                else:
                    raise ValueError("Unknown run command")
            except (ValueError, KeyError) as exc:
                result = {"status": "rejected", "detail": str(exc)}
            self.journal.set(key, result)
        acknowledgment = await self.client.acknowledge(self.session_id, command, **result)
        if (result["status"] == "applied" and command["type"] in ("start", "resume")
                and command["epoch"] == self.epoch and not acknowledgment.get("stale")
                and self.phase == "preparing" and time.monotonic() < self._lease_until):
            self.phase = "running"
            self._enabled.set()
        self.after_command_id = max(self.after_command_id, command["id"])
        self.journal.set(f"ack:{self.session_id}", self.after_command_id)

    async def flush(self):
        async with self._flush_lock:
            for receipt in self.journal.pending():
                if receipt["kind"] == "transition_pending":
                    raise RuntimeError("An actual transition is waiting for its measured bootstrap action")
                item = receipt["payload"]
                response = await self.client.send_encoded(item["endpoint"], base64.b64decode(item["npz"]),
                                                          contract_hash=item["contract_hash"])
                self.journal.set("reply:" + receipt["id"], response)
                if self.run and item.get("run_id") == self.run["run_id"]:
                    await self._apply_rpc_result(item, response)
                self.journal.acknowledge(receipt["id"])

    async def _apply_rpc_result(self, item, response):
        endpoint, effect = item["endpoint"], item.get("effect", {})
        self._adopt_epoch(response)
        if endpoint == "episode/begin":
            self.episode_id = response["episode_id"]
        elif endpoint == "phase":
            self.stage = effect["target"]
            self.steps = self.calls = 0
            if self.stage == "reset":
                await self.bridge.set_phase("reset")
                self.history.reset(await self.bridge.observe())
        elif endpoint == "episode/end":
            if response["finished"]:
                await self._hold("run complete", phase="completed")
            else:
                self.episode_index = response["completed_episodes"] + 1
                self.episode_id, self.stage, self.steps, self.calls = None, "task", 0, 0
                await self.bridge.set_phase("task")
                self.history.reset(await self.bridge.observe())

    async def _rpc(self, path, metadata, observations):
        await self.flush()
        data = self.client.encode(metadata, observations)
        identifier = metadata["request_id"]
        self.journal.enqueue(identifier, "rpc", {"endpoint": path, "npz": base64.b64encode(data).decode(),
            "contract_hash": self.contract["hash"], "run_id": self.run["run_id"], "effect": metadata})
        await self.flush()
        return self.journal.get("reply:" + identifier)

    async def _work_loop(self):
        last_preview = 0.
        while not self._closed.is_set():
            try:
                if self._live():
                    await self._work_once()
                else:
                    pending = self.journal.pending()
                    if pending and not any(item["kind"] == "transition_pending" for item in pending):
                        # Uploading already-executed evidence is safe while held;
                        # a lost final ACK must not cause another arm reset.
                        await self.flush()
                    if self.bridge and self.phase == "idle" and time.monotonic() - last_preview >= 1:
                        observation = await self.bridge.observe()
                        response = await self.client.send_encoded("observe", self.client.encode(self._meta(), [observation]))
                        self._adopt_epoch(response)
                        last_preview = time.monotonic()
                    await asyncio.sleep(.05)
            except (RemoteError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                if not self._enabled.is_set() and self.phase in ("paused", "stopped", "lease_expired"):
                    await asyncio.sleep(.05)
                    continue
                self.error = str(exc)
                await self._hold("run needs operator attention", phase="error")
                await asyncio.sleep(.1)
            except asyncio.CancelledError:
                if self._closed.is_set():
                    raise
                # Pause canceled a speculative DP request, not this worker's lifetime.
                await asyncio.sleep(.01)

    async def _work_once(self):
        generation = self._generation
        if any(item["kind"] == "transition_pending" for item in self.journal.pending()):
            await self._resolve_learning()
        await self.flush()
        if not self._live(generation):
            return
        if self.episode_id is None:
            result = await self._rpc("episode/begin", self._meta(index=self.episode_index, seed=0,
                                     start_kind="current_state"), [self.history.latest])
            if self.run is None:
                return
            self.episode_id = result["episode_id"]
            self._adopt_epoch(result)
            return
        if self.stage == "arm_reset":
            await self._arm_reset()
            return
        if self.stage == "ending":
            raise RuntimeError("An arm reset is already recorded; reconcile its pending completion receipt")
        if self.history.latest.success and (self.stage == "task" or self._mock_reset()):
            await self._finish_phase()
            return
        steps_limit = self.run["limits"]["max_reset_steps" if self.stage == "reset" else "max_episode_steps"]
        if self.steps >= steps_limit:
            if self.stage == "task":
                await self._finish_phase(outcome="failure")
                return
            raise RuntimeError("Reset step limit reached; physical recovery requires operator attention")
        count = int(self.contract["observation_steps"])
        if not 1 <= count <= min(8, self.config.history_capacity):
            raise ValueError("Policy observation history does not fit the wire/history contract")
        while len(self.history) < count:
            await asyncio.sleep(1 / self.config.control_hz)
            if not self._live(generation):
                return
            async with self._motion:
                self.history.append(await self.bridge.sample_tick())
        call_limit = self.run["limits"]["max_reset_calls" if self.stage == "reset" else "max_task_calls"]
        if self.contract.get("kind") == "gpt" and self.calls >= call_limit:
            raise RuntimeError("Policy call limit reached")
        request_tick = self.history.latest.tick
        if self._dp_prefetch:
            task = self._dp_prefetch
            try:
                requested_generation, meta, arrays, request_tick = await task
            finally:
                if self._dp_prefetch is task:
                    self._dp_prefetch = None
            if requested_generation != generation:
                return
        elif self._prefetched and self._prefetched[0] == generation:
            _, meta, arrays = self._prefetched
            self._prefetched = None
        else:
            self.calls += 1
            meta, arrays = await self.client.infer(self._meta(operation="chunk"),
                self.history.sample(tuple(range(-count + 1, 1))), contract_hash=self.contract["hash"])
        if not self._live(generation) or meta.get("epoch") != self.epoch or meta.get("episode_id") != self.episode_id:
            return  # A late answer never revives a paused/stopped/replaced queue.
        decision = meta.get("decision") or {}
        if decision.get("status") in ("failure", "help"):
            raise RuntimeError("Policy requested failure/help; operator intervention is required")
        actions = arrays.get("actions")
        if actions is None or not len(actions):
            if decision.get("status") == "success" and self.stage == "reset":
                # Task-success bits can remain true while objects still need
                # restoration. A GPT reset completion also needs local clearance.
                await self._finish_phase()
                return
            if decision.get("status") == "success" and self.history.latest.success:
                await self._finish_phase()
                return
            raise RuntimeError("Policy provided no executable action or verified success")
        raw = {**meta, "actions": actions}
        decoded = self.decoder.decode(raw, context={"contract": self.contract, "phase": self.stage,
            "request_control_tick": request_tick, "current_control_tick": self.history.latest.tick,
            "observation_delay_ticks": self.history.latest.tick - request_tick})
        length = min(len(decoded), int(self.contract.get("execution_steps", len(decoded))), steps_limit - self.steps)
        if length < 1:
            raise ValueError("Policy chunk has no permitted execution steps")
        async with self._motion:
            if not self._live(generation):
                return
            await self.bridge.replace_pending(decoded[:length], epoch=self.epoch)
        for index, action in enumerate(decoded[:length]):
            if not self._live(generation):
                return
            if self.run.get("policy") == "residual" and self.stage == "task":
                # Correction happens on the freshest measured tick, not chunk-request state.
                self.calls += 1
                correction, values = await self.client.infer(self._meta(operation="correct",
                    base_action=action.record["values"], chunk_phase=index), [self.history.latest],
                    contract_hash=self.contract["hash"])
                if not self._live(generation) or correction.get("epoch") != self.epoch:
                    return
                corrected = self.decoder.decode({**correction, "actions": values["actions"]},
                    context={"contract": self.contract, "phase": self.stage})
                async with self._motion:
                    if not self._live(generation):
                        return
                    await self.bridge.replace_pending(corrected[:1], epoch=self.epoch)
            tick_started = time.monotonic()
            await self._tick(meta, index, generation, action,
                             decoded[index + 1] if index + 1 < length else None)
            if any(item["kind"] == "transition_pending" for item in self.journal.pending()):
                await self._resolve_learning()
            if self._terminal():
                async with self._motion:
                    await self.bridge.clear_pending("episode boundary")
                break
            if (self.stage == "task" and self.run["policy"] == "dp" and index == length - 2
                    and self._dp_prefetch is None and self._live(generation)):
                snapshot = self.history.sample(tuple(range(-count + 1, 1)))
                self._dp_prefetch = asyncio.create_task(self._prefetch_dp(generation, snapshot))
            await asyncio.sleep(max(0, 1 / self.config.control_hz - (time.monotonic() - tick_started)))
        async with self._motion:
            await self.bridge.clear_pending("chunk consumed")
            await self.bridge.hold("waiting for next policy chunk")
        await self.flush()
        if self._live(generation) and self._terminal():
            await self._finish_phase(outcome="success" if self.history.latest.success else "failure")

    def _mock_reset(self):
        return (self.stage == "reset" and self.run.get("reset_policy") == "dummy"
                and self.bridge.describe().get("simulation") is True)

    async def _prefetch_dp(self, generation, observations):
        self.calls += 1
        meta, arrays = await self.client.infer(self._meta(operation="chunk"), observations,
                                              contract_hash=self.contract["hash"])
        return generation, meta, arrays, observations[-1].tick

    def _terminal(self):
        observation = self.history.latest
        if self.stage == "task" or self._mock_reset():
            return observation.success or observation.terminated or observation.truncated
        if observation.terminated or observation.truncated:
            raise RuntimeError("Environment stopped during reset; arm reset is prohibited")
        return False

    async def _tick(self, chunk, index, generation, base_action, next_base):
        async with self._motion:
            if not self._live(generation):
                return
            before, epoch = self.history.latest, self.epoch
            seq = self.journal.next_sequence(f"transition:{self.run['run_id']}:{self.episode_id}")
            self.journal.begin_execution(self.run["run_id"], epoch, seq)
            result = await self.bridge.execute_tick(epoch=epoch)
            self.history.append(result.observation)
            self.steps += 1
            receipt_id = uuid.uuid4().hex
            annotations = {"executed": True, "control_tick": result.observation.tick,
                "base_action": base_action.record["values"], "chunk_phase": index, **result.info}
            training = (self.run["policy"] == "residual" and self.stage == "task"
                        and self.run["limits"].get("training", False))
            pending = False
            if training:
                if result.observation.success or result.observation.terminated and not result.observation.truncated:
                    annotations["bootstrap_unused"] = True
                elif next_base is not None:
                    annotations.update(next_base_action=next_base.record["values"], next_chunk_phase=index + 1)
                else:
                    pending = True
                    annotations["learning_pending"] = True
            metadata = {"session_id": self.session_id, "epoch": epoch, "request_id": receipt_id,
                "episode_id": self.episode_id, "seq": seq, "stage": self.stage,
                "action": result.action["values"], "chunk_id": chunk["chunk_id"],
                "chunk_index": base_action.record.get("chunk_index"),
                "reward": result.reward, "terminated": result.observation.terminated,
                "truncated": result.observation.truncated,
                "annotations": annotations}
            data = self.client.encode(metadata, [before, result.observation])
            self.journal.finish_execution(self.run["run_id"], epoch, seq, receipt_id,
                {"endpoint": "transition", "npz": base64.b64encode(data).decode(), "contract_hash": self.contract["hash"]},
                learning_pending=pending)

    async def _resolve_learning(self):
        pending = [item for item in self.journal.pending() if item["kind"] == "transition_pending"]
        if not pending:
            return
        generation = self._generation
        if not self._live(generation):
            return
        count = int(self.contract["observation_steps"])
        self.calls += 1
        meta, arrays = await self.client.infer(self._meta(operation="chunk"),
            self.history.sample(tuple(range(-count + 1, 1))), contract_hash=self.contract["hash"])
        if not self._live(generation) or meta.get("epoch") != self.epoch:
            return
        if "actions" not in arrays or not len(arrays["actions"]):
            raise RuntimeError("No valid next base action; actual transition retained without a fabricated bootstrap")
        decoded = self.decoder.decode({**meta, "actions": arrays["actions"]},
                                      context={"contract": self.contract, "phase": self.stage})
        for receipt in pending:
            payload = receipt["payload"]
            original, images = unpack_message(base64.b64decode(payload["npz"]))
            original["annotations"].update(learning_pending=False,
                next_base_action=decoded[0].record["values"], next_chunk_phase=0)
            self.journal.resolve_learning(receipt["id"], {**payload,
                "npz": base64.b64encode(pack_message(original, images)).decode()})
        self._prefetched = generation, meta, arrays

    async def _release_pending_learning(self):
        # A Stop cannot obtain fresh actions. Preserve actual evidence explicitly
        # marked as excluded from learning, rather than inventing bootstrap data.
        for receipt in self.journal.pending():
            if receipt["kind"] == "transition_pending":
                self.journal.resolve_learning(receipt["id"], receipt["payload"])
            elif receipt["kind"] == "rpc" and receipt["payload"]["endpoint"] in ("episode/begin", "phase"):
                self.journal.set("cancelled:" + receipt["id"], "Explicit Stop canceled a non-motion lifecycle request")
                self.journal.acknowledge(receipt["id"])

    async def _finish_phase(self, outcome="success"):
        generation = self._generation
        self._discard_dp_prefetch()
        async with self._motion:
            if not self._live(generation):
                return
            await self.bridge.clear_pending("phase boundary")
            await self.bridge.hold("phase boundary")
        if self.stage == "task":
            await self._rpc("phase", self._meta(target="reset", task_outcome=outcome,
                outcome_source="env"), [self.history.latest])
        elif self.stage == "reset":
            if outcome != "success" or not await self.bridge.safe_reset():
                raise RuntimeError("Reset is not independently verified safe; arm reset is prohibited")
            if not self._live(generation):
                return
            await self._rpc("phase", self._meta(target="arm_reset", reset_verified=True), [self.history.latest])

    async def _arm_reset(self):
        async with self._motion:
            generation = self._generation
            if not self._live(generation):
                return
            verified = await self.bridge.safe_reset()
            if not self._live(generation):
                return
            if not verified:
                raise RuntimeError("Safe-reset clearance changed; hold for operator verification")
            # Separate durable marker ensures an ambiguous arm reset is never retried.
            seq = self.journal.next_sequence(f"arm-reset:{self.run['run_id']}")
            run_id, epoch = self.run["run_id"], self.epoch
            self.journal.begin_execution(run_id + ":arm-reset", epoch, seq)
            observation = await self.bridge.arm_reset()
            self.history.reset(observation)
            identifier = uuid.uuid4().hex
            payload = self.client.encode(self._meta(request_id=identifier, reset_outcome="success",
                arm_reset_performed=True), [observation])
            self.journal.finish_execution(run_id + ":arm-reset", epoch, seq, identifier,
                {"endpoint": "episode/end", "npz": base64.b64encode(payload).decode(),
                 "contract_hash": self.contract["hash"], "run_id": run_id})
            self.stage = "ending"
        await self.flush()
