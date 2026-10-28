"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    status: null, connected: false, lastReceived: 0, pending: null,
    tasks: [], cameras: new Map(), lastObservation: null, eventSignature: "",
    formInitialized: false, formDirty: false, localError: null, serverError: null, serverOffset: 0,
  };
  // Automatic episode reset also uses "initializing". Keep Pause available
  // through that transition so a run cannot silently resume after a reset.
  const activePhases = new Set(["initializing", "thinking", "executing"]);
  const phaseLabels = {
    idle: "Not initialized", initializing: "Initializing", ready: "Ready",
    thinking: "Agent thinking", executing: "Executing action", pausing: "Pausing…",
    paused: "Paused", completed: "Completed", stopped: "Stopped", error: "Error",
  };
  const number = (value) => typeof value === "number" && Number.isFinite(value) ? value : 0;
  const count = (value) => number(value).toLocaleString();
  const dollars = (value) => typeof value === "number" && Number.isFinite(value) ? `$${value.toFixed(4)}` : "—";
  const pretty = (value) => JSON.stringify(value, null, 2);
  const text = (id, value) => { $(id).textContent = value; };
  const timeMillis = (value) => {
    if (typeof value === "number") return value < 1e12 ? value * 1000 : value;
    return Date.parse(value);
  };
  const errorMessage = (data, fallback) => {
    const detail = data?.detail ?? data?.error ?? fallback;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => item.msg || pretty(item)).join("; ");
    return pretty(detail) || fallback;
  };

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeout ?? 12000);
    try {
      const response = await fetch(path, {
        cache: "no-store", credentials: "same-origin", signal: controller.signal,
        ...options,
        headers: { "Accept": "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data, `Request failed (${response.status})`));
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The server did not respond in time. Check the activity log before retrying a command.");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function showErrors() {
    const message = ui.localError || ui.serverError;
    $("error-notice").hidden = !message;
    $("dismiss-error").hidden = !ui.localError;
    text("error-title", ui.localError ? "Request failed" : "Run needs attention");
    text("error-text", message || "");
  }

  function canUsePolicy() {
    if (!ui.status) return false;
    return ui.status.policy !== "openai" || Boolean(ui.status.api_key_configured);
  }

  function canContinueFromReset() {
    const status = ui.status;
    return status?.stage === "reset" && status.observation?.success === true
      && ["completed", "paused"].includes(status.phase)
      && $("task-select").value !== "" && $("seed").value !== ""
      && Number($("task-select").value) === status.config?.task_id
      && Number($("seed").value) === status.config?.seed;
  }

  function updateControls() {
    const status = ui.status;
    const phase = status?.phase || "idle";
    const available = ui.connected && !ui.pending;
    const active = activePhases.has(phase);
    const editable = available && !active && !["initializing", "pausing"].includes(phase) && !status?.busy;
    const runnable = available && canUsePolicy() && ["ready", "paused"].includes(phase);
    const continuing = canContinueFromReset();
    $("initialize-button").disabled = !editable;
    text("initialize-label", continuing ? "Continue from GPT reset" : "Initialize environment");
    for (const input of $("run-form").querySelectorAll("input, select")) input.disabled = !editable;
    $("start-button").disabled = !runnable;
    $("step-button").disabled = !runnable;
    $("pause-button").disabled = !ui.connected || !active || ui.pending === "pause" || ui.pending === "stop";
    $("stop-button").disabled = !ui.connected || Boolean(ui.pending) || !["ready", "initializing", "thinking", "executing", "paused", "pausing"].includes(phase);
    $("reset-button").disabled = !editable || !status?.observation || !["ready", "paused", "completed", "stopped", "error"].includes(phase);
    text("start-label", phase === "paused" ? "Resume" : "Start run");
    $("mobile-start").disabled = $("start-button").disabled;
    $("mobile-pause").disabled = $("pause-button").disabled;
    text("mobile-start", phase === "paused" ? "Resume" : "Start run");
    const resetting = status?.stage === "reset";
    text("mobile-phase", ui.connected ? `${resetting ? "Reset · " : ""}${phaseLabels[phase] || phase}` : "Disconnected");
    const messages = {
      idle: "Initialize the environment before starting.",
      initializing: number(status?.stats?.episodes) > 0 ? "Resetting for the next episode. You can pause before the next decision." : "Loading the environment and its first observation…",
      ready: "Ready. Start a run or inspect one decision.",
      thinking: resetting ? "GPT is choosing actions to restore the scene." : "Waiting for the policy to choose the next action.",
      executing: resetting ? "The robot is restoring the scene with GPT actions." : "The environment is executing the requested action.",
      pausing: "Pause requested. Waiting for the current control step or reset to finish.",
      paused: "Paused. No new actions are being executed.",
      completed: continuing ? "GPT reset verified. Continue collecting from this scene without an environment reset." : "Run complete. Inspect the results before initializing another run.",
      stopped: "Run stopped. Initialize a new run to continue.",
      error: "The run stopped. Check the error and activity log.",
    };
    text("control-message", !ui.connected ? "Server disconnected. Controls are unavailable." : ui.pending ? `Sending ${ui.pending} request…` : messages[phase] || phase);
  }

  function updateFreshness() {
    if (!ui.lastReceived) return;
    const age = (Date.now() - ui.lastReceived) / 1000;
    text("server-freshness", ui.connected ? `Server updated ${age.toFixed(1)}s ago` : "Server disconnected · reconnecting");
    const obs = ui.status?.observation;
    if (!obs) return;
    const captured = timeMillis(obs.timestamp);
    const seconds = Number.isFinite(captured) ? Math.max(0, (Date.now() + ui.serverOffset - captured) / 1000) : null;
    const frozen = ["ready", "paused", "completed", "stopped", "error"].includes(ui.status.phase);
    const label = seconds === null ? "Timestamp unavailable" : `${seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`} old`;
    text("observation-age", `${frozen ? "HELD FRAME · " : "CAPTURED · "}${label}`);
    $("observation-age").classList.toggle("old", !ui.connected || (!frozen && seconds !== null && seconds > 5));
  }

  function createCamera(name, single) {
    const view = document.createElement("div");
    view.className = `camera-view${single ? " single" : ""}`;
    const label = document.createElement("span");
    label.className = "camera-label";
    label.textContent = name.replace(/_image$/, "").replaceAll("_", " ").toUpperCase();
    const img = document.createElement("img");
    img.alt = `${name} camera observation`;
    const error = document.createElement("span");
    error.className = "camera-error";
    error.textContent = "Loading camera observation…";
    img.addEventListener("load", () => { error.hidden = true; });
    img.addEventListener("error", () => { error.hidden = false; error.textContent = "Frame unavailable. Waiting for the next observation."; });
    view.append(img, label, error);
    $("camera-grid").append(view);
    ui.cameras.set(name, { view, img, error });
  }

  function updateCameras(observation) {
    if (!observation) return;
    const names = Array.isArray(observation.cameras) ? observation.cameras.map((camera) => typeof camera === "string" ? camera : camera.name).filter(Boolean) : Object.keys(observation.cameras || {});
    const currentNames = Array.from(ui.cameras.keys());
    if (names.join("|") !== currentNames.join("|")) {
      $("camera-grid").replaceChildren();
      ui.cameras.clear();
      names.forEach((name) => createCamera(name, names.length === 1));
      ui.lastObservation = null;
      if (!names.length) {
        const placeholder = document.createElement("div");
        placeholder.className = "camera-placeholder";
        placeholder.textContent = "This adapter has not returned camera images.";
        $("camera-grid").append(placeholder);
      }
    }
    const key = `${ui.status.run_id || ""}:${observation.id}`;
    if (key !== ui.lastObservation) {
      for (const [name, camera] of ui.cameras) {
        camera.img.src = `/api/frame/${encodeURIComponent(name)}?observation_id=${encodeURIComponent(observation.id)}&t=${Date.now()}`;
      }
      ui.lastObservation = key;
    }
  }

  function updateEvents(events) {
    const visible = Array.isArray(events) ? events.slice(-40).reverse() : [];
    const signature = JSON.stringify(visible);
    if (signature === ui.eventSignature) return;
    ui.eventSignature = signature;
    text("event-count", String(visible.length));
    $("event-list").replaceChildren();
    if (!visible.length) {
      const item = document.createElement("li");
      item.className = "empty-state";
      item.textContent = "Run events will appear here.";
      $("event-list").append(item);
      return;
    }
    for (const event of visible) {
      const item = document.createElement("li");
      item.dataset.kind = event.kind || "info";
      const time = document.createElement("time");
      const millis = timeMillis(event.time ?? event.timestamp);
      if (Number.isFinite(millis)) {
        const date = new Date(millis);
        time.dateTime = date.toISOString();
        time.textContent = date.toLocaleTimeString([], { hour12: false });
      } else time.textContent = "—";
      const kind = document.createElement("span");
      kind.className = "event-kind";
      kind.textContent = event.kind || "event";
      const message = document.createElement("span");
      message.textContent = event.message || "";
      item.append(time, kind, message);
      $("event-list").append(item);
    }
  }

  function updateDecision(decision) {
    text("decision-status", decision?.status || "WAITING");
    text("decision-note", decision?.note || (decision ? "No note returned." : "The next decision will appear here with its requested actions."));
    $("action-list").replaceChildren();
    for (const [index, action] of (decision?.actions || []).entries()) {
      const row = document.createElement("div");
      row.className = "action-row";
      const ordinal = document.createElement("span");
      ordinal.className = "action-number";
      ordinal.textContent = `${String(index + 1).padStart(2, "0")}`;
      const values = document.createElement("span");
      values.className = "action-values";
      values.textContent = `[${(action.values || []).map((v) => typeof v === "number" ? Number(v.toFixed(4)) : String(v)).join(", ")}]`;
      const repeat = document.createElement("span");
      repeat.className = "action-repeat";
      repeat.textContent = `× ${action.repeat ?? 1}`;
      row.append(ordinal, values, repeat);
      $("action-list").append(row);
    }
  }

  function setMetric(id, value, limit) {
    const element = $(id);
    element.textContent = count(value);
    if (limit) {
      const suffix = document.createElement("small");
      suffix.textContent = ` / ${count(limit)}`;
      element.append(suffix);
    }
  }

  function render(status) {
    ui.status = status;
    if (typeof status.server_time === "number") ui.serverOffset = status.server_time * 1000 - Date.now();
    const phase = status.phase || "idle";
    const config = status.config || {};
    const episode = status.episode || {};
    const stats = status.stats || {};
    const obs = status.observation;
    const resetting = status.stage === "reset";
    const stageSteps = episode.stage_steps ?? (resetting ? episode.reset_steps : episode.steps);
    const stageDecisions = episode.stage_decisions ?? (resetting ? episode.reset_decisions : episode.decisions);
    const stepLimit = resetting ? config.max_reset_steps : config.max_steps;
    const decisionLimit = resetting ? config.max_reset_decisions : config.max_decisions;
    text("phase-label", phaseLabels[phase] || phase);
    $("phase-dot").className = `phase-dot ${["thinking", "executing", "initializing"].includes(phase) ? "active" : ["paused", "pausing"].includes(phase) ? "paused" : phase === "error" ? "error" : phase === "completed" ? "completed" : ""}`;
    text("backend-label", status.backend || "Environment —");
    text("suite-label", config.suite || "");
    text("policy-label", status.policy || "—");
    text("model-label", status.policy === "scripted" ? "Not used (scripted)" : status.model || "—");
    text("run-id", status.run_id || "—");
    text("tokens-value", `${count(stats.input_tokens)} / ${count(stats.output_tokens)}`);
    text("token-details-value", `${count(stats.cached_input_tokens)} / ${count(stats.reasoning_tokens)}`);
    text("cost-value", `${dollars(stats.estimated_cost_usd)}${number(stats.unknown_usage_requests) > 0 ? " + unknown" : ""}`);
    text("resets-value", `${count(stats.reset_successes)} / ${count(stats.reset_attempts)}`);
    text("hard-resets-value", count(stats.hard_resets));
    text("cost-note", number(stats.unknown_usage_requests) > 0
      ? `${count(stats.unknown_usage_requests)} request(s) have unknown usage; the estimate excludes their charges.`
      : "API usage × published pricing. Final billing may differ.");
    text("total-steps", count(stats.total_steps));
    $("key-notice").hidden = status.policy !== "openai" || Boolean(status.api_key_configured);
    $("demo-notice").hidden = status.backend !== "demo";
    text("episode-label", episode.id ? `EPISODE ${episode.index ?? 1}` : "EPISODE —");
    text("stage-label", resetting ? "GPT RESET" : "TASK");
    $("stage-label").classList.toggle("reset-stage", resetting);
    text("objective-label", resetting ? "Reset objective" : "Current task");
    text("task-instruction", (resetting ? obs?.task || episode.task : episode.task || obs?.task) || "Initialize an environment to see the task instruction.");
    if (episode.id || obs) {
      setMetric("steps-value", stageSteps, stepLimit);
      setMetric("decisions-value", stageDecisions, decisionLimit);
    } else {
      text("steps-value", "—");
      text("decisions-value", "—");
    }
    $("steps-progress").style.width = `${Math.min(100, stepLimit ? number(stageSteps) / stepLimit * 100 : 0)}%`;
    $("decisions-progress").style.width = `${Math.min(100, decisionLimit ? number(stageDecisions) / decisionLimit * 100 : 0)}%`;
    text("successes-value", count(stats.successes));
    text("episodes-value", `${count(stats.episodes)} completed episode${stats.episodes === 1 ? "" : "s"}`);
    text("observation-id", obs?.id ? `OBS ${obs.id}` : "OBS —");
    text("observation-result", obs?.success ? (resetting ? "RESET VERIFIED BY ENVIRONMENT" : "SUCCESS VERIFIED BY ENVIRONMENT") : episode.outcome && !resetting ? `OUTCOME: ${episode.outcome}` : obs ? "Environment observation" : "Awaiting environment");
    $("observation-result").classList.toggle("verified", Boolean(obs?.success));
    text("state-json", obs ? pretty(obs.state || {}) : "No observation yet.");
    text("receipt-json", status.latest_receipt ? pretty(status.latest_receipt) : "No action executed yet.");
    if (!ui.formInitialized && !ui.formDirty) {
      for (const [id, name] of [["seed", "seed"], ["episodes", "episodes"], ["max-steps", "max_steps"], ["max-decisions", "max_decisions"], ["reset-mode", "reset_mode"], ["max-reset-steps", "max_reset_steps"], ["max-reset-decisions", "max_reset_decisions"], ["budget-usd", "budget_usd"]]) {
        if (config[name] !== undefined) $(id).value = config[name];
      }
      if (config.task_id !== undefined && [...$("task-select").options].some((o) => o.value === String(config.task_id))) $("task-select").value = config.task_id;
      ui.formInitialized = true;
      resetModeDescription();
    }
    ui.serverError = status.error ? (typeof status.error === "string" ? status.error : pretty(status.error)) : null;
    updateCameras(obs);
    updateDecision(status.latest_decision);
    updateEvents(status.events);
    showErrors();
    updateControls();
    updateFreshness();
  }

  async function poll() {
    try {
      const data = await request("/api/status", { timeout: 6000 });
      ui.connected = true;
      ui.lastReceived = Date.now();
      $("connection").className = "connection connected";
      text("connection-label", "Connected");
      render(data);
    } catch (_error) {
      ui.connected = false;
      $("connection").className = "connection disconnected";
      text("connection-label", "Disconnected");
      updateControls();
      updateFreshness();
    } finally {
      window.setTimeout(poll, ui.connected ? 500 : 2000);
    }
  }

  function taskDescription() {
    const task = ui.tasks.find((item) => String(item.id) === $("task-select").value);
    text("task-description", task?.language || "Tasks are provided by the environment adapter.");
  }

  function resetModeDescription() {
    text("reset-mode-hint", $("reset-mode").value === "agent"
      ? "GPT restores the scene using robot actions after each task."
      : "The simulator resets the scene between tasks.");
  }

  async function loadTasks() {
    $("retry-tasks").disabled = true;
    try {
      const data = await request("/api/tasks", { timeout: 30000 });
      ui.tasks = data.tasks || [];
      const previous = $("task-select").value;
      $("task-select").replaceChildren();
      for (const task of ui.tasks) {
        const option = document.createElement("option");
        option.value = task.id;
        option.textContent = `${String(task.id).padStart(2, "0")} · ${(task.name || task.language || `Task ${task.id}`).replaceAll("_", " ")}`;
        $("task-select").append(option);
      }
      if (!ui.tasks.length) throw new Error("The adapter returned an empty task catalog.");
      const selected = ui.formDirty ? previous : ui.status?.config?.task_id ?? previous;
      if (ui.tasks.some((task) => String(task.id) === String(selected))) $("task-select").value = selected;
      $("catalog-error").hidden = true;
      if (data.suite) text("suite-label", data.suite);
      taskDescription();
    } catch (error) {
      $("catalog-error").hidden = false;
      text("catalog-error-text", `Task catalog unavailable: ${error.message}`);
      if (!$("task-select").options.length || $("task-select").options[0].textContent.includes("Loading")) {
        $("task-select").replaceChildren();
        const option = document.createElement("option");
        option.value = "0";
        option.textContent = "Task 0 · catalog unavailable";
        $("task-select").append(option);
      }
    } finally {
      $("retry-tasks").disabled = false;
      updateControls();
    }
  }

  async function command(name, body = {}) {
    if (!ui.connected || (ui.pending && name !== "pause")) return;
    ui.pending = name;
    ui.localError = null;
    showErrors();
    updateControls();
    try {
      const data = await request(`/api/${name}`, { method: "POST", body: JSON.stringify(body), timeout: ["initialize", "continue", "reset"].includes(name) ? 120000 : 15000 });
      if (data.phase) render(data);
    } catch (error) {
      ui.localError = error.message;
      showErrors();
    } finally {
      if (ui.pending === name) ui.pending = null;
      updateControls();
    }
  }

  $("run-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!$("run-form").reportValidity()) return;
    command(canContinueFromReset() ? "continue" : "initialize", {
      task_id: Number($("task-select").value), seed: Number($("seed").value),
      episodes: Number($("episodes").value), max_steps: Number($("max-steps").value),
      max_decisions: Number($("max-decisions").value),
      reset_mode: $("reset-mode").value,
      max_reset_steps: Number($("max-reset-steps").value),
      max_reset_decisions: Number($("max-reset-decisions").value),
      budget_usd: Number($("budget-usd").value),
    });
  });
  const formChanged = () => { ui.formDirty = true; updateControls(); };
  $("run-form").addEventListener("input", formChanged);
  $("run-form").addEventListener("change", formChanged);
  $("task-select").addEventListener("change", taskDescription);
  $("reset-mode").addEventListener("change", resetModeDescription);
  $("start-button").addEventListener("click", () => command(ui.status?.phase === "paused" ? "resume" : "start"));
  $("pause-button").addEventListener("click", () => command("pause"));
  $("step-button").addEventListener("click", () => command("step"));
  $("stop-button").addEventListener("click", () => command("stop"));
  $("reset-button").addEventListener("click", () => command("reset"));
  $("mobile-start").addEventListener("click", () => $("start-button").click());
  $("mobile-pause").addEventListener("click", () => $("pause-button").click());
  $("retry-tasks").addEventListener("click", loadTasks);
  $("dismiss-error").addEventListener("click", () => { ui.localError = null; showErrors(); });
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest("input, select, textarea, button, a, summary, [contenteditable]")) return;
    if (!$("pause-button").disabled) { event.preventDefault(); command("pause"); }
    else if (ui.status?.phase === "paused" && !$("start-button").disabled) { event.preventDefault(); command("resume"); }
  });
  window.setInterval(updateFreshness, 500);
  poll();
  loadTasks();
})();
