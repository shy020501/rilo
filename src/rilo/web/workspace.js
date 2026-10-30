"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const icons = {
    grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    plus: '<path d="M12 5v14M5 12h14"/>', close: '<path d="m6 6 12 12M6 18 18 6"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>', back: '<path d="M19 12H5m5-5-5 5 5 5"/>', chevron: '<path d="m9 5 7 7-7 7"/>',
    terminal: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3m6 0h4"/>',
    external: '<path d="M14 3h7v7m0-7-11 11M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/>',
    database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
    check: '<path d="m5 12 4 4L19 6"/>', reset: '<path d="M3 11a9 9 0 1 1 2 7M3 4v7h7"/>',
    dollar: '<path d="M12 2v20m5-16H9a4 4 0 0 0 0 8h6a4 4 0 0 1 0 8H6" transform="translate(0 -2) scale(1 .95)"/>',
    flow: '<rect x="3" y="3" width="6" height="6" rx="1.5"/><rect x="15" y="15" width="6" height="6" rx="1.5"/><path d="M6 9v9h9m-6-12h9v9"/>',
    robot: '<rect x="4" y="7" width="16" height="13" rx="4"/><path d="M12 3v4m-2-4h4M1 12v5m22-5v5M9 16h6"/><circle cx="8.5" cy="11.5" r=".7"/><circle cx="15.5" cy="11.5" r=".7"/>',
    layers: '<path d="m12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 17l10 5 10-5"/>',
    model: '<path d="m12 3 9 5v9l-9 5-9-5V8Zm0 10v9M3 8l9 5 9-5M7.5 5.5l9 5v4"/>',
    progress: '<path d="M3 20h18M5 16l4-6 5 3 5-9"/><path d="M15 4h4v4"/>',
    settings: '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/>',
    camera: '<rect x="3" y="6" width="18" height="15" rx="2"/><path d="m7 6 2-3h6l2 3"/><circle cx="12" cy="13" r="4"/>',
    play: '<path d="m8 4 12 8-12 8Z"/>', pause: '<path d="M8 5v14M16 5v14" stroke-width="3"/>', stop: '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    step: '<path d="m5 5 10 7-10 7Zm14 0v14"/>', info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>', download: '<path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/>',
    archive: '<rect x="3" y="3" width="18" height="5" rx="1"/><path d="M5 8v12h14V8M10 12h4"/>',
    restore: '<path d="M5 8v12h14V8M3 3h18v5H3zM12 17v-6m-3 3 3-3 3 3"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
  };
  const icon = (name) => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.folder}</svg>`;
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const n = (value) => typeof value === "number" && Number.isFinite(value) ? value : 0;
  const count = (value) => n(value).toLocaleString("en-US");
  const money = (value, digits = 2) => typeof value === "number" && Number.isFinite(value) ? `$${value.toFixed(digits)}` : "—";
  const date = (value, options = {}) => value ? new Date(value * 1000).toLocaleString("ko-KR", {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",...options}) : "—";
  const asset = (value) => typeof value === "string" && value.startsWith("/api/") ? value : "";
  const tag = (text, style = "neutral") => `<span class="tag ${style}">${esc(text)}</span>`;
  const phaseNames = {idle:"실행 대기",connected:"환경 연결됨",initializing:"환경 준비 중",ready:"수집 준비 완료",thinking:"GPT 추론 중",executing:"Action 실행 중",pausing:"일시정지 중",paused:"일시정지",completed:"수집 완료",stopped:"실행 중지",error:"확인 필요"};
  const activePhases = new Set(["initializing", "ready", "thinking", "executing", "pausing", "paused"]);
  const pausePhases = new Set(["initializing", "ready", "thinking", "executing"]);
  const state = {workspace:null,status:null,detail:null,route:{id:null,tab:"overview"},connected:false,pending:null,pendingProjectId:null,pausePending:false,search:"",filter:"all",dataView:"active",episodeMutation:null,deleteTarget:null,deleteBusy:false,episodeDetail:null,episodeToken:0,workspaceSignature:"",detailSignature:"",navSignature:"",barSignature:"",statusBusy:false,statusRequest:0,workspaceBusy:false,routeToken:0,taskToken:0,tasks:[],lastObservation:null,lastCameras:"",lastEvents:"",toastTimer:null,pipeline:null,pipelineProjectId:null,pipelineBusy:false,pipelineRequest:0,pipelinePending:null,pipelinePausePending:false,evaluationStage:"bc_eval",pipelineExecutionSignature:"",modelSignature:"",evaluationSignature:"",evaluationCameraSignature:"",evaluationObservationKey:"",evaluationFrameToken:0,pipelineDialog:null,stageDialog:null,trainingAlgorithms:{},collectionSettingsExpanded:false,collectionSettingsDirty:false,collectionSettingsSignature:"",collectionSettingsSaving:false};
  const routeUrl = (id, tab = "overview") => `#/projects/${encodeURIComponent(id)}${tab === "overview" ? "" : `/${tab}`}`;
  state.projectView = "active";
  state.projectMutation = null;
  state.projectDeleteTarget = null;
  state.settingsSaving = false;
  const runtime = () => state.status || state.workspace?.runtime || {};
  const activeId = () => runtime().active_project_id;
  const currentProject = () => state.detail?.project;
  const remoteRobotProject = (project = currentProject()) => project?.backend === "remote_robot";
  const backendLabel = (backend) => backend === "remote_robot" ? "원격 로봇" : backend?.toUpperCase() || "ROBOT";
  const remoteConnectionOnlyReason = "원격 로봇은 환경 연결 탭의 client 전용 제어로 실행합니다. 기존 로컬 환경·학습 파이프라인은 이 프로젝트에서 사용하지 않습니다.";
  const stageCost = (episode, stage) => episode.costs?.[stage] || {};
  const episodeCost = (episode) => n(stageCost(episode,"task").estimated_cost_usd) + n(stageCost(episode,"reset").estimated_cost_usd);
  const externalEpisode = (episode) => episode?.source?.kind === "external_dataset";
  const episodeOutcomeLabel = (episode) => externalEpisode(episode) ? tag("외부 소스","neutral external-source") : outcomeLabel(episode.outcome);
  const episodeResetLabel = (episode) => externalEpisode(episode) && (!episode.reset_outcome || episode.reset_outcome === "not_applicable") ? tag("해당 없음") : resetLabel(episode.reset_outcome);
  const complete = (episode) => episode.outcome === "success" && episode.reset_outcome === "success";
  const resetLabel = (outcome) => outcome === "success" ? tag("복구 완료","success") : outcome ? tag("복구 미완료","warning") : tag("대기");
  const outcomeLabel = (outcome) => outcome === "success" ? tag("성공","success") : !outcome ? tag("진행 중") : tag("실패","failure");
  const allEpisodes = () => [...(state.detail?.episodes || []),...(state.detail?.archived_episodes || [])];
  const findEpisode = (run,id) => allEpisodes().find(ep => ep.run_id === run && ep.id === id);
  const episodeIndex = (episode) => episode.display_index ?? episode.index ?? 1;
  const episodeLearning = (episode) => episode.learning || (state.detail?.runs || []).find(run => run.run_id === episode.run_id)?.config?.learning || (state.status?.run_id === episode.run_id ? state.status.learning : null) || {};
  const episodeLabel = (episode) => `${["bc_eval","rl","rl_eval"].includes(episodeLearning(episode).pipeline_stage) ? "Rollout" : "Episode"} ${String(episodeIndex(episode)).padStart(2,"0")}`;
  const episodePath = (project,run,id) => `/api/projects/${encodeURIComponent(project)}/episodes/${encodeURIComponent(run)}/${encodeURIComponent(id)}`;
  const policyOptions = () => state.detail?.policy_options || state.workspace?.policy_options || {models:[],defaults:{}};
  const nextPolicy = (project) => ({model:state.workspace?.runtime?.model,reasoning_effort:state.workspace?.runtime?.reasoning_effort,...policyOptions().defaults,...project?.policy_config});
  const effortLabel = (effort, current) => effort === "auto" ? `Auto${["medium","xhigh"].includes(current) ? ` · ${current}` : " · medium ↔ xhigh"}` : effort || "—";
  const policyLabel = (policy) => `${policy?.model || "—"} / ${effortLabel(policy?.reasoning_effort,policy?.current_reasoning_effort)}`;
  const qualityLabels = {within_grasp_budget:"Optimal",extra_grasp_attempts:"Sub-optimal",unknown:"분류 정보 없음",not_successful:"성공 미완료"};
  const qualityKey = (episode, stage = "task") => {
    const quality = episode.quality?.[stage] || {};
    const mapped = {optimal:"within_grasp_budget",suboptimal:"extra_grasp_attempts",unknown:"unknown",not_successful:"not_successful"}[quality.classification];
    if (mapped) return mapped;
    const label = {success_without_reported_failure:"within_grasp_budget",recovered_success:"extra_grasp_attempts"}[quality.label] || quality.label;
    return Object.prototype.hasOwnProperty.call(qualityLabels,label) ? label : "unknown";
  };
  const qualityTag = (episode, stage = "task") => {
    const key = qualityKey(episode,stage);
    return tag(qualityLabels[key],key === "within_grasp_budget" ? "success" : key === "extra_grasp_attempts" ? "warning" : "neutral");
  };

  const resetModeLabel = (mode) => ({agent:"GPT arm reset",environment:"환경 초기화",free_space:"대상 복구 + 안전 후퇴 후 환경 초기화"})[mode] || "GPT arm reset";
  const closureCount = (value) => Number.isInteger(value) && value >= 0 ? `${count(value)}회` : "미확정";

  function episodeQuality(episode) {
    const quality = episode.quality?.task || {};
    if (externalEpisode(episode) && qualityKey(episode,"task") === "unknown") return `<section class="episode-quality"><h3>성공 분류</h3>${tag("분류 정보 없음")}<p class="field-note">이 외부 데이터에는 grasp 횟수를 기준으로 한 분류 기록이 없습니다.</p></section>`;
    const historical = ["success_without_reported_failure","recovered_success"].includes(quality.label);
    return `<section class="episode-quality"><h3>성공 분류</h3><p class="field-note">${historical ? "기존 GPT 실패 보고를 기준으로 분류한 기록입니다." : "Task의 예상 grasp 횟수와 실제 gripper 닫기 횟수를 비교합니다. Reset 동작은 Task 횟수에 합산하지 않습니다."} Optimal은 최적 경로나 실제 무실패를 보장하지 않습니다.</p><div class="quality-stage"><div class="quality-stage-heading"><strong>Task</strong>${qualityTag(episode,"task")}<span>실제 ${closureCount(quality.actual_closures)} / 예상 ${closureCount(quality.expected_closures)}</span></div><p class="field-note">${historical ? "과거 GPT 실패 보고 기반 기록입니다. Grasp 횟수 기준으로 재분류하지 않았습니다." : quality.baseline_version ? `기준 버전 ${count(quality.baseline_version)} · ${quality.source === "manual" ? "사용자 설정" : quality.source === "gpt" ? "GPT 추정" : "기록된 기준"}` : "이 episode에 기록된 grasp 기준이 없습니다."}</p></div></section>`;
  }

  function episodeSourceDetails(episode) {
    if (!externalEpisode(episode)) return "";
    const source = episode.source;
    const fields = [["데이터셋",source.dataset],["Suite",source.suite],["Task",source.task_name || (typeof source.task === "string" ? source.task : source.task?.name)],["Task ID",source.task_id],["Demo",source.demo_id ?? source.demo],["원본 파일",source.file ?? source.source_file],["등록 시각",source.imported_at ? date(source.imported_at) : null]];
    return `<section class="episode-source"><h3>외부 데이터 출처</h3><p class="field-note">원본 데이터셋에서 가져온 episode입니다. RILO에서 GPT로 수행한 수집·reset 결과가 아닙니다.</p><dl>${fields.filter(([,value]) => value !== null && value !== undefined && value !== "" && ["string","number"].includes(typeof value)).map(([label,value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join("")}</dl></section>`;
  }

  function graspBaselinePanel(project) {
    const baseline = project.grasp_baseline || {}, expected = baseline.expected_closures;
    const assumptions = Array.isArray(baseline.assumptions) ? baseline.assumptions : [];
    return `<section class="panel settings-section grasp-baseline-panel"><div class="panel-header"><div><h2>Task grasp 기준</h2><p>성공 episode의 grasp 횟수 분류에 사용할 예상값</p></div>${tag(baseline.source === "manual" ? "사용자 설정" : baseline.source === "gpt" ? "GPT 추정" : "자동 추정 대기")}</div><div class="panel-body"><p class="field-note">자동 추정은 실제 수집을 시작할 때 필요한 경우 수행하고, 같은 task의 기준을 재사용합니다. 수동 변경은 다음 새 run부터 적용됩니다. 기존 episode는 당시 기준을 유지합니다.</p>${baseline.rationale ? `<p class="baseline-rationale">${esc(baseline.rationale)}</p>` : ""}${assumptions.length ? `<ul class="baseline-assumptions">${assumptions.map(item => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}<label class="field grasp-baseline-field">예상 gripper 닫기 횟수<input id="grasp-baseline-count" name="expected_grasp_closures" type="number" min="0" max="100" step="1" value="${Number.isInteger(expected) ? expected : ""}" data-saved-expected="${Number.isInteger(expected) ? expected : ""}" placeholder="자동 추정"><small>비워서 설정을 저장하면 다음 수집 시작 시 자동 추정합니다.</small></label><p class="field-note">${baseline.version ? `저장된 기준 v${count(baseline.version)}` : "저장된 기준 없음"} · 아래 설정 저장 버튼으로 함께 저장합니다.</p></div></section>`;
  }

  const pipelineStages = [
    {id:"collect",number:"01",name:"GPT 데이터 수집",short:"수집",icon:"database",description:"사용 중인 성공 task episode를 목표까지 모읍니다. 성공·실패 기록은 모두 저장합니다."},
    {id:"bc",number:"02",name:"Diffusion Policy 학습",short:"DP 학습",icon:"model",description:"수집을 멈춘 상태에서 성공 task 데이터로 BC 정책을 학습합니다."},
    {id:"bc_eval",number:"03",name:"BC 성공률 평가",short:"BC 평가",icon:"check",description:"DP로 rollout하고 선택한 평가 reset 방식으로 복구한 뒤 실제 성공률을 기록합니다."},
    {id:"progress",number:"04",name:"Progress 모델 학습",short:"Progress",icon:"progress",description:"BC 성공률 목표를 달성한 뒤 성공 데이터만 사용해 진행도를 학습합니다."},
    {id:"rl",number:"05",name:"Residual TD3 학습",short:"Residual RL",icon:"layers",description:"Frozen DP에 매 step correction을 더하고 offline·online replay를 혼합합니다."},
    {id:"rl_eval",number:"06",name:"RL 성공률 평가",short:"RL 평가",icon:"check",description:"BC+Residual로 rollout하고 선택한 평가 reset 방식으로 복구한 뒤 실제 성공률을 기록합니다."},
  ];
  const pipelinePhaseNames = {idle:"실행 대기",running:"진행 중",paused:"일시정지",pausing:"일시정지 중",completed:"완료",stopped:"중지",error:"확인 필요",interrupted:"재개 준비 필요"};
  const pipelineActive = (phase) => ["running","paused","pausing"].includes(phase);
  const currentPipeline = () => state.pipelineProjectId === state.route.id ? state.pipeline : null;
  const pipelineRuntime = () => state.status && Object.prototype.hasOwnProperty.call(state.status,"pipeline_runtime") ? state.status.pipeline_runtime : state.workspace?.pipeline_runtime || null;
  const pipelineEnabled = () => Boolean(state.detail?.capabilities?.bc || state.workspace?.capabilities?.bc || currentPipeline()?.capabilities?.bc);
  const pipelineStageName = (id) => pipelineStages.find(stage => stage.id === id)?.name || id || "단계 선택 전";
  const finiteValue = (value, digits = 3) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("en-US",{maximumFractionDigits:digits}) : "—";

  function fillIcons(root = document) {
    root.querySelectorAll("[data-icon]").forEach(node => { node.innerHTML = icon(node.dataset.icon); });
  }

  function errorText(data, fallback) {
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail)) return data.detail.map(x => `${x.loc?.slice(1).join(".") || "입력"}: ${x.msg}`).join(" · ");
    return fallback;
  }

  async function api(path, body, timeout = 15000, externalSignal) {
    const abort = new AbortController();
    const externalAbort = () => abort.abort();
    if (externalSignal?.aborted) abort.abort();
    else externalSignal?.addEventListener("abort",externalAbort,{once:true});
    const timer = setTimeout(() => abort.abort(), timeout);
    try {
      const response = await fetch(path, {method:body === undefined ? "GET" : "POST",body:body === undefined ? undefined : JSON.stringify(body),headers:body === undefined ? {Accept:"application/json"} : {Accept:"application/json","Content-Type":"application/json"},cache:"no-store",credentials:"same-origin",signal:abort.signal});
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = new Error(errorText(result, `요청을 처리하지 못했습니다 (${response.status}).`));
        error.status = response.status;
        error.detail = result.detail;
        throw error;
      }
      return result;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답을 기다리는 중입니다. 실행 상태를 확인한 뒤 다시 시도해주세요.");
      throw error;
    } finally { clearTimeout(timer);externalSignal?.removeEventListener("abort",externalAbort); }
  }

  function toast(message, error = false) {
    clearTimeout(state.toastTimer);
    $("toast").textContent = message;
    $("toast").className = `toast${error ? " error" : ""}`;
    $("toast").hidden = false;
    state.toastTimer = setTimeout(() => $("toast").hidden = true, error ? 7000 : 4000);
  }

  function showError(message) {
    $("page-error").hidden = !message;
    $("page-error").innerHTML = message ? `<span>${esc(message)}</span><button class="button small secondary" data-action="refresh">다시 연결</button>` : "";
  }

  function projectStatus(project) {
    if (project.delete_pending) return {text:"삭제 미완료",style:"warning"};
    if (project.archived) return {text:"보관됨",style:"neutral"};
    if (remoteRobotProject(project)) return {text:"원격 client",style:"neutral"};
    const learning = project.pipeline_status || (pipelineRuntime()?.project_id === project.id ? pipelineRuntime() : null);
    if (learning && pipelineActive(learning.phase)) return {text:learning.phase === "paused" ? "파이프라인 일시정지" : pipelineStageName(learning.stage),style:learning.phase === "paused" ? "warning" : "success"};
    const s = runtime();
    if (s.active_project_id === project.id && activePhases.has(s.phase)) return {text:phaseNames[s.phase],style:s.phase === "paused" ? "warning" : "success"};
    if (n(project.stats?.episodes) > n(project.stats?.reset_successes)) return {text:"복구 기록 확인",style:"warning"};
    return {text:n(project.stats?.episodes) ? "데이터 수집" : "시작 전",style:n(project.stats?.episodes) ? "success" : "neutral"};
  }

  function metric(label, value, foot, symbol, id = "") {
    return `<div class="metric"><div class="metric-label">${esc(label)}${icon(symbol)}</div><div class="metric-value"${id ? ` id="${id}"` : ""}>${value}</div><div class="metric-foot">${esc(foot)}</div></div>`;
  }

  function renderNav() {
    const projects = state.workspace?.projects || [];
    const signature = JSON.stringify(projects.map(x => [x.id,x.name])) + state.route.id;
    if (signature === state.navSignature) return;
    state.navSignature = signature;
    $("home-nav").classList.toggle("active", !state.route.id);
    $("project-nav").innerHTML = projects.length ? projects.map(p => `<a href="${routeUrl(p.id)}" class="${state.route.id === p.id ? "active" : ""}" title="${esc(p.name)}"><span class="project-bullet"></span><span>${esc(p.name)}</span></a>`).join("") : '<span class="sidebar-empty">첫 프로젝트를 만들어보세요.</span>';
  }

  function renderBreadcrumb() {
    const project = currentProject();
    $("breadcrumb").innerHTML = state.route.id && project ? `<a href="#/">워크스페이스</a>${icon("chevron")}<span class="current">${esc(project.name)}</span>` : '<span>워크스페이스</span><span class="small-muted">/</span><span class="current">Overview</span>';
  }

  function renderGlobal() {
    const s = runtime();
    const learning = pipelineRuntime();
    if (learning && pipelineActive(learning.phase)) {
      const signature = JSON.stringify(["pipeline",learning,state.pipelinePending,state.pipelinePausePending,state.connected]);
      if (signature === state.barSignature) return;
      state.barSignature = signature;
      const project = state.workspace?.projects.find(x => x.id === learning.project_id);
      $("global-run").hidden = false;
      $("global-run").innerHTML = `<div class="global-run-copy"><i class="dot ${learning.phase === "running" ? "dot-pulse" : ""}"></i><a href="${routeUrl(learning.project_id,"overview")}"><strong>${esc(project?.name || "학습 파이프라인")}</strong></a><span>· ${esc(pipelineStageName(learning.stage))} / ${esc(pipelinePhaseNames[learning.phase] || learning.phase)}</span></div><button class="button pause" data-action="pipeline-global-pause" data-project="${esc(learning.project_id)}" ${learning.phase !== "running" || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pipelinePausePending ? "일시정지 요청 중…" : learning.phase === "paused" ? "일시정지됨" : "Pause"}</button>`;
      return;
    }
    const id = s.active_project_id;
    const visible = activePhases.has(s.phase);
    const signature = JSON.stringify([visible,id,s.phase,s.stage,state.pending,state.pausePending,state.connected]);
    if (signature === state.barSignature) return;
    state.barSignature = signature;
    $("global-run").hidden = !visible;
    if (!visible) return;
    const project = state.workspace?.projects.find(x => x.id === id);
    $("global-run").innerHTML = `<div class="global-run-copy"><i class="dot ${["thinking","executing"].includes(s.phase) ? "dot-pulse" : ""}"></i><a href="${id ? routeUrl(id,"collection") : "/console"}"><strong>${esc(project?.name || (state.pending === "connect" ? "환경 연결 중" : "실행 콘솔의 수집"))}</strong></a><span>· ${esc(phaseNames[s.phase] || s.phase)}${s.stage === "reset" ? " / GPT reset" : ""}</span></div><button class="button pause" data-action="global-pause" ${!pausePhases.has(s.phase) || state.pausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pausePending ? "일시정지 요청 중…" : s.phase === "paused" ? "일시정지됨" : "Pause"}</button>`;
  }

  function projectCard(project) {
    const stats = project.stats || {};
    const goal = Math.max(1,n(project.pipeline?.demo_target) || 20);
    const progress = Math.min(100,n(stats.successes) / goal * 100);
    const status = projectStatus(project);
    const thumbnail = asset(project.thumbnail_url);
    const destination = routeUrl(project.id,project.archived ? "settings" : "overview");
    return `<a class="project-card" href="${destination}" aria-label="${esc(project.name)} 프로젝트 열기"><div class="project-image">${thumbnail ? `<img src="${esc(thumbnail)}" alt="${esc(project.name)} 최근 기록" loading="lazy">` : `<div class="project-image-placeholder">${icon("robot")}</div>`}<div class="image-meta"><span class="image-tag">${esc(backendLabel(project.backend))}</span><span class="image-note">${thumbnail ? "저장된 실행" : "새 프로젝트"}</span></div></div><div class="project-card-body"><div class="card-title-row"><h3>${esc(project.name)}</h3><span class="card-arrow">${icon("arrow")}</span></div><p class="project-description">${esc(project.description || project.task_name || "데이터 수집을 준비하는 프로젝트")}</p><div class="project-state-row">${tag(status.text,status.style)}<span class="state-value">${count(stats.successes)} / ${count(goal)} demos</span></div><div class="progress-track"><span style="width:${progress}%"></span></div></div><div class="project-card-footer"><div><strong>${count(stats.episodes)}</strong>Episodes</div><div><strong>${count(stats.reset_successes)} / ${count(stats.episodes)}</strong>GPT reset</div><div><strong>${money(stats.estimated_cost_usd)}</strong>API 비용</div></div></a>`;
  }

  function renderProjectCards() {
    const holder = $("project-cards");
    if (!holder) return;
    const query = state.search.trim().toLowerCase();
    const projects = (state.projectView === "archived" ? state.workspace?.archived_projects || [] : state.workspace?.projects || []).filter(p => [p.name,p.task_name,p.description,p.suite].join(" ").toLowerCase().includes(query));
    holder.innerHTML = projects.length ? projects.map(projectCard).join("") : `<div class="card-empty">${icon(query ? "search" : "folder")}<h3>${query ? "검색 결과가 없어요" : state.projectView === "archived" ? "보관된 프로젝트가 없습니다" : "첫 번째 프로젝트를 만들어보세요"}</h3><p>${query ? "프로젝트 이름이나 task로 다시 검색해주세요." : state.projectView === "archived" ? "보관한 프로젝트를 열어 설정에서 복원할 수 있습니다." : "Task를 선택하고 데이터 수집을 시작할 수 있어요."}</p>${query || state.projectView === "archived" ? "" : `<button class="button primary" data-action="create">${icon("plus")}새 프로젝트</button>`}</div>`;
  }

  function renderHome() {
    if (!state.workspace) return;
    const w = state.workspace, t = w.totals || {}, r = w.runtime || {};
    document.title = "RILO · 워크스페이스";
    $("content").innerHTML = `<section class="page-heading"><div><span class="eyebrow">WORKSPACE OVERVIEW</span><h1>워크스페이스</h1><p>프로젝트별로 데이터를 모으고, 학습으로 이어지는 흐름을 확인하세요.</p></div><div class="heading-action"><button class="button primary" data-action="create">${icon("plus")}새 프로젝트</button></div></section><section class="metrics" aria-label="전체 실험 지표">${metric("프로젝트",`${count(t.projects)}<span class="metric-unit">개</span>`,`${count(t.runs)}개의 실행 기록`,"folder")}${metric("수집된 성공 demo",`${count(t.successes)}<span class="metric-unit">개</span>`,`${count(t.episodes)} episodes · Task 성공 기준`,"database")}${metric("완료된 GPT reset",`${count(t.reset_successes)}<span class="metric-unit">/ ${count(t.episodes)}</span>`,"Task 후 환경 복구까지 완료", "reset")}${metric("데이터 사용 비용",money(t.estimated_cost_usd),`${count(t.total_tokens)} tokens · 사용 중·보관 중 합계`,"dollar")}</section><div class="home-layout"><section><div class="section-header"><div class="section-header-left"><div class="workspace-view-switch" role="group" aria-label="프로젝트 목록"><button type="button" data-action="projects-active" class="${state.projectView === "active" ? "active" : ""}" aria-pressed="${state.projectView === "active"}">프로젝트 <span>${count(w.projects?.length)}</span></button><button type="button" data-action="projects-archived" class="${state.projectView === "archived" ? "active" : ""}" aria-pressed="${state.projectView === "archived"}">${icon("archive")}보관함 <span>${count(w.archived_projects?.length)}</span></button></div></div><label class="search-box">${icon("search")}<input id="project-search" placeholder="프로젝트 검색" aria-label="프로젝트 검색" value="${esc(state.search)}"></label></div><div id="project-cards" class="project-grid"></div><p class="home-footnote">${icon("info")}기존 실행 기록을 프로젝트별로 연결했습니다. 성공 demo와 reset 완료는 각각 집계됩니다.</p></section><aside class="home-aside"><section class="side-panel"><h3>${icon("flow")}파이프라인 연결 상태</h3><ul class="capability-list"><li><span>GPT 데이터 수집 · Reset</span>${tag("사용 가능","success")}</li><li><span>BC · Progress 학습</span>${tag(w.capabilities?.bc ? "사용 가능" : "연결 전",w.capabilities?.bc ? "success" : "neutral")}</li><li><span>BC 정책 평가</span>${tag(w.capabilities?.evaluation ? "사용 가능" : "연결 전",w.capabilities?.evaluation ? "success" : "neutral")}</li><li><span>Residual RL</span>${tag(w.capabilities?.rl ? "사용 가능" : "연결 전",w.capabilities?.rl ? "success" : "neutral")}</li></ul><p class="side-note">${w.capabilities?.bc ? "개요에서 전체 파이프라인을 설정·실행하고, 모델 탭에서 개별 학습을, 평가 탭에서 개별 평가를 관리합니다." : "모델 탭에서 파이프라인 연결 상태를 확인할 수 있습니다."}</p></section><section class="side-panel"><h3>${icon("terminal")}연결된 실행 환경</h3><dl class="runtime-list"><div><dt>Environment</dt><dd>${esc(r.backend?.toUpperCase() || "—")}</dd></div><div><dt>Teacher model</dt><dd class="code">${esc(r.policy === "scripted" ? "Scripted demo" : r.model || "—")}</dd></div><div><dt>Reasoning</dt><dd>${esc(r.reasoning_effort || "—")}</dd></div><div><dt>Worker</dt><dd>1 · ${esc(phaseNames[r.phase] || "실행 대기")}</dd></div></dl><p class="side-note">${r.policy === "scripted" ? "UI 검증용 demo 서버입니다. GPT를 호출하지 않습니다." : "프로젝트를 준비한 뒤 수집 시작을 누르면 모델이 호출됩니다."}</p></section></aside></div>`;
    if (r.policy !== "scripted") {
      document.querySelector(".runtime-list + .side-note").textContent = "프로젝트 설정에서 모델·추론 수준을 선택할 수 있습니다. 변경값은 다음 새 run을 시작할 때 적용됩니다. 환경을 다시 연결할 필요는 없습니다.";
    }
    renderProjectCards();
    renderBreadcrumb();
  }

  function projectShell(project) {
    const tabs = [["overview","개요","flow"],["collection",remoteRobotProject(project) ? "환경 연결" : "수집",remoteRobotProject(project) ? "terminal" : "camera"],["data","데이터","database"],["models","모델","model"],["evaluation","평가","check"],["settings","설정","settings"]];
    const status = projectStatus(project);
    return `<section class="page-heading project-heading"><div><span class="eyebrow">PROJECT WORKSPACE</span><h1>${esc(project.name)}</h1><p>${esc(project.description || project.task_name || "이 프로젝트의 데이터 수집과 학습 흐름을 관리하세요.")}</p><div class="project-title-meta">${tag(backendLabel(project.backend),"outline")}${tag(project.suite || "Custom","outline")}<span class="project-task">Task ${n(project.task_id)} · ${esc(project.task_name || "")}</span></div></div><div class="project-header-actions">${tag(status.text,status.style)}${state.route.tab !== "collection" ? `<a class="button primary" href="${routeUrl(project.id,"collection")}">${icon(remoteRobotProject(project) ? "terminal" : "camera")}${remoteRobotProject(project) ? "환경 연결" : "수집 화면"}</a>` : ""}</div></section><nav class="tabs" aria-label="프로젝트 메뉴">${tabs.map(([id,label,symbol]) => `<a href="${routeUrl(project.id,id)}" class="${state.route.tab === id ? "active" : ""}" ${state.route.tab === id ? 'aria-current="page"' : ""}>${icon(symbol)}${label}${id === "data" ? `<span class="tab-count">${count(project.stats?.episodes)}</span>` : ""}</a>`).join("")}</nav><div id="tab-content"></div>`;
  }

  function overviewStageProgress(stage, project, pipeline) {
    if (stage.id === "collect") {
      const collection = pipeline?.completion?.collect || {};
      const completed = collection.completed ?? project.stats?.successes;
      const target = collection.target ?? pipeline?.config?.collection_episodes ?? project.pipeline?.demo_target;
      if (typeof completed === "number" && Number.isFinite(completed) && typeof target === "number" && Number.isFinite(target) && target > 0) {
        return {value:Math.min(100,Math.max(0,100*completed/target)),label:completed >= target ? "수집 목표 달성" : "성공 데이터 수집",detail:`${count(completed)} / ${count(target)} 성공 episodes · 사용 중 데이터`};
      }
      return {value:null,label:"상태 확인 중",detail:"성공 데이터와 목표를 확인하고 있습니다"};
    }
    if (!pipeline || pipeline.unavailable) return {value:null,label:"상태 확인 중",detail:"실행 상태를 기다리고 있습니다"};
    const status = pipeline.status || {}, current = status.stage === stage.id, active = current && pipelineActive(status.phase);
    const completion = pipeline.completion?.stages?.[stage.id];
    if (completion && !active) return completion.complete
      ? {value:100,label:"완료",detail:completion.reason || "현재 데이터·설정에 맞는 결과가 있습니다"}
      : {value:0,label:"실행 필요",detail:completion.reason || "현재 조건에 맞는 결과가 필요합니다"};
    const metrics = current ? status.metrics || {} : status.stage_metrics?.[stage.id] || {};
    const started = [...(status.events || [])].reverse().find(event => event.message === `${stage.id} 단계 시작`)?.time;
    const phaseLabel = current ? pipelinePhaseNames[status.phase] || "진행률 확인 중" : "실행 대기";
    const ratio = (done,total,detail) => typeof done === "number" && Number.isFinite(done) && typeof total === "number" && Number.isFinite(total) && total > 0
      ? {value:Math.max(0,Math.min(100,100*done/total)),label:phaseLabel,detail} : null;
    if (active && ["bc","progress"].includes(stage.id)) {
      return ratio(metrics.step,pipeline.config?.[stage.id]?.steps,`${count(metrics.step)} / ${count(pipeline.config?.[stage.id]?.steps)} 학습 steps`) || {value:null,label:phaseLabel,detail:"학습 진행률 확인 중"};
    }
    const live = state.status || {}, learning = live.learning || {};
    const liveMatches = current && live.active_project_id === project.id && Boolean(status.job_id) && learning.job_id === status.job_id && learning.pipeline_stage === stage.id && typeof started === "number" && typeof live.episode?.started_at === "number" && live.episode.started_at >= started;
    if (liveMatches) {
      const resetPending = live.config?.reset_mode === "free_space" && live.episode?.reset_outcome === "success" && !live.episode?.environment_reset_done;
      const environmentReset = live.config?.reset_mode === "environment";
      const done = environmentReset ? live.stats?.episodes : typeof live.stats?.reset_successes === "number" ? Math.max(0,live.stats.reset_successes-(resetPending ? 1 : 0)) : undefined;
      const progress = ratio(done,live.config?.episodes,`${count(done)} / ${count(live.config?.episodes)} episodes · ${environmentReset ? "환경 reset" : "reset 포함"}`);
      if (progress) return progress;
    }
    if (completion) return {value:null,label:phaseLabel,detail:"실행 진행률 확인 중"};
    const completed = detail => ({value:100,label:current ? "완료" : "최근 완료",detail});
    const modelKind = stage.id === "bc" ? "bc" : stage.id === "rl" ? "residual" : null;
    const model = modelKind && [...(pipeline.models || [])].reverse().find(item => item.kind === modelKind && item.status === "ready" && item.metrics?.partial !== true && typeof status.started_at === "number" && item.created_at >= status.started_at && (!current || typeof started === "number" && item.created_at >= started));
    if (model) {
      const steps = model.metrics?.steps_completed ?? model.metrics?.step;
      return completed(stage.id === "bc" ? typeof steps === "number" ? `${count(steps)} 학습 steps` : "학습 완료" : "Checkpoint 저장 완료");
    }
    if (metrics.run_id && typeof metrics.episodes === "number" && stage.id !== "rl") return completed(`${count(metrics.episodes)} episodes · reset 완료`);
    if (current && status.phase === "completed") return completed("요청한 단계 완료");
    if (stage.id === "rl" && metrics.run_id && typeof metrics.episodes === "number") return {value:100,label:phaseLabel,detail:"Rollout 완료 · checkpoint 확인 중"};
    if (!current && !started) return {value:0,label:"실행 대기",detail:"아직 시작하지 않았습니다"};
    return {value:null,label:phaseLabel,detail:current ? "실행 진행률 확인 중" : "진행 기록 확인 중"};
  }

  const manualPipelineStages = () => pipelineStages.filter(stage => (state.route.tab === "evaluation" ? ["bc_eval","rl_eval"] : ["bc","rl"]).includes(stage.id));
  const selectedPipelineStage = () => state.route.tab === "evaluation" ? state.evaluationStage : "bc";

  function autoPlanCopy(pipeline) {
    const plan = pipeline?.auto_plan, collection = pipeline?.completion?.collect;
    if (!plan) return "현재 데이터와 모델을 확인하고 있습니다.";
    if (plan.complete) return "현재 데이터·설정에 맞는 학습과 평가 결과가 있고 모든 성공률 목표를 충족했습니다. 추가 실행이 필요하지 않습니다.";
    if (plan.next_stage === "collect") return `사용 중인 성공 task episode ${count(collection?.completed)} / ${count(collection?.target)}개를 보유하고 있습니다. 남은 성공 episode ${count(plan.remaining_episodes)}개를 수집한 뒤 필요한 학습·평가를 이어갑니다.`;
    return `${pipelineStageName(plan.next_stage)}부터 진행합니다. 현재 데이터·설정에 맞는 완료 결과를 재사용하며 남은 단계만 실행합니다.`;
  }

  function pipelineControlStrip(status) {
    const busy = pipelineActive(status.phase), paused = status.phase === "paused";
    return `<div class="pipeline-control-strip"><div><span class="pipeline-current-label">${status.stage ? esc(pipelineStageName(status.stage)) : "아직 시작하지 않았습니다"}</span><span>${status.mode ? `${status.mode === "auto" ? "자동" : "단계별"} 실행` : ""}${status.cycle ? ` · Cycle ${count(status.cycle)}` : ""}${status.round ? ` · Round ${count(status.round)}` : ""}</span></div><div class="pipeline-control-buttons"><button class="button pause small" data-action="pipeline-pause" ${status.phase !== "running" || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pipelinePausePending ? "요청 중…" : "Pause"}</button><button class="button secondary small" data-action="pipeline-resume" ${!paused || state.pipelinePending || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("play")}재개</button><button class="button danger small" data-action="pipeline-stop" ${!busy || state.pipelinePending || !state.connected ? "disabled" : ""}>${icon("stop")}중지</button></div></div>${status.error ? `<p class="pipeline-error" role="alert">${esc(status.error)}</p>` : ""}`;
  }

  function renderPipeline(project) {
    if (remoteRobotProject(project)) return remoteConnectionOnlyPanel(project,"파이프라인 실행");
    const pipeline = currentPipeline(), status = pipeline?.status || {}, enabled = pipelineEnabled();
    const config = pipeline?.config || {}, stages = pipelineStages.filter(stage => stage.id !== "progress");
    const plan = pipeline?.auto_plan, ready = pipelineReadiness("auto"), busy = pipelineActive(status.phase), global = pipelineRuntime();
    const other = global && pipelineActive(global.phase) && global.project_id !== project.id;
    const unavailable = !state.connected || state.pipelinePending || state.pipelinePausePending || busy || other || !pipeline?.config;
    const reason = other ? "다른 프로젝트의 파이프라인이 실행 중입니다." : !plan ? "남은 실행 단계를 확인하고 있습니다." : !plan.complete && !ready.ready ? ready.reason : "";
    return `<section class="panel pipeline-panel"><div class="panel-header"><div><h2>Orchestration</h2><p>수집 → BC 학습 → BC 평가 → Residual RL → RL 평가</p></div><button class="button primary small" data-action="pipeline-start-auto" ${unavailable ? "disabled" : ""}>${icon("play")}${state.pipelinePending === "configure-start" ? "설정 저장·실행 중…" : "전체 파이프라인 실행"}</button></div><div class="overview-auto-plan"><p>${esc(autoPlanCopy(pipeline))}</p>${reason ? `<p class="pipeline-readiness" role="status">${icon("info")}${esc(reason)}</p>` : ""}</div>${busy || status.error || state.pipelinePending ? pipelineControlStrip(status) : ""}<div class="sequential-pipeline">${stages.map((stage,index) => {
      const progress = overviewStageProgress(stage,project,pipeline), percent = progress.value === null ? null : Math.round(progress.value);
      const description = stage.id === "collect" ? `${count(pipeline?.completion?.collect?.target ?? config.collection_episodes ?? project.pipeline?.demo_target ?? 20)} 성공 episodes 목표` : stage.id === "bc" ? "수집 중지 후 학습" : stage.id === "bc_eval" ? `목표 ${finiteValue(config.bc_success_target ?? project.pipeline?.bc_success_target ?? 70,1)}%` : stage.id === "rl" ? "Frozen DP + correction" : `목표 ${finiteValue(config.rl_success_target ?? project.pipeline?.rl_success_target ?? 90,1)}%`;
      return `${index === 0 ? '<div class="sequence-bc-cycle"><div class="sequence-bc-flow">' : ""}<div class="sequence-stage ${status.stage === stage.id && pipelineActive(status.phase) ? "current" : ""}" data-stage="${stage.id}"><span class="sequence-number">${String(index+1).padStart(2,"0")}</span><div><h3>${esc(stage.short)}</h3><p>${esc(description)}</p></div><div class="sequence-progress"><div class="sequence-progress-label"><span>${esc(progress.label)}</span><strong>${percent === null ? "—" : `${percent}%`}</strong></div><div class="progress-track${percent === null ? " indeterminate" : ""}" role="progressbar" aria-label="${esc(stage.short)} 진행률" aria-valuemin="0" aria-valuemax="100" ${percent === null ? 'aria-valuetext="진행률 확인 중"' : `aria-valuenow="${percent}"`}><span ${percent === null ? "" : `style="width:${progress.value}%"`}></span></div><small>${esc(progress.detail)}</small></div>${index < stages.length-1 ? `<span class="sequence-arrow">${icon("arrow")}</span>` : ""}</div>${index === 2 ? '</div><div class="sequence-return" role="img" aria-label="03 BC 평가에서 목표에 미달하면 01 수집으로 돌아갑니다"><svg viewBox="0 0 300 32" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="sequence-return-tip" viewBox="0 0 6 6" markerWidth="3.5" markerHeight="3.5" refX="5" refY="3" orient="auto"><path d="M1 1L5 3L1 5"/></marker></defs><path d="M250 1V24H50V2" marker-end="url(#sequence-return-tip)"/></svg><span>목표 미달 시 추가 수집</span></div></div>' : ""}`;
    }).join("")}</div><div class="pipeline-note">${icon("info")}${enabled ? "사용 중인 성공 task 데이터와 현재 설정에 맞는 학습·평가 결과를 반영합니다. 보관·삭제 데이터는 수집 목표에서 제외되며, 자동 실행은 남은 단계부터 이어갑니다." : "학습 worker 연결 전입니다. 프로젝트의 데이터와 기존 수집 기능은 그대로 사용할 수 있습니다."}${status.stage ? `<span class="sequence-status">${tag(pipelinePhaseNames[status.phase] || status.phase,status.phase === "running" ? "success" : status.phase === "error" ? "warning" : "neutral")} ${esc(status.stage === "progress" ? "RL 학습 준비" : pipelineStageName(status.stage))}</span>` : ""}</div></section>`;
  }

  function episodesTable(episodes, compact = false) {
    const selection = compact ? null : currentDataSelection();
    const rows = episodes.map(ep => {
      const label = episodeLabel(ep);
      const attrs = `data-run="${esc(ep.run_id)}" data-episode="${esc(ep.id)}"`;
      const disabled = ep.can_manage !== true || Boolean(state.episodeMutation);
      const reason = ep.manage_disabled_reason || (disabled ? "실행 기록이 정리된 후 관리할 수 있습니다." : "");
      const archiveLabel = ep.archived ? "보관 해제" : "보관";
      const pendingDelete = ep.delete_pending === true;
      const selectCell = selection ? `<td class="episode-select-cell"><input type="checkbox" data-episode-select ${attrs} aria-label="${esc(label)} 선택" ${selection.selected.has(episodeSelectionKey(ep)) ? "checked" : ""} ${!canSelectEpisode(ep,selection.mode) || state.episodeMutation ? "disabled" : ""}></td>` : "";
      return `<tr class="episode-clickable-row" data-episode-row="${esc(ep.id)}" data-run-row="${esc(ep.run_id)}" ${pendingDelete ? 'aria-disabled="true"' : `tabindex="0" role="button" data-action="episode" ${attrs} aria-label="${esc(label)} 보기"`}>${selectCell}<td><strong>${esc(label)}</strong><span class="table-sub">${esc(date(ep.started_at))}</span>${ep.archived ? `<span class="table-sub">보관 ${esc(date(ep.archived_at))}</span>` : ""}</td><td>${pendingDelete ? tag("삭제 미완료","warning") : episodeOutcomeLabel(ep)}${!pendingDelete && ep.outcome === "success" && (!externalEpisode(ep) || qualityKey(ep,"task") !== "unknown") ? `<span class="episode-quality-tag">${qualityTag(ep,"task")}</span>` : ""}</td><td>${pendingDelete ? "—" : episodeResetLabel(ep)}</td>${compact ? "" : `<td>${count(ep.steps)} <span class="cell-muted">/</span> ${count(ep.reset_steps)}</td>`}<td>${money(episodeCost(ep),compact ? 2 : 4)}</td><td class="episode-manage-cell"><div class="episode-row-actions"><button class="icon-button episode-icon" data-action="${ep.archived ? "restore-episode" : "archive-episode"}" ${attrs} aria-label="${esc(label)} ${archiveLabel}" title="${esc(pendingDelete ? "삭제가 진행된 데이터는 보관 해제할 수 없습니다." : reason || archiveLabel)}" ${disabled || pendingDelete ? "disabled" : ""}>${icon(ep.archived ? "restore" : "archive")}</button><button class="icon-button episode-icon delete-icon" data-action="delete-episode" ${attrs} aria-label="${esc(label)} 영구 삭제" title="${esc(reason || "영구 삭제")}" ${disabled ? "disabled" : ""}>${icon("trash")}</button></div></td></tr>`;
    }).join("");
    const empty = state.route.tab === "data" && state.dataView === "archived" ? "보관된 episode가 없습니다." : state.filter !== "all" && state.route.tab === "data" ? "선택한 결과에 해당하는 episode가 없습니다." : "아직 저장된 episode가 없습니다. 수집 화면에서 첫 실행을 준비하세요.";
    return `<div class="table-wrap"><table><thead><tr>${selection ? '<th class="episode-select-cell"><input id="episode-select-all" type="checkbox" aria-label="현재 목록에서 선택 가능한 데이터 전체 선택"></th>' : ""}<th>Episode / Rollout</th><th>Task</th><th>GPT reset</th>${compact ? "" : "<th>Step (Task / Reset)</th>"}<th>API 비용</th><th class="episode-manage-cell"><span class="sr-only">Episode 관리</span></th></tr></thead><tbody>${rows || `<tr><td class="table-empty" colspan="${(compact ? 5 : 6) + (selection ? 1 : 0)}">${empty}</td></tr>`}</tbody></table></div>`;
  }

  function renderOverview(project) {
    if (remoteRobotProject(project)) {$("tab-content").innerHTML = remoteConnectionOnlyPanel(project,"원격 로봇 PC 연결");return;}
    const stats = project.stats || {};
    const episodes = (state.detail.episodes || []).slice().sort((a,b) => n(b.started_at)-n(a.started_at));
    $("tab-content").innerHTML = `${renderPipeline(project)}<section class="metrics" aria-label="프로젝트 지표">${metric("성공 demo",`${count(stats.successes)}<span class="metric-unit">개</span>`,`${count(stats.episodes)} episodes 중 Task 성공`,"database")}${metric("GPT reset 완료",`${count(stats.reset_successes)}<span class="metric-unit">/ ${count(stats.episodes)}</span>`,"환경 복구 검증 통과","reset")}${metric("전체 실행 step",count(stats.total_steps),"Task + Reset control ticks","layers")}${metric("데이터 사용 비용",money(stats.estimated_cost_usd),`${count(stats.total_tokens)} tokens · 사용 중·보관 중 합계`,"dollar")}</section><div class="overview-bottom"><section class="panel"><div class="panel-header"><h2>최근 episodes</h2><a class="button quiet small" href="${routeUrl(project.id,"data")}">전체 보기 ${icon("arrow")}</a></div>${episodesTable(episodes.slice(0,4),true)}</section><section class="panel"><div class="panel-header"><h2>수집 설정</h2><a href="${routeUrl(project.id,"collection")}" class="icon-button" aria-label="수집 설정 편집">${icon("settings")}</a></div><div class="panel-body"><dl class="project-rules"><div><dt>수집당 episode</dt><dd>${n(project.run_config?.episodes) || 1} episodes</dd></div><div><dt>Reset 방식</dt><dd>${esc(resetModeLabel(project.run_config?.reset_mode))}</dd></div><div><dt>실행당 API 예산</dt><dd>${money(project.run_config?.budget_usd)}</dd></div></dl><p class="muted-info">설정은 새 run을 시작할 때 적용됩니다. Pause·재개는 시작한 run의 설정을 유지합니다.</p></div></section></div>`;
  }

  function collectionGoalLocked() {
    const pipeline = currentPipeline(), runtime = pipelineRuntime();
    return !pipeline?.config || pipelineActive(pipeline.status?.phase) || Boolean(runtime?.project_id === state.route.id && pipelineActive(runtime.phase));
  }

  function renderCollectionSettings(project) {
    const config = project.run_config || {}, goal = currentPipeline()?.config?.collection_episodes ?? project.pipeline?.demo_target ?? 20;
    state.collectionSettingsSignature = JSON.stringify([config,currentPipeline()?.config?.collection_episodes]);
    return `<section id="collection-settings-panel" class="panel collection-settings-panel" data-expanded="${state.collectionSettingsExpanded}"><div class="panel-header"><h2>수집 실행 설정</h2><button id="collection-settings-toggle" class="icon-button collection-settings-toggle" type="button" data-action="collection-settings-toggle" aria-expanded="${state.collectionSettingsExpanded}" aria-controls="collection-settings-form" aria-label="수집 실행 설정 ${state.collectionSettingsExpanded ? "접기" : "펼치기"}"><span aria-hidden="true">${state.collectionSettingsExpanded ? "−" : "+"}</span></button></div><form id="collection-settings-form" ${state.collectionSettingsExpanded ? "" : "hidden"}><fieldset id="collection-settings-fields" class="panel-body"><div class="collection-config-row"><label class="field">Episode 수집 목표<div class="input-unit"><input name="collection_episodes" type="number" min="1" max="100" step="1" required value="${n(goal)}" data-saved-goal="${n(goal)}" ${collectionGoalLocked() ? "disabled" : ""}><span>개</span></div></label>${numericField("episodes","Run 당 episode rollout",config.episodes ?? 1,1,100,"개")}${numericField("budget_usd","API 비용 한도",config.budget_usd ?? 15,.01,1000,"$",.01)}</div><div class="collection-config-row two">${numericField("max_steps","최대 episode 길이",config.max_steps ?? 300,1,10000,"steps")}${numericField("max_decisions","Task 당 최대 GPT 호출 횟수",config.max_decisions ?? 50,1,1000,"회")}</div><div class="collection-config-row"><label class="field">Reset 방식<select name="reset_mode"><option value="agent" ${config.reset_mode === "agent" || !config.reset_mode ? "selected" : ""}>GPT arm reset</option><option value="environment" ${config.reset_mode === "environment" ? "selected" : ""}>환경 초기화</option><option value="free_space" ${config.reset_mode === "free_space" ? "selected" : ""}>대상 복구 + 안전 후퇴 후 환경 초기화</option></select></label>${numericField("max_reset_steps","최대 reset step",config.max_reset_steps ?? 300,1,10000,"steps")}${numericField("max_reset_decisions","Reset 당 최대 GPT 호출 횟수",config.max_reset_decisions ?? 40,1,1000,"회")}</div><p id="collection-goal-summary" class="field-note"></p><p id="collection-goal-note" class="field-note"></p><p class="field-note">수집 목표에는 사용 중인 성공 task episode만 포함하며 보관·삭제 데이터는 제외합니다. BC 평가 미달로 늘어난 누적 목표는 기본 목표를 바꾸면 다시 설정됩니다.</p><p class="field-note">Run 설정은 환경 재연결 없이 다음 새 run부터 적용됩니다. 시작한 run은 Pause·재개 시에도 기존 설정을 유지합니다. 한 번만 실행은 1 episode와 reset을 수행합니다.</p><p class="field-note">안전 후퇴 모드는 시뮬레이터에서 복구·후퇴 검증 후 환경을 초기화합니다. API 비용 한도는 응답 후 확인하므로 마지막 호출 비용만큼 초과할 수 있습니다.</p></fieldset><p id="collection-settings-error" class="form-error" role="alert" hidden></p><div class="pipeline-config-actions"><p>저장만으로 환경 연결이나 수집·학습을 시작하지 않습니다.</p><button id="collection-settings-save" class="button primary" type="submit">${icon("check")}수집 설정 저장</button></div></form></section>`;
  }

  function setCollectionSettingsExpanded(expanded) {
    state.collectionSettingsExpanded = expanded;
    const panel = $("collection-settings-panel"), form = $("collection-settings-form"), button = $("collection-settings-toggle");
    if (!panel || !form || !button) return;
    panel.dataset.expanded = String(expanded);
    form.hidden = !expanded;
    button.setAttribute("aria-expanded", String(expanded));
    button.setAttribute("aria-label", `수집 실행 설정 ${expanded ? "접기" : "펼치기"}`);
    button.innerHTML = `<span aria-hidden="true">${expanded ? "−" : "+"}</span>`;
  }

  function updateCollectionSettings() {
    if (state.route.tab !== "collection" || !$("collection-settings-panel") || !currentProject()) return;
    const pipeline = currentPipeline(), goal = pipeline?.config?.collection_episodes;
    const signature = JSON.stringify([currentProject().run_config || {},goal]);
    if (!state.collectionSettingsDirty && !state.collectionSettingsSaving && signature !== state.collectionSettingsSignature) $("collection-settings-panel").outerHTML = renderCollectionSettings(currentProject());
    const blocked = !state.connected || state.collectionSettingsSaving || Boolean(state.pipelinePending);
    if ($("collection-settings-fields")) $("collection-settings-fields").disabled = blocked;
    if ($("collection-settings-save")) $("collection-settings-save").disabled = blocked;
    const input = $("collection-settings-form")?.elements?.collection_episodes;
    if (input) {
      if (!state.collectionSettingsSaving && typeof goal === "number" && Number(input.value) === Number(input.dataset.savedGoal)) {input.value = String(goal);input.dataset.savedGoal = String(goal);}
      input.disabled = blocked || collectionGoalLocked();
    }
    const collection = pipeline?.completion?.collect || {}, target = collection.target ?? goal;
    $("collection-goal-summary").textContent = typeof target === "number" ? `사용 중 성공 ${count(collection.completed ?? currentProject().stats?.successes)}개 · 현재 누적 목표 ${count(target)}개 · 추가 ${count(collection.remaining ?? Math.max(0,target-n(currentProject().stats?.successes)))}개 필요` : "저장된 수집 목표를 확인하고 있습니다.";
    $("collection-goal-note").textContent = collectionGoalLocked() ? "파이프라인 실행 중이거나 목표를 불러오는 동안에는 누적 목표를 바꿀 수 없습니다. Run 실행 한도는 다음 새 run에 적용하도록 저장할 수 있습니다." : "Episode 수집 목표는 누적 성공 데이터 기준이며, Run 당 episode rollout은 한 번의 수집 실행 횟수입니다.";
  }

  async function saveCollectionSettings(event) {
    event.preventDefault();
    const form = event.target, project = currentProject(), token = state.routeToken;
    if (!project || state.route.tab !== "collection" || state.collectionSettingsSaving || state.pipelinePending || !state.connected || !form.reportValidity()) return;
    const goalInput = form.elements.collection_episodes;
    const goal = goalInput ? Number(goalInput.value) : null;
    const goalChanged = Boolean(goalInput) && goal !== Number(goalInput.dataset.savedGoal);
    if (goalChanged && collectionGoalLocked()) {$("collection-settings-error").textContent = "누적 목표를 바꾸려면 파이프라인을 완료하거나 중지해주세요. 입력한 실행 설정은 유지됩니다.";$("collection-settings-error").hidden = false;return;}
    const data = new FormData(form), config = {...project.run_config};
    for (const key of ["episodes","budget_usd","max_steps","max_decisions","max_reset_steps","max_reset_decisions"]) config[key] = Number(data.get(key));
    config.reset_mode = String(data.get("reset_mode"));config.task_id = project.task_id;
    state.collectionSettingsDirty = true;state.collectionSettingsSaving = true;state.pipelinePending = "collection-settings";
    $("collection-settings-error").hidden = true;updateCollection();
    let goalSaved = false;
    try {
      if (goalChanged) {
        const pipeline = await api(`/api/projects/${encodeURIComponent(project.id)}/pipeline/config`,{collection_episodes:goal});
        goalSaved = true;
        if (token === state.routeToken && state.route.id === project.id) {
          state.pipelineRequest += 1;state.pipelineBusy = false;state.pipeline = pipeline;state.pipelineProjectId = project.id;
          goalInput.dataset.savedGoal = String(pipeline.config?.collection_episodes ?? goal);
        }
      }
      // Navigation changes the visible project, not the destination of this approved save.
      const result = await api(`/api/projects/${encodeURIComponent(project.id)}/settings`,{run_config:config});
      if (token !== state.routeToken || state.route.id !== project.id) return;
      state.detail.project = result.project;state.collectionSettingsDirty = false;
      toast("수집 설정을 저장했습니다. Run 설정은 다음 새 run부터 적용하며 현재 run과 Pause·재개 설정은 유지합니다.");
      await Promise.allSettled([refreshStatus(true),refreshWorkspace()]);
    } catch (error) {
      const message = `${goalSaved ? "누적 수집 목표는 저장했지만 Run 실행 설정을 저장하지 못했습니다. 입력값을 확인한 뒤 다시 저장해주세요. " : "수집 실행 설정을 저장하지 못했습니다. "}${error.message}`;
      if (token === state.routeToken && $("collection-settings-error")) {setCollectionSettingsExpanded(true);$("collection-settings-error").textContent = message;$("collection-settings-error").hidden = false;}
      else toast(message,true);
    } finally {
      state.collectionSettingsSaving = false;
      if (state.pipelinePending === "collection-settings") state.pipelinePending = null;
      if (token === state.routeToken) updateCollection();
    }
  }

  const episodeSelectionKey = (episode) => JSON.stringify([episode.run_id,episode.id]);
  const canSelectEpisode = (episode, mode) => episode.can_manage === true && (mode === "delete" || !episode.delete_pending);
  const currentDataSelection = () => state.dataSelection?.project === state.route.id && state.route.tab === "data" && state.dataSelection.view === state.dataView ? state.dataSelection : null;

  function visibleDataEpisodes() {
    const episodes = state.dataView === "archived" ? state.detail?.archived_episodes || [] : state.detail?.episodes || [];
    return episodes.filter(ep => {
      if (state.filter === "all") return true;
      if (state.filter === "complete") return complete(ep);
      if (state.filter === "incomplete") return ep.outcome && (ep.outcome !== "success" || !externalEpisode(ep) && !complete(ep));
      if (state.filter === "task_success") return ep.outcome === "success";
      return ep.outcome === "success" && qualityKey(ep,"task") === state.filter;
    }).slice().sort((a,b) => n(b.started_at)-n(a.started_at));
  }

  function selectedDataEpisodes() {
    const selection = currentDataSelection();
    return selection ? visibleDataEpisodes().filter(ep => canSelectEpisode(ep,selection.mode) && selection.selected.has(episodeSelectionKey(ep))) : [];
  }

  function bulkEpisodeToolbar(episodes) {
    const selection = currentDataSelection(), busy = Boolean(state.episodeMutation);
    if (selection) {
      const allowed = new Set(episodes.filter(ep => canSelectEpisode(ep,selection.mode)).map(episodeSelectionKey));
      selection.selected.forEach(key => {if (!allowed.has(key)) selection.selected.delete(key);});
    }
    const archiveWord = state.dataView === "archived" ? "보관 해제" : "보관";
    return `<div class="episode-bulk-toolbar"><span id="episode-bulk-status" role="status"></span><div class="episode-bulk-buttons">${["archive","delete"].map(mode => {
      const active = selection?.mode === mode, word = mode === "delete" ? "삭제" : archiveWord;
      const unavailable = busy || (active ? !selection.selected.size : !episodes.some(ep => canSelectEpisode(ep,mode)));
      return `<button id="episode-bulk-${mode}" type="button" class="button small ${mode === "delete" ? "danger" : "secondary"}" data-action="bulk-episode-select" data-mode="${mode}" aria-pressed="${active}" ${unavailable ? "disabled" : ""}>${icon(mode === "delete" ? "trash" : state.dataView === "archived" ? "restore" : "archive")}${active ? "일괄" : "선택"} ${word}</button>`;
    }).join("")}${selection ? `<button type="button" class="button quiet small" data-action="bulk-episode-cancel" ${busy ? "disabled" : ""}>취소</button>` : ""}</div></div>${selection?.error ? `<p class="form-error episode-bulk-error" role="alert">${esc(selection.error)}</p>` : ""}`;
  }

  function updateBulkEpisodeControls() {
    const selection = currentDataSelection(), busy = Boolean(state.episodeMutation);
    const eligible = selection ? visibleDataEpisodes().filter(ep => canSelectEpisode(ep,selection.mode)) : [];
    const selected = selectedDataEpisodes().length;
    const status = $("episode-bulk-status");
    if (status) status.textContent = state.bulkEpisodeBusy?.project === state.route.id ? `${state.bulkEpisodeBusy.done} / ${state.bulkEpisodeBusy.total}개 처리 중…` : selection ? `${selected}개 선택` : "";
    const all = $("episode-select-all");
    if (all) {all.checked = eligible.length > 0 && selected === eligible.length;all.indeterminate = selected > 0 && selected < eligible.length;all.disabled = busy || !eligible.length;}
    if (selection) {
      const button = $(`episode-bulk-${selection.mode}`);if (button) button.disabled = busy || !selected;
      document.querySelectorAll("input[data-episode-select]").forEach(input => {
        const ep = findEpisode(input.dataset.run,input.dataset.episode);
        input.checked = selection.selected.has(JSON.stringify([input.dataset.run,input.dataset.episode]));
        input.disabled = busy || !ep || !canSelectEpisode(ep,selection.mode);
        input.closest("tr")?.classList.toggle("episode-selected",input.checked);
      });
    }
  }

  function toggleEpisodeSelection(input) {
    const selection = currentDataSelection();
    if (!selection || state.episodeMutation) return;
    const episodes = visibleDataEpisodes().filter(ep => canSelectEpisode(ep,selection.mode));
    const targets = input.id === "episode-select-all" ? episodes : episodes.filter(ep => ep.run_id === input.dataset.run && ep.id === input.dataset.episode);
    targets.forEach(ep => {const key = episodeSelectionKey(ep);if (input.checked) selection.selected.add(key);else selection.selected.delete(key);});
    selection.error = null;updateBulkEpisodeControls();
  }

  function selectBulkEpisodes(mode) {
    const project = currentProject();
    if (!project || state.route.tab !== "data" || state.episodeMutation || !["archive","delete"].includes(mode)) return;
    const selection = currentDataSelection();
    if (selection?.mode === mode) {
      const episodes = selectedDataEpisodes();
      if (!episodes.length) return;
      const target = {project:project.id,routeToken:state.routeToken,view:state.dataView,mode,items:episodes.map(ep => ({run:ep.run_id,id:ep.id,label:episodeLabel(ep),key:episodeSelectionKey(ep)}))};
      if (mode === "archive") {mutateBulkEpisodes(target);return;}
      state.bulkEpisodeTarget = target;
      $("bulk-episode-info").innerHTML = `<strong>${esc(project.name)} · ${episodes.length}개 선택</strong><ul>${target.items.map(item => `<li>${esc(item.label)}</li>`).join("")}</ul>`;
      $("bulk-episode-confirm").disabled = false;$("bulk-episode-confirm").textContent = `${episodes.length}개 일괄 삭제`;
      $("bulk-episode-dialog").showModal();
      return;
    }
    state.dataSelection = {project:project.id,view:state.dataView,mode,selected:new Set(),error:null};
    renderData(project);
  }

  async function mutateBulkEpisodes(target = state.bulkEpisodeTarget) {
    if (!target || state.episodeMutation || target.project !== state.route.id || target.routeToken !== state.routeToken || state.route.tab !== "data" || target.view !== state.dataView || !target.items.length) return;
    if (target.mode === "delete" && (!$("bulk-episode-dialog").open || target !== state.bulkEpisodeTarget)) return;
    const selection = currentDataSelection(), failures = [];
    let batchDetail = state.detail;
    state.episodeMutation = "bulk";
    const progress = state.bulkEpisodeBusy = {project:target.project,done:0,total:target.items.length};
    $("bulk-episode-confirm").disabled = true;$("bulk-episode-confirm").textContent = "삭제 중…";
    renderData(currentProject());
    try {
      for (const [index,item] of target.items.entries()) {
        try {
          const latest = state.detail?.project?.id === target.project ? state.detail : batchDetail;
          const episode = [...(latest?.episodes || []),...(latest?.archived_episodes || [])].find(ep => ep.run_id === item.run && ep.id === item.id);
          if (!episode || !canSelectEpisode(episode,target.mode)) {
            failures.push(`${item.label}: ${episode?.manage_disabled_reason || "현재 관리할 수 없는 데이터입니다."}`);
            progress.done += 1;
            continue;
          }
          const detail = await api(`${episodePath(target.project,item.run,item.id)}/${target.mode}`,target.mode === "delete" ? {confirm:true} : {archived:target.view !== "archived"},30000);
          batchDetail = detail;
          selection?.selected.delete(item.key);
          if (state.route.id === target.project) {state.detail = detail;state.detailSignature = "";}
          if (target.mode === "delete" && state.episodeDetail?.run === item.run && state.episodeDetail?.id === item.id) $("episode-dialog").close();
        } catch (error) {
          failures.push(`${item.label}: ${error.message}`);
          if (!error.status) {
            failures.push(...target.items.slice(index+1).map(next => `${next.label}: 연결 오류로 요청하지 않았습니다.`));
            break;
          }
        }
        progress.done += 1;
        if (state.route.id === target.project && state.route.tab === "data") renderData(currentProject());
      }
    } finally {
      state.episodeMutation = null;state.bulkEpisodeBusy = null;state.bulkEpisodeTarget = null;
      $("bulk-episode-dialog").close();
      if (selection && state.dataSelection === selection) {
        if (failures.length) selection.error = `${target.items.length - failures.length}개 처리 완료 · ${failures.length}개 처리 실패. ${failures.join(" / ")}`;
        else state.dataSelection = null;
      }
      const verb = target.mode === "delete" ? "삭제" : target.view === "archived" ? "보관 해제" : "보관";
      toast(failures.length ? `${target.items.length - failures.length}개 ${verb} 완료 · ${failures.length}개 실패. 데이터 목록에서 확인해주세요.` : `${target.items.length}개 데이터를 ${verb}했습니다.`,failures.length > 0);
      if (state.route.id === target.project && state.route.tab === "data") renderData(currentProject());
      await refreshWorkspace();
    }
  }

  function renderData(project) {
    const archived = state.dataView === "archived";
    const filtered = visibleDataEpisodes();
    $("tab-content").innerHTML = `<div class="data-toolbar"><div><h2>${archived ? "보관함" : "수집 데이터"}</h2><p>${archived ? "보관한 episode는 대시보드 demo 수와 기본 목록에서 제외됩니다." : `${count(project.stats?.runs)} runs · ${count(project.stats?.episodes)} episodes · 데이터와 GPT 판단 기록`}</p></div><select id="episode-filter" class="filter-select" aria-label="Episode 결과 필터"><option value="all">전체 episodes</option><option value="complete">Task + Reset 완료</option><option value="incomplete">실패 / 복구 미완료</option><option value="task_success">Task 성공 · 전체</option><option value="within_grasp_budget">Task 성공 · Optimal</option><option value="extra_grasp_attempts">Task 성공 · Sub-optimal</option><option value="unknown">Task 성공 · 분류 정보 없음</option></select></div><div class="data-view-switch" role="group" aria-label="데이터 표시 범위"><button class="${!archived ? "selected" : ""}" data-action="data-active" aria-pressed="${!archived}">${icon("database")}데이터 <span>${count(project.stats?.episodes)}</span></button><button class="${archived ? "selected" : ""}" data-action="data-archived" aria-pressed="${archived}">${icon("archive")}보관함 <span>${count(project.archived_count ?? state.detail.archived_episodes?.length)}</span></button></div>${bulkEpisodeToolbar(filtered)}<section class="panel episode-data-panel">${episodesTable(filtered)}</section><p class="home-footnote">${icon("info")}${archived ? "보관 해제로 데이터 목록에 다시 표시할 수 있습니다. 보관은 원본 기록과 영상을 유지합니다." : "보관 아이콘은 데이터를 숨기고, 휴지통은 확인 후 episode 기록과 영상을 영구 삭제합니다."}</p><p class="home-footnote billing-note">데이터 사용 비용과 토큰 사용량은 사용 중·보관 중 episode의 합계입니다. 삭제한 episode는 집계에서 제외됩니다.</p>`;
    $("episode-filter").value = state.filter;
    updateBulkEpisodeControls();
  }

  const episodeVideoUnavailableMessage = "현재 서버에 episode 영상 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요.";

  function setEpisodeVideoError(detail, error) {
    detail.videoApiUnavailable = error.status === 404 && error.detail === "Not Found";
    detail.videoError = detail.videoApiUnavailable ? episodeVideoUnavailableMessage : error.message;
  }

  function episodeVideoBlockedReason(detail) {
    if (!detail) return "Episode를 선택해주세요.";
    if (detail.videoApiUnavailable) return episodeVideoUnavailableMessage;
    if (!state.connected) return "서버 연결을 확인해주세요.";
    const learning = pipelineRuntime();
    if (learning && pipelineActive(learning.phase) && learning.project_id === detail.project) return "이 프로젝트의 파이프라인을 중지한 뒤 영상을 관리할 수 있습니다.";
    const status = state.status || {};
    if (status.run_id === detail.run && activePhases.has(status.phase)) return "현재 run이 끝나거나 중지된 뒤 영상을 관리할 수 있습니다.";
    const episode = findEpisode(detail.run,detail.id) || detail.episode;
    return episode?.can_manage === false || episode?.delete_pending ? episode.manage_disabled_reason || "Episode 기록을 정리한 뒤 영상을 관리할 수 있습니다." : "";
  }

  function updateEpisodeVideo() {
    const detail = state.episodeDetail, panel = $("episode-video-actions"), player = $("episode-video-player");
    if (!detail || !$("episode-dialog").open || !panel || !player) return;
    const ep = findEpisode(detail.run,detail.id) || detail.episode || {}, job = detail.videoJob;
    const video = asset(ep.video_url), poster = asset(ep.poster_url), signature = `${video}|${poster}`;
    if (player.dataset.mediaSignature !== signature) {
      player.querySelector("video")?.pause();
      player.dataset.mediaSignature = signature;
      player.innerHTML = video ? `<video controls playsinline preload="metadata" src="${esc(video)}" ${poster ? `poster="${esc(poster)}"` : ""}></video><p class="episode-video-note">시뮬레이션 시간 기준 영상 · GPT 추론 대기는 제외됩니다.</p>` : '<div class="episode-video-empty">이 episode의 영상이 아직 생성되지 않았습니다.</div>';
    }
    const busy = ["queued","running"].includes(job?.state), reason = episodeVideoBlockedReason(detail);
    const controlsSignature = JSON.stringify([video,job,reason,detail.videoMutation,detail.videoError]);
    if (panel.dataset.controlsSignature === controlsSignature) return;
    panel.dataset.controlsSignature = controlsSignature;
    const status = detail.videoMutation === "delete" ? "영상을 삭제하고 있습니다." : detail.videoMutation ? "영상 생성을 요청하고 있습니다." : detail.videoError ? detail.videoError : job?.state === "error" ? job.error || "영상 생성에 실패했습니다. 다시 시도할 수 있습니다." : busy ? job.state === "queued" ? "이 episode의 영상 생성 대기 중입니다." : "이 episode의 영상을 생성하고 있습니다." : reason || (video ? "영상만 삭제할 수 있습니다. Episode의 관측·action·성공 기록은 유지됩니다." : "저장된 관측으로 이 episode의 영상을 생성합니다. 로봇 실행이나 GPT 호출은 발생하지 않습니다.");
    panel.innerHTML = `<div class="episode-video-controls">${video ? `<button class="button danger small" data-action="delete-episode-video" ${reason || busy || detail.videoMutation ? "disabled" : ""}>${icon("trash")}영상 삭제</button><a href="${esc(video)}" download class="button secondary small">${icon("download")}MP4 다운로드</a>` : `<button class="button secondary" data-action="generate-episode-video" ${reason || busy || detail.videoMutation || job?.state === "completed" ? "disabled" : ""}>${icon("camera")}${busy ? "영상 생성 중…" : "영상 생성"}</button>`}</div><p class="${detail.videoError || job?.state === "error" ? "form-error" : "field-note"}" role="status">${esc(status)}</p>`;
  }

  async function refreshEpisodeVideoJob() {
    const detail = state.episodeDetail, token = state.episodeToken;
    if (!detail || !$("episode-dialog").open || detail.videoBusy || detail.videoMutation) return;
    const request = ++detail.videoRequest;
    detail.videoBusy = true;
    try {
      const result = await api(`${episodePath(detail.project,detail.run,detail.id)}/video`);
      if (token !== state.episodeToken || request !== detail.videoRequest) return;
      const previous = detail.videoJob?.state;
      detail.videoJob = result.video_job || null;detail.videoError = null;detail.videoApiUnavailable = false;
      if (detail.videoJob?.state === "completed" && (previous !== "completed" || !(findEpisode(detail.run,detail.id) || detail.episode)?.video_url)) await loadDetail(false);
      if (token === state.episodeToken && request === detail.videoRequest) updateEpisodeVideo();
    } catch (error) {
      if (token === state.episodeToken && request === detail.videoRequest) {setEpisodeVideoError(detail,error);updateEpisodeVideo();}
    } finally {if (request === detail.videoRequest) detail.videoBusy = false;}
  }

  async function mutateEpisodeVideo(action) {
    const detail = state.episodeDetail, token = state.episodeToken;
    if (!detail || detail.videoMutation || ["queued","running"].includes(detail.videoJob?.state) || episodeVideoBlockedReason(detail)) return;
    const ep = findEpisode(detail.run,detail.id) || detail.episode;
    if (action === "generate" && (ep?.video_url || detail.videoJob?.state === "completed") || action === "delete" && !ep?.video_url) return;
    const request = ++detail.videoRequest;
    detail.videoMutation = action;detail.videoBusy = false;detail.videoError = null;updateEpisodeVideo();
    try {
      const result = await api(`${episodePath(detail.project,detail.run,detail.id)}/video${action === "delete" ? "/delete" : ""}`,{});
      if (token !== state.episodeToken || request !== detail.videoRequest) return;
      detail.videoJob = result.video_job || {state:"idle"};detail.videoApiUnavailable = false;
      if (action === "delete") {
        ep.video_url = null;ep.poster_url = null;
        toast("영상만 삭제했습니다. Episode 데이터는 유지됩니다.");
      } else toast("이 episode의 영상 생성을 요청했습니다.");
      if (action === "delete" || detail.videoJob?.state === "completed") await loadDetail(false);
    } catch (error) {
      if (token === state.episodeToken && request === detail.videoRequest) setEpisodeVideoError(detail,error);
    } finally {if (token === state.episodeToken && request === detail.videoRequest) {detail.videoMutation = null;updateEpisodeVideo();}}
  }

  function renderEvaluationLive() {
    state.evaluationCameraSignature = "";state.evaluationObservationKey = "";state.evaluationFrameToken += 1;state.evaluationDisplayedContext = null;
    return `<section id="evaluation-live-panel" class="panel evaluation-live-panel"><div class="panel-header"><div><h2>평가 rollout 화면</h2><p id="evaluation-live-subtitle">이 프로젝트의 BC 또는 RL 평가 관측을 표시합니다.</p></div><span id="evaluation-live-state" class="tag neutral">평가 대기</span></div><div id="evaluation-live-images"></div><dl class="evaluation-live-context"><div><dt>정책</dt><dd id="evaluation-live-policy">—</dd></div><div><dt>Rollout</dt><dd id="evaluation-live-episode">—</dd></div><div><dt>Task / Reset step</dt><dd id="evaluation-live-steps">—</dd></div><div><dt>현재 동작</dt><dd id="evaluation-live-stage">—</dd></div><div><dt>평가 Reset</dt><dd id="evaluation-live-reset">—</dd></div><div><dt>Run</dt><dd id="evaluation-live-run">—</dd></div></dl><p id="evaluation-live-note" class="evaluation-live-note"></p></section>`;
  }

  function evaluationLiveSource(allowInitializing = false) {
    const status = state.status, project = currentProject();
    if (state.route.tab !== "evaluation" || !state.connected || !status || !project || status.active_project_id !== project.id || !status.run_id || !status.observation?.id) return null;
    if (!["ready","thinking","executing","pausing","paused","completed","stopped","error"].includes(status.phase) && !(allowInitializing && status.phase === "initializing")) return null;
    const stage = status.learning?.pipeline_stage;
    if (!["bc_eval","rl_eval"].includes(stage)) return null;
    const pipeline = currentPipeline()?.status;
    if (pipeline?.job_id && status.learning?.job_id && pipeline.job_id !== status.learning.job_id) return null;
    if (pipelineActive(pipeline?.phase)) {
      if (pipeline.stage && pipeline.stage !== stage) return null;
      const started = [...(pipeline.events || [])].reverse().find(event => event.message === `${stage} 단계 시작`)?.time;
      if (typeof started === "number" && typeof status.episode?.started_at === "number" && status.episode.started_at < started) return null;
    }
    return status;
  }

  function updateEvaluationLive() {
    if (state.route.tab !== "evaluation" || !$("evaluation-live-images")) return;
    const status = evaluationLiveSource(), holder = $("evaluation-live-images");
    const preparing = !status && evaluationLiveSource(true), displayed = state.evaluationDisplayedContext;
    if (preparing?.phase === "initializing" && displayed
        && displayed.projectId === preparing.active_project_id && displayed.runId === preparing.run_id
        && displayed.stage === preparing.learning.pipeline_stage && displayed.jobId === preparing.learning.job_id
        && displayed.cameraSignature === state.evaluationCameraSignature) {
      // Keep the last image through the same run's reset; never reuse another run's frames.
      state.evaluationObservationKey = "";state.evaluationFrameToken += 1;
      $("evaluation-live-state").textContent = "다음 rollout 준비 중";$("evaluation-live-state").className = "tag neutral";
      $("evaluation-live-stage").textContent = "Reset 준비";
      $("evaluation-live-note").textContent = "환경을 준비하는 동안 마지막 평가 이미지를 유지합니다. 다음 관측이 도착하면 갱신됩니다.";
      return;
    }
    if (!status) {
      const key = state.connected ? "empty" : "offline";
      if (state.evaluationCameraSignature !== key) {
        holder.innerHTML = `<div class="evaluation-live-empty">${icon("camera")}<strong>${state.connected ? "표시할 평가 관측이 없습니다" : "서버 연결을 확인하고 있습니다"}</strong><p>이 프로젝트의 BC 또는 RL 평가를 실행하면 카메라 화면이 표시됩니다.</p></div>`;
        state.evaluationCameraSignature = key;state.evaluationObservationKey = "";state.evaluationFrameToken += 1;state.evaluationDisplayedContext = null;
      }
      $("evaluation-live-state").textContent = state.connected ? "평가 대기" : "연결 확인";$("evaluation-live-state").className = "tag neutral";
      $("evaluation-live-subtitle").textContent = "이 프로젝트의 BC 또는 RL 평가 관측을 표시합니다.";
      for (const field of ["policy","episode","steps","stage","reset","run"]) $("evaluation-live-"+field).textContent = "—";
      $("evaluation-live-note").textContent = "다른 프로젝트나 수집 run의 관측은 표시하지 않습니다.";
      return;
    }
    const observation = status.observation, episode = status.episode || {}, stage = status.learning.pipeline_stage;
    const policy = stage === "rl_eval" ? "BC + Residual" : "BC only";
    const finished = ["completed","stopped","error"].includes(status.phase);
    const stateLabel = status.phase === "paused" ? "평가 일시정지" : finished ? status.phase === "completed" ? "마지막 평가 관측" : "중단된 평가 관측" : "평가 진행 중";
    $("evaluation-live-state").textContent = stateLabel;$("evaluation-live-state").className = `tag ${status.phase === "paused" || ["stopped","error"].includes(status.phase) ? "warning" : finished ? "neutral" : "success"}`;
    const phaseLabel = status.phase === "thinking" && status.stage !== "reset" ? "정책 action 계산 중" : phaseNames[status.phase] || status.phase;
    $("evaluation-live-subtitle").textContent = `${currentProject().name} · ${policy} · ${phaseLabel}`;
    $("evaluation-live-policy").textContent = policy;
    $("evaluation-live-episode").textContent = `${episodeLabel({...episode,run_id:status.run_id,learning:status.learning})} · run ${count(episode.index)} / ${count(status.config?.episodes)}`;
    $("evaluation-live-steps").textContent = `${count(episode.steps)} / ${count(episode.reset_steps)}`;
    $("evaluation-live-stage").textContent = status.stage === "reset" ? "Reset" : "Task rollout";
    $("evaluation-live-reset").textContent = ["agent","environment","free_space"].includes(status.config?.reset_mode) ? resetModeLabel(status.config.reset_mode) : "기록 없음";
    $("evaluation-live-run").textContent = status.run_id;
    $("evaluation-live-note").textContent = `상태 확인 ${date(observation.timestamp,{second:"2-digit"})} · 카메라는 같은 평가 run의 최신 화면이며 상태 정보는 주기적으로 갱신됩니다.`;
    const cameras = (Array.isArray(observation.cameras) ? observation.cameras.map(item => typeof item === "string" ? item : item.name) : Object.keys(observation.cameras || {})).filter(name => typeof name === "string" && name && !/^reset_reference(?:_|$)/.test(name));
    const cameraSignature = JSON.stringify([status.active_project_id,status.run_id,stage,cameras]);
    if (cameraSignature !== state.evaluationCameraSignature) {
      holder.innerHTML = cameras.length ? `<div class="camera-grid ${cameras.length === 1 ? "single" : ""}">${cameras.map((camera,index) => `<div class="camera"><span class="camera-label">${esc(camera)}</span><div id="evaluation-frame-${index}" class="evaluation-frame-slot"><span>관측 수신 대기</span></div></div>`).join("")}</div>` : '<div class="evaluation-live-empty"><p>평가 카메라를 기다리고 있습니다.</p></div>';
      state.evaluationCameraSignature = cameraSignature;state.evaluationObservationKey = "";
    }
    const observationKey = `${cameraSignature}|${observation.id}`;
    if (observationKey === state.evaluationObservationKey) return;
    state.evaluationObservationKey = observationKey;
    const token = ++state.evaluationFrameToken;
    cameras.forEach((camera,index) => {
      const frame = new Image();
      frame.alt = `${camera} ${policy} 평가 관측`;
      frame.onload = () => {
        const current = evaluationLiveSource();
        if (token !== state.evaluationFrameToken || !current || current.run_id !== status.run_id || current.active_project_id !== status.active_project_id || current.learning.pipeline_stage !== stage) return;
        const slot = $("evaluation-frame-"+index);
        if (slot) {
          slot.replaceChildren(frame);
          state.evaluationDisplayedContext = {projectId:status.active_project_id,runId:status.run_id,
            stage,jobId:status.learning.job_id,cameraSignature};
        }
      };
      frame.onerror = () => {if (token === state.evaluationFrameToken) state.evaluationObservationKey = "";};
      frame.src = `/api/frame/${encodeURIComponent(camera)}?project_id=${encodeURIComponent(status.active_project_id)}&run_id=${encodeURIComponent(status.run_id)}&t=${encodeURIComponent(observation.id)}`;
    });
  }

  function evaluationResetField(mode = "inherit") {
    const options = [["inherit","수집 설정 따르기"],["agent","GPT arm reset"],["environment","환경 reset"],["free_space","복구·후퇴 후 환경 초기화"]];
    return `<label class="field">평가 Reset 방식<select name="evaluation_reset_mode">${options.map(([value,label]) => `<option value="${value}" ${value === mode ? "selected" : ""}>${label}</option>`).join("")}</select></label>`;
  }

  function evaluationResetProtocol(item) {
    const mode = item?.reset_mode ?? item?.provenance?.evaluation?.run_config?.reset_mode;
    return ["agent","environment","free_space"].includes(mode) ? resetModeLabel(mode) : "Reset 방식 기록 없음";
  }

  function renderModels(project) { renderLearningTab(project,false); }
  function renderEvaluation(project) { renderLearningTab(project,true); }

  function renderLearningTab(project, evaluation) {
    if (remoteRobotProject(project)) {$("tab-content").innerHTML = remoteConnectionOnlyPanel(project,evaluation ? "정책 평가" : "모델 학습");return;}
    state.modelSignature = "";state.evaluationSignature = "";state.pipelineExecutionSignature = "";
    const heading = evaluation ? "정책 평가" : "모델 학습";
    const description = evaluation ? "완료된 평가 결과를 비교하고, 정책별 rollout을 실행하세요." : "BC와 RL의 학습 설정을 확인하고 각 단계의 지표와 기록을 살펴보세요.";
    const stageSummary = `<section class="panel learning-stage-summary"><div class="panel-header"><div><h2>학습 진행 상황</h2><p>BC와 RL의 학습 상태를 확인하세요.</p></div><span id="pipeline-phase-tag"></span></div><div id="pipeline-stages" class="pipeline-stage-list learning-stage-list"></div><div id="training-pipeline-controls"></div></section>`;
    const registry = `<section class="panel model-registry-panel"><div class="panel-header"><div><h2>모델 레지스트리</h2><p>정책과 RL 보조 모델의 체크포인트·학습 데이터·설정</p></div><span id="model-count" class="tag neutral">0</span></div><div id="pipeline-models"></div></section>`;
    const training = `${stageSummary}<section class="panel training-execution-panel"><div class="panel-header"><div><h2>BC 학습</h2><p>성공 task 데이터로 기본 정책을 학습합니다.</p></div></div><div id="training-bc-execution"></div></section><div id="training-bc-monitor"></div><section class="panel training-execution-panel"><div class="panel-header"><div><h2>RL 학습</h2><p>BC 정책을 고정하고 Residual 정책을 학습합니다.</p></div></div><div id="training-rl-execution"></div></section><div id="training-rl-monitor"></div>${registry}`;
    const evaluations = `<section class="panel evaluation-panel"><div class="panel-header"><div><h2>평가 결과와 기록</h2><p>완료된 BC-only / BC+Residual 평가의 성공률과 실행 조건</p></div></div><div id="pipeline-evaluations"></div></section><section class="panel pipeline-execution-panel"><div class="panel-header"><div><h2>개별 평가 실행</h2><p>실행 전에 episode 수·목표 성공률·Reset 방식을 설정합니다.</p></div><span id="pipeline-phase-tag"></span></div><div id="pipeline-execution"></div></section>${evaluation ? renderEvaluationLive() : ""}<div id="pipeline-monitor"></div>`;
    $("tab-content").innerHTML = `<div class="learning-heading"><div><h2>${heading}</h2><p>${description}</p></div><span id="learning-connection" class="tag neutral">상태 확인 중</span></div><div id="pipeline-unavailable" hidden></div><div id="pipeline-workspace">${evaluation ? evaluations : training}</div>`;
    updatePipelineView();updateEvaluationLive();refreshPipeline(true);
  }

  function pipelineReadiness(stage) {
    if (remoteRobotProject()) return {ready:false,reason:remoteConnectionOnlyReason};
    const pipeline = currentPipeline();
    const item = pipeline?.readiness?.[stage];
    if (typeof item === "boolean") return {ready:item,reason:""};
    return item || {ready:Boolean(pipeline),reason:"실행 조건을 확인하고 있습니다."};
  }

  function updatePipelineView(forceConfig = false) {
    const pipeline = currentPipeline(), project = currentProject();
    if (!project) return;
    if (remoteRobotProject(project)) {if (state.route.tab === "collection") updateRemoteRobotConnection();return;}
    if (state.route.tab === "collection") {updateCollectionSettings();return;}
    if (state.route.tab === "overview") {
      const panel = document.querySelector(".pipeline-panel");
      if (panel) panel.outerHTML = renderPipeline(project);
      return;
    }
    if (!["models","evaluation"].includes(state.route.tab) || !$("pipeline-workspace")) return;
    const unavailable = pipeline?.unavailable === true || (!pipelineEnabled() && !pipeline);
    $("pipeline-unavailable").hidden = !unavailable;$("pipeline-workspace").hidden = unavailable;
    $("learning-connection").textContent = unavailable ? "학습 worker 연결 전" : pipeline ? "파이프라인 연결됨" : "상태 확인 중";
    $("learning-connection").className = `tag ${pipeline && !unavailable ? "success" : "neutral"}`;
    if (unavailable) {
      $("pipeline-unavailable").innerHTML = `<section class="panel"><div class="empty-models"><div class="empty-model-icon">${icon("model")}</div><h2>학습 worker 연결을 기다리고 있습니다</h2><p>BC·RL 학습과 평가는 서버 연결 후 사용할 수 있습니다.</p><button class="button secondary" data-action="pipeline-refresh">${icon("reset")}상태 다시 확인</button></div></section>`;
      return;
    }
    if (!pipeline) {
      const target = $("pipeline-execution") || $("training-bc-execution");
      if (target) target.innerHTML = '<p class="learning-empty">파이프라인 상태와 실행 조건을 불러오고 있습니다.</p>';
      return;
    }
    const status = pipeline.status || {}, phase = status.phase || "idle";
    $("pipeline-phase-tag").innerHTML = tag(pipelinePhaseNames[phase] || phase,phase === "running" ? "success" : ["error","interrupted","paused"].includes(phase) ? "warning" : "neutral");
    updatePipelineExecution();updatePipelineStages();renderPipelineMonitor();updateStageStartDialog();
    if (state.route.tab === "evaluation") {
      updateEvaluationLive();
      const signature = JSON.stringify(pipeline.evaluations || []);
      if (signature !== state.evaluationSignature) {state.evaluationSignature = signature;renderEvaluations();}
    } else {
      const signature = JSON.stringify(pipeline.models || []);
      if (signature !== state.modelSignature) {state.modelSignature = signature;renderModelRegistry();}
    }
  }

  function trainingAlgorithmOptions(stage) {
    const defaults = stage === "bc" ? [{id:"dp",label:"Diffusion Policy (DP)",available:true},{id:"pi0",label:"π0",available:false},{id:"pi0_5",label:"π0.5",available:false}] : [{id:"residual_td3",label:"Residual TD3",available:true}];
    const catalog = currentPipeline()?.algorithm_catalog?.[stage] || [];
    return defaults.map(item => ({...item,...catalog.find(option => option.id === item.id),available:item.available && catalog.find(option => option.id === item.id)?.available !== false}));
  }

  function selectedTrainingAlgorithm(stage) {
    const configured = currentPipeline()?.config?.[stage === "bc" ? "bc_algorithm" : "rl_algorithm"];
    return state.trainingAlgorithms[stage] ?? configured ?? (stage === "bc" ? "dp" : "residual_td3");
  }

  function trainingAlgorithmAvailable(stage, algorithm) {
    return algorithm === (stage === "bc" ? "dp" : "residual_td3") && trainingAlgorithmOptions(stage).some(item => item.id === algorithm && item.available);
  }

  function updatePipelineExecution() {
    if (state.route.tab === "overview") {updatePipelineView();return;}
    if (!["models","evaluation"].includes(state.route.tab)) return;
    const pipeline = currentPipeline(), status = pipeline?.status || {};
    if (!pipeline) return;
    const busy = pipelineActive(status.phase), global = pipelineRuntime();
    const other = global && pipelineActive(global.phase) && global.project_id !== state.route.id;
    const executionSignature = JSON.stringify([status,pipeline.readiness,pipeline.config,pipeline.algorithm_catalog,pipeline.rl_preparation,state.route.tab,state.evaluationStage,state.trainingAlgorithms,global,state.pipelinePending,state.pipelinePausePending,state.connected]);
    if (executionSignature === state.pipelineExecutionSignature) return;
    state.pipelineExecutionSignature = executionSignature;
    if (state.route.tab === "models") {
      for (const stage of ["bc","rl"]) {
        const target = $(`training-${stage}-execution`);if (!target) continue;
        const ready = pipelineReadiness(stage), algorithm = selectedTrainingAlgorithm(stage), options = trainingAlgorithmOptions(stage);
        const supported = trainingAlgorithmAvailable(stage,algorithm);
        const disabled = !state.connected || state.pipelinePending || state.pipelinePausePending || busy || other || !ready.ready || !supported;
        const reason = other ? "다른 프로젝트의 파이프라인이 실행 중입니다." : !supported ? "선택한 알고리즘은 아직 구현되지 않았습니다. 지원되는 알고리즘을 선택해주세요." : !ready.ready ? ready.reason || "실행 조건을 확인해주세요." : "";
        const preparing = stage === "rl" && status.stage === "progress" && busy;
        target.innerHTML = `<div class="pipeline-run-choice"><div class="pipeline-run-copy"><h3>${stage === "bc" ? "기본 정책 학습" : preparing ? "RL 준비 · Progress 보조 모델 학습 중" : "Residual 정책 학습"}</h3><p>${stage === "bc" ? "알고리즘을 선택한 뒤 학습 설정을 확인하고 실행하세요." : "필요한 Progress 보조 모델은 자동으로 준비합니다. 준비 후 RL 학습을 진행합니다."}</p><label class="field training-algorithm-field">알고리즘<select id="training-${stage}-algorithm" data-training-algorithm="${stage}" ${busy || state.pipelinePending ? "disabled" : ""}>${options.some(item => item.id === algorithm) ? "" : `<option value="${esc(algorithm)}" selected disabled>${esc(algorithm)} · 지원하지 않음</option>`}${options.map(item => `<option value="${esc(item.id)}" ${item.id === algorithm ? "selected" : ""} ${!item.available ? "disabled" : ""}>${esc(item.label)}${item.available ? "" : " · 미구현"}</option>`).join("")}</select></label></div><button id="training-${stage}-start" class="button primary" data-action="pipeline-start-manual" data-stage="${stage}" ${disabled ? "disabled" : ""}>${icon("play")}${stage.toUpperCase()} 학습 실행</button></div>${reason ? `<p class="pipeline-readiness" role="status">${icon("info")}${esc(reason)}</p>` : ""}`;
      }
      if ($("training-pipeline-controls")) $("training-pipeline-controls").innerHTML = pipelineControlStrip(status);
      return;
    }
    if (!$("pipeline-execution")) return;
    const stage = state.evaluationStage, ready = pipelineReadiness(stage);
    const disabled = !state.connected || state.pipelinePending || state.pipelinePausePending || busy || other || !ready.ready;
    const reason = other ? "다른 프로젝트의 파이프라인이 실행 중입니다." : !ready.ready ? ready.reason || "이 평가의 실행 조건을 아직 충족하지 않았습니다." : "";
    $("pipeline-execution").innerHTML = `<div class="pipeline-run-choice"><div class="pipeline-run-copy"><h3>평가할 정책 선택</h3><p>설정 팝업에서 평가 조건을 확인한 뒤 선택한 정책만 실행합니다.</p><label class="field manual-stage-field">평가 대상<select id="pipeline-manual-stage" ${busy || state.pipelinePending ? "disabled" : ""}>${manualPipelineStages().map(item => `<option value="${item.id}" ${item.id === stage ? "selected" : ""}>${esc(item.name)}</option>`).join("")}</select></label></div><button id="pipeline-start" class="button primary" data-action="pipeline-start-manual" data-stage="${stage}" ${disabled ? "disabled" : ""}>${icon("play")}평가 실행</button></div>${reason ? `<p class="pipeline-readiness" role="status">${icon("info")}${esc(reason)}</p>` : ""}${pipelineControlStrip(status)}`;
  }

  function updatePipelineStages() {
    const pipeline = currentPipeline();
    if (state.route.tab !== "models" || !pipeline || !$("pipeline-stages")) return;
    const status = pipeline.status || {}, names = {bc:"BC 학습",rl:"RL 학습"};
    $("pipeline-stages").innerHTML = pipelineStages.filter(stage => ["bc","rl"].includes(stage.id)).map((stage,index) => {
      const preparing = stage.id === "rl" && status.stage === "progress";
      const active = (status.stage === stage.id || preparing) && pipelineActive(status.phase), ready = pipelineReadiness(stage.id);
      const completion = pipeline.completion?.stages?.[stage.id]?.complete;
      const label = active ? preparing ? "RL 준비" : pipelinePhaseNames[status.phase] : completion ? "완료" : ready.ready ? "실행 가능" : "조건 대기";
      return `<div class="pipeline-stage ${active ? "active" : ""}"><span class="stage-number">${String(index+1).padStart(2,"0")}</span><div class="stage-description"><h3>${names[stage.id]}</h3></div>${tag(label,active ? status.phase === "paused" ? "warning" : "success" : completion ? "success" : "neutral")}</div>`;
    }).join("");
  }

  function stageMonitorData(stage) {
    const status = currentPipeline()?.status || {}, jobId = status.job_id;
    if (!jobId) return {metrics:{},events:[],current:false,jobId:null};
    const started = n(status.started_at), events = (status.events || []).filter(event => event.job_id === jobId && event.stage === stage && (!started || n(event.time || event.timestamp) >= started));
    const current = status.stage === stage && status.phase !== "idle";
    // Historical unscoped metrics and events cannot be assigned to a model safely.
    const metrics = current ? status.metrics || {} : events.length ? status.stage_metrics?.[stage] || {} : {};
    return {metrics,events,current,jobId};
  }

  function renderStageMonitor(stage, targetId) {
    const target = $(targetId);if (!target) return;
    const pipeline = currentPipeline(), status = pipeline?.status || {}, data = stageMonitorData(stage), metrics = data.metrics;
    const modelKind = stage === "bc" ? "bc" : stage === "rl" ? "residual" : null;
    const recentModel = modelKind ? (pipeline?.models || []).filter(model => model.kind === modelKind && model.status === "ready" && model.metrics?.partial !== true).sort((a,b) => n(b.created_at)-n(a.created_at))[0] : null;
    const recentMetrics = recentModel?.metrics || {};
    const matchingModel = recentModel && pipeline.completion?.stages?.[stage]?.complete && pipeline.completion.stages[stage].model_id === recentModel.id;
    const recentValues = stage === "bc" ? [["학습 steps",recentMetrics.steps_completed ?? recentMetrics.step],["Best validation loss",recentMetrics.best_validation_loss ?? recentMetrics.validation_loss]] : [["Critic updates",recentMetrics.critic_updates],["Actor updates",recentMetrics.actor_updates],["Online steps",recentMetrics.online_steps],["Critic loss",recentMetrics.critic_loss]];
    const recentHTML = recentModel ? `<section class="recent-training-record"><div><h3>최근 완료 모델</h3>${tag(matchingModel ? "현재 설정과 일치" : "저장된 학습 기록",matchingModel ? "success" : "neutral")}</div><p>${esc(date(recentModel.created_at))} · ${esc(recentModel.id)}</p><dl>${recentValues.map(([name,value]) => `<div><dt>${name}</dt><dd>${finiteValue(value,5)}</dd></div>`).join("")}</dl><small>저장된 모델의 완료 기록입니다. 위 지표는 현재 작업의 해당 단계만 표시합니다.</small>${stage === "bc" ? `<button type="button" class="button secondary small" data-action="bc-history" data-metric="train_loss" data-model="${esc(recentModel.id)}">${icon("progress")}이 모델의 학습 기록</button>` : ""}</section>` : "";
    const isBC = stage === "bc", isRL = stage === "rl", title = isBC ? "BC 학습 지표·로그" : isRL ? "RL 학습 지표·로그" : `${stage === "bc_eval" ? "BC" : "RL"} 평가 실행 지표·로그`;
    const value = key => finiteValue(metrics[key],5), phase = data.current ? pipelinePhaseNames[status.phase] || status.phase : Object.keys(metrics).length ? "현재 작업의 기록" : "기록 대기";
    const cards = isBC ? [["학습 step",finiteValue(metrics.step,0),"BC optimizer updates"],["Train loss",value("train_loss"),"BC 학습 loss"],["Validation loss",value("validation_loss"),"BC 검증 loss"]] : isRL ? [["Critic loss",value("critic_loss"),"Residual TD3 critic"],["Actor loss",value("actor_loss"),"Warmup·지연 update 중에는 값이 없을 수 있습니다"],["Online steps",finiteValue(metrics.online_steps,0),"누적 online transition 수"],["Critic updates",finiteValue(metrics.critic_updates,0),"누적 critic update 수"]] : [["Episode return",value("episode_return"),"해당 평가 rollout reward 합계"],["Environment steps",finiteValue(metrics.environment_steps,0),"해당 평가에서 보고한 값"]];
    const prep = isRL ? stageMonitorData("progress") : null;
    const preparing = isRL && status.stage === "progress" && pipelineActive(status.phase);
    const prepHTML = prep && (preparing || prep.events.length) ? `<section class="rl-preparation-monitor"><div><h3>RL 준비 · Progress 보조 모델</h3>${tag(preparing ? pipelinePhaseNames[status.phase] || "준비 중" : "현재 작업의 준비 기록",preparing ? "warning" : "neutral")}</div><dl><div><dt>학습 step</dt><dd>${finiteValue(prep.metrics.step,0)}</dd></div><div><dt>Progress loss</dt><dd>${finiteValue(prep.metrics.progress_loss,5)}</dd></div><div><dt>Validation MAE</dt><dd>${finiteValue(prep.metrics.progress_validation_mae,5)}</dd></div></dl>${prep.events.length ? `<ol class="learning-event-list">${prep.events.slice(-5).reverse().map(event => `<li><time>${esc(date(event.time || event.timestamp))}</time><span>${esc(event.message || event.kind || "준비 상태 갱신")}</span></li>`).join("")}</ol>` : ""}</section>` : "";
    const scalarMetrics = Object.entries(metrics).filter(([,value]) => ["number","string","boolean"].includes(typeof value));
    const expanded = target.querySelector(".learning-metric-details")?.open;
    target.innerHTML = `<section class="panel learning-monitor-panel stage-monitor-panel"><div class="panel-header"><div><h2>${title}</h2><p>${data.jobId ? `작업 ${esc(data.jobId)}` : "실제 실행 이후 지표가 표시됩니다"}</p></div>${tag(phase,data.current && status.phase === "running" ? "success" : "neutral")}</div>${prepHTML}<section class="metrics learning-metrics stage-learning-metrics" aria-label="${title}">${cards.map(([label,value,foot],index) => isBC ? `<button type="button" class="metric bc-metric-button" data-action="bc-history" data-metric="${["step","train_loss","validation_loss"][index]}" aria-haspopup="dialog" aria-label="${esc(label)} 학습 기록 열기"><span class="metric-label">${esc(label)}${icon("progress")}</span><span class="metric-value">${value}</span><span class="metric-foot">${esc(foot)} · 기록 보기 ${icon("arrow")}</span></button>` : metric(label,value,foot,"progress")).join("")}</section>${data.events.length ? `<ol class="learning-event-list">${data.events.slice(-10).reverse().map(event => `<li><time>${esc(date(event.time || event.timestamp))}</time><span>${esc(event.message || event.kind || "실행 상태 갱신")}</span></li>`).join("")}</ol>` : '<p class="learning-empty">현재 작업에서 이 단계의 기록이 없습니다.</p>'}${recentHTML}${scalarMetrics.length ? `<details class="learning-metric-details" ${expanded ? "open" : ""}><summary>이 단계의 전체 지표</summary><dl>${scalarMetrics.map(([key,value]) => `<div><dt>${esc(key)}</dt><dd>${typeof value === "number" ? finiteValue(value,6) : esc(value)}</dd></div>`).join("")}</dl></details>` : ""}</section>`;
  }

  function renderPipelineMonitor() {
    if (state.route.tab === "models") {renderStageMonitor("bc","training-bc-monitor");renderStageMonitor("rl","training-rl-monitor");return;}
    if (state.route.tab === "evaluation") {
      const actual = currentPipeline()?.status?.stage;
      renderStageMonitor(["bc_eval","rl_eval"].includes(actual) ? actual : state.evaluationStage,"pipeline-monitor");
    }
  }

  function bcHistoryIsCurrent(history = state.bcHistory) {
    return Boolean(history && state.bcHistory === history && $("bc-history-dialog")?.open && history.projectId === state.route.id && history.routeToken === state.routeToken && state.route.tab === "models");
  }

  function closeBCHistory() {
    const history = state.bcHistory;
    if (history) {clearTimeout(history.timer);history.controller?.abort();}
    state.bcHistory = null;
  }

  function openBCHistory(metric = "train_loss", modelId = null) {
    if (state.route.tab !== "models" || !state.route.id) return;
    closeBCHistory();
    state.bcHistory = {projectId:state.route.id,routeToken:state.routeToken,modelId:modelId || currentPipeline()?.bc_training?.model_id || null,jobId:null,metric:["step","train_loss","validation_loss"].includes(metric) ? metric : "train_loss",data:null,points:[],selectedIndex:0,followLatest:true,busy:false,timer:null,controller:null};
    $("bc-history-title").textContent = "BC 학습 기록";$("bc-history-error").hidden = true;
    $("bc-history-identity").textContent = "학습 기록을 불러오고 있습니다.";
    $("bc-history-content").innerHTML = '<p class="learning-empty">기록된 학습 지표와 체크포인트를 확인하고 있습니다.</p>';
    if (!$("bc-history-dialog").open) $("bc-history-dialog").showModal();
    refreshBCHistory();
  }

  async function refreshBCHistory() {
    const history = state.bcHistory;
    if (!bcHistoryIsCurrent(history) || history.busy) return;
    clearTimeout(history.timer);history.busy = true;history.controller = new AbortController();
    try {
      const suffix = history.modelId ? `?model_id=${encodeURIComponent(history.modelId)}` : "";
      const result = await api(`/api/projects/${encodeURIComponent(history.projectId)}/learning/bc/history${suffix}`,undefined,15000,history.controller.signal);
      if (!bcHistoryIsCurrent(history)) return;
      if (result.project_id && result.project_id !== history.projectId || history.modelId && result.model_id !== history.modelId || history.jobId && result.job_id && result.job_id !== history.jobId) throw new Error("다른 학습의 응답이 도착해 현재 그래프를 유지했습니다. 팝업을 다시 열어 확인해주세요.");
      if (result.model_id) history.modelId = result.model_id;
      if (result.job_id) history.jobId = result.job_id;
      history.data = result;$("bc-history-error").hidden = true;renderBCHistory();
    } catch (error) {
      if (bcHistoryIsCurrent(history)) {
        $("bc-history-error").textContent = error.message;$("bc-history-error").hidden = false;
        if (!history.data) $("bc-history-content").innerHTML = '<p class="learning-empty">기록을 불러오지 못했습니다. 연결이 돌아오면 자동으로 다시 확인합니다.</p>';
      }
    } finally {
      history.busy = false;history.controller = null;
      if (bcHistoryIsCurrent(history)) history.timer = setTimeout(refreshBCHistory,2500);
    }
  }

  function bcHistoryPoints(data) {
    return (Array.isArray(data?.points) ? data.points : []).filter(point => Number.isFinite(point.step) && point.step >= 0).map(point => ({step:point.step,time:Number.isFinite(point.time) ? point.time : null,train_loss:Number.isFinite(point.train_loss) ? point.train_loss : null,validation_loss:Number.isFinite(point.validation_loss) ? point.validation_loss : null})).sort((a,b) => a.step-b.step || (a.time ?? 0)-(b.time ?? 0));
  }

  function latestBCHistoryIndex(points, metric) {
    for (let index=points.length-1;index>=0;index--) if (metric === "step" || Number.isFinite(points[index][metric])) return index;
    return Math.max(0,points.length-1);
  }

  function bcHistoryChart(history) {
    const points = history.points, timeView = history.metric === "step";
    const finitePoints = timeView ? points.filter(point => point.time !== null) : points.filter(point => point.train_loss !== null || point.validation_loss !== null);
    history.chart = null;
    if (!finitePoints.length) return `<p class="bc-history-empty">${timeView ? "시간이 기록된 학습 step이 없습니다." : "아직 기록된 loss가 없습니다."} 기록되지 않은 값은 추정해 표시하지 않습니다.</p>`;
    const width = 760, height = 320, left = 65, right = 22, top = 22, bottom = 47, plotWidth = width-left-right, plotHeight = height-top-bottom;
    const firstTime = finitePoints.reduce((earliest,point) => Math.min(earliest,point.time ?? Infinity),Infinity);
    const xValue = point => timeView ? point.time-firstTime : point.step;
    const xMax = finitePoints.reduce((maximum,point) => Math.max(maximum,xValue(point)),1);
    const yValues = timeView ? finitePoints.map(point => point.step) : finitePoints.flatMap(point => [point.train_loss,point.validation_loss].filter(value => value !== null));
    let yMin = timeView ? 0 : yValues.reduce((minimum,value) => Math.min(minimum,value),Infinity), yMax = yValues.reduce((maximum,value) => Math.max(maximum,value),-Infinity);
    const padding = Math.max((yMax-yMin)*.08,Math.abs(yMax)*.03,.000001);
    if (!timeView) yMin -= padding;
    yMax += padding;if (yMax === yMin) yMax = yMin+1;
    const x = value => left+value/xMax*plotWidth, y = value => top+(yMax-value)/(yMax-yMin)*plotHeight;
    history.chart = {width,height,left,top,plotWidth,plotHeight,xMax,yMin,yMax,firstTime,timeView};
    const grid = Array.from({length:5},(_,index) => {
      const fraction = index/4, yy = top+plotHeight*(1-fraction), xx = left+plotWidth*fraction;
      return `<line x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}" class="bc-chart-grid"/><text x="${left-10}" y="${yy+4}" text-anchor="end">${esc(finiteValue(yMin+(yMax-yMin)*fraction,timeView ? 0 : 4))}</text><text x="${xx}" y="${height-24}" text-anchor="middle">${esc(finiteValue(xMax*fraction/(timeView && xMax>=120 ? 60 : 1),timeView ? 1 : 0))}</text>`;
    }).join("");
    const series = timeView ? [{key:"step",color:"step",line:true}] : [{key:"train_loss",color:"train",line:true},{key:"validation_loss",color:"validation",line:false}];
    const lines = series.map(item => {
      const values = finitePoints.filter(point => Number.isFinite(point[item.key]));
      const path = item.line && values.length>1 ? `<path class="bc-chart-line ${item.color}" d="${values.map((point,index) => `${index ? "L" : "M"}${x(xValue(point)).toFixed(2)},${y(point[item.key]).toFixed(2)}`).join(" ")}"/>` : "";
      const dots = !item.line || values.length<=100 ? values.map(point => `<circle class="bc-chart-dot ${item.color}" cx="${x(xValue(point)).toFixed(2)}" cy="${y(point[item.key]).toFixed(2)}" r="${item.line ? 2.7 : 3.5}"><title>Step ${esc(point.step)} · ${item.key}: ${esc(finiteValue(point[item.key],6))}</title></circle>`).join("") : "";
      return `<g class="bc-chart-series ${history.metric === item.key ? "selected" : ""}">${path}${dots}</g>`;
    }).join("");
    const observedStep = finitePoints.reduce((maximum,point) => Math.max(maximum,point.step),0);
    const markers = (Array.isArray(history.data.checkpoints) ? history.data.checkpoints : []).filter(checkpoint => Number.isFinite(checkpoint.step) && checkpoint.step>=0 && checkpoint.step<=observedStep).map(checkpoint => timeView ? `<line x1="${left}" x2="${width-right}" y1="${y(checkpoint.step)}" y2="${y(checkpoint.step)}" class="bc-chart-checkpoint"><title>${esc(checkpoint.file || "Checkpoint")} · Step ${esc(checkpoint.step)}</title></line>` : `<line x1="${x(checkpoint.step)}" x2="${x(checkpoint.step)}" y1="${top}" y2="${height-bottom}" class="bc-chart-checkpoint"><title>${esc(checkpoint.file || "Checkpoint")} · Step ${esc(checkpoint.step)}</title></line>`).join("");
    return `<div class="bc-chart-legend">${timeView ? '<span class="step">학습 step</span>' : '<span class="train">Train loss</span><span class="validation">Validation loss · 실제 관측점</span>'}</div><div class="bc-chart-wrap"><svg id="bc-history-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${timeView ? "경과 시간별 학습 step" : "학습 step별 train loss와 실제 validation loss 관측점"}">${grid}${markers}${lines}<line id="bc-history-cursor" class="bc-chart-cursor" x1="${left}" x2="${left}" y1="${top}" y2="${height-bottom}" visibility="hidden"/><circle id="bc-history-highlight" class="bc-chart-highlight" cx="${left}" cy="${top}" r="5" visibility="hidden"/><text x="${left+plotWidth/2}" y="${height-4}" text-anchor="middle">${timeView ? xMax>=120 ? "첫 표시 기록 이후 (분)" : "첫 표시 기록 이후 (초)" : "학습 step"}</text></svg></div><p class="bc-chart-caption">${finitePoints.length===1 ? "관측점이 1개라 추세선은 표시하지 않습니다. " : ""}${timeView ? "첫 표시 기록을 시간 원점으로 삼아 실제 시각과 학습 step을 표시합니다." : "Train loss는 기록된 점을 연결하며 Validation loss는 실제 관측점만 표시합니다."} 회색 점선은 실제 체크포인트의 저장 step입니다.</p>`;
  }

  function renderBCHistory() {
    const history = state.bcHistory;if (!bcHistoryIsCurrent(history) || !history.data) return;
    const data = history.data, previous = history.points[history.selectedIndex], focused = document.activeElement?.id;
    history.points = bcHistoryPoints(data);const points = history.points;
    let selected = history.followLatest ? latestBCHistoryIndex(points,history.metric) : points.findIndex(point => point.step === previous?.step && point.time === previous?.time);
    if (selected<0) selected = latestBCHistoryIndex(points,history.metric);
    history.selectedIndex = selected;
    const latest = points.at(-1), lastTrain = points[latestBCHistoryIndex(points,"train_loss")], lastValidation = points[latestBCHistoryIndex(points,"validation_loss")];
    const target = Number.isFinite(data.target_steps) && data.target_steps>0 ? data.target_steps : null;
    const progress = latest && target ? Math.min(100,latest.step/target*100) : null;
    const phase = pipelinePhaseNames[data.phase] || data.phase || "상태 기록 없음";
    $("bc-history-identity").textContent = history.modelId ? `${history.modelId}${history.jobId ? ` · 작업 ${history.jobId}` : ""} · ${phase}` : "아직 BC 학습 기록이 없습니다.";
    const summary = `<div class="bc-history-summary"><div><span>학습 step / 목표</span><strong>${latest ? count(latest.step) : "—"} / ${target ? count(target) : "—"}</strong><small>${progress === null ? "기록된 step 기준" : `${finiteValue(progress,1)}% · 기록된 step 기준`}</small></div><div><span>최근 Train loss</span><strong>${finiteValue(lastTrain?.train_loss,5)}</strong><small>${Number.isFinite(lastTrain?.train_loss) ? `Step ${count(lastTrain.step)}` : "관측 기록 없음"}</small></div><div><span>최근 Validation loss</span><strong>${finiteValue(lastValidation?.validation_loss,5)}</strong><small>${Number.isFinite(lastValidation?.validation_loss) ? `Step ${count(lastValidation.step)}` : "관측 기록 없음"}</small></div></div>`;
    const tabs = `<div class="bc-history-views" role="group" aria-label="학습 그래프 선택">${[["step","학습 step"],["train_loss","Train loss"],["validation_loss","Validation loss"]].map(([key,label]) => `<button id="bc-history-view-${key}" type="button" data-action="bc-history-view" data-metric="${key}" aria-pressed="${history.metric === key}" class="${history.metric === key ? "active" : ""}">${label}</button>`).join("")}</div>`;
    const reasons = {no_training:"아직 BC 학습을 실행한 기록이 없습니다.",metrics_not_found:"이 학습의 지표 로그 파일을 찾을 수 없습니다. 저장된 최종 모델 지표만으로 과거 그래프를 만들지는 않습니다.",no_metrics:"학습은 기록되었지만 아직 지표가 저장되지 않았습니다."};
    const chart = points.length ? `${bcHistoryChart(history)}<div class="bc-history-inspector"><div id="bc-history-point" class="bc-history-point" aria-live="polite"></div><button type="button" class="button secondary small" data-action="bc-history-latest">최신 지점</button></div>` : `<p class="bc-history-empty">${esc(reasons[data.reason] || data.reason || "아직 기록된 학습 지표가 없습니다.")}</p>`;
    const checkpoints = Array.isArray(data.checkpoints) ? data.checkpoints : [], kinds = {periodic:"주기 저장",final:"학습 종료 저장",best:"검증 최상 정책"};
    const checkpointHTML = `<section class="bc-history-checkpoints"><h3>실제 저장된 체크포인트 <span class="tag neutral">${count(checkpoints.length)}</span></h3>${checkpoints.length ? `<div class="table-wrap"><table><thead><tr><th>Step</th><th>종류</th><th>파일</th><th>저장 시각</th></tr></thead><tbody>${checkpoints.map(checkpoint => `<tr><td>${Number.isFinite(checkpoint.step) ? count(checkpoint.step) : "—"}</td><td>${esc(kinds[checkpoint.kind] || checkpoint.kind || "기록 참고")}</td><td><code>${esc(checkpoint.file || "—")}</code></td><td>${checkpoint.created_at ? esc(date(checkpoint.created_at)) : "—"}</td></tr>`).join("")}</tbody></table></div>` : '<p class="field-note">이 학습에 저장된 체크포인트 기록이 없습니다.</p>'}<p class="field-note">주기·종료 체크포인트의 학습 상태와 검증 최상 policy는 구분됩니다. 저장된 파일만 표시하며 별도의 완료 모델로 계산하지 않습니다.</p></section>`;
    $("bc-history-content").innerHTML = `${summary}${tabs}${chart}<p class="bc-history-footnote">${data.sampled ? `읽은 ${count(data.total_points)}개 중 ${count(points.length)}개 관측점을 표시합니다. ` : `${count(points.length)}개 관측점. `}${data.truncated ? "로그 크기 제한으로 일부 이전 기록은 포함되지 않았습니다. " : ""}팝업이 열린 동안 이 학습 기록만 자동 갱신합니다.</p>${checkpointHTML}`;
    updateBCHistoryPoint(selected,history.followLatest);
    if (focused?.startsWith("bc-history-")) $(focused)?.focus({preventScroll:true});
  }

  function updateBCHistoryPoint(index, followLatest = false) {
    const history = state.bcHistory;if (!bcHistoryIsCurrent(history) || !history.points.length) return;
    history.selectedIndex = Math.max(0,Math.min(history.points.length-1,index));history.followLatest = followLatest;
    const point = history.points[history.selectedIndex], chart = history.chart;
    if ($("bc-history-point")) $("bc-history-point").innerHTML = `<strong>Step ${count(point.step)}</strong><span>Train loss ${finiteValue(point.train_loss,6)}</span><span>Validation loss ${finiteValue(point.validation_loss,6)}</span><time>${point.time === null ? "시각 기록 없음" : esc(date(point.time))}</time>`;
    if (!chart || !$("bc-history-cursor")) return;
    const xValue = chart.timeView ? point.time === null ? null : point.time-chart.firstTime : point.step;
    const value = chart.timeView ? point.step : point[history.metric];
    $("bc-history-cursor").setAttribute("visibility",xValue === null ? "hidden" : "visible");$("bc-history-highlight").setAttribute("visibility",xValue === null || !Number.isFinite(value) ? "hidden" : "visible");
    if (xValue !== null) {
      const x = chart.left+xValue/chart.xMax*chart.plotWidth;
      $("bc-history-cursor").setAttribute("x1",x);$("bc-history-cursor").setAttribute("x2",x);
      $("bc-history-highlight").setAttribute("cx",x);
      if (Number.isFinite(value)) $("bc-history-highlight").setAttribute("cy",chart.top+(chart.yMax-value)/(chart.yMax-chart.yMin)*chart.plotHeight);
    }
  }

  function inspectBCHistoryPointer(event) {
    const svg = event.target.closest?.("#bc-history-chart"), history = state.bcHistory;
    if (!svg || !bcHistoryIsCurrent(history) || !history.chart) return;
    const chart = history.chart, rect = svg.getBoundingClientRect();if (!rect.width) return;
    const x = (event.clientX-rect.left)/rect.width*chart.width;
    const requested = Math.max(0,Math.min(chart.xMax,(x-chart.left)/chart.plotWidth*chart.xMax));
    let index = -1, distance = Infinity;
    history.points.forEach((point,i) => {
      if (chart.timeView ? point.time === null : !Number.isFinite(point[history.metric])) return;
      const actual = chart.timeView ? point.time-chart.firstTime : point.step;
      if (Math.abs(actual-requested)<distance) {index = i;distance = Math.abs(actual-requested);}
    });
    if (index>=0) updateBCHistoryPoint(index);
  }

  function renderModelRegistry() {
    const models = currentPipeline()?.models || [];
    $("model-count").textContent = `${count(models.length)} models`;
    if (!models.length) {
      $("pipeline-models").innerHTML = `<div class="learning-empty-models">${icon("model")}<h3>아직 학습된 모델이 없습니다</h3><p>DP, Progress, Residual checkpoint가 생성되면 학습 설정·데이터와 함께 여기에 표시됩니다.</p></div>`;
      return;
    }
    const labels = {bc:"Diffusion Policy",progress:"RL 보조 · Task Progress",residual:"Residual TD3"};
    $("pipeline-models").innerHTML = `<div class="learned-model-list">${models.slice().sort((a,b) => n(b.created_at)-n(a.created_at)).map(model => {
      const download = asset(model.checkpoint_url), metrics = model.metrics || {}, dataset = model.dataset || {};
      const episodes = dataset.episodes ?? dataset.episode_count ?? dataset.episode_ids?.length;
      const validationLoss = metrics.best_validation_loss ?? metrics.validation_loss ?? metrics.val_loss;
      return `<article class="learned-model"><div class="learned-model-header"><div class="learned-model-icon">${icon(model.kind === "progress" ? "progress" : "model")}</div><div><h3>${esc(model.name || labels[model.kind] || model.kind || "Model")}</h3><p>${esc(model.id || "")} · ${esc(date(model.created_at))}</p></div>${tag(model.status === "ready" ? "사용 가능" : model.status || "저장됨",model.status === "ready" ? "success" : "neutral")}</div><dl class="learned-model-facts"><div><dt>종류</dt><dd>${esc(model.kind === "progress" ? labels.progress : model.algorithm_label || labels[model.kind] || model.kind || "—")}</dd></div><div><dt>학습 episodes</dt><dd>${episodes === undefined ? "기록 참고" : count(Array.isArray(episodes) ? episodes.length : episodes)}</dd></div>${model.base_model_id ? `<div><dt>Frozen base</dt><dd>${esc(model.base_model_id)}</dd></div>` : ""}${typeof validationLoss === "number" ? `<div><dt>Validation loss</dt><dd>${finiteValue(validationLoss,5)}</dd></div>` : ""}</dl><details class="learned-model-details"><summary>설정·학습 데이터·지표</summary><div class="model-detail-columns">${[["설정",model.config],["학습 데이터",model.dataset],["지표",model.metrics]].filter(([,value]) => value).map(([name,value]) => `<div><h4>${name}</h4><pre>${esc(JSON.stringify(value,null,2))}</pre></div>`).join("")}</div>${model.checkpoint || model.path ? `<p class="model-checkpoint-path">${esc(model.checkpoint || model.path)}</p>` : ""}</details>${download ? `<a class="button secondary small" href="${esc(download)}" download>${icon("download")}Checkpoint 다운로드</a>` : ""}</article>`;
    }).join("")}</div>`;
  }

  function renderEvaluations() {
    const evaluations = (currentPipeline()?.evaluations || []).slice().sort((a,b) => n(b.created_at)-n(a.created_at));
    const label = value => ["residual","rl","rl_eval"].includes(value) ? "BC + Residual" : "BC only";
    const latestBC = evaluations.find(item => ["bc","bc_eval","bc_only"].includes(item.kind || item.policy));
    const latestRL = evaluations.find(item => ["residual","rl","rl_eval"].includes(item.kind || item.policy));
    const card = (title,item) => `<div class="evaluation-comparison-card"><span>${title}</span><strong>${item && typeof item.success_rate === "number" ? `${finiteValue(item.success_rate,1)}<small>%</small>` : "—"}</strong><p>${item ? `${count(item.successes)} / ${count(item.episodes)} episodes 성공` : "아직 완료된 평가가 없습니다"}</p>${item ? `<span class="evaluation-return">평균 return ${finiteValue(item.mean_return,3)}</span><span class="evaluation-reset-protocol">${esc(evaluationResetProtocol(item))}</span>` : ""}</div>`;
    $("pipeline-evaluations").innerHTML = `<div class="evaluation-comparison">${card("BC only · 최근 평가",latestBC)}${card("BC + Residual · 최근 평가",latestRL)}</div>${evaluations.length ? `<div class="table-wrap"><table class="evaluation-table"><thead><tr><th>정책</th><th>평가 시각</th><th>성공률</th><th>Episodes</th><th>평균 return</th><th>Reset 방식</th><th>모델</th></tr></thead><tbody>${evaluations.map(item => `<tr><td><strong>${label(item.kind || item.policy)}</strong></td><td>${esc(date(item.created_at))}</td><td>${finiteValue(item.success_rate,1)}%</td><td>${count(item.successes)} / ${count(item.episodes)}</td><td>${finiteValue(item.mean_return,3)}</td><td class="evaluation-protocol-cell">${esc(evaluationResetProtocol(item))}${item.evaluation_protocol ? `<span class="table-sub">${esc(item.evaluation_protocol)}</span>` : ""}</td><td><span class="evaluation-model-id">${esc(item.residual_model_id || item.model_id || item.checkpoint_id || "—")}</span></td></tr>`).join("")}</tbody></table></div>` : ""}<p class="evaluation-footnote">학습 중 loss와 실제 rollout 성공률은 별도로 기록됩니다. 평가마다 기록된 reset 방식이 표시되며, 성공·실패 episode는 데이터 탭에서 확인할 수 있습니다.</p>`;
  }

  function pipelineStartBlockedReason() {
    if (remoteRobotProject()) return remoteConnectionOnlyReason;
    const pipeline = currentPipeline(), runtime = pipelineRuntime();
    if (!state.connected) return "서버 연결을 확인해주세요.";
    if (!pipeline?.config || pipeline.unavailable) return "파이프라인 설정을 불러온 뒤 다시 시도해주세요.";
    if (pipelineActive(pipeline.status?.phase)) return "진행 중인 파이프라인을 완료하거나 중지한 뒤 설정할 수 있습니다.";
    if (runtime && pipelineActive(runtime.phase)) return "진행 중인 파이프라인을 완료하거나 중지한 뒤 실행할 수 있습니다.";
    return "";
  }

  function openPipelineStartDialog() {
    const projectId = state.route.id, pipeline = currentPipeline(), dialog = $("pipeline-start-dialog");
    if (!projectId || state.route.tab !== "overview" || state.pipelinePending || state.pipelinePausePending || dialog.open) return;
    const reason = pipelineStartBlockedReason();
    if (reason) {toast(reason,true);return;}
    const config = pipeline.config;
    state.pipelineDialog = {projectId,routeToken:state.routeToken,baseCollectionGoal:config.collection_episodes,bcSteps:config.bc?.steps,busy:false};
    const field = (name,label,min,max,unit="",step=1) => numericField(name,label,config[name],min,max,unit,step);
    $("pipeline-start-fields").innerHTML = `<div class="form-grid three">${field("collection_episodes","BC 시작 성공 episode 목표",1,100,"개")}${numericField("bc_steps","BC 학습 step 수",config.bc?.steps ?? 10000,1,1000000,"steps")}${field("evaluation_episodes","평가 episode 수",1,100,"개")}${field("bc_success_target","BC 성공률 목표",.1,100,"%",.1)}${field("rl_success_target","RL 성공률 목표",.1,100,"%",.1)}${field("rl_episodes_per_round","RL round당 episode",1,100,"개")}${evaluationResetField(config.evaluation_reset_mode || "inherit")}</div><p class="field-note">성공 episode 목표는 기본 수집 목표입니다. BC 평가 미달로 늘어난 현재 누적 목표는 아래에 표시됩니다. 기본 목표를 바꾸면 새 값으로 다시 설정됩니다.</p><details class="pipeline-start-options"><summary>반복 한도와 실행 환경</summary><div class="form-grid three">${field("max_bc_cycles","BC 최대 cycle",1,1000,"회")}${field("max_rl_rounds","RL 최대 round",1,1000,"회")}${field("run_budget_usd","Run당 API 비용 한도",.01,1000,"$",.01)}<label class="field">학습 장치<select name="device"><option value="auto">Auto</option><option value="cuda">CUDA GPU</option><option value="cuda:0">CUDA:0</option><option value="cpu">CPU</option>${config.device && !["auto","cuda","cuda:0","cpu"].includes(config.device) ? `<option value="${esc(config.device)}">${esc(config.device)}</option>` : ""}</select></label></div></details>`;
    $("pipeline-start-form").elements.device.value = config.device || "auto";
    $("pipeline-start-error").hidden = true;$("pipeline-start-error").textContent = "";
    dialog.showModal();updatePipelineStartPreview();
  }

  function updatePipelineStartPreview() {
    const pending = state.pipelineDialog, dialog = $("pipeline-start-dialog");
    if (!pending || !dialog?.open || pending.projectId !== state.route.id || pending.routeToken !== state.routeToken) return;
    const pipeline = currentPipeline(), collection = pipeline?.completion?.collect || {};
    const input = Number($("pipeline-start-form").elements.collection_episodes?.value);
    const target = input === pending.baseCollectionGoal ? collection.target ?? pending.baseCollectionGoal : input;
    const completed = collection.completed ?? currentProject()?.stats?.successes ?? 0;
    const valid = Number.isFinite(target) && target > 0;
    $("pipeline-start-summary").innerHTML = `<strong>사용 중 성공 episode ${count(completed)}개</strong><span>${valid ? `현재 누적 목표 ${count(target)}개 · 추가 성공 episode ${count(Math.max(0,target-completed))}개 필요` : "유효한 성공 episode 목표를 입력해주세요."}</span><small>보관·삭제 데이터는 제외합니다. 저장 후 현재 모델·평가 결과를 확인해 필요한 단계부터 실행합니다.</small>`;
    const reason = pipelineStartBlockedReason();
    $("pipeline-start-fields").disabled = Boolean(pending.busy || reason);
    $("pipeline-start-confirm").disabled = Boolean(pending.busy || reason);
    $("pipeline-start-confirm").textContent = pending.busy ? "설정 저장·실행 중…" : "설정 저장 후 실행";
    $("pipeline-start-runtime-note").textContent = reason;
    $("pipeline-start-runtime-note").hidden = !reason;
  }

  async function submitPipelineStart(event) {
    event.preventDefault();
    const pending = state.pipelineDialog, form = event.target;
    const current = () => state.pipelineDialog === pending && $("pipeline-start-dialog").open && pending?.projectId === state.route.id && pending.routeToken === state.routeToken;
    if (!pending || !current() || pending.busy || state.pipelinePending || !form.reportValidity()) return;
    const reason = pipelineStartBlockedReason();
    if (reason) {$("pipeline-start-error").textContent = reason;$("pipeline-start-error").hidden = false;return;}
    const data = new FormData(form), patch = {};
    for (const key of ["collection_episodes","evaluation_episodes","bc_success_target","rl_success_target","rl_episodes_per_round","max_bc_cycles","max_rl_rounds","run_budget_usd"]) patch[key] = Number(data.get(key));
    patch.device = String(data.get("device"));
    patch.evaluation_reset_mode = String(data.get("evaluation_reset_mode") || "inherit");
    const bcSteps = Number(data.get("bc_steps"));
    if (bcSteps !== pending.bcSteps) patch.bc = {...currentPipeline().config.bc,steps:bcSteps};
    pending.busy = true;state.pipelinePending = "configure-start";$("pipeline-start-error").hidden = true;updatePipelineStartPreview();updatePipelineView();
    let saved = false;
    try {
      const projectId = pending.projectId;
      const result = await api(`/api/projects/${encodeURIComponent(projectId)}/pipeline/config`,patch);
      saved = true;
      if (!current()) return;
      state.pipelineRequest += 1;state.pipelineBusy = false;state.pipeline = result;state.pipelineProjectId = projectId;
      if (result.auto_plan?.complete) {
        $("pipeline-start-dialog").close();toast("설정을 저장했습니다. 모든 목표를 달성해 추가 실행은 시작하지 않았습니다.");return;
      }
      if (!result.auto_plan || !result.readiness?.auto?.ready) throw new Error(result.readiness?.auto?.reason || "저장된 설정의 실행 가능 여부를 확인할 수 없습니다.");
      if (!current()) return;
      const started = await api(`/api/projects/${encodeURIComponent(projectId)}/pipeline/start`,{mode:"auto"},30000);
      if (!current()) return;
      if (started.config) {state.pipelineRequest += 1;state.pipelineBusy = false;state.pipeline = started;state.pipelineProjectId = projectId;}
      $("pipeline-start-dialog").close();toast("설정을 저장하고 남은 파이프라인 실행을 시작했습니다.");
    } catch (error) {
      if (current()) {
        $("pipeline-start-error").textContent = `${saved ? "설정은 저장되었습니다. 실행 상태를 확인해주세요. " : ""}${error.message}`;
        $("pipeline-start-error").hidden = false;
        if (saved) await refreshPipeline(true);
      }
    } finally {
      pending.busy = false;
      if (state.pipelinePending === "configure-start") state.pipelinePending = null;
      if (pending.projectId === state.route.id && pending.routeToken === state.routeToken) {updatePipelineView();updatePipelineStartPreview();renderGlobal();}
    }
  }



  function stageStartBlockedReason(stage, algorithm) {
    const reason = pipelineStartBlockedReason();if (reason) return reason;
    if (!manualPipelineStages().some(item => item.id === stage)) return "현재 화면에서 실행할 수 없는 단계입니다.";
    if (["bc","rl"].includes(stage) && !trainingAlgorithmAvailable(stage,algorithm)) return "선택한 알고리즘은 아직 구현되지 않았습니다. 지원되는 알고리즘을 선택해주세요.";
    const readiness = pipelineReadiness(stage);
    return readiness.ready ? "" : readiness.reason || "이 단계의 실행 조건을 아직 충족하지 않았습니다.";
  }

  function learningDeviceField(device = "auto") {
    const options = [["auto","Auto · 사용 가능한 장치"],["cuda","CUDA GPU"],["cuda:0","CUDA:0"],["cpu","CPU"]];
    if (!options.some(([value]) => value === device)) options.push([device,device]);
    return `<label class="field">실행 장치<select name="device">${options.map(([value,label]) => `<option value="${esc(value)}" ${value === device ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label>`;
  }

  function stageHyperparameterField(key, label, config, help = "") {
    return `<label class="field advanced-config-field">${esc(label)}<small>${esc(help)}</small><textarea name="config_${key}" rows="12" spellcheck="false" aria-label="${esc(label)} hyperparameter JSON">${esc(JSON.stringify(config || {},null,2))}</textarea></label>`;
  }

  function openStageStartDialog(stage) {
    const projectId = state.route.id, pipeline = currentPipeline(), dialog = $("stage-start-dialog");
    if (!projectId || !["models","evaluation"].includes(state.route.tab) || state.pipelinePending || state.pipelinePausePending || dialog.open) return;
    const algorithm = ["bc","rl"].includes(stage) ? selectedTrainingAlgorithm(stage) : null;
    const reason = stageStartBlockedReason(stage,algorithm);
    if (reason) {toast(reason,true);return;}
    const config = pipeline.config, evaluation = ["bc_eval","rl_eval"].includes(stage), isBC = stage === "bc";
    const title = evaluation ? `${stage === "bc_eval" ? "BC" : "RL"} 평가 실행 설정` : `${isBC ? "BC" : "RL"} 학습 실행 설정`;
    state.stageDialog = {projectId,routeToken:state.routeToken,tab:state.route.tab,stage,algorithm,busy:false};
    const field = (name,label,min,max,unit="",step=1) => numericField(name,label,config[name],min,max,unit,step);
    let fields;
    if (evaluation) {
      fields = `<div class="form-grid three">${field("evaluation_episodes","평가 episode 수",1,100,"개")}${field("bc_success_target","BC 성공률 목표",.1,100,"%",.1)}${field("rl_success_target","RL 성공률 목표",.1,100,"%",.1)}${evaluationResetField(config.evaluation_reset_mode || "inherit")}${field("run_budget_usd","Run당 API 비용 한도",.01,1000,"$",.01)}${learningDeviceField(config.device || "auto")}</div><p class="field-note">개별 평가와 전체 파이프라인이 같은 평가 설정을 사용합니다. 수집 설정 따르기는 실행 시 수집 탭의 reset 방식을 사용합니다. 환경 reset은 reset GPT 호출을 생략하며, RL 평가의 진행도 판단은 별도입니다.</p>`;
    } else {
      const algorithmLabel = trainingAlgorithmOptions(stage).find(item => item.id === algorithm)?.label || algorithm;
      const algorithmField = `<label class="field">알고리즘<select name="algorithm"><option value="${esc(algorithm)}" selected>${esc(algorithmLabel)}</option></select></label>`;
      if (isBC) {
        const advanced = {...config.bc};delete advanced.steps;delete advanced.checkpoint_every;
        fields = `<div class="form-grid three">${algorithmField}${numericField("bc_steps","BC 학습 step 수",config.bc?.steps ?? 10000,1,1000000,"steps")}${learningDeviceField(config.device || "auto")}${numericField("checkpoint_every","체크포인트 저장 간격",config.bc?.checkpoint_every ?? 1000,1,1000000,"steps")}</div><p class="field-note">지정한 학습 step 간격으로 학습 상태를 저장합니다. 검증 성능이 가장 좋은 policy 체크포인트는 별도로 유지합니다.</p><details class="advanced-learning-config stage-advanced-config"><summary>${icon("settings")}BC hyperparameter</summary>${stageHyperparameterField("bc","Diffusion Policy",advanced,"학습 steps와 체크포인트 저장 간격은 위 숫자 필드에서 설정합니다. 나머지 전체 설정을 JSON으로 편집하세요.")}</details>`;
      } else {
        fields = `<div class="form-grid three">${algorithmField}${field("rl_episodes_per_round","RL 학습 episode 수",1,100,"개")}${learningDeviceField(config.device || "auto")}${field("run_budget_usd","Run당 API 비용 한도",.01,1000,"$",.01)}${field("progress_gate_threshold","GPT 판단 시작 progress",0,1,"",.01)}${field("gpt_judge_interval","후반 GPT 판단 간격",1,10000,"steps")}</div><label class="required-progress-option"><input id="stage-progress-required" type="checkbox" checked disabled aria-describedby="stage-progress-required-note"><span>Progress 모델 사용</span></label><p id="stage-progress-required-note" class="field-note">현재 RL 학습에 필수이며 해제할 수 없습니다. 현재 설정에 맞는 모델을 재사용하고, 필요한 경우 학습한 뒤 RL을 시작합니다.</p><details class="advanced-learning-config stage-advanced-config"><summary>${icon("settings")}RL·Progress hyperparameter</summary>${stageHyperparameterField("residual","Residual TD3",config.residual,"보정 범위, replay 혼합, n-step, UTD, critic·target 설정")}${stageHyperparameterField("progress","Progress 보조 모델",config.progress,"성공 데이터의 진행도 학습과 encoder 설정")}</details>`;
      }
    }
    $("stage-start-title").textContent = title;$("stage-start-eyebrow").textContent = evaluation ? "POLICY EVALUATION" : isBC ? "BC TRAINING" : "RL TRAINING";
    $("stage-start-description").textContent = "확인 버튼을 누르면 이 단계의 설정을 저장한 뒤 실행합니다. 확인 전에는 저장하거나 실행하지 않습니다.";
    $("stage-start-fields").innerHTML = fields;
    $("stage-start-error").textContent = "";$("stage-start-error").hidden = true;
    dialog.showModal();updateStageStartDialog();
  }

  function updateStageStartDialog() {
    const pending = state.stageDialog, dialog = $("stage-start-dialog");
    if (!pending || !dialog?.open || pending.projectId !== state.route.id || pending.routeToken !== state.routeToken) return;
    const reason = stageStartBlockedReason(pending.stage,pending.algorithm);
    $("stage-start-fields").disabled = Boolean(pending.busy || reason);
    $("stage-start-confirm").disabled = Boolean(pending.busy || reason);
    $("stage-start-confirm").textContent = pending.busy ? "설정 저장·실행 중…" : "설정 저장 후 실행";
    $("stage-start-runtime-note").textContent = reason;$("stage-start-runtime-note").hidden = !reason;
  }

  async function submitStageStart(event) {
    event.preventDefault();
    const pending = state.stageDialog, form = event.target;
    const current = () => state.stageDialog === pending && $("stage-start-dialog").open && pending?.projectId === state.route.id && pending.routeToken === state.routeToken && pending.tab === state.route.tab;
    if (!pending || !current() || pending.busy || state.pipelinePending || !form.reportValidity()) return;
    const showError = message => {$("stage-start-error").textContent = message;$("stage-start-error").hidden = false;};
    const reason = stageStartBlockedReason(pending.stage,pending.algorithm);
    if (reason) {showError(reason);return;}
    const data = new FormData(form), patch = {}, stage = pending.stage;
    const parseConfig = key => {
      let value;try {value = JSON.parse(data.get(`config_${key}`));} catch {throw new Error(`${key.toUpperCase()} hyperparameter의 JSON 형식을 확인해주세요.`);}
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${key.toUpperCase()} 설정은 JSON object여야 합니다.`);
      return value;
    };
    try {
      patch.device = String(data.get("device"));
      if (["bc","rl"].includes(stage)) {
        const algorithm = String(data.get("algorithm"));
        if (algorithm !== pending.algorithm || !trainingAlgorithmAvailable(stage,algorithm)) throw new Error("선택한 알고리즘은 아직 구현되지 않았습니다. 지원되는 알고리즘을 선택해주세요.");
        patch[stage === "bc" ? "bc_algorithm" : "rl_algorithm"] = algorithm;
        if (stage === "bc") patch.bc = {...parseConfig("bc"),steps:Number(data.get("bc_steps")),checkpoint_every:Number(data.get("checkpoint_every"))};
        else {
          for (const key of ["rl_episodes_per_round","run_budget_usd","progress_gate_threshold","gpt_judge_interval"]) patch[key] = Number(data.get(key));
          patch.residual = parseConfig("residual");patch.progress = parseConfig("progress");
        }
      } else {
        for (const key of ["evaluation_episodes","bc_success_target","rl_success_target","run_budget_usd"]) patch[key] = Number(data.get(key));
        patch.evaluation_reset_mode = String(data.get("evaluation_reset_mode") || "inherit");
      }
    } catch (error) {showError(error.message);return;}
    pending.busy = true;state.pipelinePending = "stage-configure-start";$("stage-start-error").hidden = true;updateStageStartDialog();updatePipelineView();
    let saved = false;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(pending.projectId)}/pipeline/config`,patch);
      saved = true;if (!current()) return;
      state.pipelineRequest += 1;state.pipelineBusy = false;state.pipeline = result;state.pipelineProjectId = pending.projectId;
      const ready = result.readiness?.[stage];
      if (!(typeof ready === "boolean" ? ready : ready?.ready)) throw new Error(ready?.reason || "저장된 설정의 실행 가능 여부를 확인할 수 없습니다.");
      const blocked = stageStartBlockedReason(stage,pending.algorithm);
      if (blocked) throw new Error(blocked);
      if (!current()) return;
      const started = await api(`/api/projects/${encodeURIComponent(pending.projectId)}/pipeline/start`,{mode:"manual",stage},30000);
      if (!current()) return;
      if (started.config) {state.pipelineRequest += 1;state.pipelineBusy = false;state.pipeline = started;state.pipelineProjectId = pending.projectId;}
      $("stage-start-dialog").close();toast(`${["bc_eval","rl_eval"].includes(stage) ? "평가" : stage.toUpperCase()+" 학습"} 설정을 저장하고 실행을 시작했습니다.`);
      await Promise.allSettled([refreshPipeline(true),refreshStatus(true)]);
    } catch (error) {
      if (current()) {
        showError(`${saved ? "설정은 저장되었습니다. 실행 상태를 확인해주세요. " : ""}${error.message}`);
        if (saved) await refreshPipeline(true);
      }
    } finally {
      pending.busy = false;if (state.pipelinePending === "stage-configure-start") state.pipelinePending = null;
      if (pending.projectId === state.route.id && pending.routeToken === state.routeToken) {updatePipelineView();updateStageStartDialog();renderGlobal();}
    }
  }

  async function refreshPipeline(force = false) {
    if (remoteRobotProject()) return;
    const id = state.route.id, token = state.routeToken;
    if (!id || state.pipelineBusy && !force || !force && !["overview","models","evaluation","collection"].includes(state.route.tab)) return;
    if (!force && !pipelineEnabled()) return;
    const requestId = ++state.pipelineRequest;
    state.pipelineBusy = true;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(id)}/pipeline`);
      if (requestId !== state.pipelineRequest || token !== state.routeToken || id !== state.route.id) return;
      state.pipeline = data;state.pipelineProjectId = id;
      updatePipelineView();updatePipelineStartPreview();updateStageStartDialog();renderGlobal();updateCollection();
    } catch (error) {
      if (requestId !== state.pipelineRequest || token !== state.routeToken) return;
      if (!pipelineEnabled()) {state.pipeline = {unavailable:true};state.pipelineProjectId = id;updatePipelineView();}
      else if (force) toast(error.message,true);
    } finally {if (requestId === state.pipelineRequest) state.pipelineBusy = false;}
  }

  async function pipelineCommand(action, payload = {}, projectId = state.route.id) {
    if (projectId === state.route.id && remoteRobotProject()) {toast(remoteConnectionOnlyReason,true);return;}
    if (!projectId || (action === "pause" ? state.pipelinePausePending : state.pipelinePending)) return;
    if (action === "start" && (payload.mode === "auto" ? state.route.tab !== "overview" : !["models","evaluation"].includes(state.route.tab) || !manualPipelineStages().some(stage => stage.id === payload.stage))) return;
    if (action === "start" && payload.mode === "auto" && currentPipeline()?.auto_plan?.complete) {toast("모든 목표를 달성했습니다. 추가 실행이 필요하지 않습니다.");return;}
    if (action === "pause") state.pipelinePausePending = true;else state.pipelinePending = action;
    updatePipelineExecution();renderGlobal();
    try {
      await api(`/api/projects/${encodeURIComponent(projectId)}/pipeline/${action}`,payload,30000);
      // A delayed start response must not replace a newer Pause state.
      await Promise.allSettled([refreshPipeline(true),refreshStatus(true),refreshWorkspace()]);
      const messages = {start:"파이프라인 실행을 요청했습니다.",pause:"파이프라인 일시정지를 요청했습니다.",resume:"파이프라인을 재개했습니다.",stop:"파이프라인을 중지했습니다. 저장된 기록과 모델은 유지됩니다."};
      toast(messages[action] || "요청을 처리했습니다.");
    } catch (error) {toast(error.message,true);await refreshPipeline(true);}
    finally {if (action === "pause") state.pipelinePausePending = false;else state.pipelinePending = null;updatePipelineView();renderGlobal();}
  }



  const numericField = (name, label, value, min, max, unit = "", step = 1) => `<label class="field">${esc(label)}<div class="input-unit${unit === "$" ? " currency-prefix" : ""}">${unit === "$" ? "<span>$</span>" : ""}<input name="${name}" type="number" min="${min}" max="${max}" step="${step}" required value="${n(value)}">${unit && unit !== "$" ? `<span>${esc(unit)}</span>` : ""}</div></label>`;

  function updateProjectManagement() {
    const project = currentProject();
    if (!project || !$("project-archive")) return;
    const busy = Boolean(state.projectMutation || state.settingsSaving);
    const reason = project.manage_disabled_reason || "";
    const archive = $("project-archive"), remove = $("project-delete");
    archive.dataset.action = project.archived ? "restore-project" : "archive-project";
    archive.innerHTML = `${icon(project.archived ? "restore" : "archive")}${project.archived ? "프로젝트 복원" : "프로젝트 보관"}`;
    archive.disabled = busy || !state.connected || project.can_archive !== true;
    remove.disabled = busy || !state.connected || project.can_delete !== true;
    remove.innerHTML = `${icon("trash")}${project.delete_pending ? "삭제 재시도" : "프로젝트 삭제"}`;
    archive.title = remove.title = reason;
    $("project-management-note").textContent = reason || (project.archived
      ? "보관된 프로젝트입니다. 복원하면 다시 수집·학습할 수 있습니다."
      : "보관한 프로젝트는 워크스페이스 보관함에서 복원할 수 있습니다.");
    if ($("settings-save")) $("settings-save").disabled = busy || Boolean(project.delete_pending);
  }

  async function archiveProject(archived) {
    const project = currentProject(), token = state.routeToken;
    if (!project || project.can_archive !== true || state.projectMutation || state.settingsSaving) return;
    state.projectMutation = {projectId:project.id,action:"archive"};
    updateProjectManagement();
    try {
      const result = await api(`/api/projects/${encodeURIComponent(project.id)}/archive`,{archived});
      if (token === state.routeToken && currentProject()?.id === project.id) {
        if (archived) {state.projectView = "archived";location.hash = "#/";}
        else {state.detail = result;renderProject();}
      }
      toast(archived ? "프로젝트를 보관했습니다. 워크스페이스 보관함에서 복원할 수 있습니다." : "프로젝트를 복원했습니다.");
      await refreshWorkspace(true);
    } catch (error) {
      toast(error.message,true);
      if (token === state.routeToken) await loadDetail(false);
    } finally {state.projectMutation = null;updateProjectManagement();}
  }

  function askDeleteProject() {
    const project = currentProject();
    if (!project || project.can_delete !== true || state.projectMutation || state.settingsSaving) return;
    state.projectDeleteTarget = {id:project.id,name:project.name};
    $("project-delete-name").textContent = project.name;
    $("project-delete-error").hidden = true;
    $("project-delete-confirm").disabled = false;
    $("project-delete-dialog").showModal();
  }

  async function deleteProject() {
    const target = state.projectDeleteTarget;
    if (!target || state.projectMutation || state.settingsSaving) return;
    state.projectMutation = {projectId:target.id,action:"delete"};
    $("project-delete-confirm").disabled = true;
    $("project-delete-error").hidden = true;
    updateProjectManagement();
    try {
      await api(`/api/projects/${encodeURIComponent(target.id)}/delete`,{confirm:true},120000);
      $("project-delete-dialog").close();
      if (state.route.id === target.id) {
        state.detail = null;state.detailSignature = "";state.pipeline = null;state.pipelineProjectId = null;
        location.hash = "#/";
      }
      toast("프로젝트와 해당 프로젝트의 데이터를 삭제했습니다.");
      await refreshWorkspace(true);
    } catch (error) {
      $("project-delete-error").textContent = error.message;
      $("project-delete-error").hidden = false;
      if (state.route.id === target.id) await loadDetail(false);
    } finally {
      state.projectMutation = null;
      $("project-delete-confirm").disabled = false;
      updateProjectManagement();
    }
  }

  function renderSettings(project) {
    const selected = nextPolicy(project);
    const models = policyOptions().models || [];
    const known = models.some(item => item.model === selected.model);
    const modelChoices = `${known ? "" : `<option value="${esc(selected.model || "")}">${esc(selected.model || "서버 기본값")} · 서버 설정</option>`}${models.map(item => `<option value="${esc(item.model)}">${esc(item.label || item.model)}</option>`).join("")}`;
    $("tab-content").innerHTML = `<form id="settings-form" class="settings-layout"><section class="panel settings-section"><div class="panel-header"><div><h2>프로젝트 정보</h2><p>Task별 데이터와 실행을 구분하는 이름과 설명</p></div></div><div class="panel-body"><div class="form-grid"><label class="field full">프로젝트 이름<input name="name" required maxlength="120" value="${esc(project.name)}"></label><label class="field full">설명<textarea name="description" rows="2" maxlength="2000">${esc(project.description || "")}</textarea></label></div></div></section><section class="panel settings-section"><div class="panel-header"><div><h2>GPT 모델 설정</h2><p>이 프로젝트의 데이터 수집과 reset에 사용할 모델</p></div>${tag("프로젝트별 설정","success")}</div><div class="panel-body"><div class="form-grid"><label class="field">추론 모델<select id="policy-model" name="policy_model" required>${modelChoices}</select></label><label class="field">추론 수준<select id="policy-effort" name="reasoning_effort" required></select></label></div><div id="policy-run-status" class="policy-run-status"></div><p class="field-note">저장 후 다음 <strong>새 run 시작</strong>부터 적용됩니다. 환경을 다시 연결할 필요는 없습니다. 시작한 run은 Pause·재개 시에도 기존 모델·추론 수준을 유지합니다.</p><p id="policy-model-note" class="field-note"></p></div></section>${remoteRobotProject(project) ? "" : graspBaselinePanel(project)}<p id="settings-error" class="form-error" role="alert" hidden></p><div class="settings-actions"><div class="settings-action-notes"><p>환경 재연결 없이 다음 새 run에 적용됩니다. 진행 중인 run은 기존 설정을 유지합니다.</p><p id="project-management-note" class="field-note"></p></div><div class="settings-action-buttons"><button id="project-archive" class="button secondary" type="button" data-action="archive-project">${icon("archive")}프로젝트 보관</button><button id="project-delete" class="button danger" type="button" data-action="delete-project">${icon("trash")}프로젝트 삭제</button><button id="settings-save" class="button primary" type="submit">${icon("check")}설정 저장</button></div></div></form>`;
    $("policy-model").value = selected.model || "";
    updatePolicyEfforts(selected.reasoning_effort);
    updatePolicyStatus();
    updateProjectManagement();
  }

  function updatePolicyEfforts(preferred = $("policy-effort")?.value) {
    const select = $("policy-effort"), model = $("policy-model");
    if (!select || !model) return;
    const entry = (policyOptions().models || []).find(item => item.model === model.value);
    const values = entry?.reasoning_efforts || [nextPolicy(currentProject()).reasoning_effort].filter(Boolean);
    select.innerHTML = values.map(value => `<option value="${esc(value)}">${esc(effortLabel(value))}</option>`).join("");
    select.value = values.includes(preferred) ? preferred : entry?.default_reasoning_effort || values[0] || "";
    $("policy-model-note").textContent = entry ? "이미지 입력과 action 생성을 지원하는 모델입니다. 실제 사용 가능 여부는 서버 API key의 접근 권한에 따라 달라집니다." : "현재 서버 기본 모델입니다. 모델을 변경하려면 목록의 지원 모델을 선택하세요.";
  }

  function updatePolicyStatus() {
    const holder = $("policy-run-status"), project = currentProject();
    if (!holder || !project) return;
    const s = runtime(), active = s.active_project_id === project.id && s.run_id;
    const selected = {model:$("policy-model")?.value,reasoning_effort:$("policy-effort")?.value};
    const pending = active && (s.model !== selected.model || s.reasoning_effort !== selected.reasoning_effort);
    holder.innerHTML = `<div><span>선택한 설정</span><strong>${esc(policyLabel(selected))}</strong></div>${active ? `<div><span>현재 또는 마지막 run 설정</span><strong>${esc(policyLabel(s))}</strong>${pending ? tag("다음 실행에 적용","warning") : tag("동일한 설정")}</div>` : `<p>새 run을 시작하면 선택한 설정을 실행 기록에 함께 저장합니다.</p>`}`;
    if (selected.reasoning_effort === "auto") holder.insertAdjacentHTML("beforeend",'<p>Auto는 medium으로 시작합니다. GPT가 정밀 조작·접촉 불확실성·실패 복구를 판단하면 동작 없이 xhigh로 다시 판단하고, 안정화되면 다음 요청부터 medium으로 돌아갑니다. 재판단도 GPT 호출 수와 비용에 포함됩니다.</p>');
  }

  function remoteConnectionOnlyPanel(project, title) {
    return `<section class="panel remote-connection-only"><div class="panel-header"><div><h2>${esc(title)}</h2><p>SSH 연결과 로봇 client 실행을 별도로 관리합니다.</p></div>${tag("원격 client")}</div><div class="panel-body"><p>${esc(remoteConnectionOnlyReason)}</p><p class="field-note">서버 별명·사용자·IP/호스트·포트를 직접 입력합니다. 설정 저장과 연결은 각각 명시적으로 요청합니다.</p><a class="button primary" href="${routeUrl(project.id,"collection")}">${icon("terminal")}SSH 연결 설정</a></div></section>`;
  }

  function robotConnectionIsCurrent(session = state.robotConnection) {
    return Boolean(session && state.robotConnection === session && session.projectId === state.route.id && session.routeToken === state.routeToken && state.route.tab === "collection" && remoteRobotProject());
  }

  function stopRobotConnectionPolling() {
    const session = state.robotConnection;
    if (session) {clearTimeout(session.timer);session.controller?.abort();if (session.draft) session.draft.password = "";}
    if ($("robot-password")) $("robot-password").value = "";
    if ($("robot-host-key-dialog")?.open) $("robot-host-key-dialog").close();
    state.robotHostKey = null;
    state.robotConnection = null;
  }

  function robotConnectionError(message, code) {
    const help = {
      host_key:"호스트 키 확인 버튼에서 지문을 확인하고 등록한 뒤 다시 연결해주세요.",
      authentication:"입력한 사용자와 선택한 인증 방법의 비밀번호 또는 키를 확인해주세요.",
      timeout:"SSH 대상 주소와 네트워크 연결을 확인한 뒤 다시 시도해주세요.",
      unreachable:"대상 PC가 켜져 있는지, SSH 주소·포트와 네트워크가 올바른지 확인해주세요.",
      unsupported_os:"현재 연결 기능이 지원하는 원격 PC 운영체제를 확인해주세요.",
      protocol:"원격 PC의 SSH 응답 형식을 확인할 수 없습니다. 사용자·호스트·포트를 확인해주세요.",
      ssh_unavailable:"RILO 서버를 실행하는 PC에서 SSH 명령을 사용할 수 있는지 확인해주세요.",
      identity_file:"선택한 인증 키 파일이 RILO 컴퓨터에 있는지, 서버 실행 계정이 읽을 수 있는지 확인해주세요.",
      process_exit:"SSH 연결이 종료되었습니다. 연결 설정을 확인한 뒤 다시 연결할 수 있습니다."
    };
    const text = typeof message === "string" ? message : "";
    return help[code] ? `${help[code]}${text ? ` ${text}` : ""}` : text;
  }

  function robotConnectionConfig(value = {}) {
    const method = value?.auth_method || "password";
    return {name:String(value?.name ?? "").trim(),username:String(value?.username ?? "").trim(),host:String(value?.host ?? "").trim(),port:Number(value?.port ?? 22),auth_method:method,remember_password:method === "password" && value?.remember_password === true,identity_file:method === "key" ? String(value?.identity_file ?? "").trim() || null : null};
  }

  function robotConnectionConfigError(config) {
    if (!config.name || config.name.length>120 || /[\x00-\x1f\x7f]/.test(config.name)) return "표시용 서버 별명을 1~120자로 입력해주세요.";
    if (!config.username || config.username.length>64 || !/^[A-Za-z0-9_][A-Za-z0-9_.-]*\$?$/.test(config.username)) return "사용자는 영문·숫자·밑줄로 시작하는 계정 이름으로 입력해주세요. 최대 64자이며 끝의 $는 사용할 수 있습니다.";
    const host = config.host;
    if (!host || host.length>253 || (host.includes(":") ? (host.match(/:/g) || []).length<2 || !/^[0-9a-fA-F:.]+$/.test(host) : !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(host))) return "IP 또는 실제 호스트명을 입력해주세요. 사용자@호스트, URL, 공백과 포트 접미사는 사용할 수 없습니다. 포트는 별도 입력란에 지정하세요.";
    if (!Number.isInteger(config.port) || config.port<1 || config.port>65535) return "포트는 1~65535 사이의 정수로 입력해주세요.";
    if (!["password","key"].includes(config.auth_method)) return "비밀번호 또는 키 인증을 선택해주세요.";
    if (config.identity_file && (config.identity_file.length>4096 || !config.identity_file.startsWith("/") && !config.identity_file.startsWith("~/") || config.identity_file.includes("${") || /[%\x00-\x1f\x7f]/.test(config.identity_file))) return "인증 키 경로는 4096자 이내의 절대 경로 또는 ~/로 시작하는 경로로 입력해주세요. 변수 토큰이나 % 토큰, 제어 문자는 사용할 수 없습니다.";
    return "";
  }

  function robotConnectionDestination(config) {
    if (!config?.username || !config?.host) return "";
    return `${config.username}@${config.host.includes(":") ? `[${config.host}]` : config.host}:${config.port}`;
  }

  function robotConnectionChanged(session) {
    const config = robotConnectionConfig(session.draft), saved = session.data?.config ? robotConnectionConfig(session.data.config) : null;
    return !saved || JSON.stringify(config) !== JSON.stringify(saved) || config.auth_method === "password" && Boolean(session.draft.password);
  }

  function editRobotConnectionField(input) {
    const key = input.dataset?.robotConfig;
    if (!["name","username","host","port","auth_method","remember_password","identity_file","password"].includes(key) || !robotConnectionIsCurrent()) return;
    const session = state.robotConnection;
    session.draft[key] = key === "remember_password" ? input.checked : input.value;
    if (key === "auth_method" && input.value === "key") {session.draft.password = "";session.draft.remember_password = false;}
    session.dirty = true;session.error = null;updateRemoteRobotConnection();
  }

  function renderRemoteRobotConnection(project) {
    if (!robotConnectionIsCurrent()) {
      stopRobotConnectionPolling();
      state.robotConnection = {projectId:project.id,routeToken:state.routeToken,data:null,draft:robotConnectionConfig(),dirty:false,error:null,stale:false,busy:false,request:0,timer:null,controller:null};
    }
    const session = state.robotConnection, draft = session.draft;
    $("tab-content").innerHTML = `<section class="panel robot-connection-panel"><div class="panel-header"><div><h2>로봇 PC SSH 연결</h2><p>접속 정보를 직접 입력합니다. SSH config는 불러오지 않습니다.</p></div><span id="robot-connection-state" class="tag neutral">상태 확인 중</span></div><div class="panel-body"><p class="robot-connection-scope">SSH는 원격 PC 접속과 상태를 확인합니다. 로봇 관측·정책 실행은 아래 로봇 client 패널에서 별도로 연결합니다.</p><form id="robot-connection-form"><fieldset id="robot-connection-fields" class="robot-connection-config"><label class="field robot-server-name-field">서버 별명<input id="robot-server-name" name="name" data-robot-config="name" required maxlength="120" autocomplete="off" placeholder="예: 실험실 로봇 PC" value="${esc(draft.name)}"><small>화면에 표시할 이름입니다. 접속 주소로 사용하지 않습니다.</small></label><label class="field">사용자<input id="robot-username" name="username" data-robot-config="username" required maxlength="64" autocomplete="off" spellcheck="false" placeholder="예: seunghyo" value="${esc(draft.username)}"></label><label class="field">IP / 호스트<input id="robot-host" name="host" data-robot-config="host" required maxlength="253" autocomplete="off" spellcheck="false" placeholder="예: 192.168.0.10" value="${esc(draft.host)}"><small>실제 호스트명·IPv4·IPv6 주소를 입력하세요.</small></label><label class="field robot-port-field">포트<input id="robot-port" name="port" data-robot-config="port" type="number" required min="1" max="65535" step="1" value="${esc(draft.port)}"></label><label class="field robot-auth-method-field">인증 방법<select id="robot-auth-method" name="auth_method" data-robot-config="auth_method"><option value="password" ${draft.auth_method !== "key" ? "selected" : ""}>비밀번호</option><option value="key" ${draft.auth_method === "key" ? "selected" : ""}>SSH 키</option></select></label><div id="robot-password-fields" class="robot-password-fields"><label class="field">비밀번호<input id="robot-password" name="password" data-robot-config="password" type="password" autocomplete="current-password" spellcheck="false" placeholder="저장된 비밀번호는 다시 표시하지 않습니다"></label><label class="robot-remember-option"><input id="robot-remember-password" name="remember_password" data-robot-config="remember_password" type="checkbox" ${draft.remember_password ? "checked" : ""}><span>.env에 비밀번호 저장</span></label><p class="field-note">선택하면 artifacts/robot_ssh/.env에 비밀번호를 평문으로 저장하며 현재 계정만 읽을 수 있습니다. 선택하지 않으면 현재 RILO 서버가 종료될 때까지 사용하고, 재시작 후에는 다시 입력합니다.</p><p id="robot-password-state" class="field-note" role="status"></p></div><label id="robot-identity-field" class="field robot-identity-field">인증 키 경로 <span class="optional">선택</span><input id="robot-identity-file" name="identity_file" data-robot-config="identity_file" maxlength="4096" autocomplete="off" spellcheck="false" placeholder="예: ~/.ssh/id_ed25519" value="${esc(draft.identity_file || "")}"><small>RILO가 실행되는 컴퓨터의 키 파일 경로입니다. 비워두면 기본 키 또는 ssh-agent를 사용합니다. 키 내용이나 비밀번호를 입력하지 마세요.</small></label></fieldset><div class="robot-config-actions"><p id="robot-config-note" class="field-note" role="status"></p><button id="robot-config-save" type="submit" class="button secondary" disabled>${icon("check")}설정 저장</button></div><p id="robot-legacy-note" class="field-note" role="status" hidden></p></form><div class="robot-connection-actions"><button id="robot-host-key-check" type="button" class="button secondary" data-action="robot-host-key-check" disabled>${icon("check")}호스트 키 확인</button><button id="robot-connect" type="button" class="button primary" data-action="robot-connect" disabled>${icon("terminal")}SSH 연결</button><button id="robot-disconnect" type="button" class="button danger" data-action="robot-disconnect" disabled>${icon("close")}연결 해제</button><button id="robot-status-refresh" type="button" class="button secondary" data-action="robot-status-refresh" disabled>${icon("reset")}상태 확인</button></div><p id="robot-connection-note" class="field-note" role="status"></p><p id="robot-connection-owner" class="field-note" role="status" hidden></p><p id="robot-connection-error" class="form-error" role="alert" hidden></p><section class="robot-host-section"><h3 id="robot-host-heading">연결된 PC 정보</h3><dl class="robot-host-info"><div><dt>서버 별명</dt><dd id="robot-saved-name">—</dd></div><div><dt>접속 대상</dt><dd id="robot-destination">—</dd></div><div><dt>Hostname</dt><dd id="robot-host-name">—</dd></div><div><dt>사용자</dt><dd id="robot-host-user">—</dd></div><div><dt>운영체제</dt><dd id="robot-host-os">—</dd></div><div><dt>Architecture</dt><dd id="robot-host-arch">—</dd></div><div><dt>연결 시각</dt><dd id="robot-connected-at">—</dd></div><div><dt>최근 연결 확인</dt><dd id="robot-last-heartbeat">—</dd></div></dl></section><p class="field-note">화면을 이동해도 SSH 연결은 유지됩니다. 연결을 끝내려면 연결 해제를 누르세요.</p></div></section>`;
    $("tab-content").insertAdjacentHTML("beforeend",renderRemotePolicyPanel(project));
    updateRemoteRobotConnection();refreshRobotConnection();updateRemotePolicyPanel();refreshRemotePolicy();
  }

  function updateRemoteRobotConnection() {
    const session = state.robotConnection;
    if (!robotConnectionIsCurrent(session) || !$("robot-connection-fields")) return;
    const response = session.data, status = response?.status || {}, phase = status.phase || "disconnected";
    const active = response?.active_connection, held = Boolean(active?.project_id);
    const other = held && active.project_id !== session.projectId;
    const ownActive = held && !other || ["connecting","connected","disconnecting"].includes(phase);
    const mutation = state.robotConnectionMutation || state.remotePolicyMutation, pending = mutation?.projectId === session.projectId ? mutation.action : null;
    const managedClient = remotePolicyIsCurrent() && (remoteLauncherHeld(state.remotePolicy) || remoteRunActive(state.remotePolicy.data?.run));
    const config = robotConnectionConfig(session.draft), saved = response?.config ? robotConnectionConfig(response.config) : null;
    const configured = Boolean(saved && !robotConnectionConfigError(saved));
    const unavailable = !state.connected || !response || session.stale, changed = robotConnectionChanged(session);
    const phases = {disconnected:"SSH 연결 안 됨",connecting:"SSH 연결 중",connected:"SSH 연결됨",disconnecting:"연결 해제 중",error:"연결 확인 필요"};
    $("robot-connection-state").textContent = response && (session.stale || !state.connected) ? "상태 확인 필요" : response ? phases[phase] || phase : "상태 확인 중";
    $("robot-connection-state").className = `tag ${session.stale ? "warning" : phase === "connected" && state.connected ? "success" : phase === "error" ? "warning" : "neutral"}`;
    $("robot-connection-fields").disabled = Boolean(unavailable || ownActive || pending);
    for (const [id,key] of [["robot-server-name","name"],["robot-username","username"],["robot-host","host"],["robot-port","port"],["robot-auth-method","auth_method"],["robot-identity-file","identity_file"]]) {
      $(id).disabled = Boolean(unavailable || ownActive || pending);
      if (!session.dirty) $(id).value = String(session.draft[key] ?? "");
    }
    const passwordMode = config.auth_method === "password";
    $("robot-password-fields").hidden = !passwordMode;$("robot-identity-field").hidden = passwordMode;
    $("robot-password").disabled = Boolean(unavailable || ownActive || pending || !passwordMode);
    $("robot-remember-password").disabled = $("robot-password").disabled;
    $("robot-identity-file").disabled = Boolean(unavailable || ownActive || pending || passwordMode);
    if ($("robot-password").value !== (session.draft.password || "")) $("robot-password").value = session.draft.password || "";
    $("robot-remember-password").checked = config.remember_password;
    const credentials = response?.credentials || {}, passwordReady = credentials.password_available === true;
    $("robot-password-state").textContent = credentials.password_saved ? "저장된 접속 대상의 비밀번호가 전용 .env에 저장되어 있습니다. 비밀번호는 다시 표시하지 않습니다." : passwordReady ? "저장된 접속 대상의 비밀번호를 현재 RILO 서버에서 사용할 수 있습니다. 서버 재시작 후에는 다시 입력해야 합니다." : "저장된 비밀번호가 없습니다. 입력한 뒤 설정 저장을 누르면 연결에 사용할 수 있습니다.";
    $("robot-config-save").disabled = Boolean(unavailable || ownActive || mutation || !changed);
    $("robot-config-save").innerHTML = icon("check")+(pending === "config" ? "저장 중…" : "설정 저장");
    $("robot-config-note").textContent = ownActive ? "접속 정보 변경은 이 프로젝트의 연결을 해제한 뒤 저장하세요." : changed && session.dirty ? "저장하지 않은 변경이 있습니다. 설정을 저장한 뒤 연결할 수 있습니다." : configured ? "저장된 접속 정보입니다. 저장만으로 SSH 연결을 시작하지 않습니다." : "서버 별명·사용자·IP/호스트·포트를 입력하고 설정을 저장하세요.";
    $("robot-legacy-note").hidden = Boolean(response?.config || !response?.legacy_name);
    $("robot-legacy-note").textContent = response?.legacy_name ? `이전 별칭 '${response.legacy_name}'은 서버 별명 힌트로만 표시합니다. SSH config를 불러오지 않으므로 사용자·IP/호스트·포트를 새로 입력하고 저장해주세요.` : "";
    $("robot-connect").disabled = Boolean(unavailable || mutation || ownActive || other || changed || !configured || saved?.auth_method === "password" && !passwordReady || response?.capabilities?.ssh === false);
    $("robot-connect").innerHTML = icon("terminal")+(pending === "connect" || phase === "connecting" ? "SSH 연결 중…" : phase === "connected" ? "SSH 연결됨" : "SSH 연결");
    $("robot-disconnect").disabled = Boolean(unavailable || mutation || managedClient || !ownActive || phase === "disconnecting");
    $("robot-disconnect").title = managedClient ? "아래 client 패널에서 run을 중지한 뒤 환경 연결을 해제하세요." : "";
    $("robot-disconnect").innerHTML = icon("close")+(pending === "disconnect" || phase === "disconnecting" ? "연결 해제 중…" : ownActive && phase === "error" ? "연결 해제 다시 시도" : "연결 해제");
    $("robot-host-key-check").disabled = Boolean(unavailable || mutation || ownActive || held || changed || !configured);
    $("robot-host-key-check").innerHTML = icon("check")+(pending === "host-key" ? "호스트 키 확인 중…" : "호스트 키 확인");
    $("robot-status-refresh").disabled = Boolean(session.busy || mutation);
    const notes = {disconnected:"SSH 연결 버튼을 누르면 저장한 접속 정보로 연결합니다. 상태 확인은 새 연결을 만들지 않습니다.",connecting:"SSH 인증과 대상 PC 정보를 확인하고 있습니다. 로봇 동작은 실행하지 않습니다.",connected:"SSH 연결과 PC 정보 확인이 완료되었습니다. 로봇 관측·실행 상태는 아래 client 패널에서 확인하세요.",disconnecting:"이 프로젝트의 SSH 연결을 해제하고 있습니다.",error:"연결 오류를 확인한 뒤 다시 연결할 수 있습니다."};
    $("robot-connection-note").textContent = !state.connected ? "RILO 서버 연결을 확인해주세요." : notes[phase] || "SSH 연결 상태를 확인하고 있습니다.";
    $("robot-connection-owner").hidden = !other;
    if (other) {
      const owner = state.workspace?.projects?.find(project => project.id === active.project_id)?.name || active.project_id;
      $("robot-connection-owner").innerHTML = `다른 프로젝트 <a href="${routeUrl(active.project_id,"collection")}">${esc(owner)}</a>의 ${esc(active.name || "원격 서버")}${active.destination ? ` (${esc(active.destination)})` : ""} 연결이 아직 종료되지 않았습니다. 해당 프로젝트에서 연결을 해제한 뒤 연결하세요.`;
    }
    const error = session.error || robotConnectionError(status.error,status.error_code);
    $("robot-connection-error").textContent = error || "";$("robot-connection-error").hidden = !error;
    const host = status.host_info || {};
    $("robot-host-heading").textContent = phase === "connected" && !unavailable ? "연결된 PC 정보" : "최근 확인된 PC 정보";
    for (const [id,value] of [["robot-saved-name",status.name || response?.config?.name],["robot-destination",status.destination || robotConnectionDestination(saved)],["robot-host-name",host.hostname],["robot-host-user",host.user],["robot-host-os",host.os],["robot-host-arch",host.arch]]) $(id).textContent = value || "—";
    $("robot-connected-at").textContent = status.connected_at ? date(status.connected_at,{second:"2-digit"}) : "—";
    $("robot-last-heartbeat").textContent = status.last_heartbeat_at ? date(status.last_heartbeat_at,{second:"2-digit"}) : "—";
    updateRobotHostKeyDialog();
  }

  function installRobotConnectionResponse(session, response) {
    if (response.project_id !== session.projectId) throw new Error("다른 프로젝트의 SSH 상태 응답을 무시했습니다.");
    session.data = response;session.error = null;session.stale = false;
    if (!session.dirty) session.draft = robotConnectionConfig(response.config || {name:String(response.legacy_name || "").slice(0,120)});
  }

  function scheduleRobotConnectionRefresh(session) {
    clearTimeout(session.timer);
    if (robotConnectionIsCurrent(session)) session.timer = setTimeout(refreshRobotConnection,2500);
  }

  async function refreshRobotConnection() {
    const session = state.robotConnection;if (!robotConnectionIsCurrent(session) || session.busy) return;
    if (state.robotConnectionMutation?.projectId === session.projectId) {scheduleRobotConnectionRefresh(session);return;}
    clearTimeout(session.timer);session.busy = true;const request = ++session.request;
    session.controller = new AbortController();updateRemoteRobotConnection();
    try {
      const response = await api(`/api/projects/${encodeURIComponent(session.projectId)}/robot-connection`,undefined,10000,session.controller.signal);
      if (!robotConnectionIsCurrent(session) || request !== session.request) return;
      installRobotConnectionResponse(session,response);
    } catch (error) {
      if (robotConnectionIsCurrent(session) && request === session.request) {session.stale = true;session.error = error.status === 404 && error.detail === "Not Found" ? "현재 서버에 SSH 연결 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요." : error.message;}
    } finally {
      if (request === session.request) {session.busy = false;session.controller = null;if (robotConnectionIsCurrent(session)) {updateRemoteRobotConnection();scheduleRobotConnectionRefresh(session);}}
    }
  }

  async function robotConnectionCommand(action, event) {
    event?.preventDefault();const session = state.robotConnection;
    if (!robotConnectionIsCurrent(session) || state.robotConnectionMutation || state.remotePolicyMutation || !state.connected || !session.data || session.stale || !["config","connect","disconnect"].includes(action)) return;
    const status = session.data.status || {}, active = session.data.active_connection;
    const other = Boolean(active?.project_id) && active.project_id !== session.projectId;
    const ownActive = Boolean(active?.project_id) && !other || ["connecting","connected","disconnecting"].includes(status.phase);
    const config = robotConnectionConfig(session.draft), saved = session.data.config ? robotConnectionConfig(session.data.config) : null;
    const configured = Boolean(saved && !robotConnectionConfigError(saved)), changed = robotConnectionChanged(session);
    if (action === "disconnect" && remotePolicyIsCurrent() && (remoteLauncherHeld(state.remotePolicy) || remoteRunActive(state.remotePolicy.data?.run))) return;
    if (action === "config") {
      if (ownActive) return;
      if (!$("robot-connection-form").reportValidity()) return;
      const error = robotConnectionConfigError(config);
      if (error) {session.error = error;updateRemoteRobotConnection();return;}
      if (!changed) return;
    }
    if (action === "connect" && (ownActive || other || !configured || changed || saved?.auth_method === "password" && session.data.credentials?.password_available !== true || session.data.capabilities?.ssh === false)) return;
    if (action === "disconnect" && (!ownActive || status.phase === "disconnecting")) return;
    session.controller?.abort();session.controller = null;session.request += 1;session.busy = false;clearTimeout(session.timer);
    const payload = action === "config" ? {...config} : {};
    if (action === "config" && config.auth_method === "password" && session.draft.password) payload.password = session.draft.password;
    const mutation = {projectId:session.projectId,action};state.robotConnectionMutation = mutation;session.error = null;updateRemoteRobotConnection();
    try {
      const response = await api(`/api/projects/${encodeURIComponent(session.projectId)}/robot-connection/${action}`,payload,30000);
      if (!robotConnectionIsCurrent(session)) return;
      if (response.project_id !== session.projectId) throw new Error("다른 프로젝트의 SSH 상태 응답을 무시했습니다.");
      if (action === "config") {session.dirty = false;session.draft = {...robotConnectionConfig(response.config || config),password:""};$("robot-password").value = "";}
      installRobotConnectionResponse(session,response);
      toast(action === "config" ? "SSH 설정을 저장했습니다. 연결은 아직 시작하지 않았습니다." : action === "connect" ? "SSH 연결을 요청했습니다." : "SSH 연결 해제를 요청했습니다.");
    } catch (error) {
      if (robotConnectionIsCurrent(session)) {session.stale = true;session.error = error.status === 404 && error.detail === "Not Found" ? "현재 서버에 SSH 연결 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요." : robotConnectionError(error.message,error.detail?.code);}
    } finally {
      if (Object.hasOwn(payload,"password")) delete payload.password;
      if (state.robotConnectionMutation === mutation) state.robotConnectionMutation = null;
      if (robotConnectionIsCurrent(session)) {updateRemoteRobotConnection();scheduleRobotConnectionRefresh(session);}
      else if (robotConnectionIsCurrent()) updateRemoteRobotConnection();
    }
  }

  function robotHostKeyBlockReason(session = state.robotConnection) {
    if (!robotConnectionIsCurrent(session) || !state.connected || !session.data || session.stale) return "연결 상태를 다시 확인해주세요.";
    if (state.remotePolicyMutation || remotePolicyIsCurrent() && remoteLauncherHeld(state.remotePolicy)) return "Client 환경 연결을 해제한 뒤 호스트 키를 확인해주세요.";
    if (!session.data.config || robotConnectionConfigError(robotConnectionConfig(session.data.config))) return "접속 정보를 먼저 저장해주세요.";
    if (robotConnectionChanged(session)) return "변경한 접속 정보를 저장한 뒤 호스트 키를 다시 확인해주세요.";
    if (session.data.active_connection || ["connecting","connected","disconnecting"].includes(session.data.status?.phase)) return "진행 중인 SSH 연결을 해제한 뒤 호스트 키를 확인할 수 있습니다.";
    return "";
  }

  function updateRobotHostKeyDialog() {
    const pending = state.robotHostKey;
    if (!pending || !$("robot-host-key-dialog")?.open) return;
    const session = state.robotConnection;
    const changed = !robotConnectionIsCurrent(session) || session.projectId !== pending.projectId || session.routeToken !== pending.routeToken || JSON.stringify(robotConnectionConfig(session.data?.config)) !== pending.configSignature;
    const reason = changed ? "접속 정보가 변경되었습니다. 창을 닫고 호스트 키를 다시 확인해주세요." : pending.invalid ? "키 확인 정보가 만료되거나 변경되었습니다. 창을 닫고 다시 확인해주세요." : robotHostKeyBlockReason(session);
    $("robot-host-key-trust").disabled = Boolean(pending.busy || state.robotConnectionMutation || state.remotePolicyMutation || reason);
    $("robot-host-key-trust").textContent = pending.busy ? "호스트 키 등록 중…" : "확인한 호스트 키 등록";
    $("robot-host-key-note").textContent = reason || "등록만으로 SSH 연결을 시작하지 않습니다. 연결은 별도 버튼으로 요청하세요.";
  }

  async function checkRobotHostKey() {
    const session = state.robotConnection;
    if (!robotConnectionIsCurrent(session) || state.robotConnectionMutation || state.remotePolicyMutation || $("robot-host-key-dialog").open) return;
    const reason = robotHostKeyBlockReason(session);
    if (reason) {session.error = reason;updateRemoteRobotConnection();return;}
    const configSignature = JSON.stringify(robotConnectionConfig(session.data.config));
    session.controller?.abort();session.controller = null;session.request += 1;session.busy = false;clearTimeout(session.timer);
    const mutation = {projectId:session.projectId,action:"host-key"};state.robotConnectionMutation = mutation;session.error = null;updateRemoteRobotConnection();
    try {
      const result = await api(`/api/projects/${encodeURIComponent(session.projectId)}/robot-connection/host-key`,{},30000);
      if (!robotConnectionIsCurrent(session)) return;
      if (configSignature !== JSON.stringify(robotConnectionConfig(session.data.config)) || robotConnectionChanged(session)) throw new Error("접속 정보가 변경되었습니다. 호스트 키를 다시 확인해주세요.");
      if (typeof result.verification_id !== "string" || !result.verification_id || !Array.isArray(result.keys) || !result.keys.length || !result.keys.every(key => typeof key.algorithm === "string" && key.algorithm && typeof key.fingerprint === "string" && key.fingerprint)) throw new Error("확인할 호스트 키 지문을 받지 못했습니다. 다시 확인해주세요.");
      if (result.destination !== robotConnectionDestination(session.data.config)) throw new Error("저장한 접속 대상과 키 확인 응답이 다릅니다. 호스트 키를 다시 확인해주세요.");
      state.robotHostKey = {projectId:session.projectId,routeToken:session.routeToken,verificationId:result.verification_id,configSignature,busy:false,invalid:false};
      $("robot-host-key-content").innerHTML = `<p class="dialog-description"><strong>${esc(session.data.config.name)}</strong><br>${esc(result.destination || robotConnectionDestination(session.data.config))}</p>${result.changed ? '<p class="robot-host-key-warning" role="alert"><strong>호스트 키가 변경되었습니다.</strong> 대상 PC에서 확인한 지문과 일치하는 경우에만 등록하세요.</p>' : ""}<p class="field-note">대상 PC에서 확인한 지문 또는 관리자가 제공한 지문과 아래 값을 비교한 뒤 등록하세요.</p><ul class="robot-host-key-list">${result.keys.map(key => `<li><span>${esc(key.algorithm)}</span><code>${esc(key.fingerprint)}</code></li>`).join("")}</ul><p class="field-note">등록한 키는 RILO 전용 known_hosts에 저장합니다. 사용자 계정의 기존 known_hosts는 변경하지 않습니다.</p>`;
      $("robot-host-key-error").hidden = true;$("robot-host-key-error").textContent = "";
      $("robot-host-key-dialog").showModal();updateRobotHostKeyDialog();
    } catch (error) {
      if (robotConnectionIsCurrent(session)) {session.error = error.status === 404 && error.detail === "Not Found" ? "현재 서버에 호스트 키 확인 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요." : error.message;updateRemoteRobotConnection();}
    } finally {
      if (state.robotConnectionMutation === mutation) state.robotConnectionMutation = null;
      if (robotConnectionIsCurrent(session)) {updateRemoteRobotConnection();scheduleRobotConnectionRefresh(session);}
    }
  }

  async function trustRobotHostKey() {
    const pending = state.robotHostKey, session = state.robotConnection;
    const current = () => Boolean(pending && state.robotHostKey === pending && $("robot-host-key-dialog").open && robotConnectionIsCurrent(session) && pending.projectId === session.projectId && pending.routeToken === session.routeToken);
    if (!current() || pending.busy || pending.invalid || state.robotConnectionMutation || state.remotePolicyMutation) return;
    const reason = robotHostKeyBlockReason(session);
    if (reason || pending.configSignature !== JSON.stringify(robotConnectionConfig(session.data.config))) {updateRobotHostKeyDialog();return;}
    pending.busy = true;session.controller?.abort();session.controller = null;session.request += 1;session.busy = false;clearTimeout(session.timer);
    const mutation = {projectId:session.projectId,action:"host-key-trust"};state.robotConnectionMutation = mutation;
    $("robot-host-key-error").hidden = true;updateRemoteRobotConnection();
    try {
      const response = await api(`/api/projects/${encodeURIComponent(session.projectId)}/robot-connection/host-key/trust`,{verification_id:pending.verificationId},30000);
      if (!current()) return;
      installRobotConnectionResponse(session,response);
      $("robot-host-key-dialog").close();toast("호스트 키를 등록했습니다. SSH 연결 버튼으로 연결하세요.");
    } catch (error) {
      if (current()) {
        if (error.status === 409) pending.invalid = true;
        $("robot-host-key-error").textContent = `${error.status === 409 ? "연결 상태를 확인한 뒤 창을 닫고 호스트 키를 다시 확인해주세요. " : ""}${error.message}`;
        $("robot-host-key-error").hidden = false;
      }
    } finally {
      pending.busy = false;if (state.robotConnectionMutation === mutation) state.robotConnectionMutation = null;
      if (robotConnectionIsCurrent(session)) {updateRemoteRobotConnection();scheduleRobotConnectionRefresh(session);}
    }
  }

  function remoteLaunchConfig(value = {}) {
    return {client_directory:String(value.client_directory ?? "~/rilo_robot_client").trim(),conda_env:String(value.conda_env ?? "hilserl").trim(),python_executable:String(value.python_executable ?? "").trim() || null,env_config:String(value.env_config ?? "").trim() || null,adapter:String(value.adapter ?? "").trim() || null,mode:value.mode || "probe"};
  }

  function remoteLaunchChanged(session) {
    return JSON.stringify(remoteLaunchConfig(session.launchDraft)) !== JSON.stringify(remoteLaunchConfig(session.data?.launch_config));
  }

  function remoteLauncherHeld(session) {
    const phase = session?.data?.launcher?.phase;
    return Boolean(session?.data?.active_launcher?.project_id === session?.projectId || phase && !["idle","disconnected","stopped","error"].includes(phase));
  }

  function remoteLaunchConfigError(config) {
    if (!config.client_directory || config.client_directory.length>4096 || !config.client_directory.startsWith("/") && !config.client_directory.startsWith("~/")) return "로봇 PC의 client 디렉터리를 절대 경로 또는 ~/ 경로로 입력해주세요.";
    if (!config.conda_env || config.conda_env.length>128 || !/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test(config.conda_env)) return "Conda 환경 이름은 영문·숫자·밑줄로 시작하며 점과 하이픈을 사용할 수 있습니다.";
    if (config.python_executable && (config.python_executable.length>4096 || !config.python_executable.startsWith("/") && !config.python_executable.startsWith("~/"))) return "Python 실행 파일은 절대 경로 또는 ~/ 경로로 입력해주세요.";
    if (config.adapter && (config.adapter.length>256 || !/^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$/.test(config.adapter))) return "환경 Adapter를 package.module:factory 형식으로 입력해주세요.";
    if (config.env_config && config.env_config.length>4096) return "로봇 환경 JSON 경로는 4096자 이하로 입력해주세요.";
    if (!["probe","mock","robot"].includes(config.mode)) return "Client 실행 모드를 선택해주세요.";
    if (config.mode === "robot" && (!config.adapter || !config.env_config)) return "실제 로봇 모드는 adapter(module:factory)와 환경 JSON 경로가 필요합니다.";
    if (Object.values(config).some(value => typeof value === "string" && /[\x00-\x1f\x7f]/.test(value))) return "연결 설정에는 줄바꿈이나 제어 문자를 사용할 수 없습니다.";
    return "";
  }

  function remoteEnvironmentBlockReason(action,session = state.remotePolicy) {
    if (!remotePolicyIsCurrent(session) || !state.connected || !session.data || session.stale) return "Client 상태를 확인해주세요.";
    if (state.remotePolicyMutation || state.robotConnectionMutation) return "이전 연결 요청을 처리하고 있습니다.";
    const capability = {"launch-config":"configure_launch","environment/connect":"environment_connect","environment/disconnect":"environment_disconnect"}[action];
    if (session.data.capabilities?.[capability] !== true) return session.data.reasons?.[capability] || "현재 연결 상태에서는 이 동작을 실행할 수 없습니다.";
    if (remoteRunActive(session.data.run)) return "진행 중인 run을 먼저 중지해주세요.";
    const held = remoteLauncherHeld(session), connected = session.data.client?.connected === true;
    if (action === "launch-config") return held || connected ? "Client 환경 연결을 해제한 뒤 실행 설정을 저장하세요." : "";
    if (action === "environment/disconnect") return "";
    if (held || connected) return "이미 환경이 연결되어 있거나 연결 중입니다.";
    if (remoteLaunchChanged(session)) return "변경한 client 실행 설정을 먼저 저장해주세요.";
    const owner = state.robotConnection?.data?.active_connection;
    if (owner?.project_id && owner.project_id !== session.projectId) return "다른 프로젝트의 SSH 연결을 먼저 해제해주세요.";
    if (!state.robotConnection?.data?.config) return "위에서 로봇 PC의 SSH 접속 정보를 먼저 저장해주세요.";
    return remoteLaunchConfigError(remoteLaunchConfig(session.launchDraft));
  }

  function renderRemoteLaunchSettings(session) {
    const config = session.launchDraft;
    return `<section class="remote-launch-section"><h3>Client 환경 연결</h3><p class="field-note">로봇 PC에 client 코드를 준비한 뒤 실행 경로를 저장하세요. 환경 연결은 SSH 접속·터널·client 실행을 차례로 요청합니다. 연결만으로 episode를 시작하지 않습니다.</p><form id="remote-launch-form"><fieldset id="remote-launch-fields" class="remote-launch-fields"><label class="field">Client 디렉터리<input name="client_directory" required maxlength="4096" value="${esc(config.client_directory)}" placeholder="~/rilo_robot_client"><small>로봇 PC에서 client 코드가 위치한 경로입니다.</small></label><label class="field">Conda 환경<input name="conda_env" required maxlength="128" value="${esc(config.conda_env)}" placeholder="hilserl"></label><label class="field">Python 실행 파일 <span class="optional">선택</span><input name="python_executable" maxlength="4096" value="${esc(config.python_executable || "")}" placeholder="/절대경로/bin/python"><small>입력하면 Conda 환경 대신 이 실행 파일을 사용합니다.</small></label><label class="field">실행 모드<select name="mode"><option value="probe">진단 · 통신 확인</option><option value="mock">Mock · 시뮬레이션</option><option value="robot">실제 로봇</option></select></label><label class="field">로봇 환경 JSON 경로<input name="env_config" maxlength="4096" value="${esc(config.env_config || "")}" placeholder="/로봇PC/경로/environment.json"></label><label class="field">환경 Adapter<input name="adapter" maxlength="256" value="${esc(config.adapter || "")}" placeholder="package.module:factory"><small>실제 로봇 모드에는 해당 로봇에 맞는 adapter와 환경 JSON이 필요합니다.</small></label></fieldset><div class="robot-config-actions"><p id="remote-launch-config-note" class="field-note"></p><button id="remote-launch-save" class="button secondary" type="submit" disabled>${icon("check")}실행 설정 저장</button></div></form><div class="remote-environment-actions"><button id="remote-environment-connect" class="button primary" type="button" data-action="remote-environment-connect" disabled>${icon("robot")}환경 연결</button><button id="remote-environment-disconnect" class="button danger" type="button" data-action="remote-environment-disconnect" disabled>${icon("close")}환경 연결 해제</button><span id="remote-launch-state" class="tag neutral">대기 중</span></div><p id="remote-environment-note" class="field-note" role="status"></p><details class="remote-contract"><summary>Client 실행 로그</summary><pre id="remote-launch-log">아직 실행 기록이 없습니다.</pre></details></section>`;
  }

  function editRemoteLaunchField(input) {
    const session = state.remotePolicy;if (!remotePolicyIsCurrent(session) || !input.name || !Object.hasOwn(session.launchDraft,input.name)) return;
    session.launchDraft[input.name] = input.value;session.launchDirty = true;session.error = null;updateRemotePolicyPanel();
  }

  function updateRemoteLaunchSettings(session) {
    const launcher = session.data?.launcher || {}, config = session.launchDraft;
    const locked = !state.connected || !session.data || session.stale || session.data.capabilities?.configure_launch !== true || remoteLauncherHeld(session) || session.data?.client?.connected === true || Boolean(state.remotePolicyMutation || state.robotConnectionMutation);
    $("remote-launch-fields").disabled = locked;
    for (const key of Object.keys(config)) {const input = $("remote-launch-form").elements.namedItem(key);if (input) {input.disabled = locked;if (!session.launchDirty) input.value = config[key] ?? "";}}
    const changed = remoteLaunchChanged(session), pending = state.remotePolicyMutation?.projectId === session.projectId ? state.remotePolicyMutation.action : null;
    $("remote-launch-save").disabled = Boolean(remoteEnvironmentBlockReason("launch-config",session) || !changed);
    $("remote-launch-save").textContent = pending === "launch-config" ? "저장 중…" : "실행 설정 저장";
    $("remote-launch-config-note").textContent = changed ? "변경한 실행 설정을 저장한 뒤 환경을 연결하세요." : "저장한 실행 설정입니다. 저장만으로 SSH나 client를 시작하지 않습니다.";
    const connectReason = remoteEnvironmentBlockReason("environment/connect",session), disconnectReason = remoteEnvironmentBlockReason("environment/disconnect",session);
    $("remote-environment-connect").disabled = Boolean(connectReason);$("remote-environment-connect").title = connectReason;
    $("remote-environment-disconnect").disabled = Boolean(disconnectReason);$("remote-environment-disconnect").title = disconnectReason;
    const phaseLabels = {idle:"대기 중",disconnected:"연결 안 됨",connecting:"환경 연결 중",preparing:"실행 준비 중",starting:"Client 시작 중",running:"Client 실행 중",connected:"환경 연결됨",stopping:"연결 해제 중",disconnecting:"연결 해제 중",stopped:"연결 해제됨",error:"확인 필요"};
    $("remote-launch-state").textContent = phaseLabels[launcher.phase] || launcher.phase || "대기 중";
    $("remote-launch-state").className = `tag ${launcher.phase === "error" ? "warning" : session.data?.client?.connected ? "success" : "neutral"}`;
    $("remote-environment-note").textContent = launcher.error || connectReason || (config.mode === "probe" ? "진단 모드는 통신만 확인합니다. 관측·로봇 동작을 제공하지 않습니다." : config.mode === "mock" ? "Mock 환경에서 연결과 정책 실행 흐름을 확인합니다." : "Client와 로봇 adapter를 연결합니다. Episode는 수집 시작 버튼으로 별도 요청합니다.");
    $("remote-launch-log").textContent = Array.isArray(launcher.logs) ? launcher.logs.map(item => typeof item === "string" ? item : item.message || JSON.stringify(item)).join("\n") || "아직 실행 기록이 없습니다." : typeof launcher.logs === "string" ? launcher.logs : "아직 실행 기록이 없습니다.";
  }

  async function remoteEnvironmentCommand(action,event) {
    event?.preventDefault();
    const session = state.remotePolicy;
    if (!["launch-config","environment/connect","environment/disconnect"].includes(action) || remoteEnvironmentBlockReason(action,session)) return;
    if (action === "launch-config" && !remoteLaunchChanged(session)) return;
    const payload = action === "launch-config" ? remoteLaunchConfig(session.launchDraft) : {};
    if (action === "launch-config") {const error = remoteLaunchConfigError(payload);if (error) {session.error = error;updateRemotePolicyPanel();return;}}
    if (action === "environment/connect") session.issuedConfig = null;
    const mutation = {projectId:session.projectId,action};state.remotePolicyMutation = mutation;session.error = null;
    clearTimeout(session.timer);session.controller?.abort();session.request += 1;session.busy = false;session.controller = null;updateRemotePolicyPanel();updateRemoteRobotConnection();
    try {
      const response = await api(remotePolicyPath(session)+"/"+action,payload,30000);
      if (!remotePolicyIsCurrent(session)) return;
      if (action === "launch-config") session.launchDirty = false;
      installRemotePolicyResponse(session,response);
      toast({"launch-config":"Client 실행 설정을 저장했습니다.","environment/connect":"환경 연결을 요청했습니다. SSH와 client 상태를 확인합니다.","environment/disconnect":"Client 환경과 SSH 연결 해제를 요청했습니다."}[action]);
    } catch (error) {if (remotePolicyIsCurrent(session)) {session.error = error.message;session.stale = true;}}
    finally {if (state.remotePolicyMutation === mutation) state.remotePolicyMutation = null;if (remotePolicyIsCurrent(session)) {updateRemotePolicyPanel();updateRemoteRobotConnection();scheduleRemotePolicyRefresh(session);refreshRobotConnection();}else if (remotePolicyIsCurrent()) updateRemotePolicyPanel();}
  }

  function remotePolicyIsCurrent(session = state.remotePolicy) {
    return Boolean(session && state.remotePolicy === session && session.projectId === state.route.id && session.routeToken === state.routeToken && state.route.tab === "collection" && remoteRobotProject());
  }

  function remotePolicyPath(session = state.remotePolicy) {
    return `/api/projects/${encodeURIComponent(session.projectId)}/remote`;
  }

  function remoteRunActive(run) {
    return Boolean(run && ["preparing","running","pausing","paused","stopping"].includes(run.status));
  }

  function stopRemotePolicyPolling() {
    const session = state.remotePolicy;
    if (session) {clearTimeout(session.timer);session.controller?.abort();session.frameToken += 1;session.issuedConfig = null;}
    state.remotePolicy = null;
    if ($("remote-token-dialog")?.open) $("remote-token-dialog").close();
  }

  function remoteRunConfig(value = {}) {
    return {policy:value.policy || "gpt",model_id:value.model_id || null,residual_model_id:value.residual_model_id || null,episodes:Number(value.episodes ?? 1),device:String(value.device || "cuda:0"),max_episode_steps:Number(value.max_episode_steps ?? 300),max_reset_steps:Number(value.max_reset_steps ?? 300),max_task_calls:Number(value.max_task_calls ?? 50),max_reset_calls:Number(value.max_reset_calls ?? 40),budget_usd:Number(value.budget_usd ?? 15),training:value.training === true};
  }

  function remoteResidualTrainingAvailable(session = state.remotePolicy) {
    return session?.data?.client?.capabilities?.residual_training === true && session.data.capabilities?.residual_training !== false;
  }

  function remoteRunConfigError(config, session = state.remotePolicy) {
    if (!["gpt","dp","residual","dummy"].includes(config.policy)) return "실행할 정책을 선택해주세요.";
    if (config.policy === "dummy" && session?.data?.client?.capabilities?.simulation !== true) return "Mock 정책은 시뮬레이션 client에서만 사용할 수 있습니다.";
    const models = session?.data?.models || [];
    if (config.policy === "residual" && session?.data?.client?.capabilities?.residual_step !== true) return "이 client는 control tick마다 Residual 보정을 적용하는 기능을 지원하지 않습니다.";
    if (["dp","residual"].includes(config.policy) && !models.some(model => model.id === config.model_id && model.kind === "bc")) return "이 프로젝트의 DP 모델을 선택해주세요.";
    if (config.policy === "residual" && !models.some(model => model.id === config.residual_model_id && model.kind === "residual")) return "이 프로젝트의 Residual TD3 모델을 선택해주세요.";
    for (const [key,label,max] of [["episodes","Run 당 episode 수",100],["max_episode_steps","최대 episode 길이",10000],["max_reset_steps","최대 reset step",10000],["max_task_calls","Task GPT 호출 한도",10000],["max_reset_calls","Reset GPT 호출 한도",1000]]) {
      if (!Number.isInteger(config[key]) || config[key]<1 || config[key]>max) return `${label}는 1~${max} 사이의 정수로 입력해주세요.`;
    }
    if (!Number.isFinite(config.budget_usd) || config.budget_usd<=0 || config.budget_usd>1000) return "API 비용 한도는 $0 초과, $1,000 이하로 입력해주세요.";
    if (!config.device.trim()) return "정책 실행 장치를 선택해주세요.";
    if (config.training && (config.policy !== "residual" || !remoteResidualTrainingAvailable(session))) return "현재 연결에서는 Residual 학습을 지원하지 않습니다.";
    return "";
  }

  function remotePolicyBlockReason(action, session = state.remotePolicy) {
    if (!remotePolicyIsCurrent(session) || !state.connected || !session.data || session.stale) return "로봇 client 상태를 먼저 확인해주세요.";
    if (state.remotePolicyMutation) return "이전 요청을 처리하고 있습니다.";
    if (action === "token") return remoteRunActive(session.data.run) ? "진행 중인 run을 중지한 뒤 토큰을 재발급할 수 있습니다." : session.data.active_launcher || remoteLauncherHeld(session) ? "Client 환경 연결을 해제한 뒤 토큰을 발급하거나 재발급하세요." : "";
    if (action === "start" && session.data.client?.connected !== true) return "로봇 PC의 client를 연결해주세요. SSH 연결 상태와 별도로 확인합니다.";
    if (session.data.capabilities?.[action] !== true) return session.data.reasons?.[action] || "현재 client 상태에서는 이 동작을 실행할 수 없습니다.";
    return "";
  }

  function remoteServerUrl(value) {
    try {const parsed = new URL(String(value).trim());if (!["http:","https:"].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash || parsed.pathname.replace(/\/$/,"") !== "/robot/v1") return null;return parsed.href.replace(/\/$/,"");} catch {return null;}
  }

  function renderRemotePolicyPanel(project) {
    if (!remotePolicyIsCurrent()) {
      stopRemotePolicyPolling();
      state.remotePolicy = {projectId:project.id,routeToken:state.routeToken,data:null,draft:remoteRunConfig(),dirty:false,launchDraft:remoteLaunchConfig(),launchDirty:false,baseUrl:"",baseUrlDirty:false,issuedConfig:null,error:null,readError:null,stale:false,busy:false,request:0,timer:null,controller:null,frameToken:0,frameSignature:"",modelSignature:""};
    }
    const draft = state.remotePolicy.draft;
    state.remotePolicy.modelSignature = "";state.remotePolicy.frameSignature = "";state.remotePolicy.cameraSignature = "";state.remotePolicy.frameToken += 1;
    return `<section id="remote-policy-panel" class="panel remote-policy-panel"><div class="panel-header"><div><h2>로봇 client / 정책 서버</h2><p>로봇 PC의 client가 환경 계약과 실행 상태를 전달합니다.</p></div><span id="remote-client-state" class="tag neutral">상태 확인 중</span></div><div class="panel-body">${renderRemoteLaunchSettings(state.remotePolicy)}<section class="remote-client-setup"><p class="field-note">위쪽 SSH 연결은 PC 접속 진단입니다. 환경 연결 버튼은 SSH·터널·로봇 PC의 client를 함께 준비합니다. Client의 실행 준비가 확인된 뒤에만 아래 제어를 사용할 수 있습니다.</p><details class="remote-contract remote-manual-config"><summary>수동 client 연결 설정과 토큰</summary><label class="field">로봇 PC에서 접속할 정책 서버 URL<input id="remote-server-url" type="url" autocomplete="off" spellcheck="false" placeholder="http://RILO서버IP:8000/robot/v1"><small>SSH 터널을 사용하면 http://127.0.0.1:45587/robot/v1을 사용합니다. 직접 연결할 때는 로봇 PC에서 접근 가능한 RILO 서버 주소를 입력하세요.</small></label><div class="remote-client-actions"><button id="remote-token-issue" type="button" class="button secondary" data-action="remote-token-issue" disabled>${icon("check")}연결 토큰 발급</button><button id="remote-config-download" type="button" class="button secondary" data-action="remote-config-download" disabled>${icon("download")}연결 설정 다운로드 (.json)</button><button id="remote-status-refresh" type="button" class="button secondary" data-action="remote-status-refresh" disabled>${icon("reset")}상태 확인</button></div><p id="remote-token-note" class="remote-token-note">발급한 토큰은 이 화면에서만 다운로드할 수 있습니다. 설정 파일에는 인증 토큰이 포함되므로 로봇 PC에서 chmod 600으로 권한을 설정하세요.</p></details><dl class="remote-client-details"><div><dt>Client</dt><dd id="remote-client-id">—</dd></div><div><dt>실행 준비</dt><dd id="remote-client-phase">—</dd></div><div><dt>최근 heartbeat</dt><dd id="remote-client-heartbeat">—</dd></div><div><dt>환경</dt><dd id="remote-client-environment">—</dd></div><div><dt>세션</dt><dd id="remote-client-session">—</dd></div><div><dt>프로토콜</dt><dd id="remote-client-protocol">—</dd></div></dl><details class="remote-contract"><summary>환경 계약과 client 지원 기능</summary><pre id="remote-contract">client가 연결되면 확인할 수 있습니다.</pre></details></section><div class="remote-run-layout"><form id="remote-run-form"><fieldset id="remote-run-fields" class="remote-run-fields"><label class="field full">Run 정책<select id="remote-policy-kind" name="policy" data-remote-run="policy"><option value="gpt">GPT</option><option value="dp">Diffusion Policy</option><option value="residual">Residual TD3</option><option value="dummy" disabled>Mock · 시뮬레이션 전용</option></select></label><label id="remote-model-field" class="field full">DP 모델<select id="remote-model-id" name="model_id" data-remote-run="model_id"><option value="">모델을 선택하세요</option></select></label><label id="remote-residual-field" class="field full">Residual TD3 모델<select id="remote-residual-model-id" name="residual_model_id" data-remote-run="residual_model_id"><option value="">모델을 선택하세요</option></select></label><p id="remote-policy-note" class="field-note full"></p>${numericField("episodes","Run 당 episode 수",draft.episodes,1,100,"개")}${numericField("budget_usd","API 비용 한도",draft.budget_usd,.01,1000,"$",.01)}${numericField("max_episode_steps","최대 episode 길이",draft.max_episode_steps,1,10000,"steps")}${numericField("max_reset_steps","최대 reset step",draft.max_reset_steps,1,10000,"steps")}${numericField("max_task_calls","Task GPT 호출 한도",draft.max_task_calls,1,10000,"회")}${numericField("max_reset_calls","Reset GPT 호출 한도",draft.max_reset_calls,1,1000,"회")}<label class="field full">정책 실행 장치<select id="remote-device" name="device" data-remote-run="device"><option value="cuda:0">CUDA:0</option><option value="cuda">CUDA</option><option value="auto">자동</option><option value="cpu">CPU</option></select></label><label id="remote-training-field" class="robot-remember-option full" hidden><input id="remote-training" name="training" data-remote-run="training" type="checkbox"><span>Residual 온라인 학습</span></label></fieldset><p class="field-note">입력값은 새 run 시작 시 확정됩니다. Pause·재개는 기존 run 설정을 유지합니다. 한 번만 실행은 1 episode와 reset을 수행합니다.</p><div class="remote-run-actions"><button id="remote-run-start" class="button primary" type="submit" disabled>${icon("play")}수집 시작</button><button id="remote-run-once" class="button secondary" type="button" data-action="remote-run-once" disabled>${icon("step")}한 번만 실행</button><button id="remote-run-pause" class="button pause" type="button" data-action="remote-run-pause" disabled>${icon("pause")}Pause</button><button id="remote-run-resume" class="button secondary" type="button" data-action="remote-run-resume" disabled>${icon("play")}재개</button><button id="remote-run-stop" class="button danger" type="button" data-action="remote-run-stop" disabled>${icon("stop")}중지</button></div><p id="remote-run-note" class="field-note" role="status"></p></form><section class="remote-run-live"><h3>최근 관측과 실행 상태</h3><div id="remote-live-images" class="remote-live-images"><div class="remote-live-empty">client가 전달한 현재 프로젝트의 관측만 표시합니다.</div></div><p id="remote-run-status" class="remote-run-status" role="status">Run 대기 중</p><dl class="remote-run-progress"><div><dt>완료 / 목표 episode</dt><dd id="remote-run-episodes">—</dd></div><div><dt>Task / Reset steps</dt><dd id="remote-run-steps">—</dd></div><div><dt>Task / Reset GPT 호출</dt><dd id="remote-run-calls">—</dd></div><div><dt>추정 API 비용</dt><dd id="remote-run-cost">—</dd></div><div><dt>Run</dt><dd id="remote-run-id">—</dd></div><div><dt>Episode</dt><dd id="remote-run-episode">—</dd></div><div><dt>현재 run 정책</dt><dd id="remote-run-policy">—</dd></div><div><dt>현재 run 한도</dt><dd id="remote-run-limits">—</dd></div></dl><p class="field-note">Pause·중지는 client의 로컬 정지 처리를 요청합니다. 물리 비상정지와 controller watchdog은 로봇 PC에서 유지해야 합니다.</p></section></div><p id="remote-policy-error" class="form-error" role="alert" hidden></p></div></section>`;
  }

  function editRemoteRunField(input) {
    const session = state.remotePolicy;
    if (!remotePolicyIsCurrent(session) || !input.name || !Object.hasOwn(session.draft,input.name)) return;
    session.draft[input.name] = input.name === "training" ? input.checked : input.value;
    if (input.name === "policy" && input.value !== "residual") session.draft.training = false;
    if (input.name === "residual_model_id") {const model = session.data?.models?.find(item => item.id === input.value);if (model?.base_model_id && session.data.models.some(item => item.id === model.base_model_id && item.kind === "bc")) session.draft.model_id = model.base_model_id;}
    session.dirty = true;session.error = null;updateRemotePolicyPanel();
  }

  function updateRemotePolicyPanel() {
    const session = state.remotePolicy;
    if (!remotePolicyIsCurrent(session) || !$("remote-policy-panel")) return;
    const data = session.data, client = data?.client, run = data?.run, draft = session.draft, active = remoteRunActive(run), pending = state.remotePolicyMutation?.projectId === session.projectId ? state.remotePolicyMutation.action : null;
    const unavailable = !state.connected || !data || session.stale;
    $("remote-client-state").textContent = unavailable ? data ? "상태 확인 필요" : "상태 확인 중" : client?.connected ? "Client 연결됨" : "Client 연결 안 됨";
    $("remote-client-state").className = `tag ${unavailable ? "neutral" : client?.connected ? "success" : "neutral"}`;
    $("remote-client-id").textContent = client?.client_id || "—";
    $("remote-client-phase").textContent = client?.phase || "—";
    $("remote-client-heartbeat").textContent = client?.last_seen ? date(client.last_seen,{second:"2-digit"}) : "—";
    $("remote-client-session").textContent = client?.session_id || "—";
    $("remote-client-protocol").textContent = data?.protocol || "—";
    $("remote-client-environment").textContent = client?.environment?.name || client?.environment?.backend || "—";
    $("remote-contract").textContent = client ? JSON.stringify({environment:client.environment,capabilities:client.capabilities},null,2) : "client가 연결되면 확인할 수 있습니다.";
    if (!session.baseUrlDirty) $("remote-server-url").value = session.baseUrl;
    $("remote-token-issue").disabled = Boolean(remotePolicyBlockReason("token",session));
    $("remote-token-issue").textContent = pending === "token" ? "토큰 발급 중…" : data?.token_configured ? "연결 토큰 재발급" : "연결 토큰 발급";
    $("remote-config-download").disabled = !session.issuedConfig || !remoteServerUrl(session.baseUrl);
    $("remote-status-refresh").disabled = Boolean(session.busy || pending);
    $("remote-token-note").textContent = session.issuedConfig ? "토큰을 발급했습니다. 화면을 떠나기 전에 연결 설정을 다운로드하세요. 파일에는 토큰이 포함되며 로봇 PC에서 chmod 600으로 권한을 설정해야 합니다." : data?.token_configured ? "이미 토큰이 발급되어 있습니다. 기존 토큰은 다시 조회하지 않습니다. 재발급하면 기존 client 세션과 연결 설정을 사용할 수 없습니다." : "발급한 토큰은 이 화면에서만 다운로드할 수 있습니다. 설정 파일에는 인증 토큰이 포함되므로 로봇 PC에서 chmod 600으로 권한을 설정하세요.";
    const locked = unavailable || active || Boolean(pending);
    $("remote-run-fields").disabled = locked;
    if (![...$("remote-device").options].some(option => option.value === draft.device)) $("remote-device").add(new Option(draft.device,draft.device));
    for (const key of Object.keys(draft)) {const input = $("remote-run-form").elements.namedItem(key);if (input) {input.disabled = locked;if (key === "training") input.checked = draft.training === true;else if (!["model_id","residual_model_id"].includes(key)) input.value = draft[key] ?? "";}}
    const models = data?.models || [], signature = JSON.stringify(models.map(model => [model.id,model.name,model.kind,model.base_model_id]));
    if (signature !== session.modelSignature) {session.modelSignature = signature;for (const [id,kind] of [["remote-model-id","bc"],["remote-residual-model-id","residual"]]) $(id).innerHTML = `<option value="">${models.some(model => model.kind === kind) ? "모델을 선택하세요" : "등록된 모델이 없습니다"}</option>`+models.filter(model => model.kind === kind).map(model => `<option value="${esc(model.id)}">${esc(model.name || model.id)}</option>`).join("");}
    $("remote-model-id").value = draft.model_id || "";$("remote-residual-model-id").value = draft.residual_model_id || "";
    const policyChoices = [...$("remote-policy-kind").options];
    const dummyOption = policyChoices.find(option => option.value === "dummy");if (dummyOption) dummyOption.disabled = client?.capabilities?.simulation !== true;
    const residualOption = policyChoices.find(option => option.value === "residual");if (residualOption) residualOption.disabled = client?.capabilities?.residual_step !== true;
    $("remote-model-field").hidden = !["dp","residual"].includes(draft.policy);$("remote-residual-field").hidden = draft.policy !== "residual";
    $("remote-training-field").hidden = draft.policy !== "residual" || !remoteResidualTrainingAvailable(session);
    $("remote-training").disabled = locked || $("remote-training-field").hidden;
    $("remote-policy-note").textContent = draft.policy === "gpt" ? `GPT 설정: ${policyLabel(nextPolicy(currentProject()))}. 프로젝트 설정에서 변경할 수 있습니다.` : draft.policy === "dummy" ? "Mock는 시뮬레이션 연결과 메시지 흐름을 확인하는 테스트 정책입니다." : "선택한 checkpoint와 client의 카메라·상태·action 계약을 서버가 검증합니다. action 차원이 같다는 이유만으로 호환되는 것은 아닙니다.";
    const startReason = remotePolicyBlockReason("start",session);
    $("remote-run-start").disabled = Boolean(startReason);$("remote-run-once").disabled = Boolean(startReason);
    for (const action of ["pause","resume","stop"]) {const reason = remotePolicyBlockReason(action,session);$("remote-run-"+action).disabled = Boolean(reason);$("remote-run-"+action).title = reason;}
    $("remote-run-start").innerHTML = icon("play")+(pending === "start" ? "시작 요청 중…" : "수집 시작");
    $("remote-run-note").textContent = active ? "현재 run의 설정은 고정됩니다. Pause 후 재개해도 기존 목표와 정책을 유지합니다." : startReason || "새 run을 시작할 준비가 되었습니다. 버튼을 누르기 전까지 정책 실행을 요청하지 않습니다.";
    const statusLabels = {preparing:"정책 준비 중",running:"실행 중",pausing:"일시정지 중",paused:"일시정지",stopping:"중지 중",stopped:"중지됨",completed:"완료",error:"확인 필요"};
    const phaseLabels = {task:"Task",reset:"Reset",arm_reset:"팔 복구",awaiting_episode:"다음 episode 대기"};
    $("remote-run-status").textContent = run ? `${statusLabels[run.status] || run.status} · ${phaseLabels[run.phase] || run.phase || "—"}${client?.connected ? "" : " · client 연결 확인 필요"}` : "Run 대기 중";
    $("remote-run-episodes").textContent = run ? `${count(run.completed_episodes)} / ${count(run.target_episodes)}` : "—";
    $("remote-run-steps").textContent = run ? `${count(run.task_steps)} / ${count(run.reset_steps)}` : "—";
    $("remote-run-calls").textContent = run ? `${count(run.task_calls)} / ${count(run.reset_calls)}` : "—";
    $("remote-run-cost").textContent = run ? money(run.estimated_cost_usd,4) : "—";
    $("remote-run-id").textContent = run?.id || "—";$("remote-run-episode").textContent = run?.episode_id || "—";
    const runConfig = run?.config || {}, policyName = {gpt:"GPT",dp:"Diffusion Policy",residual:"Residual TD3",dummy:"Mock"}[run?.policy || runConfig.policy];
    $("remote-run-policy").textContent = run ? [policyName || "—",runConfig.residual_model_id || runConfig.model_id,runConfig.device].filter(Boolean).join(" · ") : "—";
    $("remote-run-limits").textContent = run?.config ? `Task ${count(runConfig.max_episode_steps)} / Reset ${count(runConfig.max_reset_steps)} steps · ${money(runConfig.budget_usd)}` : "—";
    const error = session.error || session.readError || run?.error;
    $("remote-policy-error").textContent = typeof error === "string" ? error : error ? JSON.stringify(error) : "";$("remote-policy-error").hidden = !error;
    updateRemoteLaunchSettings(session);updateRemotePolicyFrames(session);
    if ($("remote-token-dialog")?.open) $("remote-token-confirm").disabled = Boolean(remotePolicyBlockReason("token",session));
  }

  function updateRemotePolicyFrames(session) {
    const prefix = remotePolicyPath(session)+"/frame/";
    const frames = (session.data?.frames || []).filter(frame => typeof frame.camera === "string" && !frame.camera.startsWith("reset_reference") && typeof frame.url === "string" && frame.url.startsWith(prefix) && !frame.url.includes(".."));
    const context = `${session.data?.client?.session_id || ""}:${session.data?.run?.id || ""}`;
    const signature = JSON.stringify([context,frames]);
    if (signature === session.frameSignature) return;
    session.frameSignature = signature;const token = ++session.frameToken;
    const cameras = JSON.stringify([context,frames.map(frame => frame.camera)]);
    if (cameras !== session.cameraSignature) {session.cameraSignature = cameras;$("remote-live-images").innerHTML = frames.length ? frames.map((frame,index) => `<figure><img id="remote-frame-${index}" alt="${esc(frame.camera)} 최근 관측"><figcaption>${esc(frame.camera)}</figcaption></figure>`).join("") : '<div class="remote-live-empty">아직 client가 전달한 관측이 없습니다.</div>';}
    frames.forEach((frame,index) => {const img = new Image();img.onload = () => {if (!remotePolicyIsCurrent(session) || token !== session.frameToken) return;const target = $("remote-frame-"+index);if (target) target.src = frame.url;};img.onerror = () => {if (remotePolicyIsCurrent(session) && token === session.frameToken) session.frameSignature = "";};img.src = frame.url;});
  }

  function installRemotePolicyResponse(session,response) {
    if (!response || typeof response !== "object" || response.project_id && response.project_id !== session.projectId || !response.capabilities || !Array.isArray(response.models)) throw new Error("로봇 client 상태 응답을 확인할 수 없습니다.");
    session.data = response;session.stale = false;
    if (!session.dirty) session.draft = remoteRunConfig(response.defaults);
    if (!session.launchDirty && response.launch_config) session.launchDraft = remoteLaunchConfig(response.launch_config);
    if (!session.baseUrlDirty) session.baseUrl = response.client_url || "";
  }

  function scheduleRemotePolicyRefresh(session) {
    clearTimeout(session.timer);
    if (remotePolicyIsCurrent(session)) session.timer = setTimeout(refreshRemotePolicy,1500);
  }

  async function refreshRemotePolicy() {
    const session = state.remotePolicy;if (!remotePolicyIsCurrent(session) || session.busy) return;
    clearTimeout(session.timer);
    if (state.remotePolicyMutation?.projectId === session.projectId) {scheduleRemotePolicyRefresh(session);return;}
    const request = ++session.request;session.busy = true;session.controller = new AbortController();updateRemotePolicyPanel();
    try {const response = await api(remotePolicyPath(session),undefined,8000,session.controller.signal);if (!remotePolicyIsCurrent(session) || request !== session.request) return;installRemotePolicyResponse(session,response);session.readError = null;}
    catch (error) {if (remotePolicyIsCurrent(session) && request === session.request) {session.stale = true;session.readError = error.status === 404 && error.detail === "Not Found" ? "현재 서버에 로봇 client 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요." : error.message;}}
    finally {if (request === session.request) {session.busy = false;session.controller = null;if (remotePolicyIsCurrent(session)) {updateRemotePolicyPanel();scheduleRemotePolicyRefresh(session);}}}
  }

  function requestRemoteToken() {
    const session = state.remotePolicy;if (remotePolicyBlockReason("token",session)) return;
    if (!remoteServerUrl(session.baseUrl)) {session.error = "로봇 PC에서 접근 가능한 http(s)://서버주소:포트/robot/v1 URL을 입력해주세요.";updateRemotePolicyPanel();return;}
    if (session.data.token_configured) {$("remote-token-dialog").showModal();updateRemotePolicyPanel();}
    else remotePolicyCommand("token");
  }

  async function remotePolicyCommand(action,event,once = false) {
    event?.preventDefault();
    const session = state.remotePolicy;
    if (!["token","start","pause","resume","stop"].includes(action) || remotePolicyBlockReason(action,session)) return;
    let payload = {};
    if (action === "start") {
      payload = remoteRunConfig(session.draft);if (once) payload.episodes = 1;
      if (!["dp","residual"].includes(payload.policy)) payload.model_id = null;
      if (payload.policy !== "residual") {payload.residual_model_id = null;payload.training = false;}
      const error = remoteRunConfigError(payload,session);if (error) {session.error = error;updateRemotePolicyPanel();return;}
    }
    if (action === "token" && !remoteServerUrl(session.baseUrl)) {session.error = "정책 서버 URL을 확인해주세요.";updateRemotePolicyPanel();return;}
    const mutation = {projectId:session.projectId,action};state.remotePolicyMutation = mutation;session.error = null;
    clearTimeout(session.timer);session.controller?.abort();session.request += 1;session.busy = false;session.controller = null;updateRemotePolicyPanel();
    try {
      const response = await api(remotePolicyPath(session)+"/"+action,payload,30000);
      if (!remotePolicyIsCurrent(session)) return;
      if (action === "token") {
        if (typeof response.token !== "string" || !response.token || response.client_config?.project_id && response.client_config.project_id !== session.projectId) throw new Error("발급 응답의 연결 토큰과 프로젝트를 확인할 수 없습니다. 상태를 확인하고 다시 발급해주세요.");
        const config = response.client_config || {};
        session.issuedConfig = {base_url:remoteServerUrl(session.baseUrl),project_token:response.token,client_id:config.client_id || "robot-pc-1",journal_path:config.journal_path || "./state/journal.sqlite3",control_hz:config.control_hz ?? 20,history_capacity:config.history_capacity ?? 64,request_timeout_seconds:config.request_timeout_seconds ?? 30,adapter_options:config.adapter_options || {}};
        session.data.token_configured = true;if ($("remote-token-dialog").open) $("remote-token-dialog").close();
        toast("연결 토큰을 발급했습니다. 연결 설정을 다운로드하세요.");
      } else {installRemotePolicyResponse(session,response);toast({start:"원격 run 시작을 요청했습니다.",pause:"Client에 일시정지를 요청했습니다.",resume:"기존 run의 재개를 요청했습니다.",stop:"Client에 중지를 요청했습니다."}[action]);}
    } catch (error) {if (remotePolicyIsCurrent(session)) {session.error = error.status === 404 && error.detail === "Not Found" ? "현재 서버에 로봇 client 기능이 반영되지 않았습니다. RILO 서버를 재시작한 뒤 새로고침해주세요." : error.message;session.stale = true;}}
    finally {if (state.remotePolicyMutation === mutation) state.remotePolicyMutation = null;if (remotePolicyIsCurrent(session)) {updateRemotePolicyPanel();scheduleRemotePolicyRefresh(session);}else if (remotePolicyIsCurrent()) updateRemotePolicyPanel();}
  }

  function downloadRemoteConfig() {
    const session = state.remotePolicy;if (!remotePolicyIsCurrent(session) || !session.issuedConfig) return;
    const baseUrl = remoteServerUrl(session.baseUrl);if (!baseUrl) {session.error = "정책 서버 URL을 확인해주세요.";updateRemotePolicyPanel();return;}
    const data = {...session.issuedConfig,base_url:baseUrl};
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+"\n"],{type:"application/json"}));
    const link = document.createElement("a");link.href = url;link.download = `rilo-client-${session.projectId}.json`;link.click();setTimeout(() => URL.revokeObjectURL(url),1000);
  }

  function updateCollectionFrames() {
    const holder = $("live-images"), project = currentProject(), status = state.status || {};
    if (state.route.tab !== "collection" || !holder || !project) return;
    const own = status.active_project_id === project.id, runId = status.run_id || null;
    let view = state.collectionFrames;
    const sameSource = view && view.holder === holder && view.routeToken === state.routeToken
      && own && view.projectId === project.id && view.runId === runId;
    if (!sameSource) {state.collectionFrames = null;view = null;}
    const footer = $("observation-footer"), observation = own ? status.observation : null;
    const waiting = !state.connected || status.phase === "initializing" || !observation?.id;
    if (!own || waiting) {
      if (view) {
        // A reset/offline transition invalidates pending callbacks, but keeps decoded pixels.
        view.slots.forEach(slot => {slot.pending = null;});
        footer.textContent = !state.connected ? "서버 연결 확인 중 · 마지막 수집 화면을 유지합니다."
          : view.slots.some(slot => slot.displayed) ? "새 관측 준비 중 · 같은 run 또는 환경의 마지막 화면을 유지합니다."
          : "카메라 관측 수신 대기 중입니다.";
        return;
      }
      if (!view || holder.dataset.framePlaceholder !== "true") {
        holder.innerHTML = `<div class="live-placeholder">${icon("camera")}<h3>${own ? "카메라 관측을 기다리고 있습니다" : "수집 환경을 연결하세요"}</h3><p>${own ? "관측 이미지가 준비되면 화면을 표시합니다." : "현재 프로젝트에 연결된 실시간 관측이 없습니다.<br>환경 연결 후 카메라를 확인할 수 있습니다."}</p></div>`;
        holder.dataset.framePlaceholder = "true";
      }
      footer.textContent = "카메라 대기 중 · 현재 프로젝트의 관측만 표시합니다.";
      return;
    }
    const cameras = (Array.isArray(observation.cameras) ? observation.cameras.map(item => typeof item === "string" ? item : item.name) : Object.keys(observation.cameras || {})).filter(name => typeof name === "string" && name && !/^reset_reference(?:_|$)/.test(name));
    const signature = JSON.stringify(cameras), observationId = observation.id;
    if (!view || signature !== view.cameraSignature || holder.dataset.framePlaceholder === "true") {
      holder.innerHTML = cameras.length ? `<div class="camera-grid ${cameras.length === 1 ? "single" : ""}">${cameras.map((camera,index) => `<div class="camera"><span class="camera-label">${esc(camera)}</span><div id="live-camera-slot-${index}" class="collection-frame-slot"><span>관측 수신 대기</span></div></div>`).join("")}</div>` : '<div class="live-placeholder"><p>카메라 관측을 기다리고 있습니다.</p></div>';
      holder.dataset.framePlaceholder = "false";
      view = {holder,projectId:project.id,runId,routeToken:state.routeToken,cameraSignature:signature,
        slots:cameras.map((camera,index) => ({camera,element:$("live-camera-slot-"+index),pending:null,observationId:null,displayed:false,failed:false}))};
      state.collectionFrames = view;
    }
    const currentSource = () => state.collectionFrames === view && state.route.tab === "collection"
      && state.routeToken === view.routeToken && $("live-images") === holder && state.route.id === view.projectId
      && state.status?.active_project_id === view.projectId && (state.status?.run_id || null) === view.runId;
    const updateFooter = () => {
      if (!currentSource()) return;
      footer.textContent = view.slots.some(slot => slot.failed) ? "이미지 갱신 재시도 중 · 마지막 수집 화면을 유지합니다."
        : `상태 관측 ${date(state.status?.observation?.timestamp,{second:"2-digit"})} · 카메라는 같은 ${view.runId ? "run" : "환경"}의 최신 화면입니다.`;
    };
    view.slots.forEach((slot,index) => {
      // Keep one request per camera in flight. Faster status polls must not starve a slow image.
      if (slot.pending || slot.observationId === observationId) return;
      const frame = new Image();
      frame.id = "live-camera-"+index;frame.alt = `${slot.camera} 현재 관측`;
      slot.pending = frame;
      frame.onload = () => {
        if (slot.pending !== frame) return;
        slot.pending = null;
        if (!currentSource() || !state.connected || state.status?.phase === "initializing") return;
        slot.element.replaceChildren(frame);
        slot.observationId = observationId;slot.displayed = true;slot.failed = false;
        updateFooter();
      };
      frame.onerror = () => {
        if (slot.pending !== frame) return;
        slot.pending = null;slot.failed = true;
        // Do not replace the displayed image with a broken image; retry even while paused.
        updateFooter();
      };
      // The runner advances between status and JPEG requests. Scope to the owner/run,
      // using the observation ID only to bypass caches, rather than requiring an exact tick.
      frame.src = `/api/frame/${encodeURIComponent(slot.camera)}?project_id=${encodeURIComponent(view.projectId)}${view.runId ? `&run_id=${encodeURIComponent(view.runId)}` : ""}&t=${encodeURIComponent(observationId)}`;
    });
    updateFooter();
  }

  function renderCollection(project) {
    if (remoteRobotProject(project)) {renderRemoteRobotConnection(project);return;}
    state.lastObservation = null;state.lastCameras = "";state.lastEvents = "";state.collectionFrames = null;
    $("tab-content").innerHTML = `${renderCollectionSettings(project)}<div class="collection-layout"><div><section class="panel"><div class="panel-header"><div><h2>실시간 수집</h2><p id="live-subtitle">환경을 연결한 뒤 관측을 확인하세요.</p></div><span id="live-state" class="tag neutral">준비 전</span></div><div id="live-images"><div class="live-placeholder">${icon("camera")}<h3>수집 환경을 연결하세요</h3><p>환경 연결 후 현재 카메라를 확인할 수 있습니다.<br>수집 시작을 누르기 전까지 GPT는 호출되지 않습니다.</p></div></div><div id="observation-footer" class="observation-footer"><span>카메라 대기 중</span><span>현재 프로젝트의 관측만 표시합니다.</span></div><div class="execution-meta"><div><span>현재 단계</span><strong id="live-stage">—</strong></div><div><span>Task / Reset step</span><strong id="live-steps">0 / 0</strong></div><div><span>실행 API 비용</span><strong id="live-cost">—</strong></div></div><div class="collection-record-link">${icon("database")}<span>GPT 판단 요약과 action 기록은 <a href="${routeUrl(project.id,"data")}">데이터 → Episode 상세</a>에서 확인하세요.</span></div></section><section class="panel events-panel"><div class="panel-header"><h2>실행 이벤트</h2><span id="events-count" class="tag neutral">0</span></div><ol id="events-list" class="events-list"><li class="empty-event">이 프로젝트의 실행 이벤트가 여기에 표시됩니다.</li></ol></section></div><aside class="panel control-panel"><div class="panel-header"><h2>수집 제어</h2><a class="icon-button" href="#collection-settings-panel" aria-label="수집 실행 설정" data-action="collection-settings-focus">${icon("settings")}</a></div><div class="panel-body"><section class="control-section"><span class="control-label">01 · 환경 연결</span><button id="initialize-project" class="button secondary full" data-action="connect" disabled>${icon("robot")}환경 연결</button><p id="initialize-note" class="control-note">환경 관측만 연결합니다. Run 생성과 GPT 호출은 수집 시작 시 수행합니다.</p></section><section class="control-section"><span class="control-label">02 · 데이터 수집</span><button id="start-collection" class="button primary full" data-action="start" disabled>${icon("play")}수집 시작</button><div class="control-row"><button id="pause-collection" class="button pause" data-action="pause" disabled>${icon("pause")}Pause</button><button id="step-collection" class="button secondary" data-action="step" disabled>${icon("step")}한 번만 실행</button></div><button id="stop-collection" class="button danger full" data-action="stop" style="margin-top:9px" disabled>${icon("stop")}수집 중지</button><p id="control-note" class="control-note">환경을 먼저 연결해주세요.</p></section><section class="control-section"><dl class="session-rows"><div><dt id="live-goal-label">실행 목표</dt><dd id="live-goal">${n(project.run_config?.episodes) || 1} episodes / run</dd></div><div id="next-goal-row" hidden><dt>다음 run 목표</dt><dd id="next-goal">—</dd></div><div><dt>Reset</dt><dd id="live-reset-mode">${esc(resetModeLabel(project.run_config?.reset_mode))}</dd></div><div><dt>비용 한도</dt><dd id="live-budget">${money(project.run_config?.budget_usd)}</dd></div><div><dt>Teacher</dt><dd id="teacher-name">—</dd></div><div><dt>Reasoning</dt><dd id="teacher-effort">—</dd></div><div id="next-policy-row" hidden><dt>다음 모델 설정</dt><dd id="next-policy">—</dd></div><div><dt>입력 / 출력 tokens</dt><dd id="live-tokens">—</dd></div><div><dt>Run</dt><dd id="live-run">—</dd></div></dl></section></div></aside></div>`;
    updateCollection();
    refreshPipeline(true);
  }

  function updateCollection() {
    if (state.route.tab === "collection" && remoteRobotProject()) {updateRemoteRobotConnection();updateRemotePolicyPanel();return;}
    if (state.route.tab !== "collection" || !$("initialize-project")) return;
    const s = state.status || {}, p = currentProject();
    if (!p) return;
    updateCollectionSettings();
    const own = s.active_project_id === p.id;
    const learning = pipelineRuntime(), managed = learning && pipelineActive(learning.phase), managedOwn = managed && learning.project_id === p.id;
    const preparingOwn = state.pending === "connect" && state.pendingProjectId === p.id;
    const another = !own && !preparingOwn && activePhases.has(s.phase);
    const envConnected = own && s.environment_connected === true;
    const ep = own ? s.episode || {} : {};
    const cycleDone = Boolean(ep.outcome && (s.config?.reset_mode === "environment" || ep.reset_outcome));
    const batchDone = cycleDone && (n(ep.index) >= n(s.config?.episodes) || s.config?.reset_mode !== "environment" && ep.reset_outcome !== "success");
    const newRunReady = own && s.can_start_new_run === true;
    const resumable = own && ["ready","paused"].includes(s.phase) && !batchDone;
    const ready = newRunReady || resumable;
    const reconnect = own && ["stopped","error","completed"].includes(s.phase) && !newRunReady;
    const canConnect = !another && (!envConnected || reconnect) && (!own || !activePhases.has(s.phase));
    const runnable = p.runnable !== false && state.detail.runnable !== false;
    const locked = Boolean(state.pending || state.pausePending || !state.connected || managed);
    $("initialize-project").disabled = locked || !runnable || !canConnect;
    $("initialize-project").innerHTML = icon("robot") + (state.pending === "connect" ? "환경 연결 중…" : reconnect ? "환경 다시 연결" : envConnected ? "환경 연결 완료" : "환경 연결");
    $("initialize-note").textContent = reconnect ? "현재 장면을 초기화하고 환경을 다시 연결합니다. 수집은 시작하지 않습니다." : envConnected ? "환경이 연결되어 있습니다. 설정 변경은 다음 새 run 시작 시 적용됩니다." : "환경 관측만 연결합니다. Run 생성과 GPT 호출은 수집 시작 시 수행합니다.";
    const hasPolicy = s.policy !== "openai" || s.api_key_configured;
    $("start-collection").disabled = locked || state.collectionSettingsSaving || !ready || !hasPolicy;
    $("start-collection").innerHTML = icon("play") + (own && s.phase === "paused" && resumable ? "수집 재개" : "수집 시작");
    $("pause-collection").disabled = state.pausePending || !state.connected || !(own || preparingOwn) || !pausePhases.has(s.phase);
    $("pause-collection").innerHTML = icon("pause") + (state.pausePending ? "요청 중…" : "Pause");
    $("step-collection").disabled = locked || state.collectionSettingsSaving || !hasPolicy || !(newRunReady || own && s.phase === "ready" && s.run_started === false);
    $("step-collection").innerHTML = icon("step") + "한 번만 실행";
    $("stop-collection").disabled = locked || !own || !activePhases.has(s.phase);
    const note = state.pausePending ? "일시정지를 요청했습니다. 현재 control tick 또는 환경 작업이 끝나면 적용됩니다." : !runnable ? p.runnable_reason || state.detail.runnable_reason || "현재 서버에서 이 프로젝트 환경을 실행할 수 없습니다." : another ? "다른 실행에서 환경을 사용 중입니다. 해당 실행을 중지한 뒤 연결할 수 있습니다." : !hasPolicy ? "서버에 API key가 설정되지 않았습니다. 환경 관측만 확인할 수 있습니다." : own && s.error ? s.error : own && s.phase === "paused" ? "일시정지 상태입니다. 재개하면 현재 run의 모델·목표·reset 설정을 유지합니다." : own && s.phase === "thinking" ? "GPT가 다음 동작 또는 수집 기준을 판단하고 있습니다. Pause하면 대기 중 응답을 폐기합니다." : own && s.phase === "executing" ? "Action을 실행하고 있습니다. Pause는 현재 control tick이 끝난 뒤 적용됩니다." : ready ? `수집 시작은 설정된 ${n(p.run_config?.episodes) || 1} episodes를, 한 번만 실행은 1 episode와 reset을 수행합니다. 완료 후 자동으로 다음 run을 시작하지 않습니다.` : reconnect ? "안전하게 이어갈 수 없는 상태입니다. 환경 다시 연결로 현재 장면을 초기화한 뒤 새 run을 시작하세요." : "환경을 먼저 연결해주세요.";
    $("control-note").textContent = note;$("control-note").classList.toggle("alert",Boolean(!runnable || another || !hasPolicy || own && s.error));
    $("pause-collection").dataset.action = managedOwn ? "pipeline-pause" : "pause";
    $("stop-collection").dataset.action = managedOwn ? "pipeline-stop" : "stop";
    if (managedOwn) {
      $("pause-collection").disabled = learning.phase !== "running" || state.pipelinePausePending || !state.connected;
      $("pause-collection").innerHTML = icon("pause") + (state.pipelinePausePending ? "요청 중…" : "Pause");
      $("stop-collection").disabled = Boolean(state.pipelinePending || !state.connected);
      $("control-note").textContent = `${pipelineStageName(learning.stage)} 단계가 파이프라인에서 ${learning.phase === "paused" ? "일시정지되어 있습니다" : "진행 중입니다"}. 전체 실행의 Pause·재개는 개요에서 관리하세요.`;
      $("control-note").classList.remove("alert");
      $("initialize-note").textContent = "파이프라인이 환경을 관리합니다. 학습 중에는 수집을 멈추고 다음 rollout에서 이어갑니다.";
    } else if (managed) $("control-note").textContent = "다른 프로젝트의 파이프라인이 환경을 사용 중입니다. 개요에서 실행 상태를 확인하세요.";
    $("live-state").className = `tag ${own && (activePhases.has(s.phase) || envConnected) ? s.phase === "paused" ? "warning" : "success" : "neutral"}`;
    $("live-state").textContent = own ? s.phase === "thinking" && s.stage !== "reset" && ["bc_eval","rl","rl_eval"].includes(s.learning?.pipeline_stage) ? "학습 정책 action 계산 중" : phaseNames[s.phase] || s.phase : "준비 전";
    $("live-subtitle").textContent = own && s.run_id ? `Episode ${episodeIndex(ep.display_index ? ep : findEpisode(s.run_id,ep.id) || ep)} · run ${n(ep.index) || 1} / ${n(s.config?.episodes) || 1} · ${s.stage === "reset" ? "GPT가 환경을 복구하고 있습니다." : "Task를 수행하는 관측과 실행 기록"}` : envConnected ? "환경 연결됨 · 설정을 저장한 뒤 수집을 시작하세요." : "환경을 연결한 뒤 관측을 확인하세요.";
    $("live-stage").textContent = own && s.run_id && s.observation ? s.stage === "reset" ? "GPT Reset" : s.learning?.pipeline_stage && s.learning.pipeline_stage !== "collect" ? pipelineStageName(s.learning.pipeline_stage) : "Task" : "—";
    const shownRun = own && s.run_id ? s.config || {} : p.run_config || {};
    $("live-goal").textContent = `${n(shownRun.episodes) || 1} episodes / run`;
    $("live-goal-label").textContent = own && s.run_id ? activePhases.has(s.phase) ? "현재 run 목표" : "마지막 run 목표" : "수집 목표";
    $("next-goal-row").hidden = !(own && s.run_id && n(shownRun.episodes) !== n(p.run_config?.episodes));
    $("next-goal").textContent = `${n(p.run_config?.episodes) || 1} episodes / run`;
    $("next-goal-row").title = s.run_started ? "시작한 run은 Pause·재개 시에도 현재 목표를 유지합니다." : "다음 새 run에 적용되는 저장된 목표입니다.";
    $("live-reset-mode").textContent = resetModeLabel(shownRun.reset_mode);
    $("live-budget").textContent = money(shownRun.budget_usd);
    $("live-steps").textContent = `${count(ep.steps)} / ${count(ep.reset_steps)}`;
    $("live-cost").textContent = own ? money(s.stats?.estimated_cost_usd,4) : "—";
    const selectedPolicy = nextPolicy(p), shownPolicy = own && s.run_id ? s : selectedPolicy;
    $("teacher-name").textContent = s.policy === "scripted" ? "Scripted demo" : shownPolicy.model || "—";
    $("teacher-effort").textContent = s.policy === "scripted" ? "—" : effortLabel(shownPolicy.reasoning_effort,shownPolicy.current_reasoning_effort);
    $("next-policy-row").hidden = !(own && s.run_id && s.policy !== "scripted" && (s.model !== selectedPolicy.model || s.reasoning_effort !== selectedPolicy.reasoning_effort));
    $("next-policy").textContent = policyLabel(selectedPolicy);
    $("live-tokens").textContent = own ? `${count(s.stats?.input_tokens)} / ${count(s.stats?.output_tokens)}` : "—";
    $("live-run").textContent = own ? s.run_id || "—" : "—";
    updateCollectionFrames();
    const events = own ? (s.events || []).filter(event => (!event.run_id || event.run_id === s.run_id) && ["ready","episode_start","episode","reset","budget","control","error"].includes(event.kind) && !/Discarded a response/.test(event.message || "")) : [];
    const eventSignature = JSON.stringify(events.map(x => x.id));
    if (eventSignature !== state.lastEvents) {
      $("events-count").textContent = count(events.length);
      $("events-list").innerHTML = events.length ? events.slice(-8).reverse().map(event => `<li><time>${esc(new Date(event.time*1000).toLocaleTimeString("ko-KR",{hour12:false,hour:"2-digit",minute:"2-digit",second:"2-digit"}))}</time><p>${esc(dashboardEventMessage(event))}</p></li>`).join("") : '<li class="empty-event">이 프로젝트의 실행 이벤트가 여기에 표시됩니다.</li>';
      state.lastEvents = eventSignature;
    }
  }

  function dashboardEventMessage(event) {
    const message = event.message || "";
    let match = message.match(/^Episode (\d+) started/);
    if (match) return `Episode ${episodeIndex(findEpisode(event.run_id,event.episode_id) || {index:Number(match[1])})} 시작`;
    match = message.match(/^Episode (\d+) ended: (.+)\.$/);
    if (match) return `Episode ${episodeIndex(findEpisode(event.run_id,event.episode_id) || {index:Number(match[1])})} Task ${match[2] === "success" ? "성공" : `종료 (${match[2]})`}`;
    if (message.startsWith("GPT reset started")) return "GPT reset 시작 · 현재 환경에서 복구합니다.";
    if (message.startsWith("GPT reset ended: success")) return "GPT reset 완료 · 환경 복구 검증을 통과했습니다.";
    if (message.startsWith("GPT reset ended:")) return "GPT reset 미완료 · 환경을 확인해주세요.";
    if (message.startsWith("Environment initialized")) return "Run 준비 완료 · 설정을 실행 기록에 저장했습니다.";
    if (message.startsWith("Environment connected")) return "환경 연결 완료 · 수집 시작 전까지 GPT를 호출하지 않습니다.";
    if (message === "Execution resumed.") return "수집 실행 / 재개";
    if (message.startsWith("Execution stopped")) return "수집 중지 · 실행 기록을 저장했습니다.";
    if (message.startsWith("Paused.")) return "일시정지 · 재개할 때 현재 상태를 다시 관측합니다.";
    if (message === "Single decision requested.") return "한 번 실행 요청";
    if (message.startsWith("Single decision complete.")) return "한 번 실행 완료 · 일시정지";
    if (message.startsWith("API cost budget reached.")) return "API 비용 한도 도달 · 환경을 유지하며 일시정지했습니다.";
    return message;
  }

  function renderProject() {
    const project = currentProject();
    if (!project) return;
    document.title = `${project.name} · RILO`;
    $("content").innerHTML = projectShell(project);
    renderBreadcrumb();
    ({overview:renderOverview,collection:renderCollection,data:renderData,models:renderModels,evaluation:renderEvaluation,settings:renderSettings}[state.route.tab] || renderOverview)(project);
  }

  async function refreshWorkspace(force = false) {
    if (state.workspaceBusy) return;
    state.workspaceBusy = true;
    try {
      const data = await api("/api/workspace");
      const signature = JSON.stringify([data.projects,data.archived_projects,data.totals]);
      const changed = signature !== state.workspaceSignature;
      state.workspace = data;state.workspaceSignature = signature;
      renderNav();renderGlobal();
      if (!state.route.id && (changed || force) && document.activeElement?.id !== "project-search") renderHome();
      if (state.route.id && (force || !["collection","models","evaluation"].includes(state.route.tab))) await loadDetail(false);
      $("freshness").textContent = `최근 동기화 ${new Date().toLocaleTimeString("ko-KR",{hour:"2-digit",minute:"2-digit",hour12:false})}`;
      if (state.connected) showError(null);
    } catch (error) { if (!state.workspace || force) showError(error.message); }
    finally { state.workspaceBusy = false; }
  }

  async function refreshStatus(force = false) {
    if (state.statusBusy && !force) return;
    const requestId = ++state.statusRequest;
    state.statusBusy = true;
    try {
      const status = await api("/api/status",undefined,8000);
      if (requestId !== state.statusRequest) return;
      state.status = status;state.connected = true;
      $("connection").className = "connection online";$("connection").innerHTML = "<i></i>서버 연결됨";
      renderGlobal();updateCollection();updatePolicyStatus();updateEpisodeVideo();updatePipelineStartPreview();updateStageStartDialog();updateEvaluationLive();
    } catch {
      if (requestId !== state.statusRequest) return;
      state.connected = false;$("connection").className = "connection offline";$("connection").innerHTML = "<i></i>연결 확인 필요";
      renderGlobal();updateCollection();updateEpisodeVideo();updatePipelineStartPreview();updateStageStartDialog();updateEvaluationLive();
    } finally { if (requestId === state.statusRequest) state.statusBusy = false; }
  }

  async function loadDetail(force = true) {
    const id = state.route.id, token = state.routeToken;
    if (!id) return;
    try {
      const detail = await api(`/api/projects/${encodeURIComponent(id)}`);
      if (token !== state.routeToken || id !== state.route.id) return;
      const signature = JSON.stringify([detail.project,detail.episodes,detail.archived_episodes]);
      const changed = signature !== state.detailSignature;
      state.detail = detail;state.detailSignature = signature;
      if (force || changed && !["settings","collection","models","evaluation"].includes(state.route.tab)) renderProject();
      else if (state.route.tab === "collection") updateCollection();
      else if (state.route.tab === "settings") updateProjectManagement();
      else if (["models","evaluation"].includes(state.route.tab)) updatePipelineView();
      if (state.route.tab === "overview") refreshPipeline();
      if (state.episodeDetail && $("episode-dialog").open) refreshEpisodeVideoJob();
    } catch (error) {
      if (token !== state.routeToken) return;
      if (force) $("content").innerHTML = `<a class="back-link" href="#/">${icon("back")}워크스페이스로</a><section class="panel"><div class="empty-state">${esc(error.message)}</div></section>`;
      else toast(error.message,true);
    }
  }

  async function navigate() {
    stopRobotConnectionPolling();stopRemotePolicyPolling();
    const pieces = location.hash.replace(/^#\/?/,"").split("/");
    let id = null;
    try { id = pieces[0] === "projects" && pieces[1] ? decodeURIComponent(pieces[1]) : null; } catch { id = null; }
    const tabs = ["overview","collection","data","models","evaluation","settings"];
    const tab = tabs.includes(pieces[2]) ? pieces[2] : "overview";
    state.dataSelection = null;state.bulkEpisodeTarget = null;
    if ($("bulk-episode-dialog").open) $("bulk-episode-dialog").close();
    state.route = {id,tab};state.routeToken += 1;state.detailSignature = "";state.evaluationFrameToken += 1;state.evaluationCameraSignature = "";state.evaluationObservationKey = "";
    if ($("episode-dialog").open) $("episode-dialog").close();
    if (state.detail?.project?.id !== id) {state.detail = null;state.pipeline = null;state.pipelineProjectId = null;}
    state.collectionSettingsDirty = false;state.collectionSettingsSignature = "";state.collectionSettingsExpanded = false;
    if ($("pipeline-start-dialog").open) $("pipeline-start-dialog").close();
    if ($("stage-start-dialog").open) $("stage-start-dialog").close();
    if ($("bc-history-dialog").open) $("bc-history-dialog").close();
    closeBCHistory();
    state.trainingAlgorithms = {};
    renderNav();renderBreadcrumb();
    if (!id) { if (state.workspace) renderHome(); return; }
    $("content").innerHTML = '<div class="loading-view"><span class="spinner"></span><p>프로젝트를 불러오고 있어요.</p></div>';
    await loadDetail();
  }

  async function command(action, global = false) {
    if (!global && remoteRobotProject()) {toast(remoteConnectionOnlyReason,true);return;}
    if (action === "pause" ? state.pausePending : state.pending || state.pausePending) return;
    const id = global ? activeId() : state.route.id;
    const unownedPause = action === "pause" && (global && !id || state.pending === "connect" && state.pendingProjectId === id && activeId() !== id);
    if (!id && !unownedPause) return;
    let payload = {};
    if (action === "pause") state.pausePending = true;
    else {state.pending = action;state.pendingProjectId = id;}
    renderGlobal();updateCollection();
    try {
      await api(unownedPause ? "/api/pause" : `/api/projects/${encodeURIComponent(id)}/${action}`,payload,["connect","start","step"].includes(action) || action === "pause" && state.pending === "connect" ? 120000 : 15000);
      // A delayed Start response must never replace a newer Pause snapshot.
      await refreshStatus(true);
      const messages = {connect:"환경이 연결되었습니다. 설정을 확인한 뒤 수집을 시작하세요.",start:"설정한 목표로 run 하나를 시작합니다.",resume:"현재 run을 기존 설정으로 재개합니다.",pause:"일시정지했습니다.",step:"1 episode와 reset을 실행합니다. 저장된 수집 목표는 유지됩니다.",stop:"수집을 중지했습니다. 기록은 유지됩니다."};
      toast(messages[action]);
      await refreshWorkspace();
    } catch (error) { toast(error.message,true);await refreshStatus(true); }
    finally {
      if (action === "pause") state.pausePending = false;
      else {state.pending = null;state.pendingProjectId = null;}
      renderGlobal();updateCollection();
    }
  }

  function shortTask(task) {
    const text = task?.language || task?.name || "새 task";
    if (/open.*top.drawer/i.test(text)) return "상단 서랍";
    if (/open.*middle.drawer/i.test(text)) return "중간 서랍";
    return text.replaceAll("_"," ").slice(0,85);
  }

  function selectedCreateBackend() {
    return $("create-backend").value;
  }

  function updateCreateTask() {
    const backend = selectedCreateBackend(), remote = backend === "remote_robot";
    const task = remote ? {name:$("create-remote-task").value.trim() || "로봇 PC 연결"} : state.tasks.find(x => String(x.id) === $("create-task").value);
    const name = $("create-name");
    if (!name.value || name.dataset.auto === "true") {
      name.value = `[${remote ? "SSH" : backend.toUpperCase()}] ${shortTask(task)}`;name.dataset.auto = "true";
    }
    $("task-help").textContent = task?.agent_reset_supported === false ? "이 task에는 독립 GPT reset 검증이 아직 없습니다. 실행하려면 지원 task를 선택하거나 수집 설정에서 환경 초기화 모드를 명시적으로 선택하세요." : "프로젝트를 만들어도 환경이나 모델은 실행되지 않습니다.";
  }

  function updateCreateBackend() {
    const backend = selectedCreateBackend(), remote = backend === "remote_robot";
    state.taskToken += 1;state.tasks = [];$("create-error").hidden = true;
    $("create-suite-field").hidden = remote || !backend;$("create-suite").disabled = remote || backend !== "libero";
    $("create-catalog-task-field").hidden = remote || !backend;$("create-task").disabled = remote || !backend;
    $("create-remote-task-field").hidden = !remote;$("create-remote-task").disabled = !remote;$("create-remote-task").required = remote;
    $("create-goal-fields").hidden = remote || !backend;
    $("create-goal-fields").querySelectorAll("input").forEach(input => {input.disabled = remote || !backend;});
    if (!backend) {$("create-submit").disabled = true;return;}
    if (remote) {$("create-submit").disabled = Boolean(state.createBusy);updateCreateTask();}
    else loadTasks();
  }

  async function loadTasks() {
    const token = ++state.taskToken, backend = selectedCreateBackend();
    if (!backend) return;
    if (backend === "remote_robot") {state.tasks = [];updateCreateTask();return;}
    $("create-task").disabled = true;$("create-submit").disabled = true;
    $("create-task").innerHTML = '<option value="">Task를 불러오는 중…</option>';$("create-error").hidden = true;
    try {
      const data = await api(`/api/tasks${backend === "libero" ? `?suite=${encodeURIComponent($("create-suite").value)}` : ""}`,undefined,30000);
      if (token !== state.taskToken || selectedCreateBackend() !== backend || !$("create-dialog").open) return;
      state.tasks = data.tasks || [];
      $("create-task").innerHTML = state.tasks.map(task => `<option value="${n(task.id)}">${n(task.id)} · ${esc(task.language || task.name || "Task")}</option>`).join("");
      const selectedId = n(state.status?.config?.task_id);
      if (state.tasks.some(task => task.id === selectedId)) $("create-task").value = String(selectedId);
      $("create-task").disabled = !state.tasks.length;$("create-submit").disabled = !state.tasks.length || Boolean(state.createBusy);
      if (!state.tasks.length) throw new Error("사용 가능한 task가 없습니다.");
      updateCreateTask();
    } catch (error) {
      if (token !== state.taskToken || selectedCreateBackend() !== backend || !$("create-dialog").open) return;
      $("create-task").innerHTML = '<option value="">Task를 불러오지 못했습니다</option>';
      $("create-error").textContent = error.message;$("create-error").hidden = false;
    }
  }

  function openCreate() {
    if (!state.workspace || state.createBusy) return;
    $("create-form").reset();$("create-error").hidden = true;$("create-name").dataset.auto = "true";
    const backend = state.workspace.runtime.backend || "libero", suite = state.workspace.runtime.suite || "libero_90";
    $("create-backend").innerHTML = `<option value="" disabled selected>환경을 선택하세요</option>${backend === "remote_robot" ? "" : `<option value="${esc(backend)}">${esc(backendLabel(backend))}</option>`}<option value="remote_robot">원격 로봇</option>`;
    $("create-backend").value = "";
    if (![...$("create-suite").options].some(option => option.value === suite)) $("create-suite").add(new Option(suite,suite));
    $("create-suite").value = suite;$("create-dialog").showModal();updateCreateBackend();
  }

  async function createProject(event) {
    event.preventDefault();const form = event.currentTarget;
    if (state.createBusy || !form.reportValidity()) return;
    const backend = selectedCreateBackend(), remote = backend === "remote_robot", runtimeBackend = state.workspace?.runtime.backend || "libero";
    if (backend !== runtimeBackend && !remote) return;
    const data = new FormData(form), task = remote ? {id:0,name:String(data.get("remote_task_name") || "").trim()} : state.tasks.find(item => String(item.id) === data.get("task_id"));
    if (!task || remote && !task.name) return;
    const payload = {name:String(data.get("name")).trim(),description:String(data.get("description") || "").trim(),backend,suite:remote ? "custom" : $("create-suite").value,task_id:task.id,task_name:task.language || task.name};
    if (!remote) payload.pipeline = {demo_target:Number(data.get("demo_target")),bc_success_target:Number(data.get("bc_success_target")),rl_success_target:Number(data.get("rl_success_target"))};
    state.createBusy = true;$("create-submit").disabled = true;$("create-backend").disabled = true;$("create-error").hidden = true;
    try {
      const result = await api("/api/projects",payload);
      $("create-dialog").close();toast(remote ? "원격 로봇 프로젝트를 만들었습니다. 환경 연결 탭에서 SSH와 client를 설정하세요." : "프로젝트를 만들었습니다. 수집은 아직 시작되지 않았습니다.");
      await refreshWorkspace();location.hash = routeUrl(result.project.id,remote ? "collection" : "overview");
    } catch (error) {$("create-error").textContent = error.message;$("create-error").hidden = false;}
    finally {state.createBusy = false;$("create-submit").disabled = false;$("create-backend").disabled = false;}
  }

  async function saveSettings(event) {
    event.preventDefault();
    const form = event.target, project = currentProject(), routeToken = state.routeToken;
    if (!project || state.projectMutation || state.settingsSaving || project.delete_pending || !form.reportValidity()) return;
    const data = new FormData(form);
    const payload = {name:String(data.get("name")).trim(),description:String(data.get("description") || "").trim()};
    const graspInput = form.elements.expected_grasp_closures;
    if (graspInput) {
      const expected = String(data.get("expected_grasp_closures") ?? "").trim();
      const saved = String(graspInput.dataset.savedExpected ?? "").trim();
      const expectedValue = expected === "" ? null : Number(expected), savedValue = saved === "" ? null : Number(saved);
      if (expectedValue !== savedValue) payload.expected_grasp_closures = expectedValue;
    }
    if ((policyOptions().models || []).some(item => item.model === data.get("policy_model"))) {
      payload.policy_config = {model:String(data.get("policy_model")),reasoning_effort:String(data.get("reasoning_effort"))};
    }
    state.settingsSaving = true;updateProjectManagement();$("settings-error").hidden = true;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(project.id)}/settings`,payload);
      const stillHere = routeToken === state.routeToken && currentProject()?.id === project.id;
      if (stillHere) {
        state.detail.project = result.project;
        const panel = document.querySelector(".grasp-baseline-panel");
        if (panel) panel.outerHTML = graspBaselinePanel(result.project);
      }
      toast("설정을 저장했습니다. 다음 새 run부터 적용되며 Pause·재개는 현재 run의 설정을 유지합니다.");
      await refreshWorkspace();await refreshStatus(true);renderNav();renderBreadcrumb();
      if (routeToken === state.routeToken && currentProject()?.id === project.id) {
        const heading = document.querySelector(".project-heading h1");if (heading) heading.textContent = result.project.name;
      }
    } catch (error) {
      if (routeToken === state.routeToken && $("settings-error")) {$("settings-error").textContent = error.message;$("settings-error").hidden = false;}
      else toast(error.message,true);
    }
    finally {state.settingsSaving = false;if (routeToken === state.routeToken) updateProjectManagement();}
  }

  async function manageEpisode(run, id, archived) {
    const project = currentProject(), ep = findEpisode(run,id);
    if (!project || !ep || ep.can_manage !== true || ep.delete_pending || state.episodeMutation) return;
    state.episodeMutation = `${run}/${id}`;
    if (state.route.tab === "data") renderData(project);
    try {
      const detail = await api(`${episodePath(project.id,run,id)}/archive`,{archived});
      if (state.route.id === project.id) {
        state.detail = detail;state.detailSignature = "";
        if (!["settings","collection"].includes(state.route.tab)) renderProject();
      }
      toast(archived ? "Episode를 보관했습니다. 보관함에서 확인하거나 복원할 수 있습니다." : "보관을 해제했습니다. 데이터 목록에 다시 표시됩니다.");
      await refreshWorkspace();
    } catch (error) { toast(error.message,true); }
    finally {
      state.episodeMutation = null;
      if (state.route.id === project.id && state.route.tab === "data") renderData(currentProject());
      else if (state.route.id === project.id && state.route.tab === "overview") renderOverview(currentProject());
    }
  }

  function askDeleteEpisode(run, id) {
    const project = currentProject(), ep = findEpisode(run,id);
    if (!project || !ep || ep.can_manage !== true || state.episodeMutation) return;
    state.deleteTarget = {project:project.id,run,id};
    $("delete-title").textContent = `${episodeLabel(ep)}을 영구 삭제할까요?`;
    $("delete-episode-info").innerHTML = `<strong>${esc(project.name)} · ${esc(episodeLabel(ep))}</strong><span>${esc(date(ep.started_at))}</span><code>${esc(run)} / ${esc(id)}</code>`;
    $("delete-error").hidden = true;$("delete-confirm").disabled = false;
    $("delete-dialog").showModal();
  }

  async function deleteEpisode() {
    const target = state.deleteTarget;
    if (!target || state.deleteBusy || state.episodeMutation) return;
    state.deleteBusy = true;state.episodeMutation = `${target.run}/${target.id}`;
    $("delete-confirm").disabled = true;$("delete-confirm").textContent = "삭제 중…";
    $("delete-error").hidden = true;
    try {
      const detail = await api(`${episodePath(target.project,target.run,target.id)}/delete`,{confirm:true},30000);
      $("delete-dialog").close();
      if (state.episodeDetail?.run === target.run && state.episodeDetail?.id === target.id) $("episode-dialog").close();
      if (state.route.id === target.project) {
        state.detail = detail;state.detailSignature = "";
        if (!["settings","collection"].includes(state.route.tab)) renderProject();
      }
      toast("Episode 데이터와 연결된 영상을 삭제했습니다.");
      await refreshWorkspace();
    } catch (error) {
      $("delete-error").textContent = error.message;$("delete-error").hidden = false;
    } finally {
      state.deleteBusy = false;state.episodeMutation = null;
      $("delete-confirm").disabled = false;$("delete-confirm").textContent = "영구 삭제";
      if (state.route.id === target.project && state.route.tab === "data") renderData(currentProject());
      else if (state.route.id === target.project && state.route.tab === "overview") renderOverview(currentProject());
    }
  }

  function recordActions(actions) {
    if (!Array.isArray(actions) || !actions.length) return '<p class="trace-empty">실행 action 없음</p>';
    return `<div class="trace-actions"><div class="trace-action-heading"><span>요청 action vector</span><span>반복 tick</span></div>${actions.map(action => `<div class="trace-action"><code>${esc(JSON.stringify(action.values || []))}</code><span>× ${count(action.repeat)}</span></div>`).join("")}</div>`;
  }

  function traceRecord(record, index) {
    const stage = record.stage === "reset" ? "Reset" : "Task";
    const time = esc(date(record.recorded_at,{second:"2-digit"}));
    const id = `trace-record-${index}`;
    if (record.kind === "decision") {
      const decision = record.decision || {};
      const learned = record.stage !== "reset" && ["bc_eval","rl","rl_eval"].includes(state.episodeDetail?.learning?.pipeline_stage);
      return `<details class="trace-record decision-record" ${index <= 1 ? "open" : ""} id="${id}"><summary><span class="trace-record-heading">${tag(stage,record.stage === "reset" ? "warning" : "neutral")}<strong>${learned ? "정책 action" : "GPT 판단"}</strong><span class="trace-note-preview">${esc(decision.note || decision.status || "판단 기록")}</span></span><time>${time}</time></summary><div class="trace-record-body"><p class="trace-note">${esc(decision.note || "별도 판단 요약이 기록되지 않았습니다.")}</p><div class="trace-decision-meta"><span>응답 상태: ${esc(decision.status || "—")}</span><span>관측: ${esc(decision.observation_id || "—")}</span>${decision.reasoning_effort ? `<span>이번 추론: ${esc(decision.reasoning_effort)}</span>` : ""}${decision.next_reasoning_effort ? `<span>다음 추론: ${esc(decision.next_reasoning_effort)}</span>` : ""}</div>${decision.status === "reconsider" ? '<p class="field-note">동작 없이 xhigh 재판단을 요청했습니다.</p>' : ""}${recordActions(decision.actions)}</div></details>`;
    }
    if (record.kind === "receipt") {
      return `<details class="trace-record receipt-record" id="${id}"><summary><span class="trace-record-heading">${tag(stage,record.stage === "reset" ? "warning" : "neutral")}<strong>Action 실행 결과</strong><span class="trace-note-preview">${count(record.executed_steps)} / ${count(record.requested_steps)} ticks${record.interrupted ? " · 부분 실행" : ""}${record.success ? " · 성공 검증" : ""}</span></span><time>${time}</time></summary><div class="trace-record-body"><div class="trace-decision-meta"><span>요청 ${count(record.requested_steps)} ticks · 실제 실행 ${count(record.executed_steps)} ticks</span>${record.success ? tag("성공 검증","success") : ""}</div><div class="trace-states"><div><h4>실행 전 상태</h4><pre>${esc(JSON.stringify(record.state_before || {},null,2))}</pre></div><div><h4>실행 후 상태</h4><pre>${esc(JSON.stringify(record.state_after || {},null,2))}</pre></div></div></details>`;
    }
    if (record.kind === "policy_transition") {
      const data = record.info || record.data || record;
      const fields = Object.fromEntries(Object.entries(data).filter(([key]) => !["kind","record_id","recorded_at","stage","observation_id"].includes(key)));
      return `<details class="trace-record policy-transition-record" id="${id}"><summary><span class="trace-record-heading">${tag(stage)}<strong>학습 정책 기록</strong><span class="trace-note-preview">${typeof data.progress === "number" ? `Progress ${finiteValue(data.progress*100,1)}%` : "BC action · 보정 · reward"}${data.gpt_judgment ? " · GPT 판단 포함" : ""}</span></span><time>${time}</time></summary><div class="trace-record-body"><div class="trace-states single"><div><h4>BC action · Progress · GPT 판단 · 학습 지표</h4><pre>${esc(JSON.stringify(fields,null,2))}</pre></div></div></details>`;
    }
    const names = {episode_start:"Episode 시작",episode_end:"Task 종료",stage_start:`${stage} 시작`,stage_end:`${stage} 종료`,control:"실행 제어",reset:"환경 준비",environment_reset:"환경 준비",agent_reset_continuation:"복구된 환경에서 시작"};
    const message = record.message || record.outcome || record.reason || "";
    return `<div class="trace-milestone" id="${id}"><span><strong>${esc(names[record.kind] || record.kind)}</strong>${message ? `<span>${esc(message)}</span>` : ""}</span><time>${time}</time></div>`;
  }

  async function loadEpisodeRecords() {
    const detail = state.episodeDetail, token = state.episodeToken;
    if (!detail || detail.loading || detail.nextOffset === null) return;
    detail.loading = true;
    const more = $("trace-more");if (more) more.disabled = true;
    $("trace-error").hidden = true;
    try {
      const result = await api(`${episodePath(detail.project,detail.run,detail.id)}/detail?offset=${detail.nextOffset}&limit=50`);
      if (token !== state.episodeToken || !$("episode-dialog").open) return;
      const offset = detail.records.length;
      detail.records.push(...(result.records || []));
      detail.nextOffset = result.next_offset ?? null;
      detail.learning = result.config?.learning || {};
      const learning = detail.learning;
      const teacher = `${result.config?.model || "기록된 모델 없음"} / ${effortLabel(result.config?.reasoning_effort)}`;
      $("episode-policy").textContent = externalEpisode(detail.episode) ? `${detail.episode.source.dataset || "외부 데이터셋"} · 외부 소스 기록` : `${learning.pipeline_stage ? `${pipelineStageName(learning.pipeline_stage)} · GPT teacher: ` : ""}${teacher}`;
      $("episode-learning-models").textContent = [learning.base_model_id ? `Base: ${learning.base_model_id}` : "",learning.residual_model_id ? `Residual: ${learning.residual_model_id}` : ""].filter(Boolean).join(" · ");
      $("episode-learning-models").hidden = externalEpisode(detail.episode) || !learning.base_model_id && !learning.residual_model_id;
      $("episode-trace").querySelector(".trace-loading")?.remove();
      $("episode-trace").insertAdjacentHTML("beforeend",(result.records || []).map((record,index) => traceRecord(record,offset+index)).join(""));
      if (!detail.records.length) $("episode-trace").innerHTML = '<p class="trace-empty">이 episode에 저장된 판단·실행 기록이 없습니다.</p>';
      $("trace-count").textContent = `${count(detail.records.length)} / ${count(result.total_records)} 기록`;
      $("trace-more").hidden = detail.nextOffset === null;
    } catch (error) {
      if (token !== state.episodeToken) return;
      $("episode-trace").querySelector(".trace-loading")?.remove();
      $("trace-error").textContent = error.message;$("trace-error").hidden = false;
      $("trace-more").hidden = false;$("trace-more").textContent = "기록 다시 불러오기";
    } finally {
      detail.loading = false;
      if (token === state.episodeToken && $("trace-more")) $("trace-more").disabled = false;
    }
  }

  function openEpisode(run, id) {
    const ep = findEpisode(run,id), project = currentProject();
    if (!ep || !project || ep.delete_pending) return;
    state.episodeToken += 1;
    state.episodeDetail = {project:project.id,run,id,episode:ep,records:[],nextOffset:0,loading:false,videoJob:null,videoBusy:false,videoMutation:null,videoError:null,videoApiUnavailable:false,videoRequest:0};
    $("episode-title").textContent = `${episodeLabel(ep)} · ${project.name}`;
    $("episode-eyebrow").textContent = episodeLabel(ep).startsWith("Rollout") ? "ROLLOUT DETAIL" : "EPISODE DETAIL";
    const external = externalEpisode(ep);
    const wall = external ? null : ep.reset_finished_at && ep.started_at ? (ep.reset_finished_at-ep.started_at)/60 : ep.finished_at && ep.started_at ? (ep.finished_at-ep.started_at)/60 : null;
    $("episode-content").innerHTML = `<div class="episode-info">${episodeOutcomeLabel(ep)}${episodeResetLabel(ep)}${ep.archived ? tag("보관됨") : ""}<span>${esc(date(ep.started_at))}</span></div>${episodeSourceDetails(ep)}<div class="episode-policy"><span>${external ? "데이터 출처" : "실행 모델 / 추론 수준"}</span><strong id="episode-policy">기록을 불러오는 중…</strong><span id="episode-learning-models" hidden></span></div><section class="episode-video-section"><div id="episode-video-player"></div><div id="episode-video-actions"></div></section><div class="table-wrap"><table><thead><tr><th>단계</th><th>Step</th><th>GPT 호출</th><th>Input</th><th>Output</th><th>추정 비용</th></tr></thead><tbody>${(external ? ["task"] : ["task","reset"]).map(stage => {const cost = stageCost(ep,stage);return `<tr><td>${stage === "task" ? "Task" : "Reset"}</td><td>${count(stage === "task" ? ep.steps : ep.reset_steps)}</td><td>${count(cost.api_requests ?? (stage === "task" ? ep.decisions : ep.reset_decisions))}</td><td>${count(cost.input_tokens)}</td><td>${count(cost.output_tokens)}</td><td>${money(cost.estimated_cost_usd,4)}</td></tr>`;}).join("")}</tbody></table></div><div class="episode-meta"><span>${esc(run)} / ${esc(id)}</span><span>${wall !== null ? `실제 경과 ${wall.toFixed(1)}분 · ` : ""}총 ${money(episodeCost(ep),4)}</span></div>${episodeQuality(ep)}<section class="episode-trace-section"><div class="trace-heading"><h3>${external ? "외부 데이터 기록" : "정책 판단과 실행 기록"}</h3><span id="trace-count" class="tag neutral">불러오는 중</span></div><p class="field-note">${external ? "외부 데이터셋의 관측과 action 기록입니다. GPT 추론이나 RILO의 reset 실행 기록은 포함하지 않습니다." : "정책의 action과 실행 결과, GPT가 반환한 판단 요약입니다. 학습 rollout에는 BC action·진행도·보정 지표도 포함됩니다. 모델 내부 추론 원문은 포함되지 않습니다."}</p><div id="episode-trace" class="episode-trace"><p class="trace-loading">기록을 불러오고 있습니다.</p></div><p id="trace-error" class="form-error" role="alert" hidden></p><button id="trace-more" class="button secondary trace-more" data-action="trace-more" hidden>기록 더 보기</button></section>`;
    if (!$("episode-dialog").open) $("episode-dialog").showModal();
    updateEpisodeVideo();
    refreshEpisodeVideoJob();
    loadEpisodeRecords();
  }

  document.addEventListener("click", event => {
    const close = event.target.closest("[data-close-dialog]");
    if (close) {
      const dialog = close.closest("dialog");
      if (!(dialog.id === "delete-dialog" && state.deleteBusy) && !(dialog.id === "bulk-episode-dialog" && state.bulkEpisodeBusy) && !(dialog.id === "project-delete-dialog" && state.projectMutation)) dialog.close();
      return;
    }
    if (event.target.closest(".episode-select-cell")) {event.stopPropagation();return;}
    const button = event.target.closest("[data-action]");
    if (!button || button.disabled) return;
    if (button.closest(".episode-clickable-row") && button.tagName === "BUTTON") event.stopPropagation();
    const action = button.dataset.action;
    if (action === "create") openCreate();
    else if (action === "projects-active" || action === "projects-archived") {state.projectView = action === "projects-archived" ? "archived" : "active";renderHome();}
    else if (action === "archive-project") archiveProject(true);
    else if (action === "restore-project") archiveProject(false);
    else if (action === "delete-project") askDeleteProject();
    else if (action === "confirm-delete-project") deleteProject();
    else if (action === "remote-environment-connect") remoteEnvironmentCommand("environment/connect");
    else if (action === "remote-environment-disconnect") remoteEnvironmentCommand("environment/disconnect");
    else if (action === "remote-token-issue") requestRemoteToken();
    else if (action === "remote-token-confirm") remotePolicyCommand("token");
    else if (action === "remote-config-download") downloadRemoteConfig();
    else if (action === "remote-status-refresh") refreshRemotePolicy();
    else if (action === "remote-run-once") remotePolicyCommand("start",null,true);
    else if (["remote-run-pause","remote-run-resume","remote-run-stop"].includes(action)) remotePolicyCommand(action.replace("remote-run-",""));
    else if (action === "robot-connect") robotConnectionCommand("connect");
    else if (action === "robot-disconnect") robotConnectionCommand("disconnect");
    else if (action === "robot-status-refresh") refreshRobotConnection();
    else if (action === "robot-host-key-check") checkRobotHostKey();
    else if (action === "robot-host-key-trust") trustRobotHostKey();
    else if (action === "refresh") {refreshStatus();refreshWorkspace(true);}
    else if (action === "generate-episode-video") mutateEpisodeVideo("generate");
    else if (action === "delete-episode-video") mutateEpisodeVideo("delete");
    else if (action === "episode") openEpisode(button.dataset.run,button.dataset.episode);
    else if (action === "archive-episode") manageEpisode(button.dataset.run,button.dataset.episode,true);
    else if (action === "restore-episode") manageEpisode(button.dataset.run,button.dataset.episode,false);
    else if (action === "delete-episode") askDeleteEpisode(button.dataset.run,button.dataset.episode);
    else if (action === "confirm-delete") deleteEpisode();
    else if (action === "bulk-episode-select") selectBulkEpisodes(button.dataset.mode);
    else if (action === "bulk-episode-cancel" && !state.episodeMutation) {state.dataSelection = null;renderData(currentProject());}
    else if (action === "bulk-episode-confirm") mutateBulkEpisodes();
    else if (action === "trace-more") loadEpisodeRecords();
    else if (action === "pipeline-start-auto") openPipelineStartDialog();
    else if (action === "bc-history") openBCHistory(button.dataset.metric,button.dataset.model);
    else if (action === "bc-history-view" && bcHistoryIsCurrent()) {state.bcHistory.metric = button.dataset.metric;state.bcHistory.followLatest = true;renderBCHistory();}
    else if (action === "bc-history-latest" && bcHistoryIsCurrent()) updateBCHistoryPoint(latestBCHistoryIndex(state.bcHistory.points,state.bcHistory.metric),true);
    else if (action === "pipeline-start-manual") openStageStartDialog(button.dataset.stage || selectedPipelineStage());
    else if (action === "pipeline-pause") pipelineCommand("pause");
    else if (action === "pipeline-resume") pipelineCommand("resume");
    else if (action === "pipeline-stop") pipelineCommand("stop");
    else if (action === "pipeline-global-pause") pipelineCommand("pause",{},button.dataset.project);
    else if (action === "pipeline-refresh") refreshPipeline(true);
    else if (action === "collection-settings-toggle") setCollectionSettingsExpanded(!state.collectionSettingsExpanded);
    else if (action === "collection-settings-focus") {event.preventDefault();setCollectionSettingsExpanded(true);$("collection-settings-panel")?.scrollIntoView({behavior:"smooth",block:"start"});}
    else if (action === "data-active" || action === "data-archived") {state.dataSelection = null;state.dataView = action === "data-archived" ? "archived" : "active";renderData(currentProject());}
    else if (action === "global-pause") command("pause",true);
    else if (["connect","start","pause","stop","step"].includes(action)) command(action === "start" && state.status?.phase === "paused" ? "resume" : action);
  });
  document.addEventListener("keydown", event => {
    if (!["Enter"," "].includes(event.key) || event.target.closest("button,a,input,select,textarea")) return;
    const row = event.target.closest(".episode-clickable-row[data-action=episode]");
    if (!row) return;
    event.preventDefault();openEpisode(row.dataset.run,row.dataset.episode);
  });
  document.addEventListener("input", event => {
    if (event.target.dataset?.robotConfig) editRobotConnectionField(event.target);
    if (event.target.closest("#remote-run-form")) editRemoteRunField(event.target);
    if (event.target.closest("#remote-launch-form")) editRemoteLaunchField(event.target);
    if (event.target.id === "remote-server-url" && remotePolicyIsCurrent()) {state.remotePolicy.baseUrl = event.target.value;state.remotePolicy.baseUrlDirty = true;updateRemotePolicyPanel();}
    if (event.target.id === "create-remote-task") updateCreateTask();
    if (event.target.id === "project-search") {state.search = event.target.value;renderProjectCards();}
    if (event.target.id === "create-name") event.target.dataset.auto = "false";
    if (event.target.closest("#collection-settings-form")) state.collectionSettingsDirty = true;
    if (event.target.closest("#pipeline-start-form")) updatePipelineStartPreview();
  });
  document.addEventListener("change", event => {
    if (event.target.dataset?.robotConfig) editRobotConnectionField(event.target);
    if (event.target.closest("#remote-run-form")) editRemoteRunField(event.target);
    if (event.target.closest("#remote-launch-form")) editRemoteLaunchField(event.target);
    if (event.target.id === "remote-server-url" && remotePolicyIsCurrent()) {state.remotePolicy.baseUrl = event.target.value;state.remotePolicy.baseUrlDirty = true;updateRemotePolicyPanel();}
    if (event.target.id === "create-backend") updateCreateBackend();
    if (event.target.id === "create-suite") loadTasks();
    if (event.target.id === "create-task") updateCreateTask();
    if (event.target.id === "episode-select-all" || event.target.matches("input[data-episode-select]")) toggleEpisodeSelection(event.target);
    if (event.target.id === "episode-filter") {state.dataSelection = null;state.filter = event.target.value;renderData(currentProject());}
    if (event.target.id === "policy-model") {updatePolicyEfforts();updatePolicyStatus();}
    if (event.target.id === "policy-effort") updatePolicyStatus();
    if (event.target.id === "pipeline-manual-stage" && state.route.tab === "evaluation" && manualPipelineStages().some(stage => stage.id === event.target.value)) {state.evaluationStage = event.target.value;updatePipelineExecution();renderPipelineMonitor();}
    if (event.target.dataset.trainingAlgorithm && ["bc","rl"].includes(event.target.dataset.trainingAlgorithm)) {state.trainingAlgorithms[event.target.dataset.trainingAlgorithm] = event.target.value;updatePipelineExecution();}
  });
  document.addEventListener("submit", event => {
    if (event.target.id === "robot-connection-form") robotConnectionCommand("config",event);
    if (event.target.id === "remote-run-form") remotePolicyCommand("start",event);
    if (event.target.id === "remote-launch-form") remoteEnvironmentCommand("launch-config",event);
    if (event.target.id === "settings-form") saveSettings(event);
    if (event.target.id === "stage-start-form") submitStageStart(event);
    if (event.target.id === "pipeline-start-form") submitPipelineStart(event);
    if (event.target.id === "collection-settings-form") saveCollectionSettings(event);
  });
  $("pipeline-start-dialog").addEventListener("close", () => {state.pipelineDialog = null;});
  $("stage-start-dialog").addEventListener("close", () => {state.stageDialog = null;});
  $("bc-history-dialog").addEventListener("close",closeBCHistory);
  $("robot-host-key-dialog").addEventListener("close", () => {state.robotHostKey = null;});
  document.addEventListener("pointermove",inspectBCHistoryPointer);
  document.addEventListener("pointerdown",inspectBCHistoryPointer);
  $("create-form").addEventListener("submit",createProject);
  $("create-dialog").addEventListener("close", () => {state.taskToken += 1;});
  $("episode-dialog").addEventListener("close", () => {const video = $("episode-content").querySelector("video");if (video) video.pause();state.episodeToken += 1;state.episodeDetail = null;});
  $("delete-dialog").addEventListener("cancel", event => {if (state.deleteBusy) event.preventDefault();});
  $("bulk-episode-dialog").addEventListener("cancel", event => {if (state.bulkEpisodeBusy) event.preventDefault();});
  $("bulk-episode-dialog").addEventListener("close", () => {state.bulkEpisodeTarget = null;});
  $("project-delete-dialog").addEventListener("cancel", event => {if (state.projectMutation) event.preventDefault();});
  $("project-delete-dialog").addEventListener("close", () => {state.projectDeleteTarget = null;});
  window.addEventListener("hashchange",navigate);
  fillIcons();
  navigate();
  Promise.allSettled([refreshStatus(),refreshWorkspace(true)]);
  setInterval(refreshStatus,1000);
  setInterval(refreshWorkspace,5000);
  setInterval(refreshPipeline,2500);
})();
