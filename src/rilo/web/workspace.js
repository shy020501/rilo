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
  const phaseNames = {idle:"실행 대기",initializing:"환경 준비 중",ready:"수집 준비 완료",thinking:"GPT 추론 중",executing:"Action 실행 중",pausing:"일시정지 중",paused:"일시정지",completed:"수집 완료",stopped:"실행 중지",error:"확인 필요"};
  const activePhases = new Set(["initializing", "ready", "thinking", "executing", "pausing", "paused"]);
  const pausePhases = new Set(["initializing", "ready", "thinking", "executing"]);
  const state = {workspace:null,status:null,detail:null,route:{id:null,tab:"overview"},connected:false,pending:null,pendingProjectId:null,pausePending:false,search:"",filter:"all",dataView:"active",episodeMutation:null,deleteTarget:null,deleteBusy:false,episodeDetail:null,episodeToken:0,workspaceSignature:"",detailSignature:"",navSignature:"",barSignature:"",statusBusy:false,statusRequest:0,workspaceBusy:false,routeToken:0,taskToken:0,tasks:[],lastObservation:null,lastCameras:"",lastEvents:"",toastTimer:null,pipeline:null,pipelineProjectId:null,pipelineBusy:false,pipelineRequest:0,pipelinePending:null,pipelinePausePending:false,pipelineMode:"auto",pipelineStage:"collect",pipelineDirty:false,pipelineConfigSignature:"",pipelineExecutionSignature:"",modelSignature:"",evaluationSignature:""};
  const routeUrl = (id, tab = "overview") => `#/projects/${encodeURIComponent(id)}${tab === "overview" ? "" : `/${tab}`}`;
  const runtime = () => state.status || state.workspace?.runtime || {};
  const activeId = () => runtime().active_project_id;
  const currentProject = () => state.detail?.project;
  const stageCost = (episode, stage) => episode.costs?.[stage] || {};
  const episodeCost = (episode) => n(stageCost(episode,"task").estimated_cost_usd) + n(stageCost(episode,"reset").estimated_cost_usd);
  const complete = (episode) => episode.outcome === "success" && episode.reset_outcome === "success";
  const resetLabel = (outcome) => outcome === "success" ? tag("복구 완료","success") : outcome ? tag("복구 미완료","warning") : tag("대기");
  const outcomeLabel = (outcome) => outcome === "success" ? tag("성공","success") : !outcome ? tag("진행 중") : tag("미완료","warning");
  const allEpisodes = () => [...(state.detail?.episodes || []),...(state.detail?.archived_episodes || [])];
  const findEpisode = (run,id) => allEpisodes().find(ep => ep.run_id === run && ep.id === id);
  const episodePath = (project,run,id) => `/api/projects/${encodeURIComponent(project)}/episodes/${encodeURIComponent(run)}/${encodeURIComponent(id)}`;
  const policyOptions = () => state.detail?.policy_options || state.workspace?.policy_options || {models:[],defaults:{}};
  const nextPolicy = (project) => ({model:state.workspace?.runtime?.model,reasoning_effort:state.workspace?.runtime?.reasoning_effort,...policyOptions().defaults,...project?.policy_config});
  const policyLabel = (policy) => `${policy?.model || "—"} / ${policy?.reasoning_effort || "—"}`;
  const pipelineStages = [
    {id:"collect",number:"01",name:"GPT 데이터 수집",short:"수집",icon:"database",description:"정해진 수만큼 task와 GPT reset을 수행하고 성공·실패를 함께 저장합니다."},
    {id:"bc",number:"02",name:"Diffusion Policy 학습",short:"DP 학습",icon:"model",description:"수집을 멈춘 상태에서 성공 task 데이터로 BC 정책을 학습합니다."},
    {id:"bc_eval",number:"03",name:"BC 성공률 평가",short:"BC 평가",icon:"check",description:"DP로 rollout하고 GPT로 reset합니다. 목표 미달이면 수집·학습을 반복합니다."},
    {id:"progress",number:"04",name:"Progress 모델 학습",short:"Progress",icon:"progress",description:"BC 성공률 목표를 달성한 뒤 성공 데이터만 사용해 진행도를 학습합니다."},
    {id:"rl",number:"05",name:"Residual TD3 학습",short:"Residual RL",icon:"layers",description:"Frozen DP에 매 step correction을 더하고 offline·online replay를 혼합합니다."},
    {id:"rl_eval",number:"06",name:"RL 성공률 평가",short:"RL 평가",icon:"check",description:"BC+Residual의 성공률을 확인하고 목표에 도달할 때까지 online RL을 반복합니다."},
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

  async function api(path, body, timeout = 15000) {
    const abort = new AbortController();
    const timer = setTimeout(() => abort.abort(), timeout);
    try {
      const response = await fetch(path, {method:body === undefined ? "GET" : "POST",body:body === undefined ? undefined : JSON.stringify(body),headers:body === undefined ? {Accept:"application/json"} : {Accept:"application/json","Content-Type":"application/json"},cache:"no-store",credentials:"same-origin",signal:abort.signal});
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorText(result, `요청을 처리하지 못했습니다 (${response.status}).`));
      return result;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답을 기다리는 중입니다. 실행 상태를 확인한 뒤 다시 시도해주세요.");
      throw error;
    } finally { clearTimeout(timer); }
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
      $("global-run").innerHTML = `<div class="global-run-copy"><i class="dot ${learning.phase === "running" ? "dot-pulse" : ""}"></i><a href="${routeUrl(learning.project_id,"models")}"><strong>${esc(project?.name || "학습 파이프라인")}</strong></a><span>· ${esc(pipelineStageName(learning.stage))} / ${esc(pipelinePhaseNames[learning.phase] || learning.phase)}</span></div><button class="button pause" data-action="pipeline-global-pause" data-project="${esc(learning.project_id)}" ${learning.phase !== "running" || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pipelinePausePending ? "일시정지 요청 중…" : learning.phase === "paused" ? "일시정지됨" : "Pause"}</button>`;
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
    $("global-run").innerHTML = `<div class="global-run-copy"><i class="dot ${["thinking","executing"].includes(s.phase) ? "dot-pulse" : ""}"></i><a href="${id ? routeUrl(id,"collection") : "/console"}"><strong>${esc(project?.name || (state.pending === "initialize" ? "환경 준비 중" : "실행 콘솔의 수집"))}</strong></a><span>· ${esc(phaseNames[s.phase] || s.phase)}${s.stage === "reset" ? " / GPT reset" : ""}</span></div><button class="button pause" data-action="global-pause" ${!pausePhases.has(s.phase) || state.pausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pausePending ? "일시정지 요청 중…" : s.phase === "paused" ? "일시정지됨" : "Pause"}</button>`;
  }

  function projectCard(project) {
    const stats = project.stats || {};
    const goal = Math.max(1,n(project.pipeline?.demo_target) || 20);
    const progress = Math.min(100,n(stats.successes) / goal * 100);
    const status = projectStatus(project);
    const thumbnail = asset(project.thumbnail_url);
    return `<a class="project-card" href="${routeUrl(project.id)}" aria-label="${esc(project.name)} 프로젝트 열기"><div class="project-image">${thumbnail ? `<img src="${esc(thumbnail)}" alt="${esc(project.name)} 최근 기록" loading="lazy">` : `<div class="project-image-placeholder">${icon("robot")}</div>`}<div class="image-meta"><span class="image-tag">${esc(project.backend?.toUpperCase() || "ROBOT")}</span><span class="image-note">${thumbnail ? "저장된 실행" : "새 프로젝트"}</span></div></div><div class="project-card-body"><div class="card-title-row"><h3>${esc(project.name)}</h3><span class="card-arrow">${icon("arrow")}</span></div><p class="project-description">${esc(project.description || project.task_name || "데이터 수집을 준비하는 프로젝트")}</p><div class="project-state-row">${tag(status.text,status.style)}<span class="state-value">${count(stats.successes)} / ${count(goal)} demos</span></div><div class="progress-track"><span style="width:${progress}%"></span></div></div><div class="project-card-footer"><div><strong>${count(stats.episodes)}</strong>Episodes</div><div><strong>${count(stats.reset_successes)} / ${count(stats.episodes)}</strong>GPT reset</div><div><strong>${money(stats.estimated_cost_usd)}</strong>API 비용</div></div></a>`;
  }

  function renderProjectCards() {
    const holder = $("project-cards");
    if (!holder) return;
    const query = state.search.trim().toLowerCase();
    const projects = (state.workspace?.projects || []).filter(p => [p.name,p.task_name,p.description,p.suite].join(" ").toLowerCase().includes(query));
    holder.innerHTML = projects.length ? projects.map(projectCard).join("") : `<div class="card-empty">${icon(query ? "search" : "folder")}<h3>${query ? "검색 결과가 없어요" : "첫 번째 프로젝트를 만들어보세요"}</h3><p>${query ? "프로젝트 이름이나 task로 다시 검색해주세요." : "Task를 선택하고 데이터 수집을 시작할 수 있어요."}</p>${query ? "" : `<button class="button primary" data-action="create">${icon("plus")}새 프로젝트</button>`}</div>`;
  }

  function renderHome() {
    if (!state.workspace) return;
    const w = state.workspace, t = w.totals || {}, r = w.runtime || {};
    document.title = "RILO · 워크스페이스";
    $("content").innerHTML = `<section class="page-heading"><div><span class="eyebrow">WORKSPACE OVERVIEW</span><h1>워크스페이스</h1><p>프로젝트별로 데이터를 모으고, 학습으로 이어지는 흐름을 확인하세요.</p></div><div class="heading-action"><button class="button primary" data-action="create">${icon("plus")}새 프로젝트</button></div></section><section class="metrics" aria-label="전체 실험 지표">${metric("프로젝트",`${count(t.projects)}<span class="metric-unit">개</span>`,`${count(t.runs)}개의 실행 기록`,"folder")}${metric("수집된 성공 demo",`${count(t.successes)}<span class="metric-unit">개</span>`,`${count(t.episodes)} episodes · Task 성공 기준`,"database")}${metric("완료된 GPT reset",`${count(t.reset_successes)}<span class="metric-unit">/ ${count(t.episodes)}</span>`,"Task 후 환경 복구까지 완료", "reset")}${metric("누적 사용 비용",money(t.estimated_cost_usd),`${count(t.total_tokens)} tokens · 보관·삭제 포함`,"dollar")}</section><div class="home-layout"><section><div class="section-header"><div class="section-header-left"><h2>프로젝트</h2><span class="count-tag">${count(t.projects)}</span></div><label class="search-box">${icon("search")}<input id="project-search" placeholder="프로젝트 검색" aria-label="프로젝트 검색" value="${esc(state.search)}"></label></div><div id="project-cards" class="project-grid"></div><p class="home-footnote">${icon("info")}기존 실행 기록을 프로젝트별로 연결했습니다. 성공 demo와 reset 완료는 각각 집계됩니다.</p></section><aside class="home-aside"><section class="side-panel"><h3>${icon("flow")}파이프라인 연결 상태</h3><ul class="capability-list"><li><span>GPT 데이터 수집 · Reset</span>${tag("사용 가능","success")}</li><li><span>BC · Progress 학습</span>${tag(w.capabilities?.bc ? "사용 가능" : "연결 전",w.capabilities?.bc ? "success" : "neutral")}</li><li><span>BC 정책 평가</span>${tag(w.capabilities?.evaluation ? "사용 가능" : "연결 전",w.capabilities?.evaluation ? "success" : "neutral")}</li><li><span>Residual RL</span>${tag(w.capabilities?.rl ? "사용 가능" : "연결 전",w.capabilities?.rl ? "success" : "neutral")}</li></ul><p class="side-note">${w.capabilities?.bc ? "프로젝트의 모델 탭에서 전체 자동 실행과 단계별 실행을 선택할 수 있습니다. 학습 중에는 수집을 멈춥니다." : "모델 탭에서 파이프라인 연결 상태를 확인할 수 있습니다."}</p></section><section class="side-panel"><h3>${icon("terminal")}연결된 실행 환경</h3><dl class="runtime-list"><div><dt>Environment</dt><dd>${esc(r.backend?.toUpperCase() || "—")}</dd></div><div><dt>Teacher model</dt><dd class="code">${esc(r.policy === "scripted" ? "Scripted demo" : r.model || "—")}</dd></div><div><dt>Reasoning</dt><dd>${esc(r.reasoning_effort || "—")}</dd></div><div><dt>Worker</dt><dd>1 · ${esc(phaseNames[r.phase] || "실행 대기")}</dd></div></dl><p class="side-note">${r.policy === "scripted" ? "UI 검증용 demo 서버입니다. GPT를 호출하지 않습니다." : "프로젝트를 준비한 뒤 수집 시작을 누르면 모델이 호출됩니다."}</p></section></aside></div>`;
    if (r.policy !== "scripted") {
      document.querySelector(".runtime-list + .side-note").textContent = "프로젝트 설정에서 모델·추론 수준을 선택할 수 있습니다. 변경값은 다음 환경 준비 또는 복구 후 이어 수집부터 적용됩니다.";
    }
    renderProjectCards();
    renderBreadcrumb();
  }

  function projectShell(project) {
    const tabs = [["overview","개요","flow"],["collection","수집","camera"],["data","데이터","database"],["models","모델","model"],["settings","설정","settings"]];
    const status = projectStatus(project);
    return `<section class="page-heading project-heading"><div><span class="eyebrow">PROJECT WORKSPACE</span><h1>${esc(project.name)}</h1><p>${esc(project.description || project.task_name || "이 프로젝트의 데이터 수집과 학습 흐름을 관리하세요.")}</p><div class="project-title-meta">${tag(project.backend?.toUpperCase() || "ROBOT","outline")}${tag(project.suite || "Custom","outline")}<span class="project-task">Task ${n(project.task_id)} · ${esc(project.task_name || "")}</span></div></div><div class="project-header-actions">${tag(status.text,status.style)}${state.route.tab !== "collection" ? `<a class="button primary" href="${routeUrl(project.id,"collection")}">${icon("camera")}수집 화면</a>` : ""}</div></section><nav class="tabs" aria-label="프로젝트 메뉴">${tabs.map(([id,label,symbol]) => `<a href="${routeUrl(project.id,id)}" class="${state.route.tab === id ? "active" : ""}" ${state.route.tab === id ? 'aria-current="page"' : ""}>${icon(symbol)}${label}${id === "data" ? `<span class="tab-count">${count(project.stats?.episodes)}</span>` : ""}</a>`).join("")}</nav><div id="tab-content"></div>`;
  }

  function renderPipeline(project) {
    const pipeline = currentPipeline(), status = pipeline?.status || {}, enabled = pipelineEnabled();
    const config = pipeline?.config || {};
    return `<section class="panel pipeline-panel"><div class="panel-header"><div><h2>Orchestration</h2><p>수집 → DP 학습 → 평가 → Progress → Residual RL</p></div><a href="${routeUrl(project.id,"models")}" class="button secondary small">${icon("model")}모델 · 파이프라인</a></div><div class="sequential-pipeline">${pipelineStages.map((stage,index) => `<div class="sequence-stage ${status.stage === stage.id && pipelineActive(status.phase) ? "current" : ""}"><span class="sequence-number">${stage.number}</span><div><h3>${esc(stage.short)}</h3><p>${stage.id === "collect" ? `${count(config.collection_episodes ?? project.pipeline?.demo_target ?? 20)} episodes · 성공/실패` : stage.id === "bc" ? "수집 중지 후 학습" : stage.id === "bc_eval" ? `목표 ${finiteValue(config.bc_success_target ?? project.pipeline?.bc_success_target ?? 70,1)}%` : stage.id === "progress" ? "목표 통과 · 성공 데이터" : stage.id === "rl" ? "Frozen DP + correction" : `목표 ${finiteValue(config.rl_success_target ?? project.pipeline?.rl_success_target ?? 90,1)}%`}</p></div>${index < pipelineStages.length-1 ? `<span class="sequence-arrow">${icon("arrow")}</span>` : ""}</div>`).join("")}</div><div class="pipeline-note">${icon("info")}${enabled ? "BC 평가 목표에 미달하면 수집·학습·평가를 반복합니다. Progress 학습 후 online RL로 이어지며 reset은 GPT가 수행합니다." : "학습 worker 연결 전입니다. 프로젝트의 데이터와 기존 수집 기능은 그대로 사용할 수 있습니다."}${status.stage ? `<span class="sequence-status">${tag(pipelinePhaseNames[status.phase] || status.phase,status.phase === "running" ? "success" : status.phase === "error" ? "warning" : "neutral")} ${esc(pipelineStageName(status.stage))}</span>` : ""}</div></section>`;
  }

  function episodesTable(episodes, compact = false) {
    const rows = episodes.map(ep => {
      const label = `Episode ${String(ep.index || 1).padStart(2,"0")}`;
      const attrs = `data-run="${esc(ep.run_id)}" data-episode="${esc(ep.id)}"`;
      const disabled = ep.can_manage !== true || Boolean(state.episodeMutation);
      const reason = ep.manage_disabled_reason || (disabled ? "실행 기록이 정리된 후 관리할 수 있습니다." : "");
      const archiveLabel = ep.archived ? "보관 해제" : "보관";
      const pendingDelete = ep.delete_pending === true;
      return `<tr data-episode-row="${esc(ep.id)}" data-run-row="${esc(ep.run_id)}"><td><strong>${esc(label)}</strong><span class="table-sub">${esc(date(ep.started_at))}</span>${ep.archived ? `<span class="table-sub">보관 ${esc(date(ep.archived_at))}</span>` : ""}</td><td>${pendingDelete ? tag("삭제 미완료","warning") : outcomeLabel(ep.outcome)}</td><td>${pendingDelete ? "—" : resetLabel(ep.reset_outcome)}</td>${compact ? "" : `<td>${count(ep.steps)} <span class="cell-muted">/</span> ${count(ep.reset_steps)}</td>`}<td>${money(episodeCost(ep),compact ? 2 : 4)}</td><td class="episode-detail-cell"><button class="table-action" data-action="episode" ${attrs} aria-label="${esc(label)} 상세 보기" ${pendingDelete ? 'disabled title="삭제가 진행된 데이터입니다. 휴지통을 눌러 삭제를 완료하세요."' : ""}>${icon(ep.video_url ? "play" : "external")}${ep.video_url ? "상세 / 영상" : "상세"}</button></td><td class="episode-manage-cell"><div class="episode-row-actions"><button class="icon-button episode-icon" data-action="${ep.archived ? "restore-episode" : "archive-episode"}" ${attrs} aria-label="${esc(label)} ${archiveLabel}" title="${esc(pendingDelete ? "삭제가 진행된 데이터는 보관 해제할 수 없습니다." : reason || archiveLabel)}" ${disabled || pendingDelete ? "disabled" : ""}>${icon(ep.archived ? "restore" : "archive")}</button><button class="icon-button episode-icon delete-icon" data-action="delete-episode" ${attrs} aria-label="${esc(label)} 영구 삭제" title="${esc(reason || "영구 삭제")}" ${disabled ? "disabled" : ""}>${icon("trash")}</button></div></td></tr>`;
    }).join("");
    const empty = state.route.tab === "data" && state.dataView === "archived" ? "보관된 episode가 없습니다." : state.filter !== "all" && state.route.tab === "data" ? "선택한 결과에 해당하는 episode가 없습니다." : "아직 저장된 episode가 없습니다. 수집 화면에서 첫 실행을 준비하세요.";
    return `<div class="table-wrap"><table><thead><tr><th>Episode</th><th>Task</th><th>GPT reset</th>${compact ? "" : "<th>Step (Task / Reset)</th>"}<th>API 비용</th><th class="episode-detail-cell"><span class="sr-only">상세 / 영상</span></th><th class="episode-manage-cell"><span class="sr-only">Episode 관리</span></th></tr></thead><tbody>${rows || `<tr><td class="table-empty" colspan="${compact ? 6 : 7}">${empty}</td></tr>`}</tbody></table></div>`;
  }

  function renderOverview(project) {
    const stats = project.stats || {};
    const episodes = (state.detail.episodes || []).slice().sort((a,b) => n(b.started_at)-n(a.started_at));
    $("tab-content").innerHTML = `${renderPipeline(project)}<section class="metrics" aria-label="프로젝트 지표">${metric("성공 demo",`${count(stats.successes)}<span class="metric-unit">개</span>`,`${count(stats.episodes)} episodes 중 Task 성공`,"database")}${metric("GPT reset 완료",`${count(stats.reset_successes)}<span class="metric-unit">/ ${count(stats.episodes)}</span>`,"환경 복구 검증 통과","reset")}${metric("전체 실행 step",count(stats.total_steps),"Task + Reset control ticks","layers")}${metric("누적 사용 비용",money(stats.estimated_cost_usd),`${count(stats.total_tokens)} tokens · 보관·삭제 포함`,"dollar")}</section><div class="overview-bottom"><section class="panel"><div class="panel-header"><h2>최근 episodes</h2><a class="button quiet small" href="${routeUrl(project.id,"data")}">전체 보기 ${icon("arrow")}</a></div>${episodesTable(episodes.slice(0,4),true)}</section><section class="panel"><div class="panel-header"><h2>수집 설정</h2><a href="${routeUrl(project.id,"settings")}" class="icon-button" aria-label="수집 설정 편집">${icon("settings")}</a></div><div class="panel-body"><dl class="project-rules"><div><dt>수집당 episode</dt><dd>${n(project.run_config?.episodes) || 1} episodes</dd></div><div><dt>Reset 방식</dt><dd>${project.run_config?.reset_mode === "environment" ? "환경 초기화" : "GPT가 팔을 조작하여 복구"}</dd></div><div><dt>실행당 API 예산</dt><dd>${money(project.run_config?.budget_usd)}</dd></div></dl><p class="muted-info">설정은 다음 실행부터 적용됩니다. 프로젝트를 열거나 편집해도 수집은 시작되지 않습니다.</p></div></section></div>`;
  }

  function renderData(project) {
    const archived = state.dataView === "archived";
    const episodes = (archived ? state.detail.archived_episodes || [] : state.detail.episodes || []).slice().sort((a,b) => n(b.started_at)-n(a.started_at));
    const filtered = episodes.filter(ep => state.filter === "all" || (state.filter === "complete" ? complete(ep) : state.filter === "incomplete" ? ep.outcome && !complete(ep) : ep.outcome === "success"));
    $("tab-content").innerHTML = `<div class="data-toolbar"><div><h2>${archived ? "보관함" : "수집 데이터"}</h2><p>${archived ? "보관한 episode는 대시보드 demo 수와 기본 목록에서 제외됩니다." : `${count(project.stats?.runs)} runs · ${count(project.stats?.episodes)} episodes · 데이터와 GPT 판단 기록`}</p></div><select id="episode-filter" class="filter-select" aria-label="Episode 결과 필터"><option value="all">전체 episodes</option><option value="complete">Task + Reset 완료</option><option value="incomplete">실패 / 복구 미완료</option><option value="task_success">Task 성공</option></select></div><div class="data-view-switch" role="group" aria-label="데이터 표시 범위"><button class="${!archived ? "selected" : ""}" data-action="data-active" aria-pressed="${!archived}">${icon("database")}데이터 <span>${count(project.stats?.episodes)}</span></button><button class="${archived ? "selected" : ""}" data-action="data-archived" aria-pressed="${archived}">${icon("archive")}보관함 <span>${count(project.archived_count ?? state.detail.archived_episodes?.length)}</span></button></div><section class="panel">${episodesTable(filtered)}</section><p class="home-footnote">${icon("info")}${archived ? "보관 해제로 데이터 목록에 다시 표시할 수 있습니다. 보관은 원본 기록과 영상을 유지합니다." : "보관 아이콘은 데이터를 숨기고, 휴지통은 확인 후 episode 기록과 영상을 영구 삭제합니다."}</p><p class="home-footnote billing-note">이미 발생한 API 비용과 토큰 사용량은 보관·삭제 후에도 누적 사용 비용에 포함됩니다.</p>`;
    $("episode-filter").value = state.filter;
  }

  function renderModels(project) {
    state.modelSignature = "";state.evaluationSignature = "";state.pipelineConfigSignature = "";state.pipelineExecutionSignature = "";
    $("tab-content").innerHTML = `<div class="learning-heading"><div><h2>모델과 학습 파이프라인</h2><p>데이터 수집부터 평가까지 단계별 상태와 실제 학습 결과를 확인하세요.</p></div><span id="learning-connection" class="tag neutral">상태 확인 중</span></div><div id="pipeline-unavailable" hidden></div><div id="pipeline-workspace"><section class="panel pipeline-execution-panel"><div class="panel-header"><div><h2>파이프라인 실행</h2><p>실행 버튼을 누르기 전에는 학습이나 rollout을 시작하지 않습니다.</p></div><div id="pipeline-phase-tag"></div></div><div class="pipeline-mode-switch" role="group" aria-label="파이프라인 실행 방식"><button id="pipeline-mode-auto" data-action="pipeline-mode-auto" class="selected" aria-pressed="true">${icon("flow")}전체 자동 실행</button><button id="pipeline-mode-manual" data-action="pipeline-mode-manual" aria-pressed="false">${icon("step")}단계별 실행</button></div><div id="pipeline-execution"></div><div id="pipeline-stages" class="pipeline-stage-list"></div></section><div id="pipeline-monitor"></div><section class="panel evaluation-panel"><div class="panel-header"><div><h2>정책 성능 비교</h2><p>실제로 완료된 BC-only / BC+Residual 평가 결과</p></div></div><div id="pipeline-evaluations"></div></section><section class="panel model-registry-panel"><div class="panel-header"><div><h2>모델 레지스트리</h2><p>체크포인트, 학습 데이터와 적용된 설정</p></div><span id="model-count" class="tag neutral">0</span></div><div id="pipeline-models"></div></section><section class="panel pipeline-config-panel"><div class="panel-header"><div><h2>학습 설정</h2><p>저장한 설정은 다음 단계 실행부터 적용됩니다.</p></div><span id="pipeline-config-state" class="tag neutral">저장된 설정</span></div><div id="pipeline-config"></div></section></div>`;
    updatePipelineView(true);
    refreshPipeline(true);
  }

  function pipelineReadiness(stage) {
    const pipeline = currentPipeline();
    const item = pipeline?.readiness?.[stage];
    if (typeof item === "boolean") return {ready:item,reason:""};
    return item || {ready:Boolean(pipeline),reason:"실행 조건을 확인하고 있습니다."};
  }

  function updatePipelineView(forceConfig = false) {
    const pipeline = currentPipeline(), project = currentProject();
    if (!project) return;
    if (state.route.tab === "overview") {
      const panel = document.querySelector(".pipeline-panel");
      if (panel) panel.outerHTML = renderPipeline(project);
      return;
    }
    if (state.route.tab !== "models" || !$("pipeline-workspace")) return;
    const unavailable = pipeline?.unavailable === true || (!pipelineEnabled() && !pipeline);
    $("pipeline-unavailable").hidden = !unavailable;
    $("pipeline-workspace").hidden = unavailable;
    $("learning-connection").textContent = unavailable ? "학습 worker 연결 전" : pipeline ? "파이프라인 연결됨" : "상태 확인 중";
    $("learning-connection").className = `tag ${pipeline && !unavailable ? "success" : "neutral"}`;
    if (unavailable) {
      $("pipeline-unavailable").innerHTML = `<section class="panel"><div class="empty-models"><div class="empty-model-icon">${icon("model")}</div><h2>학습 worker 연결을 기다리고 있습니다</h2><p>DP·Progress·Residual RL 파이프라인은 서버 연결 후 사용할 수 있습니다. 기존 데이터와 수집 기능은 계속 확인할 수 있습니다.</p><button class="button secondary" data-action="pipeline-refresh">${icon("reset")}상태 다시 확인</button></div></section>`;
      return;
    }
    if (!pipeline) {
      $("pipeline-execution").innerHTML = '<p class="learning-empty">파이프라인 상태와 실행 조건을 불러오고 있습니다.</p>';
      return;
    }
    const status = pipeline.status || {}, phase = status.phase || "idle";
    $("pipeline-phase-tag").innerHTML = tag(pipelinePhaseNames[phase] || phase,phase === "running" ? "success" : ["error","interrupted","paused"].includes(phase) ? "warning" : "neutral");
    updatePipelineExecution();
    updatePipelineStages();
    renderPipelineMonitor();
    const modelSignature = JSON.stringify(pipeline.models || []);
    if (modelSignature !== state.modelSignature) {state.modelSignature = modelSignature;renderModelRegistry();}
    const evalSignature = JSON.stringify(pipeline.evaluations || []);
    if (evalSignature !== state.evaluationSignature) {state.evaluationSignature = evalSignature;renderEvaluations();}
    const configSignature = JSON.stringify(pipeline.config || {});
    if (forceConfig || !$("pipeline-config-form") || !state.pipelineDirty && configSignature !== state.pipelineConfigSignature) renderPipelineConfig();
    if ($("pipeline-config-save")) $("pipeline-config-save").disabled = Boolean(state.pipelinePending || pipelineActive(phase));
  }

  function updatePipelineExecution() {
    if (!$("pipeline-execution")) return;
    const pipeline = currentPipeline(), status = pipeline?.status || {}, config = pipeline?.config || {};
    if (!pipeline) return;
    const auto = state.pipelineMode === "auto", busy = pipelineActive(status.phase), paused = status.phase === "paused";
    const global = pipelineRuntime(), other = global && pipelineActive(global.phase) && global.project_id !== state.route.id;
    const ready = pipelineReadiness(auto ? "auto" : state.pipelineStage);
    const unavailable = Boolean(!state.connected || state.pipelinePending || state.pipelinePausePending || busy || other || !ready.ready || state.pipelineDirty);
    $("pipeline-mode-auto").classList.toggle("selected",auto);$("pipeline-mode-auto").setAttribute("aria-pressed",String(auto));
    $("pipeline-mode-manual").classList.toggle("selected",!auto);$("pipeline-mode-manual").setAttribute("aria-pressed",String(!auto));
    const executionSignature = JSON.stringify([status.phase,status.stage,status.mode,status.cycle,status.round,status.error,config.collection_episodes,config.bc_success_target,config.max_bc_cycles,config.max_rl_rounds,state.pipelineMode,state.pipelineStage,ready,global,state.pipelinePending,state.pipelinePausePending,state.connected,state.pipelineDirty]);
    if (executionSignature === state.pipelineExecutionSignature) return;
    state.pipelineExecutionSignature = executionSignature;
    const reason = other ? "다른 프로젝트의 파이프라인이 실행 중입니다." : state.pipelineDirty ? "변경한 학습 설정을 저장한 뒤 실행하세요." : !ready.ready ? ready.reason || "이 단계의 실행 조건을 아직 충족하지 않았습니다." : "";
    $("pipeline-execution").innerHTML = `<div class="pipeline-run-choice"><div class="pipeline-run-copy"><h3>${auto ? "목표 성공률까지 순서대로 진행" : "선택한 단계만 실행"}</h3><p>${auto ? `GPT 수집 ${count(config.collection_episodes)}회 → DP 학습 → BC 평가를 반복합니다. BC ${finiteValue(config.bc_success_target,1)}%를 통과하면 Progress와 Residual RL로 이어집니다.` : "학습 데이터와 이전 모델을 확인하며 각 단계를 개별 실행할 수 있습니다. 다른 단계로 자동 전환하지 않습니다."}</p>${auto ? `<div class="pipeline-run-limits">${tag(`BC 최대 ${count(config.max_bc_cycles)} cycles`)}${tag(`RL 최대 ${count(config.max_rl_rounds)} rounds`)}</div>` : `<label class="field manual-stage-field">실행할 단계<select id="pipeline-manual-stage" ${busy || state.pipelinePending ? "disabled" : ""}>${pipelineStages.map(stage => `<option value="${stage.id}" ${stage.id === state.pipelineStage ? "selected" : ""}>${stage.number} · ${esc(stage.name)}</option>`).join("")}</select></label>`}</div><button id="pipeline-start" class="button primary" data-action="${auto ? "pipeline-start-auto" : "pipeline-start-manual"}" ${unavailable ? "disabled" : ""}>${icon(auto ? "play" : "step")}${state.pipelinePending === "start" ? "시작 요청 중…" : auto ? "전체 파이프라인 시작" : "선택 단계 실행"}</button></div>${reason ? `<p class="pipeline-readiness" role="status">${icon("info")}${esc(reason)}</p>` : ""}<div class="pipeline-control-strip"><div><span class="pipeline-current-label">${status.stage ? esc(pipelineStageName(status.stage)) : "아직 시작하지 않았습니다"}</span><span>${status.mode ? `${status.mode === "auto" ? "자동" : "단계별"} 실행` : ""}${status.cycle ? ` · Cycle ${count(status.cycle)}` : ""}${status.round ? ` · Round ${count(status.round)}` : ""}</span></div><div class="pipeline-control-buttons"><button class="button pause small" data-action="pipeline-pause" ${status.phase !== "running" || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("pause")}${state.pipelinePausePending ? "요청 중…" : "Pause"}</button><button class="button secondary small" data-action="pipeline-resume" ${!paused || state.pipelinePending || state.pipelinePausePending || !state.connected ? "disabled" : ""}>${icon("play")}재개</button><button class="button danger small" data-action="pipeline-stop" ${!busy || state.pipelinePending || !state.connected ? "disabled" : ""}>${icon("stop")}중지</button></div></div>${status.error ? `<p class="pipeline-error" role="alert">${esc(status.error)}</p>` : ""}`;
  }

  function updatePipelineStages() {
    const pipeline = currentPipeline();
    if (!pipeline || !$("pipeline-stages")) return;
    const status = pipeline.status || {};
    $("pipeline-stages").innerHTML = pipelineStages.map(stage => {
      const active = status.stage === stage.id && pipelineActive(status.phase), ready = pipelineReadiness(stage.id);
      const label = active ? pipelinePhaseNames[status.phase] : ready.ready ? "실행 가능" : "조건 대기";
      return `<div class="pipeline-stage ${active ? "active" : ""}"><span class="stage-number">${stage.number}</span><div class="stage-description"><h3>${esc(stage.name)}</h3><p>${esc(stage.description)}</p>${!ready.ready && ready.reason ? `<small>${esc(ready.reason)}</small>` : ""}</div>${tag(label,active ? status.phase === "paused" ? "warning" : "success" : "neutral")}</div>`;
    }).join("");
  }

  function renderPipelineMonitor() {
    const pipeline = currentPipeline();
    if (!pipeline || !$("pipeline-monitor")) return;
    const status = pipeline.status || {}, metrics = status.metrics || {}, events = status.events || [];
    const value = (...keys) => {for (const key of keys) if (typeof metrics[key] === "number") return metrics[key];return undefined;};
    const progress = value("step","critic_updates","update_step","updates","global_step","environment_steps");
    const loss = value("critic_loss","val_loss","validation_loss","progress_validation_mae","progress_loss","train_loss","loss");
    const lossLabel = typeof metrics.critic_loss === "number" ? "Critic loss" : typeof metrics.progress_validation_mae === "number" ? "Progress validation MAE" : typeof metrics.val_loss === "number" || typeof metrics.validation_loss === "number" ? "Validation loss" : "학습 loss";
    const live = status.phase && status.phase !== "idle";
    const metricData = Object.entries(metrics).filter(([key,val]) => ["number","string","boolean"].includes(typeof val));
    const expandedMetrics = $("pipeline-monitor").querySelector(".learning-metric-details")?.open;
    $("pipeline-monitor").innerHTML = `<section class="metrics learning-metrics" aria-label="파이프라인 실행 지표">${metric("현재 단계",`<span class="metric-text">${live ? esc(pipelineStageName(status.stage)) : "실행 대기"}</span>`,`${status.cycle ? `Cycle ${status.cycle} · ` : ""}${progress !== undefined ? `${count(progress)} updates / steps` : "실제 실행 이후 지표가 표시됩니다"}`,"flow")}${metric(lossLabel,finiteValue(loss,5),"현재 단계에서 보고한 값","progress")}${metric("Episode return",finiteValue(value("episode_return","mean_return","return"),3),"Task rollout reward 합계","layers")}${metric("Residual magnitude",finiteValue(value("residual_magnitude","residual_norm","residual_l2"),5),"실행 action에 적용한 보정량","model")}</section><section class="panel learning-monitor-panel"><div class="panel-header"><h2>파이프라인 실행 기록</h2><span class="tag neutral">${count(events.length)} events</span></div>${events.length ? `<ol class="learning-event-list">${events.slice(-10).reverse().map(event => `<li><time>${esc(date(event.time || event.timestamp))}</time><span>${esc(event.message || event.kind || "실행 상태 갱신")}</span></li>`).join("")}</ol>` : '<p class="learning-empty">아직 실행한 단계가 없습니다. 모델이나 평가 결과를 생성하지 않은 상태입니다.</p>'}${metricData.length ? `<details class="learning-metric-details" ${expandedMetrics ? "open" : ""}><summary>현재 단계의 전체 지표</summary><dl>${metricData.map(([key,val]) => `<div><dt>${esc(key)}</dt><dd>${typeof val === "number" ? finiteValue(val,6) : esc(val)}</dd></div>`).join("")}</dl></details>` : ""}</section>`;
  }

  function renderModelRegistry() {
    const models = currentPipeline()?.models || [];
    $("model-count").textContent = `${count(models.length)} models`;
    if (!models.length) {
      $("pipeline-models").innerHTML = `<div class="learning-empty-models">${icon("model")}<h3>아직 학습된 모델이 없습니다</h3><p>DP, Progress, Residual checkpoint가 생성되면 학습 설정·데이터와 함께 여기에 표시됩니다.</p></div>`;
      return;
    }
    const labels = {bc:"Diffusion Policy",progress:"Task Progress",residual:"Residual TD3"};
    $("pipeline-models").innerHTML = `<div class="learned-model-list">${models.slice().sort((a,b) => n(b.created_at)-n(a.created_at)).map(model => {
      const download = asset(model.checkpoint_url), metrics = model.metrics || {}, dataset = model.dataset || {};
      const episodes = dataset.episodes ?? dataset.episode_count ?? dataset.episode_ids?.length;
      const validationLoss = metrics.best_validation_loss ?? metrics.validation_loss ?? metrics.val_loss;
      return `<article class="learned-model"><div class="learned-model-header"><div class="learned-model-icon">${icon(model.kind === "progress" ? "progress" : "model")}</div><div><h3>${esc(model.name || labels[model.kind] || model.kind || "Model")}</h3><p>${esc(model.id || "")} · ${esc(date(model.created_at))}</p></div>${tag(model.status === "ready" ? "사용 가능" : model.status || "저장됨",model.status === "ready" ? "success" : "neutral")}</div><dl class="learned-model-facts"><div><dt>종류</dt><dd>${esc(labels[model.kind] || model.kind || "—")}</dd></div><div><dt>학습 episodes</dt><dd>${episodes === undefined ? "기록 참고" : count(Array.isArray(episodes) ? episodes.length : episodes)}</dd></div>${model.base_model_id ? `<div><dt>Frozen base</dt><dd>${esc(model.base_model_id)}</dd></div>` : ""}${typeof validationLoss === "number" ? `<div><dt>Validation loss</dt><dd>${finiteValue(validationLoss,5)}</dd></div>` : ""}</dl><details class="learned-model-details"><summary>설정·학습 데이터·지표</summary><div class="model-detail-columns">${[["설정",model.config],["학습 데이터",model.dataset],["지표",model.metrics]].filter(([,value]) => value).map(([name,value]) => `<div><h4>${name}</h4><pre>${esc(JSON.stringify(value,null,2))}</pre></div>`).join("")}</div>${model.checkpoint || model.path ? `<p class="model-checkpoint-path">${esc(model.checkpoint || model.path)}</p>` : ""}</details>${download ? `<a class="button secondary small" href="${esc(download)}" download>${icon("download")}Checkpoint 다운로드</a>` : ""}</article>`;
    }).join("")}</div>`;
  }

  function renderEvaluations() {
    const evaluations = (currentPipeline()?.evaluations || []).slice().sort((a,b) => n(b.created_at)-n(a.created_at));
    const label = value => ["residual","rl","rl_eval"].includes(value) ? "BC + Residual" : "BC only";
    const latestBC = evaluations.find(item => ["bc","bc_eval","bc_only"].includes(item.kind || item.policy));
    const latestRL = evaluations.find(item => ["residual","rl","rl_eval"].includes(item.kind || item.policy));
    const card = (title,item) => `<div class="evaluation-comparison-card"><span>${title}</span><strong>${item && typeof item.success_rate === "number" ? `${finiteValue(item.success_rate,1)}<small>%</small>` : "—"}</strong><p>${item ? `${count(item.successes)} / ${count(item.episodes)} episodes 성공` : "아직 완료된 평가가 없습니다"}</p>${item ? `<span class="evaluation-return">평균 return ${finiteValue(item.mean_return,3)}</span>` : ""}</div>`;
    $("pipeline-evaluations").innerHTML = `<div class="evaluation-comparison">${card("BC only · 최근 평가",latestBC)}${card("BC + Residual · 최근 평가",latestRL)}</div>${evaluations.length ? `<div class="table-wrap"><table class="evaluation-table"><thead><tr><th>정책</th><th>평가 시각</th><th>성공률</th><th>Episodes</th><th>평균 return</th><th>모델</th></tr></thead><tbody>${evaluations.map(item => `<tr><td><strong>${label(item.kind || item.policy)}</strong></td><td>${esc(date(item.created_at))}</td><td>${finiteValue(item.success_rate,1)}%</td><td>${count(item.successes)} / ${count(item.episodes)}</td><td>${finiteValue(item.mean_return,3)}</td><td><span class="evaluation-model-id">${esc(item.residual_model_id || item.model_id || item.checkpoint_id || "—")}</span></td></tr>`).join("")}</tbody></table></div>` : ""}<p class="evaluation-footnote">학습 중 loss와 실제 rollout 성공률은 별도로 기록됩니다. 평가의 성공·실패 episode는 데이터 탭에서 확인할 수 있습니다.</p>`;
  }

  function renderPipelineConfig() {
    const pipeline = currentPipeline();
    if (!pipeline || !$("pipeline-config")) return;
    const config = pipeline.config || {}, deps = pipeline.dependencies || {};
    state.pipelineConfigSignature = JSON.stringify(config);state.pipelineDirty = false;
    $("pipeline-config-state").textContent = "저장된 설정";$("pipeline-config-state").className = "tag neutral";
    const field = (name,label,min,max,unit="",step=1) => numericField(name,label,config[name],min,max,unit,step);
    $("pipeline-config").innerHTML = `<form id="pipeline-config-form" class="pipeline-config-form"><div class="pipeline-config-group"><h3>수집과 평가</h3><div class="form-grid three">${field("collection_episodes","Cycle당 GPT 수집",1,100,"episodes")}${field("evaluation_episodes","평가 episode 수",1,100,"episodes")}${field("run_budget_usd","Rollout run당 API 예산",.01,1000,"$",.01)}${field("bc_success_target","BC 성공률 목표",1,100,"%",1)}${field("rl_success_target","RL 성공률 목표",1,100,"%",1)}<label class="field">학습 장치<select name="device"><option value="auto">Auto · 사용 가능한 장치</option><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option>${config.device && !["auto","cuda","cpu"].includes(config.device) ? `<option value="${esc(config.device)}">${esc(config.device)}</option>` : ""}</select></label></div><p class="field-note">${deps.torch === false ? "PyTorch가 설치되지 않아 학습을 실행할 수 없습니다." : deps.cuda ? "CUDA GPU를 사용할 수 있습니다." : "현재 CUDA GPU가 확인되지 않았습니다. Auto는 CPU를 사용합니다."} API 예산은 응답 수신 후 확인하므로 마지막 호출 비용만큼 초과할 수 있습니다.</p></div><div class="pipeline-config-group"><h3>반복 한도와 진행도 판단</h3><div class="form-grid three">${field("max_bc_cycles","BC 최대 cycle",1,1000,"회")}${field("max_rl_rounds","RL 최대 round",1,1000,"회")}${field("rl_episodes_per_round","RL round당 episode",1,100,"개")}${field("progress_gate_threshold","GPT 판단 시작 progress",0,1,"",.01)}${field("gpt_judge_interval","후반 GPT 판단 간격",1,10000,"steps")}</div><p class="field-note">Task 진행도가 설정값에 도달하면 지정한 간격으로 GPT 판단을 요청합니다. Reset은 GPT가 수행하며, 복구하지 못하면 중지합니다.</p></div><details class="advanced-learning-config"><summary>${icon("settings")}모델별 hyperparameter</summary><p class="field-note">각 모델의 전체 설정을 JSON으로 편집합니다. 저장은 학습을 시작하지 않습니다.</p>${[["bc","Diffusion Policy","영상 encoder, observation/action horizon, diffusion·학습 설정"],["progress","Progress 모델","성공 데이터의 진행도 학습과 encoder 설정"],["residual","Residual TD3","보정 범위, replay 혼합, n-step, UTD, critic·target 설정"]].map(([key,label,help]) => `<label class="field advanced-config-field">${label}<small>${help}</small><textarea id="config-${key}" name="config_${key}" rows="12" spellcheck="false" aria-label="${label} hyperparameter JSON">${esc(JSON.stringify(config[key] || {},null,2))}</textarea></label>`).join("")}</details><p id="pipeline-config-error" class="form-error" role="alert" hidden></p><div class="pipeline-config-actions"><p>진행 중인 파이프라인의 설정은 고정됩니다. 중지하거나 완료한 뒤 다음 실행 설정을 저장하세요.</p><div><button class="button secondary" type="button" data-action="pipeline-config-discard">편집 취소</button><button id="pipeline-config-save" class="button primary" type="submit">${icon("check")}학습 설정 저장</button></div></div></form>`;
    $("pipeline-config-form").elements.device.value = config.device || "auto";
  }

  async function refreshPipeline(force = false) {
    const id = state.route.id, token = state.routeToken;
    if (!id || state.pipelineBusy && !force || !force && !["overview","models"].includes(state.route.tab)) return;
    if (!force && !pipelineEnabled()) return;
    const requestId = ++state.pipelineRequest;
    state.pipelineBusy = true;
    try {
      const data = await api(`/api/projects/${encodeURIComponent(id)}/pipeline`);
      if (requestId !== state.pipelineRequest || token !== state.routeToken || id !== state.route.id) return;
      state.pipeline = data;state.pipelineProjectId = id;
      updatePipelineView();renderGlobal();updateCollection();
    } catch (error) {
      if (requestId !== state.pipelineRequest || token !== state.routeToken) return;
      if (!pipelineEnabled()) {state.pipeline = {unavailable:true};state.pipelineProjectId = id;updatePipelineView();}
      else if (force) toast(error.message,true);
    } finally {if (requestId === state.pipelineRequest) state.pipelineBusy = false;}
  }

  async function pipelineCommand(action, payload = {}, projectId = state.route.id) {
    if (!projectId || (action === "pause" ? state.pipelinePausePending : state.pipelinePending)) return;
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

  async function savePipelineConfig(event) {
    event.preventDefault();
    const pipeline = currentPipeline(), projectId = state.route.id, token = state.routeToken;
    const form = event.target;
    if (!pipeline || !form.reportValidity() || state.pipelinePending) return;
    const data = new FormData(form), config = structuredClone(pipeline.config || {});
    try {
      for (const key of ["collection_episodes","evaluation_episodes","run_budget_usd","bc_success_target","rl_success_target","max_bc_cycles","max_rl_rounds","rl_episodes_per_round","progress_gate_threshold","gpt_judge_interval"]) config[key] = Number(data.get(key));
      config.device = data.get("device");
      for (const key of ["bc","progress","residual"]) {
        try {config[key] = JSON.parse(data.get(`config_${key}`));}
        catch {throw new Error(`${key.toUpperCase()} hyperparameter의 JSON 형식을 확인해주세요.`);}
        if (!config[key] || typeof config[key] !== "object" || Array.isArray(config[key])) throw new Error(`${key.toUpperCase()} 설정은 JSON object여야 합니다.`);
      }
      state.pipelinePending = "config";$("pipeline-config-save").disabled = true;$("pipeline-config-error").hidden = true;
      const result = await api(`/api/projects/${encodeURIComponent(projectId)}/pipeline/config`,config);
      if (token !== state.routeToken || state.route.id !== projectId) return;
      state.pipelineDirty = false;
      if (result.config) {state.pipeline = result;state.pipelineProjectId = projectId;}
      await refreshPipeline(true);
      if (token === state.routeToken) updatePipelineView(true);
      toast("학습 설정을 저장했습니다. 실행은 아직 시작하지 않았습니다.");
    } catch (error) {
      if (token === state.routeToken && $("pipeline-config-error")) {$("pipeline-config-error").textContent = error.message;$("pipeline-config-error").hidden = false;}
      else toast(error.message,true);
    } finally {state.pipelinePending = null;if (token === state.routeToken) updatePipelineView();}
  }

  const numericField = (name, label, value, min, max, unit = "", step = 1) => `<label class="field">${esc(label)}<div class="input-unit"><input name="${name}" type="number" min="${min}" max="${max}" step="${step}" required value="${n(value)}">${unit ? `<span>${esc(unit)}</span>` : ""}</div></label>`;

  function renderSettings(project) {
    const config = project.run_config || {}, pipe = project.pipeline || {}, selected = nextPolicy(project);
    const models = policyOptions().models || [];
    const known = models.some(item => item.model === selected.model);
    const modelChoices = `${known ? "" : `<option value="${esc(selected.model || "")}">${esc(selected.model || "서버 기본값")} · 서버 설정</option>`}${models.map(item => `<option value="${esc(item.model)}">${esc(item.label || item.model)}</option>`).join("")}`;
    $("tab-content").innerHTML = `<form id="settings-form" class="settings-layout"><section class="panel settings-section"><div class="panel-header"><div><h2>프로젝트 정보</h2><p>Task별 데이터와 실행을 구분하는 이름과 설명</p></div></div><div class="panel-body"><div class="form-grid"><label class="field full">프로젝트 이름<input name="name" required maxlength="120" value="${esc(project.name)}"></label><label class="field full">설명<textarea name="description" rows="2" maxlength="2000">${esc(project.description || "")}</textarea></label><label class="field">환경<input readonly value="${esc(project.backend?.toUpperCase())} · ${esc(project.suite)}"></label><label class="field">Task<input readonly value="${n(project.task_id)} · ${esc(project.task_name || "")}"></label></div></div></section><section class="panel settings-section"><div class="panel-header"><div><h2>GPT 모델 설정</h2><p>이 프로젝트의 데이터 수집과 reset에 사용할 모델</p></div>${tag("프로젝트별 설정","success")}</div><div class="panel-body"><div class="form-grid"><label class="field">추론 모델<select id="policy-model" name="policy_model" required>${modelChoices}</select></label><label class="field">추론 수준<select id="policy-effort" name="reasoning_effort" required></select></label></div><div id="policy-run-status" class="policy-run-status"></div><p class="field-note">저장 후 다음 <strong>환경 준비</strong> 또는 <strong>복구된 환경에서 이어 수집</strong>부터 적용됩니다. 이미 준비했거나 일시정지한 실행은 기존 모델·추론 수준을 유지합니다.</p><p id="policy-model-note" class="field-note"></p></div></section><section class="panel settings-section"><div class="panel-header"><div><h2>데이터 수집</h2><p>다음 실행에 적용될 episode 수와 실행 한도</p></div>${tag("사용 가능","success")}</div><div class="panel-body"><div class="form-grid three">${numericField("episodes","Episode 수",config.episodes || 1,1,100,"개")}${numericField("budget_usd","API 비용 한도",config.budget_usd || 15,.01,1000,"$",.01)}${numericField("seed","초기 seed",config.seed || 0,0,2147473648)}${numericField("max_steps","Task 최대 step",config.max_steps || 300,1,10000)}${numericField("max_decisions","Task 최대 GPT 호출",config.max_decisions || 50,1,1000)}<label class="field">Reset 방식<select name="reset_mode"><option value="agent">GPT arm reset</option><option value="environment">환경 초기화</option></select></label>${numericField("max_reset_steps","Reset 최대 step",config.max_reset_steps || 300,1,10000)}${numericField("max_reset_decisions","Reset 최대 GPT 호출",config.max_reset_decisions || 40,1,1000)}</div><p class="field-note">GPT arm reset은 같은 물리 상태에서 복구를 수행합니다. 비용 한도는 응답 후 확인하므로 마지막 요청 비용만큼 초과할 수 있습니다.</p></div></section><section class="panel settings-section"><div class="panel-header"><div><h2>학습 파이프라인 목표</h2><p>Orchestration에서 확인할 단계별 전환 조건</p></div>${tag("계획 · 미구현")}</div><div class="panel-body"><div class="form-grid three">${numericField("demo_target","BC 시작 성공 demo 수",pipe.demo_target || 20,1,100000,"개")}${numericField("bc_success_target","BC 성공률 목표",pipe.bc_success_target || 70,1,100,"%")}${numericField("rl_success_target","RL 성공률 목표",pipe.rl_success_target || 90,1,100,"%")}</div><p class="field-note">목표는 저장되지만 학습과 평가는 아직 실행되지 않습니다. 현재 수집된 demo 수만 실제 기록에서 집계합니다.</p></div></section><p id="settings-error" class="form-error" role="alert" hidden></p><div class="settings-actions"><p>저장해도 실행 중인 수집 설정은 바뀌지 않습니다.</p><button id="settings-save" class="button primary" type="submit">${icon("check")}설정 저장</button></div></form>`;
    $("settings-form").elements.reset_mode.value = config.reset_mode || "agent";
    if (pipelineEnabled()) {
      const section = [...document.querySelectorAll("#settings-form .settings-section")].find(node => node.querySelector("h2")?.textContent === "학습 파이프라인 목표");
      if (section) section.innerHTML = `<div class="panel-header"><div><h2>학습 파이프라인 설정</h2><p>수집 수·성공률 목표와 모델별 hyperparameter</p></div>${tag("모델 탭에서 관리","success")}</div><div class="panel-body learning-settings-link"><p>DP, Progress, Residual TD3의 학습 설정과 자동·단계별 실행은 모델 탭에서 관리합니다. 학습 중에는 데이터 수집을 멈춥니다.</p><a class="button secondary" href="${routeUrl(project.id,"models")}">${icon("model")}모델 · 파이프라인 설정</a></div>`;
    }
    $("policy-model").value = selected.model || "";
    updatePolicyEfforts(selected.reasoning_effort);
    updatePolicyStatus();
  }

  function updatePolicyEfforts(preferred = $("policy-effort")?.value) {
    const select = $("policy-effort"), model = $("policy-model");
    if (!select || !model) return;
    const entry = (policyOptions().models || []).find(item => item.model === model.value);
    const values = entry?.reasoning_efforts || [nextPolicy(currentProject()).reasoning_effort].filter(Boolean);
    select.innerHTML = values.map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join("");
    select.value = values.includes(preferred) ? preferred : entry?.default_reasoning_effort || values[0] || "";
    $("policy-model-note").textContent = entry ? "이미지 입력과 action 생성을 지원하는 모델입니다. 실제 사용 가능 여부는 서버 API key의 접근 권한에 따라 달라집니다." : "현재 서버 기본 모델입니다. 모델을 변경하려면 목록의 지원 모델을 선택하세요.";
  }

  function updatePolicyStatus() {
    const holder = $("policy-run-status"), project = currentProject();
    if (!holder || !project) return;
    const s = runtime(), active = s.active_project_id === project.id && s.run_id;
    const selected = {model:$("policy-model")?.value,reasoning_effort:$("policy-effort")?.value};
    const pending = active && (s.model !== selected.model || s.reasoning_effort !== selected.reasoning_effort);
    holder.innerHTML = `<div><span>선택한 설정</span><strong>${esc(policyLabel(selected))}</strong></div>${active ? `<div><span>준비된 실행의 설정</span><strong>${esc(policyLabel(s))}</strong>${pending ? tag("다음 실행에 적용","warning") : tag("동일한 설정")}</div>` : `<p>실행을 준비하면 선택한 설정을 실행 기록에 함께 저장합니다.</p>`}`;
  }

  function renderCollection(project) {
    state.lastObservation = null;state.lastCameras = "";state.lastEvents = "";
    $("tab-content").innerHTML = `<div class="collection-layout"><div><section class="panel"><div class="panel-header"><div><h2>실시간 수집</h2><p id="live-subtitle">실행 환경을 준비한 뒤 관측을 확인하세요.</p></div><span id="live-state" class="tag neutral">준비 전</span></div><div id="live-images"><div class="live-placeholder">${icon("camera")}<h3>수집 환경을 준비하세요</h3><p>환경 준비 후 현재 카메라를 확인할 수 있습니다.<br>수집 시작을 누르기 전까지 GPT는 호출되지 않습니다.</p></div></div><div id="observation-footer" class="observation-footer"><span>카메라 대기 중</span><span>현재 프로젝트의 관측만 표시합니다.</span></div><div class="execution-meta"><div><span>현재 단계</span><strong id="live-stage">—</strong></div><div><span>Task / Reset step</span><strong id="live-steps">0 / 0</strong></div><div><span>실행 API 비용</span><strong id="live-cost">—</strong></div></div><div class="collection-record-link">${icon("database")}<span>GPT 판단 요약과 action 기록은 <a href="${routeUrl(project.id,"data")}">데이터 → Episode 상세</a>에서 확인하세요.</span></div></section><section class="panel events-panel"><div class="panel-header"><h2>실행 이벤트</h2><span id="events-count" class="tag neutral">0</span></div><ol id="events-list" class="events-list"><li class="empty-event">이 프로젝트의 실행 이벤트가 여기에 표시됩니다.</li></ol></section></div><aside class="panel control-panel"><div class="panel-header"><h2>수집 제어</h2><a class="icon-button" href="${routeUrl(project.id,"settings")}" aria-label="수집 설정">${icon("settings")}</a></div><div class="panel-body"><section class="control-section"><span class="control-label">01 · 환경 준비</span><button id="initialize-project" class="button secondary full" data-action="initialize" disabled>${icon("robot")}환경 준비</button><p id="initialize-note" class="control-note">새 실행을 위한 환경을 초기화합니다. GPT는 아직 호출하지 않습니다.</p></section><section class="control-section"><span class="control-label">02 · 데이터 수집</span><button id="start-collection" class="button primary full" data-action="start" disabled>${icon("play")}수집 시작</button><div class="control-row"><button id="pause-collection" class="button pause" data-action="pause" disabled>${icon("pause")}Pause</button><button id="step-collection" class="button secondary" data-action="step" disabled>${icon("step")}한 번 실행</button></div><button id="stop-collection" class="button danger full" data-action="stop" style="margin-top:9px" disabled>${icon("stop")}수집 중지</button><p id="control-note" class="control-note">환경을 먼저 준비해주세요.</p></section><section class="control-section"><dl class="session-rows"><div><dt>실행 목표</dt><dd id="live-goal">${n(project.run_config?.episodes) || 1} episodes / run</dd></div><div><dt>Reset</dt><dd id="live-reset-mode">${project.run_config?.reset_mode === "environment" ? "환경 초기화" : "GPT arm reset"}</dd></div><div><dt>비용 한도</dt><dd id="live-budget">${money(project.run_config?.budget_usd)}</dd></div><div><dt>Teacher</dt><dd id="teacher-name">—</dd></div><div><dt>Reasoning</dt><dd id="teacher-effort">—</dd></div><div id="next-policy-row" hidden><dt>다음 실행</dt><dd id="next-policy">—</dd></div><div><dt>입력 / 출력 tokens</dt><dd id="live-tokens">—</dd></div><div><dt>Run</dt><dd id="live-run">—</dd></div></dl></section></div></aside></div>`;
    updateCollection();
  }

  function updateCollection() {
    if (state.route.tab !== "collection" || !$("initialize-project")) return;
    const s = state.status || {}, p = currentProject();
    if (!p) return;
    const own = s.active_project_id === p.id;
    const learning = pipelineRuntime(), managed = learning && pipelineActive(learning.phase), managedOwn = managed && learning.project_id === p.id;
    const preparingOwn = state.pending === "initialize" && state.pendingProjectId === p.id;
    const another = !own && !preparingOwn && activePhases.has(s.phase);
    const ready = own && ["ready","paused"].includes(s.phase);
    const ep = own ? s.episode || {} : {};
    const cycleDone = Boolean(ep.outcome && (s.config?.reset_mode === "environment" || ep.reset_outcome));
    const batchDone = cycleDone && (n(ep.index) >= n(s.config?.episodes) || s.config?.reset_mode === "agent" && ep.reset_outcome !== "success");
    const continuing = own && ["completed","paused"].includes(s.phase) && s.stage === "reset" && s.observation?.success && s.config?.seed === p.run_config?.seed && s.config?.task_id === p.task_id;
    const canInit = !another && (!own || ["idle","stopped","completed","error"].includes(s.phase) || continuing);
    const runnable = p.runnable !== false && state.detail.runnable !== false;
    const locked = Boolean(state.pending || state.pausePending || !state.connected || managed);
    $("initialize-project").disabled = locked || !runnable || !canInit;
    $("initialize-project").innerHTML = icon("robot") + (state.pending === "initialize" ? "환경 준비 중…" : continuing ? "복구된 환경에서 이어 수집" : own && activePhases.has(s.phase) ? "환경 준비 완료" : "환경 준비");
    $("initialize-note").textContent = continuing ? "검증된 GPT reset 상태에서 새 실행을 준비합니다. 환경을 초기화하지 않습니다." : "새 실행을 위한 환경을 초기화합니다. GPT는 아직 호출하지 않습니다.";
    const hasPolicy = s.policy !== "openai" || s.api_key_configured;
    $("start-collection").disabled = locked || !ready || batchDone || !hasPolicy;
    $("start-collection").innerHTML = icon("play") + (own && s.phase === "paused" ? "수집 재개" : "수집 시작");
    $("pause-collection").disabled = state.pausePending || !state.connected || !(own || preparingOwn) || !pausePhases.has(s.phase);
    $("pause-collection").innerHTML = icon("pause") + (state.pausePending ? "요청 중…" : "Pause");
    $("step-collection").disabled = locked || !ready || batchDone || !hasPolicy;
    $("stop-collection").disabled = locked || !own || !activePhases.has(s.phase);
    const note = state.pausePending ? "일시정지를 요청했습니다. 환경 초기화 중이면 초기화가 끝난 뒤 적용됩니다." : !runnable ? p.runnable_reason || state.detail.runnable_reason || "현재 서버에서 이 프로젝트 환경을 실행할 수 없습니다." : another ? "다른 실행에서 환경을 사용 중입니다. 활성 프로젝트 또는 실행 콘솔에서 해당 실행을 중지한 뒤 준비할 수 있습니다." : !hasPolicy ? "서버에 API key가 설정되지 않았습니다. 환경 관측만 확인할 수 있습니다." : own && s.error ? s.error : own && s.phase === "paused" ? "일시정지 상태입니다. 재개하면 새 관측으로 다음 동작을 판단합니다." : own && s.phase === "completed" ? "실행이 종료되었습니다. 결과는 데이터 탭에서 확인하세요." : own && s.phase === "thinking" ? "GPT가 다음 동작을 판단하고 있습니다. Pause하면 대기 중 응답을 폐기합니다." : own && s.phase === "executing" ? "Action을 실행하고 있습니다. Pause는 현재 control tick이 끝난 뒤 적용됩니다." : ready ? "수집 시작은 연속 실행, 한 번 실행은 GPT 판단 1회 후 일시정지합니다." : "환경을 먼저 준비해주세요.";
    $("control-note").textContent = note;$("control-note").classList.toggle("alert",Boolean(!runnable || another || !hasPolicy || own && s.error));
    $("pause-collection").dataset.action = managedOwn ? "pipeline-pause" : "pause";
    $("stop-collection").dataset.action = managedOwn ? "pipeline-stop" : "stop";
    if (managedOwn) {
      $("pause-collection").disabled = learning.phase !== "running" || state.pipelinePausePending || !state.connected;
      $("pause-collection").innerHTML = icon("pause") + (state.pipelinePausePending ? "요청 중…" : "Pause");
      $("stop-collection").disabled = Boolean(state.pipelinePending || !state.connected);
      $("control-note").textContent = `${pipelineStageName(learning.stage)} 단계가 파이프라인에서 ${learning.phase === "paused" ? "일시정지되어 있습니다" : "진행 중입니다"}. 재개와 단계 실행은 모델 탭에서 관리하세요.`;
      $("control-note").classList.remove("alert");
      $("initialize-note").textContent = "파이프라인이 환경을 관리합니다. 학습 중에는 수집을 멈추고 다음 rollout에서 이어갑니다.";
    } else if (managed) $("control-note").textContent = "다른 프로젝트의 파이프라인이 환경을 사용 중입니다. 모델 탭에서 실행 상태를 확인하세요.";
    $("live-state").className = `tag ${own && activePhases.has(s.phase) ? s.phase === "paused" ? "warning" : "success" : "neutral"}`;
    $("live-state").textContent = own ? s.phase === "thinking" && s.stage !== "reset" && ["bc_eval","rl","rl_eval"].includes(s.learning?.pipeline_stage) ? "학습 정책 action 계산 중" : phaseNames[s.phase] || s.phase : "준비 전";
    $("live-subtitle").textContent = own ? `Episode ${n(ep.index) || 1} / ${n(s.config?.episodes) || 1} · ${s.stage === "reset" ? "GPT가 환경을 복구하고 있습니다." : "Task를 수행하는 관측과 실행 기록"}` : "실행 환경을 준비한 뒤 관측을 확인하세요.";
    $("live-stage").textContent = own && s.observation ? s.stage === "reset" ? "GPT Reset" : s.learning?.pipeline_stage && s.learning.pipeline_stage !== "collect" ? pipelineStageName(s.learning.pipeline_stage) : "Task" : "—";
    const shownRun = own && s.run_id ? s.config || {} : p.run_config || {};
    $("live-goal").textContent = `${n(shownRun.episodes) || 1} episodes / run`;
    $("live-reset-mode").textContent = shownRun.reset_mode === "environment" ? "환경 초기화" : "GPT arm reset";
    $("live-budget").textContent = money(shownRun.budget_usd);
    $("live-steps").textContent = `${count(ep.steps)} / ${count(ep.reset_steps)}`;
    $("live-cost").textContent = own ? money(s.stats?.estimated_cost_usd,4) : "—";
    const selectedPolicy = nextPolicy(p), shownPolicy = own && s.run_id ? s : selectedPolicy;
    $("teacher-name").textContent = s.policy === "scripted" ? "Scripted demo" : shownPolicy.model || "—";
    $("teacher-effort").textContent = s.policy === "scripted" ? "—" : shownPolicy.reasoning_effort || "—";
    $("next-policy-row").hidden = !(own && s.run_id && s.policy !== "scripted" && (s.model !== selectedPolicy.model || s.reasoning_effort !== selectedPolicy.reasoning_effort));
    $("next-policy").textContent = policyLabel(selectedPolicy);
    $("live-tokens").textContent = own ? `${count(s.stats?.input_tokens)} / ${count(s.stats?.output_tokens)}` : "—";
    $("live-run").textContent = own ? s.run_id || "—" : "—";
    const observation = own ? s.observation : null;
    if (observation) {
      const cameras = Array.isArray(observation.cameras) ? observation.cameras.map(x => typeof x === "string" ? x : x.name).filter(Boolean) : Object.keys(observation.cameras || {});
      const signature = cameras.join("|");
      if (signature !== state.lastCameras) {
        $("live-images").innerHTML = `<div class="camera-grid ${cameras.length === 1 ? "single" : ""}">${cameras.map((camera,index) => `<div class="camera ${camera.includes("reference") ? "reference" : ""}"><span class="camera-label">${esc(camera.includes("reference") ? "초기 상태 참고 이미지" : camera)}</span><img id="live-camera-${index}" alt="${esc(camera)} 현재 관측"></div>`).join("")}</div>`;
        state.lastCameras = signature;
      }
      if (observation.id !== state.lastObservation) {
        cameras.forEach((camera,index) => {$("live-camera-"+index).src = `/api/frame/${encodeURIComponent(camera)}?observation_id=${encodeURIComponent(observation.id)}`;});
        state.lastObservation = observation.id;
      }
      $("observation-footer").innerHTML = `<span>마지막 관측 ${esc(date(observation.timestamp,{second:"2-digit"}))}</span><span>관측 ID ${esc(String(observation.id).split(":").slice(-2).join(":"))}</span>`;
    } else if (state.lastObservation !== null || state.lastCameras) {
      $("live-images").innerHTML = `<div class="live-placeholder">${icon("camera")}<h3>수집 환경을 준비하세요</h3><p>현재 프로젝트에 연결된 실시간 관측이 없습니다.<br>환경 준비 후 카메라를 확인할 수 있습니다.</p></div>`;
      $("observation-footer").innerHTML = "<span>카메라 대기 중</span><span>현재 프로젝트의 관측만 표시합니다.</span>";
      state.lastObservation = null;state.lastCameras = "";
    }
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
    if (match) return `Episode ${match[1]} 시작`;
    match = message.match(/^Episode (\d+) ended: (.+)\.$/);
    if (match) return `Episode ${match[1]} Task ${match[2] === "success" ? "성공" : `종료 (${match[2]})`}`;
    if (message.startsWith("GPT reset started")) return "GPT reset 시작 · 현재 환경에서 복구합니다.";
    if (message.startsWith("GPT reset ended: success")) return "GPT reset 완료 · 환경 복구 검증을 통과했습니다.";
    if (message.startsWith("GPT reset ended:")) return "GPT reset 미완료 · 환경을 확인해주세요.";
    if (message.startsWith("Environment initialized")) return "환경 준비 완료 · 관측을 확인한 뒤 수집을 시작하세요.";
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
    ({overview:renderOverview,collection:renderCollection,data:renderData,models:renderModels,settings:renderSettings}[state.route.tab] || renderOverview)(project);
  }

  async function refreshWorkspace(force = false) {
    if (state.workspaceBusy) return;
    state.workspaceBusy = true;
    try {
      const data = await api("/api/workspace");
      const signature = JSON.stringify([data.projects,data.totals]);
      const changed = signature !== state.workspaceSignature;
      state.workspace = data;state.workspaceSignature = signature;
      renderNav();renderGlobal();
      if (!state.route.id && (changed || force) && document.activeElement?.id !== "project-search") renderHome();
      if (state.route.id && (force || !["settings","collection","models"].includes(state.route.tab))) await loadDetail(false);
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
      renderGlobal();updateCollection();updatePolicyStatus();
    } catch {
      if (requestId !== state.statusRequest) return;
      state.connected = false;$("connection").className = "connection offline";$("connection").innerHTML = "<i></i>연결 확인 필요";
      renderGlobal();updateCollection();
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
      if (force || changed && !["settings","collection","models"].includes(state.route.tab)) renderProject();
      else if (state.route.tab === "collection") updateCollection();
      else if (state.route.tab === "models") updatePipelineView();
      if (state.route.tab === "overview") refreshPipeline();
    } catch (error) {
      if (token !== state.routeToken) return;
      if (force) $("content").innerHTML = `<a class="back-link" href="#/">${icon("back")}워크스페이스로</a><section class="panel"><div class="empty-state">${esc(error.message)}</div></section>`;
      else toast(error.message,true);
    }
  }

  async function navigate() {
    const pieces = location.hash.replace(/^#\/?/,"").split("/");
    let id = null;
    try { id = pieces[0] === "projects" && pieces[1] ? decodeURIComponent(pieces[1]) : null; } catch { id = null; }
    const tabs = ["overview","collection","data","models","settings"];
    const tab = tabs.includes(pieces[2]) ? pieces[2] : "overview";
    state.route = {id,tab};state.routeToken += 1;state.detailSignature = "";
    if (state.detail?.project?.id !== id) {state.detail = null;state.pipeline = null;state.pipelineProjectId = null;}
    state.pipelineDirty = false;
    renderNav();renderBreadcrumb();
    if (!id) { if (state.workspace) renderHome(); return; }
    $("content").innerHTML = '<div class="loading-view"><span class="spinner"></span><p>프로젝트를 불러오고 있어요.</p></div>';
    await loadDetail();
  }

  async function command(action, global = false) {
    if (action === "pause" ? state.pausePending : state.pending || state.pausePending) return;
    const id = global ? activeId() : state.route.id;
    const unownedPause = action === "pause" && (global && !id || state.pending === "initialize" && state.pendingProjectId === id && activeId() !== id);
    if (!id && !unownedPause) return;
    let payload = {};
    if (action === "initialize") {
      const s = state.status || {}, p = currentProject();
      payload.preserve_environment = s.active_project_id === id && ["completed","paused"].includes(s.phase) && s.stage === "reset" && Boolean(s.observation?.success) && s.config?.seed === p?.run_config?.seed && s.config?.task_id === p?.task_id;
    }
    if (action === "pause") state.pausePending = true;
    else {state.pending = action;state.pendingProjectId = id;}
    renderGlobal();updateCollection();
    try {
      await api(unownedPause ? "/api/pause" : `/api/projects/${encodeURIComponent(id)}/${action}`,payload,action === "initialize" || action === "pause" && state.pending === "initialize" ? 120000 : 15000);
      // A delayed Start response must never replace a newer Pause snapshot.
      await refreshStatus(true);
      const messages = {initialize:"환경이 준비되었습니다. 관측을 확인한 뒤 수집을 시작하세요.",start:"수집 실행을 요청했습니다.",resume:"수집 재개를 요청했습니다.",pause:"일시정지했습니다.",step:"GPT 판단 한 번을 실행합니다.",stop:"수집을 중지했습니다. 기록은 유지됩니다."};
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

  function updateCreateTask() {
    const task = state.tasks.find(x => String(x.id) === $("create-task").value);
    const name = $("create-name");
    if (!name.value || name.dataset.auto === "true") {
      name.value = `[${(state.workspace?.runtime.backend || "libero").toUpperCase()}] ${shortTask(task)}`;
      name.dataset.auto = "true";
    }
    $("task-help").textContent = task?.agent_reset_supported === false ? "이 task에는 독립 GPT reset 검증이 아직 없습니다. 실행하려면 지원 task를 선택하거나 설정에서 환경 초기화 모드를 명시적으로 선택하세요." : "프로젝트를 만들어도 환경이나 모델은 실행되지 않습니다.";
  }

  async function loadTasks() {
    const token = ++state.taskToken;
    const backend = state.workspace?.runtime.backend || "libero";
    $("create-task").disabled = true;$("create-submit").disabled = true;
    $("create-task").innerHTML = '<option value="">Task를 불러오는 중…</option>';
    $("create-error").hidden = true;
    try {
      const data = await api(`/api/tasks${backend === "libero" ? `?suite=${encodeURIComponent($("create-suite").value)}` : ""}`,undefined,30000);
      if (token !== state.taskToken) return;
      state.tasks = data.tasks || [];
      $("create-task").innerHTML = state.tasks.map(task => `<option value="${n(task.id)}">${n(task.id)} · ${esc(task.language || task.name || "Task")}</option>`).join("");
      const selectedId = n(state.status?.config?.task_id);
      if (state.tasks.some(task => task.id === selectedId)) $("create-task").value = String(selectedId);
      $("create-task").disabled = !state.tasks.length;$("create-submit").disabled = !state.tasks.length;
      if (!state.tasks.length) throw new Error("사용 가능한 task가 없습니다.");
      updateCreateTask();
    } catch (error) {
      if (token !== state.taskToken) return;
      $("create-task").innerHTML = '<option value="">Task를 불러오지 못했습니다</option>';
      $("create-error").textContent = error.message;$("create-error").hidden = false;
    }
  }

  function openCreate() {
    if (!state.workspace) return;
    $("create-form").reset();$("create-error").hidden = true;$("create-name").dataset.auto = "true";
    const backend = state.workspace.runtime.backend || "libero", suite = state.workspace.runtime.suite || "libero_90";
    $("create-backend").value = backend.toUpperCase();$("create-suite").disabled = backend !== "libero";
    if (![...$("create-suite").options].some(o => o.value === suite)) $("create-suite").add(new Option(suite,suite));
    $("create-suite").value = suite;
    $("create-dialog").showModal();
    loadTasks();
  }

  async function createProject(event) {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    const data = new FormData(form), task = state.tasks.find(x => String(x.id) === data.get("task_id"));
    if (!task) return;
    const payload = {name:String(data.get("name")).trim(),description:String(data.get("description") || "").trim(),backend:state.workspace.runtime.backend,suite:$("create-suite").value,task_id:task.id,task_name:task.language || task.name,pipeline:{demo_target:Number(data.get("demo_target")),bc_success_target:Number(data.get("bc_success_target")),rl_success_target:Number(data.get("rl_success_target"))}};
    $("create-submit").disabled = true;$("create-error").hidden = true;
    try {
      const result = await api("/api/projects",payload);
      $("create-dialog").close();toast("프로젝트를 만들었습니다. 수집은 아직 시작되지 않았습니다.");
      await refreshWorkspace();location.hash = routeUrl(result.project.id);
    } catch (error) {$("create-error").textContent = error.message;$("create-error").hidden = false;}
    finally {$("create-submit").disabled = false;}
  }

  async function saveSettings(event) {
    event.preventDefault();
    const form = event.target, project = currentProject(), routeToken = state.routeToken;
    if (!project || !form.reportValidity()) return;
    const data = new FormData(form), config = {...project.run_config};
    for (const key of ["episodes","budget_usd","seed","max_steps","max_decisions","max_reset_steps","max_reset_decisions"]) config[key] = Number(data.get(key));
    config.reset_mode = data.get("reset_mode");config.task_id = project.task_id;
    const payload = {name:String(data.get("name")).trim(),description:String(data.get("description") || "").trim(),run_config:config,pipeline:{demo_target:Number(data.get("demo_target")),bc_success_target:Number(data.get("bc_success_target")),rl_success_target:Number(data.get("rl_success_target"))}};
    if (!form.elements.demo_target) delete payload.pipeline;
    if ((policyOptions().models || []).some(item => item.model === data.get("policy_model"))) {
      payload.policy_config = {model:String(data.get("policy_model")),reasoning_effort:String(data.get("reasoning_effort"))};
    }
    $("settings-save").disabled = true;$("settings-error").hidden = true;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(project.id)}/settings`,payload);
      const stillHere = routeToken === state.routeToken && currentProject()?.id === project.id;
      if (stillHere) state.detail.project = result.project;
      toast("설정을 저장했습니다. 다음 환경 준비 또는 복구 후 이어 수집부터 적용됩니다.");
      await refreshWorkspace();renderNav();renderBreadcrumb();
      if (routeToken === state.routeToken && currentProject()?.id === project.id) {
        const heading = document.querySelector(".project-heading h1");if (heading) heading.textContent = result.project.name;
      }
    } catch (error) {
      if (routeToken === state.routeToken && $("settings-error")) {$("settings-error").textContent = error.message;$("settings-error").hidden = false;}
      else toast(error.message,true);
    }
    finally { if (routeToken === state.routeToken && $("settings-save")) $("settings-save").disabled = false; }
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
    $("delete-episode-info").innerHTML = `<strong>${esc(project.name)} · Episode ${String(ep.index || 1).padStart(2,"0")}</strong><span>${esc(date(ep.started_at))}</span><code>${esc(run)} / ${esc(id)}</code>`;
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
      return `<details class="trace-record decision-record" ${index <= 1 ? "open" : ""} id="${id}"><summary><span class="trace-record-heading">${tag(stage,record.stage === "reset" ? "warning" : "neutral")}<strong>${learned ? "정책 action" : "GPT 판단"}</strong><span class="trace-note-preview">${esc(decision.note || decision.status || "판단 기록")}</span></span><time>${time}</time></summary><div class="trace-record-body"><p class="trace-note">${esc(decision.note || "별도 판단 요약이 기록되지 않았습니다.")}</p><div class="trace-decision-meta"><span>응답 상태: ${esc(decision.status || "—")}</span><span>관측: ${esc(decision.observation_id || "—")}</span></div>${recordActions(decision.actions)}</div></details>`;
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
      const teacher = `${result.config?.model || "기록된 모델 없음"} / ${result.config?.reasoning_effort || "—"}`;
      $("episode-policy").textContent = `${learning.pipeline_stage ? `${pipelineStageName(learning.pipeline_stage)} · GPT teacher: ` : ""}${teacher}`;
      $("episode-learning-models").textContent = [learning.base_model_id ? `Base: ${learning.base_model_id}` : "",learning.residual_model_id ? `Residual: ${learning.residual_model_id}` : ""].filter(Boolean).join(" · ");
      $("episode-learning-models").hidden = !learning.base_model_id && !learning.residual_model_id;
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
    state.episodeDetail = {project:project.id,run,id,records:[],nextOffset:0,loading:false};
    $("episode-title").textContent = `Episode ${String(ep.index || 1).padStart(2,"0")} · ${project.name}`;
    const video = asset(ep.video_url), poster = asset(ep.poster_url);
    const wall = ep.reset_finished_at && ep.started_at ? (ep.reset_finished_at-ep.started_at)/60 : ep.finished_at && ep.started_at ? (ep.finished_at-ep.started_at)/60 : null;
    $("episode-content").innerHTML = `<div class="episode-info">${outcomeLabel(ep.outcome)}${resetLabel(ep.reset_outcome)}${ep.archived ? tag("보관됨") : ""}<span>${esc(date(ep.started_at))}</span></div><div class="episode-policy"><span>실행 모델 / 추론 수준</span><strong id="episode-policy">기록을 불러오는 중…</strong><span id="episode-learning-models" hidden></span></div>${video ? `<video controls playsinline preload="metadata" src="${esc(video)}" ${poster ? `poster="${esc(poster)}"` : ""}></video><p class="episode-video-note">시뮬레이션 시간 기준 영상 · GPT 추론 대기는 제외됩니다. 영상 재생은 로봇 제어와 연결되지 않습니다.</p>` : '<div class="episode-video-empty">이 episode의 MP4가 아직 생성되지 않았습니다.</div>'}<div class="table-wrap"><table><thead><tr><th>단계</th><th>Step</th><th>GPT 호출</th><th>Input</th><th>Output</th><th>추정 비용</th></tr></thead><tbody>${["task","reset"].map(stage => {const cost = stageCost(ep,stage);return `<tr><td>${stage === "task" ? "Task" : "Reset"}</td><td>${count(stage === "task" ? ep.steps : ep.reset_steps)}</td><td>${count(cost.api_requests ?? (stage === "task" ? ep.decisions : ep.reset_decisions))}</td><td>${count(cost.input_tokens)}</td><td>${count(cost.output_tokens)}</td><td>${money(cost.estimated_cost_usd,4)}</td></tr>`;}).join("")}</tbody></table></div><div class="episode-meta"><span>${esc(run)} / ${esc(id)}</span><span>${wall !== null ? `실제 경과 ${wall.toFixed(1)}분 · ` : ""}총 ${money(episodeCost(ep),4)}</span>${video ? `<a href="${esc(video)}" download class="table-action">${icon("download")}MP4 다운로드</a>` : ""}</div><section class="episode-trace-section"><div class="trace-heading"><h3>정책 판단과 실행 기록</h3><span id="trace-count" class="tag neutral">불러오는 중</span></div><p class="field-note">정책의 action과 실행 결과, GPT가 반환한 판단 요약입니다. 학습 rollout에는 BC action·진행도·보정 지표도 포함됩니다. 모델 내부 추론 원문은 포함되지 않습니다.</p><div id="episode-trace" class="episode-trace"><p class="trace-loading">기록을 불러오고 있습니다.</p></div><p id="trace-error" class="form-error" role="alert" hidden></p><button id="trace-more" class="button secondary trace-more" data-action="trace-more" hidden>기록 더 보기</button></section>`;
    if (!$("episode-dialog").open) $("episode-dialog").showModal();
    loadEpisodeRecords();
  }

  document.addEventListener("click", event => {
    const close = event.target.closest("[data-close-dialog]");
    if (close) {if (close.closest("dialog").id !== "delete-dialog" || !state.deleteBusy) close.closest("dialog").close();return;}
    const button = event.target.closest("[data-action]");
    if (!button || button.disabled) return;
    const action = button.dataset.action;
    if (action === "create") openCreate();
    else if (action === "refresh") {refreshStatus();refreshWorkspace(true);}
    else if (action === "episode") openEpisode(button.dataset.run,button.dataset.episode);
    else if (action === "archive-episode") manageEpisode(button.dataset.run,button.dataset.episode,true);
    else if (action === "restore-episode") manageEpisode(button.dataset.run,button.dataset.episode,false);
    else if (action === "delete-episode") askDeleteEpisode(button.dataset.run,button.dataset.episode);
    else if (action === "confirm-delete") deleteEpisode();
    else if (action === "trace-more") loadEpisodeRecords();
    else if (action === "pipeline-mode-auto" || action === "pipeline-mode-manual") {state.pipelineMode = action === "pipeline-mode-auto" ? "auto" : "manual";updatePipelineExecution();}
    else if (action === "pipeline-start-auto") pipelineCommand("start",{mode:"auto"});
    else if (action === "pipeline-start-manual") pipelineCommand("start",{mode:"manual",stage:state.pipelineStage});
    else if (action === "pipeline-pause") pipelineCommand("pause");
    else if (action === "pipeline-resume") pipelineCommand("resume");
    else if (action === "pipeline-stop") pipelineCommand("stop");
    else if (action === "pipeline-global-pause") pipelineCommand("pause",{},button.dataset.project);
    else if (action === "pipeline-refresh") refreshPipeline(true);
    else if (action === "pipeline-config-discard") {renderPipelineConfig();updatePipelineExecution();}
    else if (action === "data-active" || action === "data-archived") {state.dataView = action === "data-archived" ? "archived" : "active";renderData(currentProject());}
    else if (action === "global-pause") command("pause",true);
    else if (["initialize","start","pause","stop","step"].includes(action)) command(action === "start" && state.status?.phase === "paused" ? "resume" : action);
  });
  document.addEventListener("input", event => {
    if (event.target.id === "project-search") {state.search = event.target.value;renderProjectCards();}
    if (event.target.id === "create-name") event.target.dataset.auto = "false";
    if (event.target.closest("#pipeline-config-form")) {
      state.pipelineDirty = true;
      $("pipeline-config-state").textContent = "저장하지 않은 변경";$("pipeline-config-state").className = "tag warning";
      updatePipelineExecution();
    }
  });
  document.addEventListener("change", event => {
    if (event.target.id === "create-suite") loadTasks();
    if (event.target.id === "create-task") updateCreateTask();
    if (event.target.id === "episode-filter") {state.filter = event.target.value;renderData(currentProject());}
    if (event.target.id === "policy-model") {updatePolicyEfforts();updatePolicyStatus();}
    if (event.target.id === "policy-effort") updatePolicyStatus();
    if (event.target.id === "pipeline-manual-stage") {state.pipelineStage = event.target.value;updatePipelineExecution();}
  });
  document.addEventListener("submit", event => {
    if (event.target.id === "settings-form") saveSettings(event);
    if (event.target.id === "pipeline-config-form") savePipelineConfig(event);
  });
  $("create-form").addEventListener("submit",createProject);
  $("episode-dialog").addEventListener("close", () => {const video = $("episode-content").querySelector("video");if (video) video.pause();state.episodeToken += 1;state.episodeDetail = null;});
  $("delete-dialog").addEventListener("cancel", event => {if (state.deleteBusy) event.preventDefault();});
  window.addEventListener("hashchange",navigate);
  fillIcons();
  navigate();
  Promise.allSettled([refreshStatus(),refreshWorkspace(true)]);
  setInterval(refreshStatus,1000);
  setInterval(refreshWorkspace,5000);
  setInterval(refreshPipeline,2500);
})();
