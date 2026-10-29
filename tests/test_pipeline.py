"""Pipeline state-machine tests; all rollout and neural training work is faked."""
import asyncio
import hashlib
import json
import shutil
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from rilo.config import Settings
from rilo.pipeline import PipelineEngine, PipelineStart
from rilo.projects import ProjectCreate
from rilo.runner import StateConflict
from test_learning_data import make_learning_fixture


class IdleRunner:
    def __init__(self):
        self.phase = "idle"
        self.stats = {"estimated_cost_usd": 0}
        self.config = SimpleNamespace(budget_usd=15)
        self.controls = []

    def _safe_error(self, exc):
        return str(exc)

    async def pause(self):
        self.controls.append("pause")
        self.phase = "paused"

    async def stop(self):
        self.controls.append("stop")
        self.phase = "stopped"

    async def start(self):
        self.controls.append("start")
        self.phase = "ready"


async def until(predicate, timeout=3):
    async def poll():
        while not predicate():
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), timeout)


def engine_fixture(tmp_path, *, count=3):
    data, projects, project, _ = make_learning_fixture(tmp_path, count=count)
    settings = Settings(backend="demo", policy="scripted", suite="demo", data_dir=data)
    runner = IdleRunner()
    async def forbid_real_run(*args, **kwargs):
        raise AssertionError("A state-machine test must not prepare an actual rollout")
    engine = PipelineEngine(settings, projects, lambda: runner, forbid_real_run)
    engine.configure(project["id"], {"collection_episodes": 2, "evaluation_episodes": 2,
        "max_bc_cycles": 3, "max_rl_rounds": 3, "device": "cpu",
        "bc": {"image_size":16, "backbone":"tiny", "imagenet_init":False, "steps":1,
               "feature_dim":16, "keypoints":4, "unet_dims":[8,16,32], "horizon":4,
               "action_steps":2, "diffusion_steps":8, "inference_steps":2},
        "progress":{"steps":1,"hidden_dim":8}})
    return engine, projects, project, runner


def add_recorded_attempt(engine, projects, project, *, success=True):
    """Copy complete local fixture evidence into a new recorded attempt."""
    source = next(engine.settings.data_dir.glob("*/episode-0001/transitions.jsonl")).parent.parent
    run_id = "fixture-new-" + uuid.uuid4().hex
    target = engine.settings.data_dir / run_id
    shutil.copytree(source, target)
    summary_path = target / "episode-0001" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(outcome="success" if success else "step_budget", reset_outcome="success")
    summary_path.write_text(json.dumps(summary))
    projects.link_run(project["id"], run_id)
    return run_id


def fake_train_bc(snapshot, output, config, *, callback=None, control=None):
    if control: control()
    checkpoint = output / "fake-only.pt"
    checkpoint.write_bytes(b"Mock BC checkpoint; no neural weights")
    selected = snapshot.successful() if config.get("successful_only", True) else snapshot
    return {"checkpoint":str(checkpoint), "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "config":config, "dataset":selected.manifest(), "metrics":{"validation_loss":.1}}


def fake_train_progress(snapshot, base, output, config, *, callback=None, control=None):
    if control: control()
    checkpoint = output / "fake-only.pt"
    checkpoint.write_bytes(b"Mock progress checkpoint; no neural weights")
    return {"checkpoint":str(checkpoint), "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "config":config,
            "dataset":snapshot.successful().manifest(), "metrics":{"progress_validation_mae":.1}}


def fake_evaluation(engine, projects, project, stage, rate, bc, residual=None, progress=None):
    from rilo.pipeline_plan import evaluation_signature
    return {"success_rate":rate, "model_id":bc["id"],
            "run_id":"mock-evaluation-run",
            "residual_model_id":residual["id"] if residual else None,
            "kind":"bc" if stage == "bc_eval" else "residual",
            "episodes":engine._state["config"]["evaluation_episodes"],
            "provenance":{"evaluation":evaluation_signature(projects.get(project["id"]), engine._state["config"], stage),
                          "base_model_id":bc["id"], "base_checkpoint_sha256":bc["checkpoint_sha256"],
                          "residual_model_id":residual["id"] if residual else None,
                          "residual_checkpoint_sha256":residual["checkpoint_sha256"] if residual else None,
                          "progress_model_id":progress["id"] if progress else None,
                          "progress_checkpoint_sha256":progress["checkpoint_sha256"] if progress else None}}


@pytest.mark.asyncio
async def test_auto_runs_sequential_cycles_then_progress_and_rl_targets(tmp_path, monkeypatch):
    engine, projects, project, runner = engine_fixture(tmp_path)
    order, activity, evaluations = [], [], {"bc_eval":iter([50.,80.]), "rl_eval":iter([50.,95.])}
    original = engine._stage
    async def record_stage(stage, **kwargs):
        order.append(stage)
        return await original(stage, **kwargs)
    monkeypatch.setattr(engine, "_stage", record_stage)
    def bc_train(*args, **kwargs):
        assert runner.phase not in ("ready", "thinking", "executing", "paused")
        activity.append("bc")
        return fake_train_bc(*args, **kwargs)
    monkeypatch.setattr("rilo.learning.bc_train.train_bc",bc_train)
    monkeypatch.setattr("rilo.learning.progress.train_progress",fake_train_progress)
    async def rollout(stage, *, episodes=None):
        assert not activity or activity[-1] != "collecting"
        runner.phase="executing"
        await asyncio.sleep(0)
        runner.phase="completed"
        bc=engine._latest(engine._state,"bc")
        if stage == "collect":
            assert episodes is not None
            for _ in range(episodes):
                add_recorded_attempt(engine, projects, project)
            return {"success_rate":100., "episodes":episodes, "successes":episodes}
        progress = engine._latest(engine._state, "progress", bc["id"])
        if stage=="rl":
            identifier, folder=engine._artifact_folder("residual")
            checkpoint = folder / "fake.pt"
            checkpoint.write_bytes(identifier.encode())
            snapshot = engine._snapshot(bc["config"]["image_size"])
            engine._register("residual",identifier,{"checkpoint":str(checkpoint),
                "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "config":engine._state["config"]["residual"],"metrics":{},
                "dataset":{"dataset_fingerprint":snapshot.fingerprint, "base_fingerprint":bc["checkpoint_sha256"]},
                "provenance":{"base_model_id":bc["id"],"base_checkpoint_sha256":bc["checkpoint_sha256"],
                              "progress_model_id":progress["id"],"progress_checkpoint_sha256":progress["checkpoint_sha256"],
                              "progress_gate_threshold":engine._state["config"]["progress_gate_threshold"],
                              "gpt_judge_interval":engine._state["config"]["gpt_judge_interval"],
                              "dataset":snapshot.manifest()}},bc["id"])
        residual = engine._latest(engine._state, "residual", bc["id"])
        result = (fake_evaluation(engine, projects, project, stage, next(evaluations[stage]), bc,
                                 residual if stage == "rl_eval" else None,
                                 progress if stage == "rl_eval" else None)
                  if stage in evaluations else {"success_rate":50.})
        if stage in evaluations: engine._state["evaluations"].append(result)
        return result
    monkeypatch.setattr(engine,"_rollout_stage",rollout)
    await engine.start(project["id"],PipelineStart(mode="auto"))
    await engine._task
    assert order==["bc","bc_eval","collect","bc","bc_eval","progress","rl","rl_eval","rl","rl_eval"]
    state=engine.view(project["id"])
    assert state["status"]["phase"]=="completed"
    assert [m["kind"] for m in state["models"]]==["bc","bc","progress","residual","residual"]
    bc=engine._latest(state,"bc")
    assert engine._latest(state,"progress")["base_model_id"]==bc["id"]
    assert state["readiness"]["progress"]["ready"]
    assert len(activity)==2
    assert not runner.controls  # No simulator/GPT controls were called by these tests.


@pytest.mark.asyncio
async def test_manual_prerequisites_and_current_bc_progress_binding(tmp_path,monkeypatch):
    engine,projects,project,runner=engine_fixture(tmp_path)
    pid=project["id"]
    for stage in ("bc_eval","progress","rl","rl_eval"):
        with pytest.raises(StateConflict):
            await engine.start(pid,PipelineStart(mode="manual",stage=stage))
    assert not engine.active
    monkeypatch.setattr("rilo.learning.bc_train.train_bc",fake_train_bc)
    await engine.start(pid,PipelineStart(mode="manual",stage="bc"))
    await engine._task
    assert engine.view(pid)["status"]["phase"]=="completed"
    bc=engine._latest(engine._state,"bc")
    assert not engine.view(pid)["readiness"]["progress"]["ready"]
    engine._state["evaluations"].append(fake_evaluation(engine, projects, project, "bc_eval", 80, bc))
    progress_id, folder=engine._artifact_folder("progress")
    result = fake_train_progress(engine._snapshot(bc["config"]["image_size"]), bc["checkpoint"], folder,
                                 engine._state["config"]["progress"])
    result["provenance"] = {"base_model_id":bc["id"], "base_checkpoint_sha256":bc["checkpoint_sha256"]}
    engine._register("progress",progress_id,result,bc["id"])
    assert engine.view(pid)["readiness"]["rl"]["ready"]
    await engine.start(pid,PipelineStart(mode="manual",stage="bc"))
    await engine._task
    assert not engine.view(pid)["readiness"]["rl"]["ready"]
    assert not engine.view(pid)["readiness"]["progress"]["ready"]


@pytest.mark.asyncio
async def test_pause_stop_training_keeps_owner_and_blocks_parallel_work(tmp_path,monkeypatch):
    engine,projects,project,runner=engine_fixture(tmp_path)
    entered=threading.Event()
    updates=[]
    def controlled_train(snapshot,output,config,*,callback,control):
        for index in range(1000):
            control()
            updates.append(index)
            entered.set()
            time.sleep(.01)
        return fake_train_bc(snapshot,output,config)
    monkeypatch.setattr("rilo.learning.bc_train.train_bc",controlled_train)
    await engine.start(project["id"],PipelineStart(mode="manual",stage="bc"))
    await until(entered.is_set)
    await engine.pause(project["id"])
    await asyncio.sleep(.04)
    count=len(updates)
    await asyncio.sleep(.04)
    assert len(updates)==count
    assert engine.view(project["id"])["status"]["phase"]=="paused"
    assert engine.active
    with pytest.raises(StateConflict): engine.configure(project["id"],{"collection_episodes":3})
    other=projects.create(ProjectCreate(name="Another",backend="demo",suite="demo"))
    with pytest.raises(StateConflict): await engine.start(other["id"],PipelineStart(mode="manual",stage="collect"))
    owner=engine.runtime()["project_id"]
    engine.configure(other["id"],{"collection_episodes":4})
    assert engine.runtime()["project_id"]==owner
    assert engine.view(project["id"])["status"]["phase"]=="paused"
    await engine.resume(project["id"])
    await until(lambda:len(updates)>count)
    await engine.stop(project["id"])
    await engine._task
    assert engine.view(project["id"])["status"]["phase"]=="stopped"
    assert not engine.view(project["id"])["models"]
    assert not runner.controls


@pytest.mark.asyncio
async def test_auto_limit_exits_with_error_before_progress_or_rl(tmp_path,monkeypatch):
    engine,projects,project,runner=engine_fixture(tmp_path)
    engine.configure(project["id"],{"max_bc_cycles":1})
    stages=[]
    original = engine._stage
    async def stage(name, **kwargs):
        stages.append(name)
        return await original(name, **kwargs)
    async def failed_evaluation(name, **kwargs):
        assert name == "bc_eval"
        bc = engine._latest(engine._state, "bc")
        result = fake_evaluation(engine, projects, project, name, 0., bc)
        engine._state["evaluations"].append(result)
        return result
    monkeypatch.setattr("rilo.learning.bc_train.train_bc", fake_train_bc)
    monkeypatch.setattr(engine,"_stage",stage)
    monkeypatch.setattr(engine,"_rollout_stage",failed_evaluation)
    await engine.start(project["id"],PipelineStart(mode="auto"))
    await engine._task
    assert stages==["bc","bc_eval"]
    assert engine.view(project["id"])["status"]["phase"]=="error"
    assert "BC" in engine.view(project["id"])["status"]["error"]


def test_restart_marks_interrupted_without_automatic_execution(tmp_path):
    engine,projects,project,runner=engine_fixture(tmp_path)
    path=engine._path(project["id"])/"registry.json"
    state=json.loads(path.read_text())
    state["status"].update(phase="running",stage="rl",mode="auto")
    path.write_text(json.dumps(state))
    restarted=PipelineEngine(engine.settings,projects,lambda:runner,engine.prepare_run)
    view=restarted.view(project["id"])
    assert view["status"]["phase"]=="interrupted"
    assert restarted._task is None
    assert not restarted.active
    assert not runner.controls


@pytest.mark.asyncio
async def test_existing_robot_collection_blocks_pipeline_start(tmp_path):
    engine,projects,project,runner=engine_fixture(tmp_path)
    runner.phase="paused"
    with pytest.raises(StateConflict,match="기존 수집"):
        await engine.start(project["id"],PipelineStart(mode="manual",stage="bc"))
    assert not engine.active
