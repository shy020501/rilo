"""Pipeline state-machine tests; all rollout and neural training work is faked."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

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


def engine_fixture(tmp_path):
    data, projects, project, _ = make_learning_fixture(tmp_path)
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


def fake_train_bc(snapshot, output, config, *, callback=None, control=None):
    if control: control()
    return {"checkpoint":str(output/"fake-only.pt"), "config":config,
            "dataset":snapshot.successful().manifest(), "metrics":{"validation_loss":.1}}


def fake_train_progress(snapshot, base, output, config, *, callback=None, control=None):
    if control: control()
    return {"checkpoint":str(output/"fake-only.pt"), "config":config,
            "dataset":snapshot.successful().manifest(), "metrics":{"progress_validation_mae":.1}}


@pytest.mark.asyncio
async def test_auto_runs_sequential_cycles_then_progress_and_rl_targets(tmp_path, monkeypatch):
    engine, projects, project, runner = engine_fixture(tmp_path)
    order, activity, evaluations = [], [], {"bc_eval":iter([50.,80.]), "rl_eval":iter([50.,95.])}
    original = engine._stage
    async def record_stage(stage):
        order.append(stage)
        return await original(stage)
    monkeypatch.setattr(engine, "_stage", record_stage)
    def bc_train(*args, **kwargs):
        assert runner.phase not in ("ready", "thinking", "executing", "paused")
        activity.append("bc")
        return fake_train_bc(*args, **kwargs)
    monkeypatch.setattr("rilo.learning.bc_train.train_bc",bc_train)
    monkeypatch.setattr("rilo.learning.progress.train_progress",fake_train_progress)
    async def rollout(stage):
        assert not activity or activity[-1] != "collecting"
        runner.phase="executing"
        await asyncio.sleep(0)
        runner.phase="completed"
        bc=engine._latest(engine._state,"bc")
        if stage=="rl":
            identifier, folder=engine._artifact_folder("residual")
            engine._register("residual",identifier,{"checkpoint":str(folder/"fake.pt"),"config":{},"metrics":{}},bc["id"])
        result={"success_rate":next(evaluations[stage]) if stage in evaluations else 50.,
                "model_id":bc["id"] if bc else None,"kind":"bc" if stage=="bc_eval" else "residual"}
        if stage in evaluations: engine._state["evaluations"].append(result)
        return result
    monkeypatch.setattr(engine,"_rollout_stage",rollout)
    await engine.start(project["id"],PipelineStart(mode="auto"))
    await engine._task
    assert order==["collect","bc","bc_eval","collect","bc","bc_eval","progress","rl","rl_eval","rl","rl_eval"]
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
    engine._state["evaluations"].append({"kind":"bc","model_id":bc["id"],"success_rate":80})
    progress_id, folder=engine._artifact_folder("progress")
    engine._register("progress",progress_id,{"checkpoint":str(folder/"fake.pt"),"config":{},"metrics":{}},bc["id"])
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
    async def stage(name):
        stages.append(name)
        return {"success_rate":0.}
    monkeypatch.setattr(engine,"_stage",stage)
    await engine.start(project["id"],PipelineStart(mode="auto"))
    await engine._task
    assert stages==["collect","bc","bc_eval"]
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
