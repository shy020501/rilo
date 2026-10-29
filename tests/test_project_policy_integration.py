"""Saved policy choices must reach the next run, never mutate a prepared run."""
import json
import time

from fastapi.testclient import TestClient

from rilo.config import Settings
from rilo.policies.scripted import ScriptedPolicy
from rilo.server import create_app


def test_project_policy_snapshot_and_real_factory_selection(tmp_path, monkeypatch):
    constructed = []

    def policy_factory(settings):
        constructed.append((settings.model, settings.reasoning_effort))
        return ScriptedPolicy()

    monkeypatch.setattr("rilo.runner.make_policy", policy_factory)
    settings = Settings(backend="demo", suite="demo", policy="openai", api_key="fixture-only",
                        data_dir=tmp_path / "runs", reasoning_effort="medium")
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/projects", json={"name": "Policy snapshot", "backend": "demo",
                              "suite": "demo", "task_id": 0})
        assert created.status_code == 201
        project_id = created.json()["project"]["id"]
        url = f"/api/projects/{project_id}"
        selected = {"model": "gpt-6.1-sol", "reasoning_effort": "low"}
        response = client.post(url + "/settings", json={"policy_config": selected})
        assert response.status_code == 200
        assert client.get(url).json()["project"]["policy_config"] == selected
        assert not constructed
        ready = client.post(url + "/initialize", json={}).json()
        assert {k: ready[k] for k in selected} == selected
        run_id = ready["run_id"]
        observation_id = ready["observation"]["id"]

        following = {"model": "gpt-6-astra", "reasoning_effort": "high"}
        assert client.post(url + "/settings", json={"policy_config": following}).status_code == 200
        unchanged = client.get("/api/status").json()
        assert unchanged["run_id"] == run_id
        assert unchanged["observation"]["id"] == observation_id
        assert {k: unchanged[k] for k in selected} == selected
        assert not constructed
        assert client.post(url + "/start", json={}).status_code == 200
        assert client.post(url + "/pause", json={}).status_code == 200
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            status = client.get("/api/status").json()
            if status["phase"] == "paused":
                break
            time.sleep(.01)
        assert status["phase"] == "paused"
        assert constructed == [("gpt-6.1-sol", "low")]
        assert {k: status[k] for k in selected} == selected
        manifest = json.loads((settings.data_dir / run_id / "manifest.json").read_text())
        assert {k: manifest["config"][k] for k in selected} == selected

        assert client.post(url + "/stop", json={}).status_code == 200
        next_run = client.post(url + "/initialize", json={}).json()
        assert next_run["run_id"] != run_id
        assert {k: next_run[k] for k in following} == following
        assert client.post(url + "/start", json={}).status_code == 200
        assert constructed[-1] == ("gpt-6-astra", "high")


def test_invalid_policy_setting_does_not_change_project_or_scene(tmp_path):
    settings = Settings(backend="demo", suite="demo", policy="scripted", data_dir=tmp_path / "runs")
    with TestClient(create_app(settings)) as client:
        project = client.post("/api/projects", json={"name": "Validation", "backend": "demo",
                              "suite": "demo", "task_id": 0}).json()["project"]
        url = f"/api/projects/{project['id']}"
        before = client.post(url + "/initialize", json={}).json()
        for invalid in ({"model": "unknown", "reasoning_effort": "medium"},
                        {"model": "gpt-6-astra", "reasoning_effort": "none"}):
            assert client.post(url + "/settings", json={"policy_config": invalid}).status_code == 422
        after = client.get("/api/status").json()
        assert after["run_id"] == before["run_id"]
        assert after["observation"]["id"] == before["observation"]["id"]
        assert client.get(url).json()["project"]["policy_config"] == project["policy_config"]


def test_old_suite_pending_usage_still_blocks_episode_deletion(tmp_path):
    settings = Settings(backend="demo", suite="demo", policy="scripted", data_dir=tmp_path / "runs")
    app = create_app(settings)

    class PendingReceipt:
        def done(self):
            return False

    with TestClient(app) as client:
        first = client.post("/api/projects", json={"name": "First suite", "backend": "demo",
                            "suite": "demo", "task_id": 0}).json()["project"]
        first_url = f"/api/projects/{first['id']}"
        initial = client.post(first_url + "/initialize", json={}).json()
        assert client.post(first_url + "/stop", json={}).status_code == 200
        old = app.state.runner
        old._retired_requests.add(PendingReceipt())
        second = client.post("/api/projects", json={"name": "Second suite", "backend": "demo",
                             "suite": "demo-2", "task_id": 0}).json()["project"]
        assert client.post(f"/api/projects/{second['id']}/initialize", json={}).status_code == 200
        assert app.state.runner is not old
        delete_url = first_url + f"/episodes/{initial['run_id']}/episode-0001/delete"
        assert client.post(delete_url, json={"confirm": True}).status_code == 409
        old._retired_requests.clear()
        assert client.post(delete_url, json={"confirm": True}).status_code == 200
