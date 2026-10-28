import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from rilo.config import Settings
from rilo.projects import ProjectCreate, ProjectStore
from rilo.server import create_app


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def recorded_run(root, name, *, task_id=7, suite="libero_90", count=1, reset=True, metered=True):
    folder = root / name
    write_json(folder / "manifest.json", {"run_id": name, "created_at": 100,
        "config": {"backend": "libero", "suite": suite, "task_id": task_id, "policy": "openai"},
        "environment": {"task": "open the top drawer" if task_id == 7 else "open the middle drawer"}})
    ledger = []
    for index in range(1, count + 1):
        episode_id = f"episode-{index:04d}"
        summary = {"id": episode_id, "index": index, "steps": 10, "reset_steps": 5,
            "outcome": "success", "reset_outcome": "success" if reset else "stopped",
            "costs": {"task": {"total_tokens": 100, "estimated_cost_usd": 1},
                      "reset": {"total_tokens": 50, "estimated_cost_usd": .5}}}
        write_json(folder / episode_id / "summary.json", summary)
        for stage, tokens, cost in [("task", 100, 1), ("reset", 50, .5)]:
            ledger.append({"attempt_id": f"{name}-{index}-{stage}", "episode_id": episode_id,
                "stage": stage, "tokens": {"total_tokens": tokens, "input_tokens": tokens-10, "output_tokens": 10},
                "estimated_cost_usd": cost, "cost_unknown": False})
    # Last episode also appears in live run summary; it must not be counted twice.
    write_json(folder / "summary.json", {"episode": summary})
    if metered:
        (folder / "api_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    return folder


def digest_files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_import_is_idempotent_and_preserves_original_recordings(tmp_path):
    root = tmp_path / "runs"
    recorded_run(root, "middle-1", task_id=0, suite="libero_goal", reset=False)
    recorded_run(root, "middle-2", task_id=0, suite="libero_goal", reset=False)
    recorded_run(root, "top", count=3)
    recorded_run(root, "smoke", metered=False)
    before = digest_files(root)
    store = ProjectStore(root)
    projects = store.list()
    assert len(projects) == 2
    assert [p["id"] for p in ProjectStore(root).list()] == [p["id"] for p in projects]
    cards = [store.details(p)["project"] for p in projects]
    assert sum(p["stats"]["runs"] for p in cards) == 3
    assert sum(p["stats"]["episodes"] for p in cards) == 5
    assert sum(p["stats"]["successes"] for p in cards) == 5
    assert sum(p["stats"]["reset_successes"] for p in cards) == 3
    assert sum(p["stats"]["estimated_cost_usd"] for p in cards) == 7.5
    assert digest_files(root) == before
    assert store.path == tmp_path / "projects.json"


def test_late_api_receipts_reconcile_episode_and_project_costs(tmp_path):
    root = tmp_path / "runs"
    run = recorded_run(root, "top")
    event = {"attempt_id": "late", "episode_id": "episode-0001", "stage": "reset",
             "tokens": {"total_tokens": 25, "input_tokens": 20, "output_tokens": 5},
             "estimated_cost_usd": .25, "cost_unknown": False}
    with (run / "api_calls.jsonl").open("a") as handle:
        handle.write(json.dumps(event) + "\n" + json.dumps(event) + "\n")
        handle.write(json.dumps({"attempt_id": "unknown", "episode_id": "episode-0001", "stage": "reset",
                                "tokens": None, "estimated_cost_usd": None, "cost_unknown": True}) + "\n")
    store = ProjectStore(root)
    result = store.details(store.list()[0])
    assert result["project"]["stats"]["estimated_cost_usd"] == 1.75
    assert result["project"]["stats"]["total_tokens"] == 175
    assert result["project"]["stats"]["unknown_usage_requests"] == 1
    reset = result["episodes"][0]["costs"]["reset"]
    assert reset["estimated_cost_usd"] == .75
    assert reset["total_tokens"] == 75
    assert reset["api_requests"] == 3
    assert reset["unknown_usage_requests"] == 1


def test_create_same_task_projects_are_independent_and_persistent(tmp_path):
    store = ProjectStore(tmp_path / "runs")
    a = store.create(ProjectCreate(name="First", task_id=7))
    b = store.create(ProjectCreate(name="Second", task_id=7))
    assert a["id"] != b["id"]
    assert a["run_config"]["task_id"] == 7
    assert len(ProjectStore(tmp_path / "runs").list()) == 2
    assert not store.data_dir.exists()  # Creating metadata never initializes a robot.


def demo_client(tmp_path):
    settings = Settings(backend="demo", policy="scripted", suite="demo", data_dir=tmp_path / "runs")
    return TestClient(create_app(settings))


def create_demo(client, name="Demo"):
    response = client.post("/api/projects", json={"name": name, "backend": "demo", "suite": "demo", "task_id": 0})
    assert response.status_code == 201
    return response.json()["project"]


def test_create_settings_and_workspace_without_starting_environment(tmp_path):
    with demo_client(tmp_path) as client:
        project = create_demo(client)
        assert client.get("/api/status").json()["phase"] == "idle"
        assert client.get("/api/workspace").json()["totals"]["projects"] == 1
        assert client.get("/api/workspace").json()["capabilities"]["bc"] is True
        assert not (tmp_path / "runs").exists()
        url = f"/api/projects/{project['id']}/settings"
        response = client.post(url, json={"name": "Updated", "pipeline": {"demo_target": 30, "bc_success_target": 75}})
        assert response.status_code == 200
        assert response.json()["project"]["name"] == "Updated"
        assert response.json()["project"]["pipeline"]["demo_target"] == 30
        assert client.post(url, json={"pipeline": {"demo_target": 0}}).status_code == 422
        assert client.post(url, json={"run_config": {"budget_usd": -1}}).status_code == 422
        assert client.post("/api/projects", json={"name": "   "}).status_code == 422
        assert client.get("/api/projects/missing").status_code == 404


def test_project_controls_cannot_steal_another_projects_ready_or_paused_run(tmp_path):
    with demo_client(tmp_path) as client:
        a, b = create_demo(client, "A"), create_demo(client, "B")
        a_url, b_url = f"/api/projects/{a['id']}", f"/api/projects/{b['id']}"
        response = client.post(a_url + "/initialize", json={})
        assert response.status_code == 200
        original_run = response.json()["run_id"]
        assert response.json()["active_project_id"] == a["id"]
        assert response.json()["phase"] == "ready"
        assert client.post(b_url + "/start", json={}).status_code == 409
        assert client.post(b_url + "/initialize", json={}).status_code == 409
        assert client.post(a_url + "/pause", json={}).json()["phase"] == "paused"
        assert client.post(b_url + "/initialize", json={}).status_code == 409
        assert client.get("/api/status").json()["run_id"] == original_run
        assert client.post(a_url + "/initialize", json={"preserve_environment": True}).status_code == 409
        assert client.get("/api/status").json()["active_project_id"] == a["id"]
        assert client.post(a_url + "/stop", json={}).status_code == 200
        assert client.post(b_url + "/initialize", json={}).status_code == 200
        assert client.post(a_url + "/start", json={}).status_code == 409
        assert client.get(a_url).json()["project"]["run_ids"] == [original_run]
        assert client.get("/api/workspace").json()["runtime"]["active_project_id"] == b["id"]


def test_legacy_initialize_clears_project_ownership_and_reset_preserves_it(tmp_path):
    with demo_client(tmp_path) as client:
        project = create_demo(client)
        url = f"/api/projects/{project['id']}"
        first = client.post(url + "/initialize", json={}).json()["run_id"]
        second = client.post("/api/reset", json={}).json()["run_id"]
        assert client.get(url).json()["project"]["run_ids"] == [first, second]
        assert client.get("/api/status").json()["active_project_id"] == project["id"]
        client.post("/api/initialize", json={"task_id": 0})
        assert client.get("/api/status").json()["active_project_id"] is None
        assert client.post(url + "/start", json={}).status_code == 409


def test_backend_mismatch_is_browsable_but_not_executable(tmp_path):
    with demo_client(tmp_path) as client:
        project = client.post("/api/projects", json={"name": "Real robot", "backend": "robot:factory", "suite": "", "task_id": 0}).json()["project"]
        assert project["runnable"] is False
        assert client.post(f"/api/projects/{project['id']}/initialize", json={}).status_code == 409
        assert client.get("/api/status").json()["phase"] == "idle"


def test_media_range_and_traversal_protection(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "rollout.mp4").write_bytes(b"0123456789")
    secret = tmp_path / "outside.jpg"
    secret.write_bytes(b"hidden")
    (folder / "escape.jpg").symlink_to(secret)
    with demo_client(tmp_path) as client:
        response = client.get("/api/media/rollout.mp4", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert client.get("/api/media/escape.jpg").status_code == 404
        assert client.get("/api/media/projects.json").status_code == 404
        assert client.get("/api/media/%2e%2e%2foutside.jpg").status_code == 404


def test_registry_corruption_is_not_overwritten(tmp_path):
    store = ProjectStore(tmp_path / "runs")
    store.path.write_text('{"schema_version": 999, "projects": []}')
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.create(ProjectCreate(name="New"))
    assert store.path.read_bytes() == before
