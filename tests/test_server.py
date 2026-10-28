from fastapi.testclient import TestClient

from rilo.config import Settings
from rilo.server import create_app


def test_console_preview_and_controls_without_api_key(tmp_path):
    app = create_app(Settings(backend="demo", policy="openai", data_dir=tmp_path))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert len(client.get("/api/tasks").json()["tasks"]) == 1
        response = client.post("/api/initialize", json={"task_id": 0})
        assert response.status_code == 200
        state = response.json()
        assert state["phase"] == "ready"
        assert state["api_key_configured"] is False
        frame = client.get("/api/frame/overview")
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"
        response = client.post("/api/start", json={})
        assert response.status_code == 400
        assert "OPENAI_API_KEY" in response.json()["detail"]
        assert client.get("/api/status").json()["episode"]["steps"] == 0
        assert client.post("/api/pause", json={}).json()["phase"] == "paused"
        assert client.post("/api/reset", json={}).json()["phase"] == "ready"


def test_cross_origin_commands_and_non_json_rejected(tmp_path):
    app = create_app(Settings(backend="demo", policy="scripted", data_dir=tmp_path))
    with TestClient(app) as client:
        assert client.post("/api/initialize", json={}, headers={"Origin": "https://unrelated.example"}).status_code == 403
        assert client.post("/api/initialize", data="{}").status_code == 415
        assert client.get("/api/status").json()["phase"] == "idle"


def test_server_snapshot_never_contains_configured_secret(tmp_path):
    key = "sk-this-must-not-be-exposed"
    app = create_app(Settings(backend="demo", api_key=key, data_dir=tmp_path))
    with TestClient(app) as client:
        assert key not in client.get("/api/status").text
        response = client.post("/api/initialize", json={})
        assert key not in response.text
        assert key not in client.get("/api/runs").text
