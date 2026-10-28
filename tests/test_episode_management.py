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


def fixture_store(tmp_path, *, metered=True):
    root = tmp_path / "runs" / "recorded"
    write_json(root / "manifest.json", {"created_at": 100, "config": {
        "backend": "demo", "suite": "demo", "task_id": 0,
        "model": "gpt-6-astra", "reasoning_effort": "xhigh"}, "environment": {"task": "Fixture"}})
    ledger = []
    for index in (1, 2):
        episode_id = f"episode-{index:04d}"
        summary = {"id": episode_id, "index": index, "outcome": "success", "reset_outcome": "success",
            "steps": 10, "reset_steps": 5, "costs": {"task": {"input_tokens": 100, "output_tokens": 20,
            "total_tokens": 120, "estimated_cost_usd": 1.5, "api_requests": 1}}}
        write_json(root / episode_id / "summary.json", summary)
        image = root / episode_id / "frames" / "camera" / "000000.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"private-frame")
        rows = [{"kind": "environment_reset", "recorded_at": 1, "seed": 0, "observation": {"images": {"camera": "frames/camera/000000.jpg"}}},
            {"kind": "observation", "observation": {"state": [1, 2, 3]}},
            {"kind": "decision", "recorded_at": 2, "stage": "task", "decision": {"note": "Move to the handle", "status": "continue", "actions": [{"values": [1, 0], "repeat": 6}]}},
            {"kind": "transition", "requested_action": [1, 0], "next_observation": {"state": [2, 3, 4]}},
            {"kind": "receipt", "recorded_at": 3, "stage": "task", "note": "Move to the handle", "executed_steps": 6, "requested_steps": 6},
            {"kind": "stage_end", "recorded_at": 4, "stage": "task", "outcome": "success"},
            {"kind": "episode_end", "recorded_at": 5, **summary}]
        (root / episode_id / "transitions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        ledger.append({"attempt_id": str(index), "episode_id": episode_id, "stage": "task",
            "tokens": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "estimated_cost_usd": 1.5, "cost_unknown": False})
    write_json(root / "summary.json", {"episode": summary})
    if metered:
        (root / "api_calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    videos = tmp_path / "videos"
    for index in (1, 2):
        write_json(videos / f"episode-{index}.json", {"run_id": "recorded", "episode_id": f"episode-{index:04d}",
            "video": f"episode-{index}.mp4", "poster": "shared.jpg"})
        (videos / f"episode-{index}.mp4").write_bytes(f"video-{index}".encode())
    (videos / "shared.jpg").write_bytes(b"shared-poster")
    store = ProjectStore(tmp_path / "runs")
    if metered:
        project = store.list()[0]
    else:
        project = store.create(ProjectCreate(name="Unmetered", backend="demo", suite="demo"))
        store.link_run(project["id"], "recorded")
    return store, project["id"], root


def digest(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_archive_restore_persistence_counts_and_lifetime_spend(tmp_path):
    store, project_id, root = fixture_store(tmp_path)
    original = digest(root)
    store.archive_episode(project_id, "recorded", "episode-0001", archived=True)
    result = ProjectStore(store.data_dir).details(store.get(project_id))
    assert [ep["id"] for ep in result["episodes"]] == ["episode-0002"]
    assert [ep["id"] for ep in result["archived_episodes"]] == ["episode-0001"]
    assert result["project"]["archived_count"] == 1
    assert result["project"]["stats"]["successes"] == 1
    assert result["project"]["stats"]["total_steps"] == 15
    assert result["project"]["stats"]["estimated_cost_usd"] == 3
    assert digest(root) == original
    store.archive_episode(project_id, "recorded", "episode-0001", archived=False)
    assert len(store.details(store.get(project_id))["episodes"]) == 2
    assert digest(root) == original


@pytest.mark.parametrize("metered", [True, False])
def test_delete_payload_and_unique_media_preserves_other_episode_and_expense(tmp_path, metered):
    store, project_id, root = fixture_store(tmp_path, metered=metered)
    other = digest(root / "episode-0001")
    ledger = (root / "api_calls.jsonl").read_bytes() if metered else None
    stale_summary = (root / "summary.json").read_text()
    store.delete_episode(project_id, "recorded", "episode-0002")
    assert not (root / "episode-0002").exists()
    assert not (tmp_path / "videos" / "episode-2.mp4").exists()
    assert not (tmp_path / "videos" / "episode-2.json").exists()
    assert (tmp_path / "videos" / "shared.jpg").exists()
    assert (tmp_path / "videos" / "episode-1.mp4").exists()
    assert digest(root / "episode-0001") == other
    assert json.loads((root / "summary.json").read_text())["episode"] == {}
    if metered:
        assert (root / "api_calls.jsonl").read_bytes() == ledger
    # A stopped legacy runner may still write an old live summary. It cannot
    # resurrect the data entry or its old thumbnail.
    (root / "summary.json").write_text(stale_summary)
    result = ProjectStore(store.data_dir).details(store.get(project_id))
    assert [ep["id"] for ep in result["episodes"]] == ["episode-0001"]
    assert result["project"]["stats"]["estimated_cost_usd"] == 3
    assert result["project"]["stats"]["total_tokens"] == 240
    with pytest.raises(KeyError):
        store.episode_detail(project_id, "recorded", "episode-0002")
    store.delete_episode(project_id, "recorded", "episode-0002")  # Idempotent.


def test_detail_public_notes_and_actions_pagination_and_archive_access(tmp_path):
    store, project_id, _ = fixture_store(tmp_path)
    store.archive_episode(project_id, "recorded", "episode-0001", archived=True)
    first = store.episode_detail(project_id, "recorded", "episode-0001", offset=0, limit=2)
    assert first["episode"]["archived"] is True
    assert first["config"]["reasoning_effort"] == "xhigh"
    assert first["records"][0]["kind"] == "episode_start"
    assert first["records"][1]["decision"]["note"] == "Move to the handle"
    assert first["records"][1]["decision"]["actions"][0]["repeat"] == 6
    assert first["next_offset"] == 2
    assert first["total_records"] == 5
    rest = store.episode_detail(project_id, "recorded", "episode-0001", offset=2, limit=10)
    assert rest["next_offset"] is None
    assert not any(row["kind"] in ("observation", "transition") for row in first["records"] + rest["records"])


def test_management_api_confirmation_membership_origin_and_pagination(tmp_path):
    store, project_id, root = fixture_store(tmp_path)
    another = store.create(ProjectCreate(name="Another", backend="demo", suite="demo"))
    settings = Settings(backend="demo", policy="scripted", suite="demo", data_dir=store.data_dir)
    base = f"/api/projects/{project_id}/episodes/recorded/episode-0001"
    with TestClient(create_app(settings)) as client:
        assert client.post(base + "/delete", json={"confirm": False}).status_code == 400
        assert client.post(base + "/delete", json={"confirm": "true"}).status_code == 422
        assert client.post(base + "/archive", json={"archived": True}, headers={"Origin": "https://bad.example"}).status_code == 403
        assert client.post(base.replace(project_id, another["id"]) + "/archive", json={"archived": True}).status_code == 404
        assert client.get(base + "/detail?limit=1000").status_code == 422
        assert client.get(base + "/detail?offset=-1").status_code == 422
        assert client.post(base + "/archive", json={"archived": True}).status_code == 200
        result = client.get(f"/api/projects/{project_id}").json()
        assert result["archived_episodes"][0]["can_manage"] is True
        assert client.get(base + "/detail").json()["episode"]["archived"] is True
        assert client.post(base + "/delete", json={"confirm": True}).json()["deleted"] is True
        assert client.get(base + "/detail").status_code == 404
        assert not (root / "episode-0001").exists()


def test_active_run_and_retired_request_are_protected(tmp_path):
    store, project_id, _ = fixture_store(tmp_path)
    settings = Settings(backend="demo", policy="scripted", suite="demo", data_dir=store.data_dir)
    app = create_app(settings)
    base = f"/api/projects/{project_id}"
    with TestClient(app) as client:
        response = client.post(base + "/initialize", json={})
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        route = base + f"/episodes/{run_id}/episode-0001"
        for action, payload in [("archive", {"archived": True}), ("delete", {"confirm": True})]:
            assert client.post(route + "/" + action, json=payload).status_code == 409
        client.post(base + "/pause", json={})
        assert client.post(route + "/delete", json={"confirm": True}).status_code == 409
        client.post(base + "/stop", json={})
        assert client.post(route + "/archive", json={"archived": True}).status_code == 200
        class Pending:
            def done(self):
                return False
        pending = Pending()
        app.state.runner._retired_requests.add(pending)
        try:
            assert client.post(route + "/delete", json={"confirm": True}).status_code == 409
            historical = base + "/episodes/recorded/episode-0001/delete"
            assert client.post(historical, json={"confirm": True}).status_code == 409
        finally:
            app.state.runner._retired_requests.clear()
        assert client.post(route + "/delete", json={"confirm": True}).status_code == 200


def test_deletion_failure_is_pending_and_retry_completes(tmp_path, monkeypatch):
    store, project_id, root = fixture_store(tmp_path)
    import rilo.projects as project_module
    real_remove = project_module.shutil.rmtree
    def fail(path):
        raise PermissionError("Fixture deletion denied")
    monkeypatch.setattr(project_module.shutil, "rmtree", fail)
    with pytest.raises(PermissionError):
        store.delete_episode(project_id, "recorded", "episode-0001")
    assert store._episode_state("recorded", "episode-0001")["delete_pending"] is True
    assert (root / "episode-0001").exists()
    pending = store.details(store.get(project_id))["archived_episodes"][0]
    assert pending["delete_pending"] is True
    assert pending["id"] == "episode-0001"
    assert "costs" not in pending
    assert "outcome" not in pending
    with pytest.raises(KeyError):
        store.archive_episode(project_id, "recorded", "episode-0001", archived=False)
    monkeypatch.setattr(project_module.shutil, "rmtree", real_remove)
    store.delete_episode(project_id, "recorded", "episode-0001")
    assert not (root / "episode-0001").exists()
    assert store._episode_state("recorded", "episode-0001")["delete_pending"] is False


def test_summary_symlink_is_rejected_before_deletion_and_temp_symlink_unused(tmp_path):
    store, project_id, root = fixture_store(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"private":"stay"}')
    summary = root / "summary.json"
    original = summary.read_bytes()
    summary.unlink()
    summary.symlink_to(outside)
    with pytest.raises(ValueError):
        store.delete_episode(project_id, "recorded", "episode-0002")
    assert (root / "episode-0002").exists()
    summary.unlink()
    summary.write_bytes(original)
    (root / "summary.delete.tmp").symlink_to(outside)
    store.delete_episode(project_id, "recorded", "episode-0002")
    assert outside.read_text() == '{"private":"stay"}'


def test_symlink_or_traversal_cannot_delete_outside_recording(tmp_path):
    store, project_id, root = fixture_store(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"must-stay")
    video = tmp_path / "videos" / "episode-1.mp4"
    video.unlink()
    video.symlink_to(outside)
    with pytest.raises(ValueError):
        store.delete_episode(project_id, "recorded", "episode-0001")
    assert outside.read_bytes() == b"must-stay"
    assert (root / "episode-0001").exists()
    assert not store._episode_state("recorded", "episode-0001").get("deleted_at")
    for run_id, episode_id in [("..", "episode-0001"), ("recorded", "../outside"), ("recorded", "episode-0001/../../outside")]:
        with pytest.raises(ValueError):
            store.delete_episode(project_id, run_id, episode_id)
