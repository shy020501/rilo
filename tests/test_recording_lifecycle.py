"""Deleting a completed episode must survive controller summary rewrites."""
import json
import shutil

from rilo.adapters.demo import DemoAdapter
from rilo.recording import Recorder


def test_excluded_episode_stays_deleted_when_old_controller_writes_again(tmp_path):
    adapter = DemoAdapter(image_size=32)
    recorder = Recorder(tmp_path, {}, {})
    episode = {"id": "episode-0001", "index": 1, "outcome": "success"}
    summary = {"episode": episode, "stats": {"estimated_cost_usd": 1.25}, "phase": "completed"}
    try:
        recorder.begin_episode(1, adapter.reset(seed=0, task_id=0), 0)
        recorder.finish_episode(episode)
        recorder.write_summary(summary)
        shutil.rmtree(recorder.episode_path)
        recorder.exclude_episode(episode["id"])
        # Runner._stop_locked writes its last in-memory summary on the next
        # initialize and on shutdown, after the explicit deletion has finished.
        recorder.write_summary(summary)
        recorder.finish_episode(episode)
        persisted = json.loads((recorder.path / "summary.json").read_text())
        assert persisted["episode"] == {}
        assert persisted["stats"]["estimated_cost_usd"] == 1.25
        assert not recorder.episode_path.exists()
        assert recorder.summaries == []
        assert summary["episode"] is episode  # Controller state is preserved.
        # API usage receipts remain independent of demonstration payloads.
        recorder.write_usage({"episode_id": episode["id"], "estimated_cost_usd": 1.25})
        assert json.loads((recorder.path / "api_calls.jsonl").read_text())["estimated_cost_usd"] == 1.25
    finally:
        recorder.close()
        adapter.close()
