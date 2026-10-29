#!/usr/bin/env python3
"""Export recorded control ticks as an annotated MP4 without running a policy.

Requires Pillow and ffmpeg with libx264 (or imageio-ffmpeg). The initial image
gets one frame; each recorded transition gets exactly one frame. Observation
refreshes and model inference waits are omitted. No simulation is re-run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import textwrap

from PIL import Image, ImageDraw, ImageFont


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    if executable := shutil.which("ffmpeg"):
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg / imageio-ffmpeg or pass --ffmpeg PATH") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def samples(episode: Path):
    counts: Counter = Counter()
    first = True
    note = "Initial observation"
    with (episode / "transitions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            stage = event.get("stage", "task")
            if first and "observation" in event:
                first = False
                yield {"observation": event["observation"], "stage": "task",
                       "step": 0, "call": 0, "note": note}
            if event["kind"] == "decision":
                counts[stage] += 1
                note = event["decision"].get("note", "")
            elif event["kind"] == "transition":
                yield {"observation": event["next_observation"], "stage": stage,
                       "step": event["step"], "call": counts[stage], "note": note}


def safe_image(episode: Path, relative: str) -> Image.Image:
    resolved = (episode / relative).resolve()
    if not resolved.is_relative_to(episode.resolve()):
        raise ValueError("Recorded image path must remain inside its episode")
    with Image.open(resolved) as source:
        return source.convert("RGB")


class ExportCancelled(RuntimeError):
    pass


class ExportCancellation:
    """Stop a worker and its current encoder without abandoning a thread."""
    def __init__(self):
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process = None

    def check(self):
        if self._stop.is_set():
            raise ExportCancelled("Video generation stopped during server shutdown.")

    def attach(self, process):
        with self._lock:
            self._process = process
            if self._stop.is_set() and process.poll() is None:
                process.terminate()
        self.check()

    def detach(self, process):
        with self._lock:
            if self._process is process:
                self._process = None

    def cancel(self):
        self._stop.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                # Kill also releases a worker blocked writing raw frames into
                # stdin. This process only owns an unpublished temporary file.
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass


def _stop_process(process):
    if process.poll() is None:
        process.kill()
    process.wait()


def render(episode, sample, summary, index, fps, cameras, width):
    tile = width // len(cameras)
    header, footer = 112, 116
    canvas = Image.new("RGB", (width, tile + header + footer), "#0d1421")
    draw = ImageDraw.Draw(canvas)
    stage = sample["stage"].upper()
    accent = "#6ee7b7" if stage == "TASK" else "#fbbf24"
    draw.text((18, 12), f"{episode.parent.name} / {episode.name}", font=font(17), fill="#b5c3d6")
    draw.text((18, 38), f"{stage}   step {sample['step']}   GPT call {sample['call']}",
              font=font(25), fill=accent)
    draw.text((18, 75), f"Control time: {index / fps:.2f} s | {fps:g} fps | Inference waits omitted",
              font=font(18), fill="white")
    for i, camera in enumerate(cameras):
        path = sample["observation"]["images"].get(camera)
        if not path:
            raise ValueError(f"Missing current camera {camera} in {episode}")
        source = safe_image(episode, path)
        source.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(source, (i * tile + (tile - source.width) // 2,
                              header + (tile - source.height) // 2))
        draw.rectangle((i * tile, header, (i + 1) * tile, header + 28), fill="#19283d")
        draw.text((i * tile + 10, header + 4), camera, font=font(16), fill="white")
    y = header + tile + 10
    outcome = f"Episode result: TASK {summary['outcome']} | RESET {summary.get('reset_outcome', 'not attempted')}"
    draw.text((18, y), outcome, font=font(18), fill=accent)
    note = textwrap.shorten(" ".join(sample["note"].split()), width=245, placeholder="...")
    for i, line in enumerate(textwrap.wrap(note, width=110)[:3]):
        draw.text((18, y + 29 + i * 23), line, font=font(17), fill="#d7e1ed")
    return canvas


def export_episode(episode, output, basename, ffmpeg, cameras, fps, *, cancellation=None):
    episode, output = Path(episode), Path(output)
    cancellation = cancellation or ExportCancellation()
    cancellation.check()
    if not basename or basename in (".", "..") or Path(basename).name != basename:
        raise ValueError("Video basename must be a plain filename")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video frame rate must be finite and positive")
    if output.is_symlink():
        raise ValueError("Video output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((episode / "summary.json").read_text())
    frames = []
    for sample in samples(episode):
        cancellation.check()
        frames.append(sample)
    if not frames:
        raise ValueError(f"No recorded observations in {episode}")
    if cameras is None:
        cameras = [name for name in frames[0]["observation"]["images"]
                   if not name.startswith("reset_reference_")]
    if not cameras or len(cameras) > 8:
        raise ValueError("Video requires one to eight current cameras")
    stage_counts = Counter(frame["stage"] for frame in frames[1:])
    expected = {"task": summary["steps"], "reset": summary.get("reset_steps", 0)}
    if any(stage_counts[stage] != count for stage, count in expected.items()):
        raise ValueError(f"Transition counts disagree with summary in {episode}")
    width = 512 * len(cameras)
    first = render(episode, frames[0], summary, 0, fps, cameras, width)
    with tempfile.TemporaryDirectory(prefix=".rilo-video-", dir=output) as temporary_dir:
        return _encode_episode(episode, output, Path(temporary_dir), basename,
                               ffmpeg, cameras, fps, summary, frames, stage_counts, first, width, cancellation)


def _encode_episode(episode, output, staging, basename, ffmpeg, cameras, fps,
                    summary, frames, stage_counts, first, width, cancellation):
    video = staging / f"{basename}.mp4"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{first.width}x{first.height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-threads", "2", str(video)]
    with tempfile.TemporaryFile() as errors_file, subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors_file) as process:
        try:
            cancellation.attach(process)
            for index, sample in enumerate(frames):
                cancellation.check()
                frame = first if index == 0 else render(episode, sample, summary, index, fps, cameras, width)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait():
                errors_file.seek(0)
                errors = errors_file.read(4000).decode(errors="replace")
                raise RuntimeError(f"ffmpeg encoding failed: {errors}")
        except BaseException:
            _stop_process(process)
            raise
        finally:
            cancellation.detach(process)
    cancellation.check()
    reset_index = next((i for i, sample in enumerate(frames) if sample["stage"] == "reset"), None)
    poster_index = max(0, (reset_index or len(frames)) - 1)
    poster = staging / f"{basename}.jpg"
    render(episode, frames[poster_index], summary, poster_index, fps, cameras, width).save(poster, quality=92)
    # Decode the complete file with ffmpeg, independently confirming the muxed
    # video's actual frame count instead of trusting the write loop.
    with subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
                           "-map", "0:v:0", "-f", "framemd5", "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            cancellation.attach(process)
            decoded_output, errors = process.communicate()
            cancellation.check()
            if process.returncode:
                raise RuntimeError("Encoded video failed independent decoding: " + errors[:4000].decode(errors="replace"))
        except BaseException:
            _stop_process(process)
            raise
        finally:
            cancellation.detach(process)
    decoded = sum(1 for line in decoded_output.splitlines() if line and not line.startswith(b"#"))
    if decoded != len(frames):
        raise RuntimeError(f"Decoded {decoded} frames; expected {len(frames)}")
    result = {"run_id": episode.parent.name, "episode_id": episode.name,
              "task": summary["task"], "video": video.name, "poster": poster.name,
              "fps": fps, "frame_count": len(frames), "decoded_frame_count": decoded,
              "duration_seconds": len(frames) / fps,
              "executed_control_seconds": (len(frames) - 1) / fps,
              "initial_frame_seconds": 1 / fps,
              "task_steps": stage_counts["task"], "reset_steps": stage_counts["reset"],
              "reset_start_frame": reset_index,
              "reset_start_seconds": reset_index / fps if reset_index is not None else None,
              "task_outcome": summary["outcome"], "reset_outcome": summary.get("reset_outcome"),
              "cameras": cameras, "width": first.width, "height": first.height,
              "timeline": "One initial frame + one frame per executed control tick; observation refreshes and inference waits omitted.",
              "source": str(episode.resolve()), "bytes": video.stat().st_size}
    metadata = staging / f"{basename}.json"
    metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    cancellation.check()
    for artifact in (video, poster, metadata):
        if (output / artifact.name).is_symlink():
            raise ValueError("Cannot replace a video output symlink")
    previous = output / metadata.name
    if previous.exists():
        identity = json.loads(previous.read_text(encoding="utf-8"))
        if (identity.get("run_id"), identity.get("episode_id")) != (result["run_id"], result["episode_id"]):
            raise ValueError("Video output belongs to another episode")
    for artifact in (video, poster, metadata):
        artifact.replace(output / artifact.name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("artifacts/videos"))
    parser.add_argument("--prefix", help="Output prefix; valid only when exporting one run")
    parser.add_argument("--start-index", type=int, default=1, help="First numbered suffix when using --prefix")
    parser.add_argument("--ffmpeg", help="Path to an ffmpeg binary with libx264")
    parser.add_argument("--cameras", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    args = parser.parse_args()
    if args.prefix and (len(args.runs) != 1 or Path(args.prefix).name != args.prefix):
        parser.error("--prefix must be a plain filename prefix and used with exactly one run")
    if args.start_index < 1:
        parser.error("--start-index must be positive")
    executable = find_ffmpeg(args.ffmpeg)
    args.output.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        fps = float(manifest["environment"]["action_spec"]["control_hz"])
        for index, episode in enumerate(sorted(run.glob("episode-*")), args.start_index):
            if (episode / "summary.json").is_file():
                basename = f"{args.prefix}-{index:02d}" if args.prefix else f"{run.name}-{episode.name}"
                result = export_episode(episode, args.output, basename,
                                        executable, args.cameras, fps)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
