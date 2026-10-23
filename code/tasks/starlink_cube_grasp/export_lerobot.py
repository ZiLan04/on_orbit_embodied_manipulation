"""Convert raw Genesis episode dumps into a LeRobot v2.1 dataset for StarVLA.

The contract implemented here was read straight out of
``starVLA/dataloader/gr00t_lerobot/datasets.py`` (the loader calls this layout
"v2.0"; `info.json` reports ``v2.1``):

* ``meta/info.json``      -- ``data_path`` / ``video_path`` patterns,
                             ``chunks_size``, ``fps``, and ``features`` keyed by
                             the *original* LeRobot key.  Each video feature needs
                             ``shape`` + ``names`` containing ``height``/``width``
                             /``channel``, plus an ``info.video.*`` block.
* ``meta/modality.json``  -- maps ``state.<key>`` / ``action.<key>`` / ``video.<key>``
                             onto a parquet column (``original_key``) with a
                             ``start``/``end`` slice.
* ``meta/tasks.jsonl``    -- one ``{"task_index", "task"}`` per line.
* ``meta/episodes.jsonl`` -- one ``{"episode_index", "tasks", "length"}`` per line.
* ``data/chunk-XXX/episode_XXXXXX.parquet``
* ``videos/chunk-XXX/<original_key>/episode_XXXXXX.mp4``

``meta/stats_gr00t.json`` is deliberately NOT written: the loader computes it
from the parquet files and caches it on first use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# raw npz key -> LeRobot "original_key"
VIDEO_KEYS = {
    "base_view": "observation.images.base_view",
    "ego_view": "observation.images.ego_view",
}

STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def find_ffmpeg(explicit: str | None) -> str | None:
    """Locate an ffmpeg binary without requiring the imageio-ffmpeg module.

    The export environment has pyarrow but not imageio-ffmpeg, while the
    simulator environment has the opposite, so the encoder is used across
    environments by pointing at a binary explicitly.
    """
    import os
    import shutil

    for candidate in (explicit, os.environ.get("FFMPEG_BINARY")):
        if candidate and Path(candidate).exists():
            return candidate
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def encode_with_ffmpeg(frames: np.ndarray, path: Path, fps: int, binary: str) -> None:
    """Encode HWC uint8 frames through a rawvideo pipe."""
    height, width = frames.shape[1:3]
    command = [
        binary, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    _, err = process.communicate(frames.tobytes())
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')[:400]}")


def write_video(frames: np.ndarray, path: Path, fps: int, ffmpeg: str | None = None) -> None:
    """Encode HWC uint8 frames to an H.264 mp4 the torchvision_av backend can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[1] % 2 or frames.shape[2] % 2:
        raise ValueError(f"libx264 needs even dimensions, got {frames.shape}")

    try:
        import imageio

        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            quality=8,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except Exception as exc:
        print(f"    imageio unavailable ({type(exc).__name__}); using ffmpeg binary", flush=True)

    binary = ffmpeg or find_ffmpeg(None)
    if not binary:
        raise RuntimeError("no ffmpeg available; pass --ffmpeg /path/to/ffmpeg")
    encode_with_ffmpeg(frames, path, fps, binary)


def build_info(total_episodes: int, total_frames: int, fps: int, chunk_size: int,
               resolutions: dict[str, tuple[int, int]], tasks: int = 1) -> dict:
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
    }
    for raw_key, original_key in VIDEO_KEYS.items():
        height, width = resolutions.get(raw_key, (224, 224))
        features[original_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps, "video.channels": 3, "video.height": height,
                     "video.width": width},
            # older loaders look here instead
            "video_info": {"video.fps": fps, "video.channels": 3},
        }
    for name, dtype in (("timestamp", "float32"), ("episode_index", "int64"),
                        ("frame_index", "int64"), ("task_index", "int64"),
                        ("index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}

    return {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": tasks,
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": max(1, (total_episodes + chunk_size - 1) // chunk_size),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def build_modality(has_ego: bool) -> dict:
    """Mirror the official Franka DataConfig's key names exactly.

    ``SingleFrankaRobotiqDeltaEefDataConfig`` asks for ``state.eef_position`` /
    ``state.eef_rotation`` and ``action.delta_eef_position`` /
    ``action.delta_eef_rotation`` / ``action.gripper_close``, so the sub-keys here
    must match or the loader's integrity check fails.
    """
    video = {"base_view": {"original_key": VIDEO_KEYS["base_view"]}}
    if has_ego:
        video["ego_view"] = {"original_key": VIDEO_KEYS["ego_view"]}
    return {
        "state": {
            "eef_position": {"start": 0, "end": 3, "original_key": "observation.state"},
            "eef_rotation": {"start": 3, "end": 6, "original_key": "observation.state"},
            "gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
        },
        "action": {
            "delta_eef_position": {"start": 0, "end": 3, "original_key": "action"},
            "delta_eef_rotation": {"start": 3, "end": 6, "original_key": "action"},
            "gripper_close": {"start": 6, "end": 7, "original_key": "action"},
        },
        "video": video,
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="directory of episode_*.npz dumps")
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--task", default="grasp the red cube")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--no-videos", action="store_true", help="skip mp4 encoding (parquet only)")
    parser.add_argument("--ffmpeg", default=None, help="explicit ffmpeg binary for video encoding")
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    episodes = sorted(raw_dir.glob("episode_*.npz"))
    if not episodes:
        raise SystemExit(f"no episode_*.npz found under {raw_dir}")

    (out / "meta").mkdir(parents=True, exist_ok=True)
    tasks = {args.task: 0}
    episode_records: list[dict] = []
    total_frames = 0
    resolutions: dict[str, tuple[int, int]] = {}
    has_ego = False
    global_index = 0

    for episode_index, path in enumerate(episodes):
        data = np.load(path, allow_pickle=True)
        task = str(data["task"]) if "task" in data else args.task
        tasks.setdefault(task, len(tasks))
        task_index = tasks[task]

        state = np.asarray(data["state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        length = int(state.shape[0])
        chunk = episode_index // args.chunk_size

        parquet_path = out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({
            "observation.state": [row for row in state],
            "action": [row for row in action],
            "timestamp": np.arange(length, dtype=np.float32) / float(args.fps),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + length, dtype=np.int64),
        })
        frame.to_parquet(parquet_path, index=False)
        global_index += length
        total_frames += length

        for raw_key, original_key in VIDEO_KEYS.items():
            if raw_key not in data:
                continue
            frames = np.asarray(data[raw_key])
            if frames.size == 0:
                continue
            resolutions[raw_key] = (frames.shape[1], frames.shape[2])
            if raw_key == "ego_view":
                has_ego = True
            if not args.no_videos:
                video_path = (out / "videos" / f"chunk-{chunk:03d}" / original_key
                              / f"episode_{episode_index:06d}.mp4")
                write_video(frames, video_path, args.fps, args.ffmpeg)

        episode_records.append({
            "episode_index": episode_index,
            "tasks": [task],
            "length": length,
        })
        print(f"  episode {episode_index}: {length} frames, task='{task}'", flush=True)

    info = build_info(len(episodes), total_frames, args.fps, args.chunk_size,
                      resolutions, tasks=len(tasks))
    available_video_keys = {
        original_key for raw_key, original_key in VIDEO_KEYS.items()
        if raw_key in resolutions and not args.no_videos
    }
    for feature_key in list(info["features"]):
        if info["features"][feature_key].get("dtype") == "video" and feature_key not in available_video_keys:
            del info["features"][feature_key]
    info["total_videos"] = len(episodes) * len(available_video_keys)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "meta" / "modality.json").write_text(
        json.dumps(build_modality(has_ego), indent=2), encoding="utf-8")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for name, index in sorted(tasks.items(), key=lambda item: item[1]):
            stream.write(json.dumps({"task_index": index, "task": name}) + "\n")
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in episode_records:
            stream.write(json.dumps(record) + "\n")

    print(f"\nwrote {len(episodes)} episodes / {total_frames} frames -> {out}")
    print(f"  data    : {out / 'data'}")
    print(f"  videos  : {out / 'videos'}  ({'skipped' if args.no_videos else 'encoded'})")
    print(f"  meta    : {out / 'meta'}  (stats_gr00t.json is generated by the loader)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
