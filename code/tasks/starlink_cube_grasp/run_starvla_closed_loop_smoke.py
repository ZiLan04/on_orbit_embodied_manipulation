"""Run a bounded StarVLA closed-loop rollout in the Starlink cube task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import websockets.sync.client

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = True
TPV_CAM_SETTINGS["camera"]["res"] = (224, 224)

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


_connect = websockets.sync.client.connect


def _compatible_connect(*args, **kwargs):
    """Support the Genesis environment's older synchronous WebSocket client."""
    try:
        return _connect(*args, **kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        kwargs.pop("ping_interval", None)
        kwargs.pop("ping_timeout", None)
        return _connect(*args, **kwargs)


websockets.sync.client.connect = _compatible_connect
from starvla_genesis_bridge import StarVLAPolicyClient


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def observation(env: StarlinkCubeDeltaEefEnv) -> tuple[np.ndarray, np.ndarray]:
    rendered = env.gsim.cam.render(rgb=True, force_render=True)
    image = np.asarray(rendered[0] if isinstance(rendered, tuple) else rendered)
    if image.ndim == 4:
        image = image[0]
    eef_pos, eef_quat = env._eef_pose()
    gripper = env.last_gripper_open[0].detach().cpu().numpy().reshape(1).astype(np.float32)
    state = np.concatenate((
        eef_pos[0].detach().cpu().numpy().astype(np.float32),
        quat_to_rpy(eef_quat[0].detach().cpu().numpy()),
        gripper,
    )).reshape(1, 7)
    return image.astype(np.uint8, copy=False), state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6695)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", default="grasp and lift the cube")
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if not 1 <= args.execution_horizon <= 8:
        raise ValueError("--execution-horizon must be between 1 and 8")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = StarlinkCubeDeltaEefEnv(num_envs=1)
    client = None
    records = []
    try:
        if not env.initialize():
            raise RuntimeError("Starlink cube environment failed to initialize")
        client = StarVLAPolicyClient(args.host, args.port)
        metadata = client.metadata
        if metadata.get("action_chunk_size") != 8:
            raise RuntimeError(f"Expected 8 action steps, got {metadata}")
        if "new_embodiment" not in metadata.get("available_unnorm_keys", []):
            raise RuntimeError(f"Expected new_embodiment normalization, got {metadata}")

        episode_summaries = []
        for episode in range(args.episodes):
            if episode:
                env.reset()
            cube_position = env._cube_pose()[0][0].detach().cpu().tolist()
            chunk = None
            chunk_index = 0
            episode_steps = 0
            for step in range(args.steps):
                image, state = observation(env)
                if chunk is None or chunk_index >= args.execution_horizon:
                    response = client.predict_action({
                        "examples": [{"image": [image], "lang": args.instruction, "state": state}],
                        "unnorm_key": "new_embodiment",
                        "do_sample": False,
                        "use_ddim": True,
                        "num_ddim_steps": 4,
                    })
                    action_batch = np.asarray(response["data"]["actions"], dtype=np.float32)
                    if action_batch.shape != (1, 8, 7) or not np.isfinite(action_batch).all():
                        raise RuntimeError(f"Unexpected policy action shape/range: {action_batch.shape}")
                    chunk = action_batch[0]
                    chunk_index = 0

                action = chunk[chunk_index].copy()
                # Binary unapply returns 0=closed and 1=open; the controller
                # expects the PPO signed convention (negative=closed).
                action[6] = 2.0 * action[6] - 1.0
                if not env.step(action=torch.as_tensor(action, dtype=env.datatype, device=env.device).reshape(1, 7)):
                    raise RuntimeError("Starlink cube environment rejected policy action")
                records.append({"episode": episode, "step": step, "state": state.reshape(-1).tolist(), "action": action.tolist()})
                chunk_index += 1
                episode_steps = step + 1
                if bool(env.get_done_mask()[0].item()):
                    break

            episode_summaries.append({
                "episode": episode,
                "success": bool(env.get_done_mask()[0].item()),
                "steps": episode_steps,
                "cube_position": cube_position,
            })

        success_count = sum(item["success"] for item in episode_summaries)
        summary = {
            "episodes": args.episodes,
            "successes": success_count,
            "success_rate": success_count / args.episodes,
            "episode_summaries": episode_summaries,
            "metadata": metadata,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (args.output_dir / "actions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        print(json.dumps(summary))
        return 0
    finally:
        if client is not None:
            client.close()
        camera = getattr(env.gsim, "cam", None)
        if camera is not None and getattr(camera, "_recorded_imgs", None):
            camera.stop_recording(save_to_filename=str(args.output_dir / "closed-loop.mp4"), fps=20)
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
