"""Collect successful RGB PPO rollouts in the raw format consumed by export_lerobot.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = True
TPV_CAM_SETTINGS["camera"]["res"] = (224, 224)

from stable_baselines3 import PPO

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def observation_state(env: StarlinkCubeDeltaEefEnv) -> np.ndarray:
    eef_pos, eef_quat = env._eef_pose()
    gripper = env.last_gripper_open[0].detach().cpu().numpy().reshape(1).astype(np.float32)
    return np.concatenate((
        eef_pos[0].detach().cpu().numpy().astype(np.float32),
        quat_to_rpy(eef_quat[0].detach().cpu().numpy()),
        gripper,
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--task", default="grasp and lift the cube")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = StarlinkCubeDeltaEefEnv(num_envs=1)
    if not env.initialize():
        return 1
    saved = 0
    try:
        model = PPO.load(str(args.model), device="cpu")
        for attempt in range(args.max_attempts):
            if saved >= args.episodes:
                break
            env.reset()
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            for _ in range(env.max_steps):
                rendered = env.gsim.cam.render(rgb=True, force_render=True)
                frame = rendered[0] if isinstance(rendered, tuple) else rendered
                frame = np.asarray(frame)
                if frame.ndim == 4:
                    frame = frame[0]
                obs = env.get_observation()[0].detach().cpu().numpy().astype(np.float32)
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(7)
                states.append(observation_state(env))
                actions.append(action.copy())
                frames.append(frame.astype(np.uint8, copy=False))
                action_tensor = torch.as_tensor(action, dtype=env.datatype, device=env.device).unsqueeze(0)
                if not env.step(action=action_tensor):
                    raise RuntimeError("PPO rollout step failed")
                if bool(env.get_done_mask()[0].item()) or bool(env.get_truncated_mask()[0].item()):
                    break
            if bool(env.get_done_mask()[0].item()):
                path = args.out / f"episode_{saved:06d}.npz"
                np.savez_compressed(
                    path,
                    state=np.stack(states),
                    action=np.stack(actions),
                    base_view=np.stack(frames),
                    task=np.asarray(args.task),
                )
                print(f"saved_success episode={saved} steps={len(states)} path={path}")
                saved += 1
            else:
                print(f"discarded_attempt attempt={attempt} steps={len(states)}")
        print(f"collection_complete successes={saved} attempts={attempt + 1}")
        return 0 if saved == args.episodes else 3
    finally:
        camera = getattr(env.gsim, "cam", None)
        if camera is not None and getattr(camera, "_recorded_imgs", None):
            camera.stop_recording(save_to_filename=str(args.out / "collection-preview.mp4"), fps=30)
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())