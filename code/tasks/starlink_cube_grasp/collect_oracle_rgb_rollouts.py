"""Collect vision-conditioned, successful Starlink cube demonstrations.

The oracle uses simulator-only cube pose for supervision.  Its recorded
actions use normalized delta-EEF semantics, so the visual policy must learn to
steer towards each randomized cube placement rather than replay one sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = True
TPV_CAM_SETTINGS["camera"]["res"] = (224, 224)

from delta_eef_control import DeltaEefLimits
from starlink_cube_delta_eef_env import CubeEnvConfig, StarlinkCubeDeltaEefEnv


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--approach-steps", type=int, default=50)
    parser.add_argument("--close-steps", type=int, default=35)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--task", default="grasp and lift the cube")
    parser.add_argument("--cube-noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if min(args.episodes, args.max_attempts, args.approach_steps, args.close_steps, args.max_steps) <= 0:
        raise ValueError("episode and phase lengths must be positive")
    if args.cube_noise_scale <= 0:
        raise ValueError("--cube-noise-scale must be positive")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    limits = DeltaEefLimits(normalized_actions=True)
    noise = tuple(value * args.cube_noise_scale for value in CubeEnvConfig().cube_spawn_noise_m)
    env = StarlinkCubeDeltaEefEnv(num_envs=1, control_limits=limits, config=CubeEnvConfig(cube_spawn_noise_m=noise))
    saved = 0
    try:
        if not env.initialize():
            return 1
        for attempt in range(args.max_attempts):
            if saved >= args.episodes:
                break
            env.reset()
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            cube_positions: list[np.ndarray] = []
            for step in range(args.max_steps):
                rendered = env.gsim.cam.render(rgb=True, force_render=True)
                frame = np.asarray(rendered[0] if isinstance(rendered, tuple) else rendered)
                if frame.ndim == 4:
                    frame = frame[0]

                action = torch.zeros((1, 7), dtype=env.datatype, device=env.device)
                cube_pos, _ = env._cube_pose()
                if step < args.approach_steps:
                    # The controller scales this normalized direction by its
                    # calibrated 1 cm Cartesian step limit.
                    action[:, :3] = (cube_pos - env._grasp_center()) / limits.max_xyz_step_m
                    action[:, :3].clamp_(-1.0, 1.0)
                    action[:, 6] = 1.0
                elif step < args.approach_steps + args.close_steps:
                    action[:, 6] = -1.0
                else:
                    action[:, 2] = 1.0
                    action[:, 6] = -1.0

                states.append(observation_state(env))
                actions.append(action[0].detach().cpu().numpy().astype(np.float32))
                frames.append(frame.astype(np.uint8, copy=False))
                cube_positions.append(cube_pos[0].detach().cpu().numpy().astype(np.float32))
                if not env.step(action=action):
                    raise RuntimeError("oracle rollout step failed")
                if bool(env.get_done_mask()[0].item()) or bool(env.get_truncated_mask()[0].item()):
                    break

            if bool(env.get_done_mask()[0].item()):
                path = args.out / f"episode_{saved:06d}.npz"
                np.savez_compressed(
                    path,
                    state=np.stack(states),
                    action=np.stack(actions),
                    base_view=np.stack(frames),
                    cube_position=np.stack(cube_positions),
                    task=np.asarray(args.task),
                )
                print(f"saved_success episode={saved} steps={len(states)} path={path}")
                saved += 1
            else:
                terms = env.last_reward_terms
                diagnostic = {
                    key: float(value[0].detach().cpu())
                    for key, value in terms.items()
                    if key in {"distance", "lift", "hold_count"}
                }
                print(
                    f"discarded_attempt attempt={attempt} steps={len(states)} "
                    f"diagnostic={diagnostic}"
                )
        print(f"collection_complete successes={saved} attempts={attempt + 1}")
        return 0 if saved == args.episodes else 3
    finally:
        camera = getattr(env.gsim, "cam", None)
        if camera is not None and getattr(camera, "_recorded_imgs", None):
            camera.stop_recording(save_to_filename=str(args.out / "collection-preview.mp4"), fps=20)
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
