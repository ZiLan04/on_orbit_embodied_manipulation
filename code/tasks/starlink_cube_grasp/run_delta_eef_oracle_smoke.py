"""Physical feasibility check for the reward success criterion, not a data policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    video_path = args.video.resolve() if args.video is not None else None
    TPV_CAM_SETTINGS["enable_recording"] = video_path is not None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    env = StarlinkCubeDeltaEefEnv(num_envs=1)
    if not env.initialize():
        return 1
    try:
        action = torch.zeros((1, 7), dtype=env.datatype, device=env.device)
        for step in range(260):
            action.zero_()
            if step < 50:
                cube_pos, _ = env._cube_pose()
                action[:, :3] = cube_pos - env._grasp_center()
                action[:, 6] = 1.0
            elif step < 85:
                action[:, 6] = -1.0
            else:
                action[:, 2] = 1.0
                action[:, 6] = -1.0
            if not env.step(action=action):
                return 2

        terms = env.last_reward_terms
        print(
            "oracle_smoke",
            f"success={terms['success'].detach().cpu().tolist()}",
            f"distance_m={terms['distance'].detach().cpu().tolist()}",
            f"lift_m={terms['lift'].detach().cpu().tolist()}",
            f"hold_steps={terms['hold_count'].detach().cpu().tolist()}",
        )
        return 0 if bool(terms["success"].item()) else 3
    finally:
        if video_path is not None:
            camera = getattr(env.gsim, "cam", None)
            if camera is None or not getattr(camera, "_recorded_imgs", None):
                raise RuntimeError("Oracle video was requested but no camera frames were recorded")
            camera.stop_recording(save_to_filename=str(video_path), fps=30)
            print(f"oracle_video={video_path}")
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())