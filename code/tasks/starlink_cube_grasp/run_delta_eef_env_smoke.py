"""Smoke test for the isolated physical Starlink cube RL environment."""

import argparse
import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    env = StarlinkCubeDeltaEefEnv(num_envs=args.num_envs)
    if not env.initialize():
        return 1
    try:
        action = torch.zeros((args.num_envs, 7), dtype=env.datatype, device=env.device)
        action[:, 6] = 1.0
        for _ in range(args.steps):
            if not env.step(action=action):
                return 2
            assert env.get_observation().shape == (args.num_envs, env.observation_dim)
            assert env.reward().shape == (args.num_envs,)
            assert torch.isfinite(env.reward()).all()
            assert env.get_done_mask().shape == (args.num_envs,)
        if args.num_envs > 1:
            env.reset_done_envs(torch.tensor([0, args.num_envs - 1], device=env.device))
            assert env.get_observation().shape == (args.num_envs, env.observation_dim)
            assert not env.get_done_mask().any()
        print(
            "smoke_ok",
            f"observation_shape={tuple(env.get_observation().shape)}",
            f"reward={env.reward().detach().cpu().tolist()}",
            f"done={env.get_done_mask().detach().cpu().tolist()}",
        )
        return 0
    finally:
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())