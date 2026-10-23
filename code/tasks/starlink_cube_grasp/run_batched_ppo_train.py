"""Checkpointed batched PPO launcher for Starlink cube grasping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_vecenv_adapter import StarlinkCubeSB3VecEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-steps", type=int, default=25_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    success_dir = run_dir / "success_trajectories"
    tensorboard_dir = run_dir / "tensorboard"
    for path in (checkpoint_dir, success_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    (run_dir / "train-config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    env = StarlinkCubeSB3VecEnv(num_envs=args.num_envs, success_export_dir=success_dir)
    try:
        rollout_steps = args.rollout_steps
        callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_steps // args.num_envs),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_starlink_cube",
        )
        model = PPO("MlpPolicy", env, n_steps=rollout_steps, batch_size=min(1024, rollout_steps * args.num_envs), n_epochs=4, learning_rate=3e-4, device="cpu", tensorboard_log=str(tensorboard_dir), verbose=1)
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
        model.save(str(run_dir / "ppo_starlink_cube_final"))
        print(f"training_complete run_dir={run_dir}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())