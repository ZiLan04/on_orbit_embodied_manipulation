"""Independent deterministic evaluation for a trained Starlink cube PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from stable_baselines3 import PPO

from sb3_vecenv_adapter import StarlinkCubeSB3VecEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=256)
    args = parser.parse_args()

    env = StarlinkCubeSB3VecEnv(num_envs=args.num_envs)
    try:
        model = PPO.load(str(args.model), env=env, device="cpu")
        observations = env.reset()
        running_returns = [0.0] * args.num_envs
        running_lengths = [0] * args.num_envs
        completed = 0
        successes = 0
        returns: list[float] = []
        lengths: list[int] = []
        while completed < args.episodes:
            actions, _ = model.predict(observations, deterministic=True)
            observations, rewards, dones, infos = env.step(actions)
            for env_id in range(args.num_envs):
                running_returns[env_id] += float(rewards[env_id])
                running_lengths[env_id] += 1
                if not dones[env_id]:
                    continue
                if completed < args.episodes:
                    successes += int(bool(infos[env_id].get("success", False)))
                    returns.append(running_returns[env_id])
                    lengths.append(running_lengths[env_id])
                    completed += 1
                running_returns[env_id] = 0.0
                running_lengths[env_id] = 0
        report = {
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
            "mean_return": sum(returns) / len(returns),
            "mean_episode_steps": sum(lengths) / len(lengths),
            "deterministic": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("evaluation_complete", json.dumps(report, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())