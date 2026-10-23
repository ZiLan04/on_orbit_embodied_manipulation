"""Short batched SB3/PPO integration check. This is not a training job."""

from __future__ import annotations

import argparse

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from stable_baselines3 import PPO

from sb3_vecenv_adapter import StarlinkCubeSB3VecEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=256)
    args = parser.parse_args()
    env = StarlinkCubeSB3VecEnv(num_envs=args.num_envs)
    try:
        rollout_steps = max(2, args.timesteps // args.num_envs)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=rollout_steps,
            batch_size=args.timesteps,
            n_epochs=1,
            learning_rate=3e-4,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        observations = env.reset()
        actions, _ = model.predict(observations, deterministic=True)
        observations, rewards, dones, infos = env.step(actions)
        print(
            "batched_ppo_integration_smoke_ok",
            f"num_envs={args.num_envs}",
            f"timesteps={args.timesteps}",
            f"observation_shape={observations.shape}",
            f"reward_shape={rewards.shape}",
            f"done_shape={dones.shape}",
            f"distance_m={infos[0].get('distance')}",
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())