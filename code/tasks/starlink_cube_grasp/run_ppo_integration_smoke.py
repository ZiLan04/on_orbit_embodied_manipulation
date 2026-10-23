"""Short SB3/PPO integration check. This is not a training job."""

from __future__ import annotations

import argparse

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from stable_baselines3 import PPO

from gymnasium_adapter import StarlinkCubeGymEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=64)
    args = parser.parse_args()

    env = StarlinkCubeGymEnv()
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            learning_rate=3e-4,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        observation, _ = env.reset()
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            "ppo_integration_smoke_ok",
            f"timesteps={args.timesteps}",
            f"observation_shape={observation.shape}",
            f"reward={reward:.6f}",
            f"terminated={terminated}",
            f"truncated={truncated}",
            f"distance_m={info.get('distance')}",
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())