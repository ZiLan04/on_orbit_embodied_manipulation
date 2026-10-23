"""Gymnasium bridge for a single physical Starlink cube-grasp environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


class StarlinkCubeGymEnv(gym.Env):
    """CPU-facing Gymnasium adapter; simulator state and reward stay on GPU."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 300) -> None:
        super().__init__()
        self.sim_env = StarlinkCubeDeltaEefEnv(num_envs=1)
        self.sim_env.max_steps = max_episode_steps
        if not self.sim_env.initialize():
            raise RuntimeError("Failed to initialize the Starlink cube environment")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.sim_env.observation_dim,),
            dtype=np.float32,
        )
        self._closed = False

    def _observation(self) -> np.ndarray:
        return self.sim_env.get_observation()[0].detach().cpu().numpy().astype(np.float32)

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for name, value in self.sim_env.last_reward_terms.items():
            item = value.reshape(-1)[0].detach().cpu().item()
            info[name] = item
        return info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if not self.sim_env.reset():
            raise RuntimeError("Starlink cube environment reset failed")
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action_np.shape}")
        action_tensor = torch.as_tensor(
            action_np, dtype=self.sim_env.datatype, device=self.sim_env.device
        ).unsqueeze(0)
        if not self.sim_env.step(action=action_tensor):
            raise RuntimeError("Starlink cube environment step failed")
        reward = float(self.sim_env.reward()[0].detach().cpu().item())
        terminated = bool(self.sim_env.get_done_mask()[0].detach().cpu().item())
        truncated = bool(self.sim_env.get_truncated_mask()[0].detach().cpu().item())
        return self._observation(), reward, terminated, truncated, self._info()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sim_env.starlink_manipulator.stop()
        self.sim_env.gsim.stop()