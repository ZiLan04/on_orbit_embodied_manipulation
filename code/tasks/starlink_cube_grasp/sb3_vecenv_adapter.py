"""SB3 VecEnv bridge that keeps a batch of Genesis cube environments in one scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from starlink_cube_delta_eef_env import StarlinkCubeDeltaEefEnv


class StarlinkCubeSB3VecEnv(VecEnv):
    """One GPU-resident Genesis batch exposed through Stable-Baselines3's VecEnv API."""

    def __init__(self, num_envs: int = 32, max_episode_steps: int = 300, success_export_dir: str | Path | None = None) -> None:
        self.sim_env = StarlinkCubeDeltaEefEnv(num_envs=num_envs)
        self.sim_env.max_steps = max_episode_steps
        if not self.sim_env.initialize():
            raise RuntimeError("Failed to initialize the batched Starlink cube environment")
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim_env.observation_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        super().__init__(num_envs, observation_space, action_space)
        self._pending_actions: np.ndarray | None = None
        self._closed = False
        self.success_export_dir = Path(success_export_dir) if success_export_dir else None
        if self.success_export_dir:
            self.success_export_dir.mkdir(parents=True, exist_ok=True)
        self._episode_buffers: list[list[tuple[np.ndarray, np.ndarray, float]]] = [[] for _ in range(num_envs)]
        self._success_index = 0

    def _observations(self) -> np.ndarray:
        return self.sim_env.get_observation().detach().cpu().numpy().astype(np.float32)

    def reset(self) -> np.ndarray:
        if not self.sim_env.reset():
            raise RuntimeError("Starlink cube batch reset failed")
        self._episode_buffers = [[] for _ in range(self.num_envs)]
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self._observations()

    def step_async(self, actions: np.ndarray) -> None:
        actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.shape != (self.num_envs, 7):
            raise ValueError(f"Expected action shape ({self.num_envs}, 7), got {actions_np.shape}")
        self._pending_actions = actions_np

    def _export_success(self, env_id: int) -> None:
        if self.success_export_dir is None or not self._episode_buffers[env_id]:
            return
        trajectory = self._episode_buffers[env_id]
        filename = self.success_export_dir / f"success_{self._success_index:06d}.npz"
        self._success_index += 1
        np.savez_compressed(
            filename,
            observation=np.stack([item[0] for item in trajectory]),
            action=np.stack([item[1] for item in trajectory]),
            reward=np.asarray([item[2] for item in trajectory], dtype=np.float32),
        )

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        actions_np = self._pending_actions
        observations_before = self._observations()
        actions = torch.as_tensor(actions_np, dtype=self.sim_env.datatype, device=self.sim_env.device)
        self._pending_actions = None
        if not self.sim_env.step(action=actions):
            raise RuntimeError("Starlink cube batch step failed")
        terminal_observations = self._observations()
        rewards = self.sim_env.reward().detach().cpu().numpy().astype(np.float32)
        terminated = self.sim_env.get_done_mask().detach().cpu().numpy().astype(bool)
        truncated = self.sim_env.get_truncated_mask().detach().cpu().numpy().astype(bool)
        dones = np.logical_or(terminated, truncated)
        for env_id in range(self.num_envs):
            self._episode_buffers[env_id].append((observations_before[env_id].copy(), actions_np[env_id].copy(), float(rewards[env_id])))
        reward_terms = {name: value.detach().cpu().numpy() for name, value in self.sim_env.last_reward_terms.items()}
        infos: list[dict[str, Any]] = []
        for env_id in range(self.num_envs):
            info = {name: values[env_id].item() for name, values in reward_terms.items()}
            if dones[env_id]:
                if bool(terminated[env_id]):
                    self._export_success(env_id)
                info["terminal_observation"] = terminal_observations[env_id]
                info["TimeLimit.truncated"] = bool(truncated[env_id] and not terminated[env_id])
                self._episode_buffers[env_id] = []
            infos.append(info)
        if np.any(dones):
            self.sim_env.reset_done_envs(torch.as_tensor(np.flatnonzero(dones), dtype=torch.long, device=self.sim_env.device))
        return self._observations(), rewards, dones, infos

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.sim_env.starlink_manipulator.stop()
            self.sim_env.gsim.stop()

    def get_attr(self, attr_name: str, indices: Iterable[int] | None = None):
        return [getattr(self.sim_env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Iterable[int] | None = None) -> None:
        setattr(self.sim_env, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Iterable[int] | None = None, **method_kwargs: Any):
        result = getattr(self.sim_env, method_name)(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: Iterable[int] | None = None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [None for _ in range(self.num_envs)]