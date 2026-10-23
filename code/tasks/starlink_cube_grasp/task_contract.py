"""GPU-friendly reward and success contract for Starlink cube grasping.

This module is intentionally independent of Genesis. The environment supplies
pinch/cube positions and the binary gripper state; this module owns the dense
reward, hold counter, and success criterion used by PPO and rollout filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CubeRewardConfig:
    approach_scale: float = 6.0
    close_scale: float = 0.2
    lift_scale: float = 12.0
    hold_bonus: float = 1.0
    success_bonus: float = 20.0
    action_penalty_scale: float = 0.002
    grasp_distance_m: float = 0.035
    stable_distance_m: float = 0.030
    success_lift_m: float = 0.030
    hold_steps: int = 10


class CubeGraspReward:
    """Stateful, vectorized reward with no host synchronization in ``compute``."""

    def __init__(self, num_envs: int, device: torch.device, dtype: torch.dtype,
                 config: CubeRewardConfig = CubeRewardConfig()) -> None:
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype
        self.config = config
        self.initial_cube_pos = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        self.previous_distance = torch.zeros(num_envs, device=device, dtype=dtype)
        self.previous_lift = torch.zeros(num_envs, device=device, dtype=dtype)
        self.hold_count = torch.zeros(num_envs, device=device, dtype=torch.int64)

    def reset(self, cube_pos: torch.Tensor, pinch_pos: torch.Tensor,
              env_ids: torch.Tensor | None = None) -> None:
        """Start new episodes using simulator tensors already resident on GPU."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        cube = cube_pos.to(device=self.device, dtype=self.dtype)
        pinch = pinch_pos.to(device=self.device, dtype=self.dtype)
        self.initial_cube_pos[env_ids] = cube[env_ids]
        self.previous_distance[env_ids] = torch.linalg.vector_norm(
            pinch[env_ids] - cube[env_ids], dim=-1
        )
        self.previous_lift[env_ids] = 0.0
        self.hold_count[env_ids] = 0

    def compute(self, pinch_pos: torch.Tensor, cube_pos: torch.Tensor,
                gripper_closed: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return reward and tensor-only diagnostics for one vectorized simulator step."""
        pinch = pinch_pos.to(device=self.device, dtype=self.dtype)
        cube = cube_pos.to(device=self.device, dtype=self.dtype)
        closed = gripper_closed.to(device=self.device, dtype=torch.bool)
        actions = action.to(device=self.device, dtype=self.dtype)
        if pinch.shape != (self.num_envs, 3) or cube.shape != (self.num_envs, 3):
            raise ValueError(f"Expected pinch/cube shapes ({self.num_envs}, 3), got {pinch.shape}/{cube.shape}")
        if actions.shape != (self.num_envs, 7):
            raise ValueError(f"Expected delta-EEF action shape ({self.num_envs}, 7), got {actions.shape}")

        distance = torch.linalg.vector_norm(pinch - cube, dim=-1)
        lift = cube[:, 2] - self.initial_cube_pos[:, 2]
        approach_progress = self.previous_distance - distance
        lift_progress = lift - self.previous_lift
        grasp_ready = closed & (distance <= self.config.grasp_distance_m)
        stable = (
            grasp_ready
            & (distance <= self.config.stable_distance_m)
            & (lift >= self.config.success_lift_m)
        )
        self.hold_count = torch.where(stable, self.hold_count + 1, torch.zeros_like(self.hold_count))
        success = self.hold_count >= self.config.hold_steps

        approach_reward = self.config.approach_scale * approach_progress
        close_reward = self.config.close_scale * grasp_ready.to(self.dtype)
        lift_reward = self.config.lift_scale * torch.clamp(lift_progress, min=0.0) * grasp_ready.to(self.dtype)
        hold_reward = self.config.hold_bonus * stable.to(self.dtype)
        success_reward = self.config.success_bonus * success.to(self.dtype)
        action_penalty = self.config.action_penalty_scale * torch.sum(actions[:, :6].square(), dim=-1)
        reward = approach_reward + close_reward + lift_reward + hold_reward + success_reward - action_penalty

        self.previous_distance = distance.detach()
        self.previous_lift = lift.detach()
        return reward, {
            "distance": distance,
            "lift": lift,
            "approach_reward": approach_reward,
            "close_reward": close_reward,
            "lift_reward": lift_reward,
            "hold_reward": hold_reward,
            "success_reward": success_reward,
            "action_penalty": action_penalty,
            "success": success,
            "hold_count": self.hold_count,
        }
