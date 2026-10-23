"""Shared 7-D delta-EEF action semantics for Starlink RL and VLA rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeltaEefLimits:
    max_xyz_step_m: float = 0.01
    max_rpy_step_rad: float = 0.05
    max_displacement_m: float = 0.45
    normalized_actions: bool = False


def _rpy_quaternion(rpy: torch.Tensor) -> torch.Tensor:
    half = rpy * 0.5
    roll, pitch, yaw = half.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dim=-1)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    result = torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-8)


class DeltaEefController:
    """Map normalized action ``[dx,dy,dz,droll,dpitch,dyaw,gripper]`` safely."""

    def __init__(self, initial_position: torch.Tensor, limits: DeltaEefLimits = DeltaEefLimits()) -> None:
        if initial_position.ndim != 2 or initial_position.shape[-1] != 3:
            raise ValueError(f"initial_position must be (N, 3), got {initial_position.shape}")
        self.initial_position = initial_position.detach().clone()
        self.limits = limits

    def targets(self, position: torch.Tensor, quaternion: torch.Tensor,
                action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position.shape != self.initial_position.shape or quaternion.shape != (*position.shape[:-1], 4):
            raise ValueError("position/quaternion batch shapes do not match controller state")
        if action.shape != (*position.shape[:-1], 7):
            raise ValueError(f"Expected action shape {(*position.shape[:-1], 7)}, got {action.shape}")
        if self.limits.normalized_actions:
            # New datasets use the standard [-1, 1] action convention. Scale
            # before integrating so sub-maximal commands retain their meaning.
            xyz = torch.clamp(action[:, :3], -1.0, 1.0) * self.limits.max_xyz_step_m
            rpy = torch.clamp(action[:, 3:6], -1.0, 1.0) * self.limits.max_rpy_step_rad
        else:
            # Preserve the original physical-delta convention for the first
            # PPO/VLA checkpoint, whose labels were recorded in this space.
            xyz = torch.clamp(action[:, :3], -self.limits.max_xyz_step_m, self.limits.max_xyz_step_m)
            rpy = torch.clamp(action[:, 3:6], -self.limits.max_rpy_step_rad, self.limits.max_rpy_step_rad)
        lower = self.initial_position - self.limits.max_displacement_m
        upper = self.initial_position + self.limits.max_displacement_m
        target_position = torch.maximum(torch.minimum(position + xyz, upper), lower)
        target_quaternion = _quaternion_multiply(quaternion, _rpy_quaternion(rpy))
        # Starlink control_gripper uses unit semantics: 1=open, 0=closed.
        gripper_open = (action[:, 6] >= 0.0).to(dtype=position.dtype)
        return target_position, target_quaternion, gripper_open
