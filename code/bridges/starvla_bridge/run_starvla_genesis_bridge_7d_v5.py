"""Run the Genesis bridge with the simulator's required tensor, batch, and state adapters."""

from __future__ import annotations

import torch

import run_starvla_genesis_bridge_7d_v4
from starvla_genesis_bridge import StarVLAGenesisBridge, main


_initialize = StarVLAGenesisBridge.initialize


def _initialize_with_tensor_ik(self):
    """Convert bounded NumPy targets to the tensors required by SmoothIKSolver."""
    if not _initialize(self):
        return False

    def control_joints(position, quaternion):
        robot = self.starlink_manipulator
        position_tensor = torch.as_tensor(position, dtype=robot.datatype, device=robot.device)
        quaternion_tensor = torch.as_tensor(
            quaternion, dtype=robot.datatype, device=robot.device
        )
        qpos = robot.IK.solve(position_tensor, quaternion_tensor)
        if not torch.isfinite(qpos).all():
            self.logger.error("IK returned a non-finite joint solution")
            return False
        motor_qpos = qpos[robot.motors_qs] if qpos.dim() == 1 else qpos[:, robot.motors_qs]
        return robot.control_joint_pos(motor_qpos)

    self.starlink_manipulator.control_joints = control_joints
    return True


StarVLAGenesisBridge.initialize = _initialize_with_tensor_ik


if __name__ == "__main__":
    raise SystemExit(main())
