"""Run the Genesis bridge with host-specific WebSocket, state, and IK adapters."""

from __future__ import annotations

import torch

import run_starvla_genesis_bridge_7d_v3
from starvla_genesis_bridge import StarVLAGenesisBridge, main


_initialize = StarVLAGenesisBridge.initialize


def _initialize_with_safe_ik(self):
    """Use the existing IK solver while indexing its batched qpos result correctly."""
    if not _initialize(self):
        return False

    def control_joints(position, quaternion):
        qpos = self.starlink_manipulator.IK.solve(position, quaternion)
        if not torch.isfinite(qpos).all():
            self.logger.error("IK returned a non-finite joint solution")
            return False
        if qpos.dim() == 1:
            motor_qpos = qpos[self.starlink_manipulator.motors_qs]
        else:
            motor_qpos = qpos[:, self.starlink_manipulator.motors_qs]
        return self.starlink_manipulator.control_joint_pos(motor_qpos)

    self.starlink_manipulator.control_joints = control_joints
    return True


StarVLAGenesisBridge.initialize = _initialize_with_safe_ik


if __name__ == "__main__":
    raise SystemExit(main())
