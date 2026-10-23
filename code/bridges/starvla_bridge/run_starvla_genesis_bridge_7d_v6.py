"""Run the Genesis bridge with its verified WebSocket, state, and batched-IK adapters."""

from __future__ import annotations

import torch

import run_starvla_genesis_bridge_7d_v5
from starvla_genesis_bridge import StarVLAGenesisBridge, main


_initialize = StarVLAGenesisBridge.initialize


def _initialize_with_batched_ik(self):
    """Use Genesis 0.4.3 batched IK and preserve the solver's joint-change bound."""
    if not _initialize(self):
        return False

    previous_motor_qpos = None
    max_joint_change = float(self.starlink_manipulator.params["ik_params"]["max_joint_change"])

    def control_joints(position, quaternion):
        nonlocal previous_motor_qpos
        robot = self.starlink_manipulator
        position_tensor = torch.as_tensor(position, dtype=robot.datatype, device=robot.device)
        quaternion_tensor = torch.as_tensor(
            quaternion, dtype=robot.datatype, device=robot.device
        )
        qpos = robot.robot.inverse_kinematics(
            link=robot.end_effector,
            pos=position_tensor,
            quat=quaternion_tensor,
        )
        if not torch.isfinite(qpos).all():
            self.logger.error("IK returned a non-finite joint solution")
            return False
        motor_qpos = qpos[robot.motors_qs] if qpos.dim() == 1 else qpos[:, robot.motors_qs]
        if previous_motor_qpos is not None:
            delta = motor_qpos - previous_motor_qpos
            delta_norm = torch.linalg.vector_norm(delta)
            if delta_norm > max_joint_change:
                motor_qpos = previous_motor_qpos + delta * (max_joint_change / delta_norm)
        previous_motor_qpos = motor_qpos.detach().clone()
        return robot.control_joint_pos(motor_qpos)

    self.starlink_manipulator.control_joints = control_joints
    return True


StarVLAGenesisBridge.initialize = _initialize_with_batched_ik


if __name__ == "__main__":
    raise SystemExit(main())
