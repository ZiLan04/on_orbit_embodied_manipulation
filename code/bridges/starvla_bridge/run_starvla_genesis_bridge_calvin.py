"""Run the Starlink bridge against a CALVIN-compatible StarVLA service."""

from __future__ import annotations

import numpy as np
import torch
import websockets.sync.client


_connect = websockets.sync.client.connect


def _compatible_connect(*args, **kwargs):
    """Support the host client when it rejects newer keepalive options."""
    try:
        return _connect(*args, **kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        kwargs.pop("ping_interval", None)
        kwargs.pop("ping_timeout", None)
        return _connect(*args, **kwargs)


websockets.sync.client.connect = _compatible_connect

from starvla_genesis_bridge_calvin import StarVLAGenesisBridge, main

import starvla_genesis_bridge_calvin as bridge


_configure_scene = bridge._configure_scene


def _configure_headless_scene():
    """Render camera observations without requiring an X display."""
    _configure_scene()
    bridge.SCENE_SETTINGS["show_viewer"] = False


bridge._configure_scene = _configure_headless_scene


_bounded_target = StarVLAGenesisBridge._bounded_target


def _batched_target(self, action, position, quaternion):
    """Supply the leading environment axis expected by Genesis 0.4.3 IK."""
    target_position, target_quaternion, gripper = _bounded_target(
        self, action, position, quaternion
    )
    return (
        np.asarray(target_position).reshape(1, 3),
        np.asarray(target_quaternion).reshape(1, 4),
        gripper,
    )


StarVLAGenesisBridge._bounded_target = _batched_target
_initialize = StarVLAGenesisBridge.initialize


def _initialize_with_batched_ik(self):
    """Install the validated direct joint controller after simulator startup."""
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
