"""Headless smoke test for the existing physical Starlink cube grasp task.

The upstream task's debug formatting assumes unbatched tensors. This adapter
replaces only those logging helpers in memory; it does not modify the server
source task or its physical control sequence.
"""

from __future__ import annotations

import types

import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = False

from tasks import robot_assisted_grasp_task as source
from utils.utils import as_rotation_matrix


def _rotate_local_vector(rotation, vector):
    """Rotate a task-local offset while preserving an optional env batch."""
    if rotation.ndim == 2:
        return rotation @ vector
    if vector.ndim == 1:
        vector = vector.unsqueeze(0).expand(rotation.shape[0], -1)
    return torch.bmm(rotation, vector.unsqueeze(-1)).squeeze(-1)


def _grasp_center(self):
    links = [self.starlink_manipulator.robot.get_link(name) for name in self.grasp_finger_link_names]
    finger_midpoint = 0.5 * (links[0].get_pos().detach() + links[1].get_pos().detach())
    rotation = as_rotation_matrix(self.goal_quat, order="wxyz")
    return finger_midpoint + _rotate_local_vector(rotation, self.object_center_offset_local)


def _ee_target(self, grasp_center_target):
    rotation = as_rotation_matrix(self.goal_quat, order="wxyz")
    offset_world = _rotate_local_vector(rotation, self.ee_to_grasp_center_offset_local)
    return grasp_center_target - offset_world


def _initialize(self):
    try:
        if self.starlink_manipulator is None:
            self.starlink_manipulator = source.SatelliteManipulator(name="franka_merge", sensors=[], backends=[])
        if self.target_object is None:
            self.target_object = self._create_target_cube()

        self.gsim.start()
        self.starlink_manipulator.initialize()
        self.target_object.initialize()
        self.target_link_name = self.target_object.params["base"]
        self.starlink_manipulator.update_state()
        ee_pos = self.starlink_manipulator.ee_global_position.detach().clone()
        self.goal_quat = self.starlink_manipulator.ee_global_quaternion.detach().clone()
        self.grasp_center_pos = self._compute_grasp_center_from_fingers()
        ee_rotation = as_rotation_matrix(self.goal_quat, order="wxyz")
        self.ee_to_grasp_center_offset_local = _rotate_local_vector(
            ee_rotation.transpose(-1, -2), self.grasp_center_pos - ee_pos
        )
        self.last_gripper_command = 1.0

        self.target_object.robot.set_pos(self.grasp_center_pos.detach().clone(), zero_velocity=True)
        object_quat = self.object_quat.unsqueeze(0).expand(ee_pos.shape[0], -1)
        self.target_object.robot.set_quat(object_quat, zero_velocity=True)
        self.gsim.step()

        self.target_initial_pos = self.target_object.robot.get_link(self.target_link_name).get_pos().detach().clone()
        self.phase = "approach"
        self.status = source.TaskStatus.RUNNING
        self.success = False
        self.debug_escape_reported = False
        return True
    except Exception as error:
        self.logger.error("Batch-safe initialization failed: %s", error)
        self.status = source.TaskStatus.FAILED
        return False


def _quat_distance(self, q1, q2):
    if q1.ndim == 1:
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
    q1 = q1 / torch.linalg.vector_norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.linalg.vector_norm(q2, dim=-1, keepdim=True)
    dot = torch.clamp(torch.sum(q1 * q2, dim=-1).abs(), -1.0, 1.0)
    return (2.0 * torch.acos(dot)).max()


def _control_gripper_batched(self, gripper_open):
    force_scale = self.starlink_manipulator.gripper_force_scale
    force_scalar = force_scale if bool(gripper_open) else -force_scale
    finger_force = force_scalar * self.starlink_manipulator.config_gripper_joints
    if finger_force.ndim == 1:
        finger_force = finger_force.unsqueeze(0).expand(self.grasp_center_pos.shape[0], -1)
    self.starlink_manipulator.robot.control_dofs_force(
        force=finger_force,
        dofs_idx_local=self.starlink_manipulator.fingers_dof,
    )


def _control_cartesian_batched(self, target_pos):
    if target_pos.ndim == 1:
        target_pos = target_pos.unsqueeze(0)
    qpos = self.starlink_manipulator.IK.solve(target_pos, self.goal_quat)
    if qpos.ndim == 1:
        qpos = qpos.unsqueeze(0)
    motor_qpos = qpos[:, self.starlink_manipulator.motors_qs]
    self.starlink_manipulator.robot.control_dofs_position(
        position=motor_qpos,
        dofs_idx_local=self.starlink_manipulator.motors_dof,
    )


def _check_termination(self):
    object_pos = self.target_object.robot.get_link(self.target_link_name).get_pos().detach()
    lift_height = object_pos[:, 2] - self.target_initial_pos[:, 2]
    final_target = self.grasp_center_pos + self.transport_offset_stage2
    transport_error = torch.norm(object_pos - final_target, dim=-1)
    success_mask = (
        (lift_height > self.lift_success_height)
        & (transport_error < self.transport_success_threshold)
    )
    reached_hold = self.step_count >= self.transport_hold_step
    if reached_hold and bool(torch.all(success_mask).item()):
        self.success = True
        return True
    if self.step_count >= self.max_steps:
        self.success = False
        return True
    return False


def _step(self, dt=None):
    if self.status != source.TaskStatus.RUNNING:
        return False
    try:
        target_grasp_center, gripper_open = self._compute_cartesian_target()
        target_pos = self._compute_ee_target_from_grasp_center(target_grasp_center)
        _control_gripper_batched(self, gripper_open)
        self.last_gripper_command = gripper_open
        _control_cartesian_batched(self, target_pos)
        self.starlink_manipulator.step()
        self.target_object.step()
        self.gsim.step()
        self.step_count += 1
        if self.check_termination():
            self.status = source.TaskStatus.SUCCEED if self.success else source.TaskStatus.COMPLETED
            return False
        return True
    except Exception as error:
        self.logger.error("Batch-safe step failed: %s", error)
        self.status = source.TaskStatus.FAILED
        return False


def main() -> int:
    task = source.RobotAssistedGraspTask()
    task.debug_pose_logging = False
    task._compute_grasp_center_from_fingers = types.MethodType(_grasp_center, task)
    task._compute_ee_target_from_grasp_center = types.MethodType(_ee_target, task)
    task.initialize = types.MethodType(_initialize, task)
    task.step = types.MethodType(_step, task)
    task.check_termination = types.MethodType(_check_termination, task)
    if not task.initialize():
        return 1
    task.starlink_manipulator.IK.quat_distance = types.MethodType(
        _quat_distance, task.starlink_manipulator.IK
    )
    try:
        while task.status == source.TaskStatus.RUNNING and task.step():
            pass
        cube = task.target_object.robot.get_link(task.target_link_name).get_pos().detach()
        lift = cube[:, 2] - task.target_initial_pos[:, 2]
        final_target = task.grasp_center_pos + task.transport_offset_stage2
        transport_error = torch.norm(cube - final_target, dim=-1)
        print(f"status={task.status.value} success={task.success} steps={task.step_count} lift_m={lift.cpu().tolist()} transport_error_m={transport_error.cpu().tolist()}")
        return 0 if task.success else 2
    finally:
        task.stop()


if __name__ == "__main__":
    raise SystemExit(main())
