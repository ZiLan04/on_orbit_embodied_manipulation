"""Independent batched Genesis environment for physical Starlink cube grasping.

This module deliberately does not import the scripted grasp task.  Its action
contract is the VLA-compatible 7-D delta end-effector command and its rewards
remain torch tensors on the simulator device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import genesis as gs
import torch

from robots.satellite_manipulator import SatelliteManipulator
from tasks.RL_task import RLTask, TaskStatus
from utils.utils import as_rotation_matrix

from delta_eef_control import DeltaEefController, DeltaEefLimits
from task_contract import CubeGraspReward, CubeRewardConfig


class _EntityAdapter:
    """Minimal task-facing adapter for a Genesis rigid entity."""

    def __init__(self, robot, base_link_name: str) -> None:
        self.robot = robot
        self.params = {"base": base_link_name}

    def initialize(self) -> bool:
        return True

    def step(self) -> bool:
        return True

    def on_episode_end(self, *_args, **_kwargs) -> bool:
        return True

    def stop(self) -> bool:
        return True


@dataclass(frozen=True)
class CubeEnvConfig:
    max_episode_steps: int = 300
    cube_size_m: float = 0.03
    cube_spawn_offset_m: tuple[float, float, float] = (0.0, -0.15, 0.0)
    cube_spawn_noise_m: tuple[float, float, float] = (0.02, 0.02, 0.01)


class StarlinkCubeDeltaEefEnv(RLTask):
    """GPU-batched physical cube grasp environment with a 7-D action space."""

    action_dim = 7
    observation_dim = 21

    def __init__(
        self,
        num_envs: int = 1,
        *,
        env_spacing=None,
        n_envs_per_row=None,
        center_envs_at_origin: Optional[bool] = None,
        control_limits: DeltaEefLimits = DeltaEefLimits(),
        reward_config: CubeRewardConfig = CubeRewardConfig(),
        config: CubeEnvConfig = CubeEnvConfig(),
    ) -> None:
        super().__init__(
            target_asset_name="starlink_cube",
            num_envs=num_envs,
            env_spacing=env_spacing,
            n_envs_per_row=n_envs_per_row,
            center_envs_at_origin=center_envs_at_origin,
            target_position_noise=False,
        )
        self.task_name = "starlink_cube_delta_eef"
        self.debug_grasp_marker = False
        self.max_steps = config.max_episode_steps
        self.config_cube = config
        self.control_limits = control_limits
        self.reward_config = reward_config
        self.target_link_name = "box_baselink"
        self.grasp_finger_link_names = (
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
            "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
        )
        self.delta_controller: Optional[DeltaEefController] = None
        self.reward_model: Optional[CubeGraspReward] = None
        self.initial_dofs: Optional[torch.Tensor] = None
        self.env_step_count: Optional[torch.Tensor] = None
        self.env_done: Optional[torch.Tensor] = None
        self.env_truncated: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_gripper_open: Optional[torch.Tensor] = None
        self.last_reward: Optional[torch.Tensor] = None
        self.last_reward_terms: dict[str, torch.Tensor] = {}
        self.last_eef_target: Optional[torch.Tensor] = None
        self.last_ik_arm_target: Optional[torch.Tensor] = None
        self.last_ik_qpos_width: Optional[int] = None

    def _create_cube(self) -> _EntityAdapter:
        cube = self._scene.add_entity(
            morph=gs.morphs.Box(
                pos=(-1.1, 0.3, 1.0),
                size=(self.config_cube.cube_size_m,) * 3,
                fixed=False,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
            name="starlink_rl_cube",
        )
        return _EntityAdapter(cube, self.target_link_name)

    def _as_batch(self, value: torch.Tensor, trailing_dim: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=self.datatype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape != (self.num_envs, trailing_dim):
            raise ValueError(
                f"Expected ({self.num_envs}, {trailing_dim}), got {tuple(tensor.shape)}"
            )
        return tensor

    def _eef_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        link = self.starlink_manipulator.robot.get_link(
            self.starlink_manipulator.params["end_effector"]
        )
        return self._as_batch(link.get_pos(), 3), self._as_batch(link.get_quat(), 4)

    def _pinch_position(self) -> torch.Tensor:
        first = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[0])
        second = self.starlink_manipulator.robot.get_link(self.grasp_finger_link_names[1])
        return 0.5 * (self._as_batch(first.get_pos(), 3) + self._as_batch(second.get_pos(), 3))

    def _grasp_center(self) -> torch.Tensor:
        """Object-center target implied by the calibrated open finger geometry."""
        pinch = self._pinch_position()
        _, eef_quat = self._eef_pose()
        offset = torch.tensor(
            self.config_cube.cube_spawn_offset_m,
            dtype=self.datatype,
            device=self.device,
        ).expand(self.num_envs, -1)
        rotation = as_rotation_matrix(eef_quat, order="wxyz")
        offset_world = torch.bmm(rotation, offset.unsqueeze(-1)).squeeze(-1)
        return pinch + offset_world

    def _cube_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        link = self.target_object.robot.get_link(self.target_link_name)
        return self._as_batch(link.get_pos(), 3), self._as_batch(link.get_quat(), 4)

    def get_observation(self) -> torch.Tensor:
        eef_pos, eef_quat = self._eef_pose()
        cube_pos, _ = self._cube_pose()
        grasp_center = self._grasp_center()
        return torch.cat(
            [
                eef_pos,
                eef_quat,
                cube_pos,
                grasp_center,
                self.last_gripper_open.unsqueeze(-1),
                self.last_action,
            ],
            dim=-1,
        )

    def _sample_cube_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        grasp_center = self._grasp_center()[env_ids]

        noise = torch.tensor(
            self.config_cube.cube_spawn_noise_m,
            dtype=self.datatype,
            device=self.device,
        )
        jitter = (2.0 * torch.rand((env_ids.numel(), 3), device=self.device, dtype=self.datatype) - 1.0) * noise
        return grasp_center + jitter

    def _set_initial_robot_state(self, env_ids: torch.Tensor) -> None:
        self.starlink_manipulator.robot.set_dofs_position(
            self.initial_dofs[env_ids],
            dofs_idx_local=(
                self.starlink_manipulator.motors_dof
                + self.starlink_manipulator.fingers_dof
            ),
            envs_idx=env_ids,
            zero_velocity=True,
        )
        self.starlink_manipulator.robot.set_dofs_velocity(
            torch.zeros_like(self.initial_dofs[env_ids]),
            dofs_idx_local=(
                self.starlink_manipulator.motors_dof
                + self.starlink_manipulator.fingers_dof
            ),
            envs_idx=env_ids,
        )

    def _reset_envs(self, env_ids: torch.Tensor) -> bool:
        if env_ids.numel() == 0:
            return True
        self._set_initial_robot_state(env_ids)
        self.starlink_manipulator.update_state()
        cube_pos = self._sample_cube_positions(env_ids)
        cube_quat = torch.zeros((env_ids.numel(), 4), dtype=self.datatype, device=self.device)
        cube_quat[:, 0] = 1.0
        self.target_object.robot.set_pos(cube_pos, envs_idx=env_ids, zero_velocity=True)
        self.target_object.robot.set_quat(cube_quat, envs_idx=env_ids, zero_velocity=True)
        self.gsim.step()
        self.starlink_manipulator.update_state()

        grasp_center = self._grasp_center()
        cube, _ = self._cube_pose()
        self.reward_model.reset(cube, grasp_center, env_ids)
        self.env_step_count[env_ids] = 0
        self.env_done[env_ids] = False
        self.env_truncated[env_ids] = False
        self.last_gripper_open[env_ids] = 1.0
        self.last_action[env_ids] = 0.0
        self.last_reward[env_ids] = 0.0
        return True

    def initialize(self) -> bool:
        try:
            if self.starlink_manipulator is None:
                self.starlink_manipulator = SatelliteManipulator(name="franka_merge", sensors=[], backends=[])
            if self.target_object is None:
                self.target_object = self._create_cube()
            self.gsim.start()
            self.starlink_manipulator.initialize()
            self.target_object.initialize()
            self.starlink_manipulator.update_state()

            current_dofs = self.starlink_manipulator.config["initial_dofs"].to(
                dtype=self.datatype, device=self.device
            )
            self.initial_dofs = current_dofs.unsqueeze(0).expand(self.num_envs, -1).clone()
            eef_pos, _ = self._eef_pose()
            self.delta_controller = DeltaEefController(eef_pos, self.control_limits)
            self.reward_model = CubeGraspReward(
                self.num_envs, self.device, self.datatype, self.reward_config
            )
            self.env_step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.env_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.env_truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.last_action = torch.zeros((self.num_envs, self.action_dim), dtype=self.datatype, device=self.device)
            self.last_gripper_open = torch.ones(self.num_envs, dtype=self.datatype, device=self.device)
            self.last_reward = torch.zeros(self.num_envs, dtype=self.datatype, device=self.device)
            self._reset_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
            self.status = TaskStatus.RUNNING
            return True
        except Exception as error:
            self.logger.error("Cube RL initialization failed: %s", error)
            self.status = TaskStatus.FAILED
            return False

    def apply_action(self, action: torch.Tensor) -> None:
        if self.delta_controller is None:
            raise RuntimeError("initialize() must be called before apply_action()")
        action = torch.as_tensor(action, dtype=self.datatype, device=self.device)
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape ({self.num_envs}, 7), got {tuple(action.shape)}")
        eef_pos, eef_quat = self._eef_pose()
        target_pos, target_quat, gripper_open = self.delta_controller.targets(eef_pos, eef_quat, action)
        # Direct Genesis IK is batch-native; do not route RL through SmoothIKSolver.
        qpos = self.starlink_manipulator.robot.inverse_kinematics(
            link=self.starlink_manipulator.end_effector,
            pos=target_pos,
            quat=target_quat,
            # The robot has a free-floating base. Restrict IK to the six arm
            # joints so an end-effector command cannot be satisfied by moving
            # the base coordinates that this environment intentionally does not command.
            dofs_idx_local=self.starlink_manipulator.motors_dof,
        )
        qpos = self._as_batch(qpos, qpos.shape[-1])
        # Genesis can return either generalized coordinates (n_qs) or degrees
        # of freedom (n_dofs).  Extract arm targets in the matching basis.
        if qpos.shape[-1] == self.starlink_manipulator.robot.n_dofs:
            arm_indices = self.starlink_manipulator.motors_dof
        elif qpos.shape[-1] == self.starlink_manipulator.robot.n_qs:
            arm_indices = self.starlink_manipulator.motors_qs
        else:
            raise RuntimeError(
                f"Unexpected IK output width {qpos.shape[-1]}; "
                f"n_dofs={self.starlink_manipulator.robot.n_dofs}, "
                f"n_qs={self.starlink_manipulator.robot.n_qs}"
            )
        motor_qpos = qpos[:, arm_indices]
        self.last_eef_target = target_pos.detach().clone()
        self.last_ik_arm_target = motor_qpos.detach().clone()
        self.last_ik_qpos_width = qpos.shape[-1]
        self.starlink_manipulator.robot.control_dofs_position(
            motor_qpos, dofs_idx_local=self.starlink_manipulator.motors_dof
        )
        self.starlink_manipulator.control_gripper(gripper_open)
        self.last_action = action.detach().clone()
        self.last_gripper_open = gripper_open.detach().clone()

    def step(self, dt: float = None, action: Optional[torch.Tensor] = None) -> bool:
        if self.status != TaskStatus.RUNNING:
            return False
        try:
            if action is None:
                action = self.last_action
            self.apply_action(action)
            self.starlink_manipulator.step()
            self.target_object.step()
            self.gsim.step()
            self.starlink_manipulator.update_state()

            grasp_center = self._grasp_center()
            cube, _ = self._cube_pose()
            closed = self.last_gripper_open <= 0.5
            self.last_reward, self.last_reward_terms = self.reward_model.compute(
                grasp_center, cube, closed, self.last_action
            )
            self.env_step_count += 1
            self.env_done = self.last_reward_terms["success"].detach().clone()
            self.env_truncated = self.env_step_count >= self.max_steps
            self.last_done_mask = self.env_done.detach().clone()
            self.last_truncated_mask = self.env_truncated.detach().clone()
            self.step_count += 1
            return True
        except Exception as error:
            self.logger.error("Cube RL step failed: %s", error)
            self.status = TaskStatus.FAILED
            return False

    def reward(self) -> torch.Tensor:
        return self.last_reward

    def is_success(self) -> torch.Tensor:
        return self.last_reward_terms.get(
            "success", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def reset(self) -> bool:
        return self._reset_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def reset_done_envs(self, env_ids=None) -> bool:
        if env_ids is None:
            return self.reset()
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        return self._reset_envs(env_ids)

    def get_done_mask(self) -> torch.Tensor:
        return self.env_done.detach().clone()

    def get_truncated_mask(self) -> torch.Tensor:
        return self.env_truncated.detach().clone()