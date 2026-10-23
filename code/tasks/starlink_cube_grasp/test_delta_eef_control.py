"""Tests for Starlink delta-EEF control safety and gripper semantics."""

import torch

from delta_eef_control import DeltaEefController, DeltaEefLimits


def main() -> None:
    initial = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    limits = DeltaEefLimits(max_xyz_step_m=0.01, max_rpy_step_rad=0.05, max_displacement_m=0.02)
    controller = DeltaEefController(initial, limits)
    pos = torch.tensor([[0.019, 0.0, 0.0], [0.981, 1.0, 1.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    action = torch.tensor([[1.0, -1.0, 0.0, 9.0, 0.0, 0.0, -1.0], [-1.0, 1.0, 0.0, 0.0, 9.0, 0.0, 1.0]])
    target_pos, target_quat, gripper = controller.targets(pos, quat, action)
    assert torch.allclose(target_pos[0], torch.tensor([0.02, -0.01, 0.0]))
    assert torch.allclose(target_pos[1], torch.tensor([0.98, 1.01, 1.0]))
    assert torch.allclose(torch.linalg.vector_norm(target_quat, dim=-1), torch.ones(2))
    assert torch.equal(gripper, torch.tensor([0.0, 1.0]))

    normalized = DeltaEefController(
        initial[:1],
        DeltaEefLimits(
            max_xyz_step_m=0.01,
            max_rpy_step_rad=0.05,
            max_displacement_m=0.02,
            normalized_actions=True,
        ),
    )
    target_pos, _, _ = normalized.targets(
        torch.zeros((1, 3)),
        quat[:1],
        torch.tensor([[0.5, -0.5, 0.0, 0.5, 0.0, 0.0, -1.0]]),
    )
    assert torch.allclose(target_pos[0], torch.tensor([0.005, -0.005, 0.0]))
    print("delta-EEF control tests passed")


if __name__ == "__main__":
    main()
