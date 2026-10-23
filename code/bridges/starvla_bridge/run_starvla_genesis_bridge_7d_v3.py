"""Run the Genesis bridge with the state, transport, and IK batch compatibility fixes."""

from __future__ import annotations

import numpy as np

import run_starvla_genesis_bridge_7d_v2
from starvla_genesis_bridge import StarVLAGenesisBridge, main


_bounded_target = StarVLAGenesisBridge._bounded_target


def _batched_target(self, action, position, quaternion):
    """Match Genesis 0.4.3 IK's required leading environment dimension."""
    target_position, target_quaternion, gripper = _bounded_target(
        self, action, position, quaternion
    )
    return (
        np.asarray(target_position).reshape(1, 3),
        np.asarray(target_quaternion).reshape(1, 4),
        gripper,
    )


StarVLAGenesisBridge._bounded_target = _batched_target


if __name__ == "__main__":
    raise SystemExit(main())
