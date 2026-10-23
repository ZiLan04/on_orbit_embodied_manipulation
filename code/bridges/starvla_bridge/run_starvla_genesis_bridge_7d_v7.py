"""Record a clearer, visibly moving Genesis-to-StarVLA bridge demonstration.

This launcher layers presentation and bounded action-scaling controls on top of
the validated v6 compatibility launcher. The policy response, workspace limit,
and Genesis 0.4.3 batched-IK path remain unchanged.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import run_starvla_genesis_bridge_7d_v6
import starvla_genesis_bridge as bridge


def _parse_demo_args() -> argparse.Namespace:
    """Consume demonstration-only arguments before the bridge parses its CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--translation-scale", type=float, default=3.0)
    parser.add_argument("--rotation-scale", type=float, default=1.5)
    parser.add_argument("--video-resolution", type=int, default=512)
    args, remaining = parser.parse_known_args()
    if args.translation_scale <= 0 or args.rotation_scale <= 0:
        parser.error("action scales must be positive")
    if args.video_resolution < 224:
        parser.error("video resolution must be at least 224 pixels")
    sys.argv = [sys.argv[0], *remaining]
    return args


DEMO_ARGS = _parse_demo_args()
_configure_scene = bridge._configure_scene
_bounded_target = bridge.StarVLAGenesisBridge._bounded_target


def _configure_demo_scene() -> None:
    """Use a higher-resolution camera for both policy observations and recording."""
    _configure_scene()
    bridge.TPV_CAM_SETTINGS["camera"].update(
        {"res": (DEMO_ARGS.video_resolution, DEMO_ARGS.video_resolution), "GUI": False}
    )


def _scaled_bounded_target(
    self: bridge.StarVLAGenesisBridge,
    action: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale policy deltas before the bridge applies its existing safety limits."""
    scaled_action = np.asarray(action, dtype=np.float64).copy()
    scaled_action[:3] *= DEMO_ARGS.translation_scale
    scaled_action[3:6] *= DEMO_ARGS.rotation_scale
    return _bounded_target(self, scaled_action, position, quaternion)


bridge._configure_scene = _configure_demo_scene
bridge.StarVLAGenesisBridge._bounded_target = _scaled_bounded_target


if __name__ == "__main__":
    raise SystemExit(bridge.main())
