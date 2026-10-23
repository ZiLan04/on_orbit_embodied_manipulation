"""Drive a stock Genesis Franka Panda from a StarVLA policy server.

This is a deliberately self-contained companion to the original
``starvla_bridge`` experiment.  The original bridge swapped the policy into an
existing satellite scene and inherited that scene's ``Task``/``GenesisSim``
machinery.  This script instead builds a minimal scene from scratch -- ground
plane, stock Franka Panda, one coloured cube -- so the robot, the object and the
camera are all visible and editable in one file.

What is reused verbatim from the original bridge (that part already works):

* ``_pack_numpy`` / ``_unpack_numpy`` -- StarVLA's msgpack-numpy wire format.
* ``StarVLAPolicyClient`` -- synchronous WebSocket client.
* the quaternion / rpy helpers and ``_to_uint8_rgb``.

What is new:

* the scene (Franka Panda + coloured cube + table plane),
* IK through Genesis' own ``inverse_kinematics`` on the ``hand`` link,
* a policy profile table, so switching checkpoint does not mean editing code.

Known contract facts carried over from the validated Bridge run: the loaded
checkpoint wants a ``(1, 7)`` state (``xyz + rpy + normalised gripper``); the
eighth pad field advertised in the server metadata must NOT be sent, or the
``state_encoder`` fails on a ``1x8`` -> ``7x1024`` matrix multiply.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Genesis renders headless on this host through EGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

try:  # noqa: E402
    import msgpack
except ModuleNotFoundError:  # pragma: no cover - isolated venv fallback
    from pip._vendor import msgpack

import genesis as gs  # noqa: E402


# --------------------------------------------------------------------------- #
# Policy profiles
#
# Each profile describes how to talk to one StarVLA checkpoint.  ``chunk`` is
# the action horizon the server must report in its metadata; ``unnorm_key`` is
# the dataset key used for action un-normalisation; ``cameras`` names the
# observations the checkpoint expects, in the order it expects them.
# --------------------------------------------------------------------------- #

POLICY_PROFILES: dict[str, dict[str, Any]] = {
    # Verified 2026-09: server metadata reports action_chunk_size == 16 and
    # "oxe_bridge" among available_unnorm_keys.  Trained on WidowX/Bridge.
    "bridge": {
        "unnorm_key": "oxe_bridge",
        "chunk": 16,
        "cameras": ["primary"],
        "send_state": True,
        "state_dim": 7,
        "gripper": "unit",
        "description": "Qwen3VL-GR00T-Bridge-RT-1 (WidowX/Bridge RT-1)",
    },
    # Verified 2026-09-20 against examples/LIBERO/eval_files/model2libero_interface.py
    # and examples/calvin/eval_calvin.py:
    #   * two images, primary first then wrist, both 224x224
    #   * NO state field at all
    #   * the server still answers data["actions"] as (1, T, 7); the client
    #     slices one row per env step itself, so caching works the same way
    #   * open_gripper is 1 = open, 0 = close
    # Trained on CALVIN -- a real Franka, so the embodiment matches this scene.
    "calvin": {
        "unnorm_key": "franka",
        "chunk": 8,
        "cameras": ["primary", "wrist"],
        "send_state": False,
        "state_dim": 7,
        "gripper": "official",
        "description": "Calvin_D_D-Qwen2.5-VL-3B (Franka, CALVIN task D->D)",
    },
}


# --------------------------------------------------------------------------- #
# Wire format (copied from starvla_genesis_bridge.py)
# --------------------------------------------------------------------------- #


def _pack_numpy(value: Any) -> Any:
    """Encode NumPy values using StarVLA's msgpack-numpy wire representation."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in {"V", "O", "c"}:
        raise ValueError(f"Unsupported msgpack NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    """Decode StarVLA's msgpack-numpy wire representation."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


class StarVLAPolicyClient:
    """Minimal synchronous client for StarVLA's WebSocket policy endpoint."""

    def __init__(self, host: str, port: int) -> None:
        # Imported lazily so that ``--dry-run`` works in a simulator-only venv
        # that has no WebSocket client installed.
        import websockets.sync.client

        for name in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(name, None)
        self._connection = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=30,
            ping_interval=None,
            ping_timeout=60,
        )
        self.metadata = self._unpack(self._connection.recv())

    @staticmethod
    def _unpack(payload: bytes | str) -> dict[str, Any]:
        if isinstance(payload, str):
            raise RuntimeError(f"StarVLA server error: {payload}")
        return msgpack.unpackb(payload, object_hook=_unpack_numpy)

    def predict_action(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = msgpack.packb(request, default=_pack_numpy)
        self._connection.send(payload)
        return self._unpack(self._connection.recv())

    def close(self) -> None:
        """Close the underlying WebSocket connection."""
        self._connection.close()


def _quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert a Genesis [qw, qx, qy, qz] quaternion to roll, pitch, yaw."""
    qw, qx, qy, qz = quaternion
    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rpy_to_quaternion(rpy: np.ndarray) -> np.ndarray:
    """Return the [qw, qx, qy, qz] quaternion for an XYZ roll-pitch-yaw delta."""
    half_roll, half_pitch, half_yaw = rpy / 2.0
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two [qw, qx, qy, qz] quaternions and normalize the result."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a [qw, qx, qy, qz] quaternion."""
    qw, qx, qy, qz = quaternion
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return rotation @ np.asarray(vector, dtype=np.float64)


def _to_uint8_rgb(rendered: Any) -> np.ndarray:
    """Normalize a Genesis camera result into one HWC uint8 RGB image."""
    image = np.asarray(rendered)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected RGB render shape: {image.shape}")
    if image.dtype != np.uint8:
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _to_numpy(value: Any) -> np.ndarray:
    """Move any simulator scalar/array/tensor to a host NumPy array.

    Genesis returns CUDA torch tensors from ``get_pos``/``get_quat``/
    ``get_dofs_position``; those must be copied to the host before NumPy can
    touch them.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _as_vector(value: Any, size: int) -> np.ndarray:
    """Convert one simulator tensor or array into a single fixed-size vector."""
    array = _to_numpy(value)
    if array.ndim > 1:
        array = array[0]
    array = array.reshape(-1)
    if array.size != size:
        raise ValueError(f"Expected {size} values, got {array.shape}")
    return array


# --------------------------------------------------------------------------- #
# Official Franka contract helpers
#
# Copied from StarVLA's examples/Franka/eval_files/inference_single_example.py so
# that this client speaks exactly the contract the Franka case documents.
# --------------------------------------------------------------------------- #


def load_action_norm_stats(path: str | Path, embodiment_key: str = "franka") -> dict[str, Any]:
    """Load min/max/mask action statistics from a ``dataset_statistics.json``."""
    with open(path, encoding="utf-8") as stream:
        stats = json.load(stream)
    if embodiment_key in stats:
        stats = stats[embodiment_key]
    if "action" in stats:
        stats = stats["action"]
    norm_stats: dict[str, Any] = {
        "min": np.array(stats.get("min", stats.get("low", []))),
        "max": np.array(stats.get("max", stats.get("high", []))),
    }
    if "mask" in stats:
        norm_stats["mask"] = np.array(stats["mask"], dtype=bool)
    return norm_stats


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict[str, Any]) -> np.ndarray:
    """Map normalized actions in [-1, 1] back to the real action space.

    ``action = 0.5 * (normalized + 1) * (max - min) + min`` with the gripper
    dimension binarized first (``< 0.5 -> -1`` close, ``>= 0.5 -> +1`` open).

    IMPORTANT: apply this **only** when the server returns normalized actions.
    The StarVLA policy server deployed here already unnormalizes server-side
    (``PolicyNormProcessor.unapply_actions``), so running this on top of it
    would invert the transform twice and corrupt the action scale.
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])

    normalized_actions = np.clip(np.asarray(normalized_actions, dtype=np.float64), -1.0, 1.0)
    if normalized_actions.shape[-1] >= 7:
        normalized_actions[..., 6] = np.where(normalized_actions[..., 6] < 0.5, -1.0, 1.0)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def parse_response(result: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Extract an action chunk and report which response key carried it.

    StarVLA has shipped at least two response shapes -- ``data.actions`` (this
    deployment) and ``data.normalized_actions`` (the Franka README) -- so try
    every documented key instead of assuming one.
    """
    data = result.get("data", result)
    for key in ("normalized_actions", "actions", "action"):
        if key in data:
            actions = np.asarray(data[key], dtype=np.float64)
            if actions.ndim == 3:
                actions = actions[0]
            elif actions.ndim == 1:
                actions = actions.reshape(1, -1)
            return actions, key
    raise KeyError(f"Could not extract actions; keys present: {list(data.keys())}")


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #


class FrankaCubeScene:
    """Ground plane + stock Franka Panda + one coloured cube + cameras."""

    #: finger travel, metres (fully open)
    FINGER_OPEN = 0.04

    def __init__(self, args: argparse.Namespace, profile: dict[str, Any]) -> None:
        self.args = args
        self.profile = profile
        self.scene: Any = None
        self.franka: Any = None
        self.cube: Any = None
        self.cams: dict[str, Any] = {}
        self.ee_link: Any = None
        self.motors_dof: np.ndarray = np.arange(7)
        self.fingers_dof: np.ndarray = np.arange(7, 9)
        self.initial_position: np.ndarray | None = None
        self._last_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self.cube_home: np.ndarray = np.asarray(args.cube_pos, dtype=np.float64)

    @property
    def cam(self) -> Any:
        """The primary camera (kept as a convenience alias)."""
        return self.cams.get("primary")

    def build(self) -> None:
        """Initialise Genesis and construct the scene."""
        gs.init(
            backend=gs.gpu,
            precision="32",
            logging_level="warning",
            seed=self.args.seed,
        )

        scene_kwargs: dict[str, Any] = {
            "sim_options": gs.options.SimOptions(dt=1.0 / 100.0, substeps=2),
            "rigid_options": gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
            ),
            "vis_options": gs.options.VisOptions(rendered_envs_idx=[0]),
            "show_viewer": False,
        }
        # The rasterizer is what makes an offscreen camera work headless.
        try:
            scene_kwargs["renderer"] = gs.renderers.Rasterizer()
        except AttributeError:  # pragma: no cover - older Genesis
            pass
        self.scene = gs.Scene(**scene_kwargs)

        # Ground / table surface.
        self.scene.add_entity(gs.morphs.Plane())

        # Stock Franka Panda: joints joint1..joint7 + finger_joint1/2,
        # links link0..link7 + hand + left_finger + right_finger.
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        )
        self.ee_link = self.franka.get_link("hand")

        # The coloured cube the policy is asked to grasp.
        colour = tuple(float(c) for c in self.args.cube_color)
        size = float(self.args.cube_size)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(size, size, size),
                pos=tuple(self.args.cube_pos),
                fixed=False,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(color=colour),
        )

        primary = self.scene.add_camera(
            res=(224, 224),
            pos=tuple(self.args.cam_pos),
            lookat=tuple(self.args.cam_lookat),
            fov=self.args.cam_fov,
            GUI=False,
        )
        self.cams["primary"] = primary

        if "wrist" in self.profile["cameras"]:
            # A wrist camera is created at the primary pose and re-aimed every
            # step from the hand link (see ``aim_wrist_camera``).
            wrist = self.scene.add_camera(
                res=(224, 224),
                pos=tuple(self.args.cam_pos),
                lookat=tuple(self.args.cam_lookat),
                fov=60,
                GUI=False,
            )
            self.cams["wrist"] = wrist

        self.scene.build(n_envs=0)
        if self.args.record_video:
            for camera in self.cams.values():
                try:
                    camera.start_recording()
                except Exception:  # pragma: no cover - recording is best effort
                    pass

        self.aim_wrist_camera()

        for _ in range(self.args.warmup_steps):
            self.scene.step()

        self._settle_ready_pose()

        self.initial_position = self.ee_position()
        self._last_quat = self.ee_quaternion()

    def _settle_ready_pose(self) -> None:
        """Drive the arm to the ready pose before the episode starts.

        The MJCF neutral pose folds the arm straight up, almost a metre from
        anything on the table.  Servoing it to a pose above the cube first
        gives the policy (and the scripted dry-run) a reachable starting point.
        """
        steps = int(self.args.ready_steps)
        if steps <= 0:
            return
        ready_pos = np.asarray(self.args.ready_pose, dtype=np.float64)
        # Point the tool down: a rotation of pi about X maps +Z to -Z.
        ready_quat = _rpy_to_quaternion(np.array([math.pi, 0.0, 0.0]))
        for _ in range(steps):
            qpos, _ = self.franka.inverse_kinematics(
                link=self.ee_link,
                pos=ready_pos,
                quat=ready_quat,
                return_error=True,
            )
            qpos = _to_numpy(qpos).reshape(-1)
            self.franka.control_dofs_position(qpos[:7], self.motors_dof)
            self.franka.control_dofs_position(
                np.full(2, self.FINGER_OPEN, dtype=np.float64), self.fingers_dof
            )
            self.scene.step()
        residual = float(np.linalg.norm(self.ee_position() - ready_pos))
        print(
            f"[franka-cube] ready pose settled: EE={np.round(self.ee_position(), 4).tolist()} "
            f"target={np.round(ready_pos, 4).tolist()} residual={residual:.4f} m",
            flush=True,
        )

    def set_cube_position(self, position: np.ndarray) -> None:
        """Move the cube (and remember it as the episode's home position)."""
        self.cube_home = np.asarray(position, dtype=np.float64).copy()
        try:
            self.cube.set_pos(self.cube_home)
        except Exception:
            pass

    def reset(self) -> None:
        """``env.reset()``: restore the cube and return the arm to the ready pose."""
        try:
            self.cube.set_pos(self.cube_home)
        except Exception:
            pass
        try:
            self.cube.zero_all_dofs_velocity()
        except Exception:
            pass
        self.franka.control_dofs_position(
            np.full(2, self.FINGER_OPEN, dtype=np.float64), self.fingers_dof
        )
        self._settle_ready_pose()
        self.initial_position = self.ee_position()
        self._last_quat = self.ee_quaternion()

    # -- observation -------------------------------------------------------- #

    def ee_position(self) -> np.ndarray:
        """Current Cartesian position of the ``hand`` link."""
        return _to_numpy(self.ee_link.get_pos()).reshape(-1)[:3]

    def ee_quaternion(self) -> np.ndarray:
        """Current [qw, qx, qy, qz] of the ``hand`` link, cached on failure.

        Reading link orientation right after an IK-driven step has previously
        tripped a device-side assert on this stack, so a failure falls back to
        the last known-good orientation rather than killing the episode.
        """
        try:
            quat = _to_numpy(self.ee_link.get_quat()).reshape(-1)[:4]
            if quat.size == 4 and np.isfinite(quat).all():
                self._last_quat = quat
        except Exception:
            pass
        return self._last_quat

    def cube_position(self) -> np.ndarray:
        """Current Cartesian position of the cube."""
        return _to_numpy(self.cube.get_pos()).reshape(-1)[:3]

    def gripper_width(self) -> float:
        """Current finger separation in metres (0 closed .. FINGER_OPEN open)."""
        try:
            values = _to_numpy(self.franka.get_dofs_position(self.fingers_dof)).reshape(-1)
            return float(np.mean(np.abs(values)))
        except Exception:
            return float("nan")

    def fingertip_position(self) -> np.ndarray:
        """Midpoint between the two finger links -- where a grasp actually happens.

        The ``hand`` link origin sits roughly 0.10 m behind the fingertips, so
        distance-to-object must be measured from here, not from the wrist.
        """
        try:
            left = _to_numpy(self.franka.get_link("left_finger").get_pos()).reshape(-1)[:3]
            right = _to_numpy(self.franka.get_link("right_finger").get_pos()).reshape(-1)[:3]
            return 0.5 * (left + right)
        except Exception:
            return self.ee_position()

    def render(self) -> np.ndarray:
        """One 224x224 uint8 RGB frame from the primary camera."""
        rgb, *_ = self.cam.render(rgb=True, force_render=True)
        return _to_uint8_rgb(rgb)

    def aim_wrist_camera(self) -> None:
        """Point the wrist camera along the tool axis of the hand link.

        Cheap and orientation-aware: the hand's local +Z is the approach
        direction, so the camera sits just behind the wrist looking forward
        past the fingers.
        """
        wrist = self.cams.get("wrist")
        if wrist is None:
            return
        position = self.ee_position()
        quat = self.ee_quaternion()
        forward = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
        up = _quat_rotate(quat, np.array([0.0, 1.0, 0.0]))
        cam_pos = position - forward * 0.06 + up * 0.02
        lookat = position + forward * 0.20
        try:
            wrist.set_pose(pos=cam_pos, lookat=lookat, up=up)
        except Exception:
            pass

    def render_all(self) -> dict[str, np.ndarray]:
        """One uint8 RGB frame per camera the active profile asks for."""
        self.aim_wrist_camera()
        frames: dict[str, np.ndarray] = {}
        for name in self.profile["cameras"]:
            rgb, *_ = self.cams[name].render(rgb=True, force_render=True)
            frames[name] = _to_uint8_rgb(rgb)
        return frames

    def cube_visible_pixels(self, image: np.ndarray) -> int:
        """Count clearly-red pixels, to check whether the cube is actually visible.

        A 5 cm cube filling ~40 px across should give roughly 1500 red pixels;
        at the old wide framing it was closer to 120.
        """
        rgb = image.astype(np.int16)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (red > 90) & (red - green > 45) & (red - blue > 45)
        return int(mask.sum())

    # -- control ------------------------------------------------------------ #

    def command(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        gripper_open: bool,
    ) -> float:
        """Servo the arm to a Cartesian target; return the IK residual in metres."""
        qpos, ik_error = self.franka.inverse_kinematics(
            link=self.ee_link,
            pos=np.asarray(target_position, dtype=np.float64),
            quat=np.asarray(target_quaternion, dtype=np.float64),
            return_error=True,
        )
        qpos = _to_numpy(qpos).reshape(-1)
        if qpos.size < 7:
            raise RuntimeError(f"Unexpected IK output shape: {qpos.shape}")
        self.franka.control_dofs_position(qpos[:7], self.motors_dof)

        width = self.FINGER_OPEN if gripper_open else 0.0
        self.franka.control_dofs_position(
            np.array([width, width], dtype=np.float64), self.fingers_dof
        )
        return float(_to_numpy(ik_error).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #


class FrankaCubeStarvlaRunner:
    """Closed loop: render -> request chunk -> bounded IK step -> repeat."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.profile = POLICY_PROFILES[args.profile]
        self.scene_engine = FrankaCubeScene(args, self.profile)
        self.client: StarVLAPolicyClient | None = None
        self.action_chunk: np.ndarray | None = None
        self.action_index = 0
        self.step_count = 0
        self.gripper_mode = args.gripper_mode
        self.last_response_key: str | None = None
        # Held gripper state: the official Franka gate keeps the previous state
        # when the command sits between -0.9 and +0.9.
        self.gripper_open = True
        self._resolved_gripper_mode: str | None = None
        self.initial_visibility: dict[str, int] = {}
        self.action_norm_stats: dict[str, Any] | None = None
        # Scripted-grasp state machine, used only by --dry-run.
        self.scripted_phase: str | None = None
        self.scripted_phase_chunks = 0
        self.scripted_target: np.ndarray | None = None
        self.scripted_close = False
        if args.action_space == "normalized":
            self.action_norm_stats = load_action_norm_stats(args.action_stats, args.unnorm_key_override or "franka")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "franka-cube-run.jsonl"
        self.video_file = self.output_dir / "franka-cube-run.mp4"
        # Per-episode (image, state, action) dump for LeRobot export.
        self.raw_export_dir = Path(args.export_raw) if args.export_raw else None
        if self.raw_export_dir is not None:
            self.raw_export_dir.mkdir(parents=True, exist_ok=True)
        self.record_frames: dict[str, list[Any]] | None = None

    # -- helpers ------------------------------------------------------------ #

    def _log(self, message: str) -> None:
        print(f"[franka-cube] {message}", flush=True)

    def _write_event(self, event: str, **fields: Any) -> None:
        record = {"event": event, "step": self.step_count, **fields}
        with self.log_file.open("a", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=True)
            stream.write("\n")

    def observation_state(self) -> np.ndarray:
        """The 7-value proprioceptive state: ``xyz + rpy + normalised gripper``.

        Computed unconditionally, because dataset export needs it even for
        profiles that do not put state on the wire (CALVIN sends images only).
        """
        position = self.scene_engine.ee_position()
        rpy = _quaternion_to_rpy(self.scene_engine.ee_quaternion())
        width = self.scene_engine.gripper_width()
        normalised = 1.0 if not math.isfinite(width) else float(
            np.clip(width / FrankaCubeScene.FINGER_OPEN, 0.0, 1.0)
        )
        return np.array([*position, *rpy, normalised], dtype=np.float32)

    def _state(self) -> np.ndarray:
        """The state in the profile's wire format (empty when unused).

        The eighth pad value advertised by the server metadata must be omitted:
        sending it makes the ``state_encoder`` fail on a ``1x8`` -> ``7x1024``
        matrix multiply.
        """
        if not self.profile["send_state"]:
            return np.zeros((0,), dtype=np.float32)
        state = self.observation_state().reshape(1, 7)
        expected = self.profile["state_dim"]
        if state.shape[1] != expected:
            raise RuntimeError(f"State has {state.shape[1]} values, profile wants {expected}")
        return state

    def _extract_actions(self, response: dict[str, Any]) -> np.ndarray:
        chunk = int(self.profile["chunk"])
        actions, key = parse_response(response)
        self.last_response_key = key
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(f"Expected (T, 7) actions, got {actions.shape} (key '{key}')")
        if actions.shape[0] != chunk:
            raise RuntimeError(
                f"Expected {chunk} steps from key '{key}', got {actions.shape[0]}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("StarVLA returned non-finite actions")

        # Only invert the normalization when the server did NOT already do it.
        # The deployed server unnormalizes server-side, so the default is "raw".
        if self.args.action_space == "normalized":
            if self.action_norm_stats is None:
                raise RuntimeError(
                    "--action-space normalized needs --action-stats pointing at dataset_statistics.json"
                )
            before = actions.copy()
            actions = unnormalize_actions(actions, self.action_norm_stats)
            self._log(
                f"unnormalized chunk: before range=[{before.min():.4f}, {before.max():.4f}] "
                f"after range=[{actions.min():.4f}, {actions.max():.4f}]"
            )
        return actions

    # -- scripted grasp (--dry-run only) ------------------------------------ #
    #
    # A deterministic policy replacement: approach the cube from above, descend
    # until the fingers straddle it, close, then lift.  Actions servo the
    # *fingertip* position rather than the wrist, so the ~0.10 m tool offset
    # needs no manual compensation -- the same error term works for every phase.
    #
    # Each phase is entered once, and its world-frame target is frozen at entry
    # so a lifted cube cannot drag the target along with it.

    def _set_scripted_phase(self, phase: str) -> None:
        cube = self.scene_engine.cube_position()
        tip = self.scene_engine.fingertip_position()
        self.scripted_phase = phase
        self.scripted_phase_chunks = 0
        if phase == "approach":
            self.scripted_target = cube + np.array([0.0, 0.0, self.args.pre_grasp_height])
            self.scripted_close = False
        elif phase == "descend":
            # Aim the fingertips at the cube centre so they straddle it.
            self.scripted_target = cube.copy()
            self.scripted_close = False
        elif phase == "close":
            self.scripted_target = tip.copy()
            self.scripted_close = True
        elif phase == "lift":
            self.scripted_target = tip + np.array([0.0, 0.0, self.args.lift_height])
            self.scripted_close = True
        else:  # hold
            self.scripted_target = tip.copy()
            self.scripted_close = True
        self._log(
            f"scripted phase -> {phase} target={np.round(self.scripted_target, 4).tolist()} "
            f"gripper={'close' if self.scripted_close else 'open'}"
        )

    def _advance_scripted_phase(self) -> None:
        """Move to the next phase once the current one is satisfied or stalls."""
        cube = self.scene_engine.cube_position()
        tip = self.scene_engine.fingertip_position()
        phase = self.scripted_phase
        tol = self.args.reach_tolerance
        timed_out = self.scripted_phase_chunks >= self.args.phase_timeout
        assert self.scripted_target is not None
        reached = float(np.linalg.norm(self.scripted_target - tip)) < tol

        if phase == "approach":
            if reached or timed_out:
                self._set_scripted_phase("descend")
        elif phase == "descend":
            # The fingers grip best once their tips are at the cube centre.
            if tip[2] <= cube[2] + 0.006 or timed_out:
                self._set_scripted_phase("close")
        elif phase == "close":
            if self.scripted_phase_chunks >= self.args.close_chunks:
                self._set_scripted_phase("lift")
        elif phase == "lift":
            if reached or timed_out:
                self._set_scripted_phase("hold")

    def _scripted_chunk(self) -> np.ndarray:
        """One deterministic action chunk: servo the fingertips to the phase target."""
        if self.scripted_phase is None:
            self._set_scripted_phase("approach")
        else:
            self._advance_scripted_phase()
            self.scripted_phase_chunks += 1

        assert self.scripted_target is not None
        error = self.scripted_target - self.scene_engine.fingertip_position()
        translation = np.clip(error, -self.args.max_xyz_step, self.args.max_xyz_step)
        gripper = -1.0 if self.scripted_close else 1.0
        return np.array(
            [[*translation, 0.0, 0.0, 0.0, gripper]] * int(self.profile["chunk"]),
            dtype=np.float64,
        )

    def _request_chunk(self, frames: dict[str, np.ndarray], state: np.ndarray) -> None:
        if self.args.dry_run:
            self.action_chunk = self._scripted_chunk()
            self.action_index = 0
            return
        if self.client is None:
            raise RuntimeError("Policy client is not initialized")
        # Images go in the order the profile declares: primary first, wrist
        # second for CALVIN.
        images = [frames[name] for name in self.profile["cameras"]]
        example: dict[str, Any] = {
            "image": images,
            "lang": self.args.instruction,
        }
        if self.profile["send_state"]:
            example["state"] = state
        response = self.client.predict_action(
            {
                "examples": [example],
                "unnorm_key": self.profile["unnorm_key"],
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": self.args.num_ddim_steps,
            }
        )
        self.action_chunk = self._extract_actions(response)
        self.action_index = 0
        self._log(
            f"chunk@{self.step_count} range=[{self.action_chunk.min():.5f}, "
            f"{self.action_chunk.max():.5f}] gripper={self.action_chunk[:, 6].tolist()}"
        )

    def _bounded_target(
        self, action: np.ndarray, position: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        assert self.scene_engine.initial_position is not None
        translation = np.clip(action[:3], -self.args.max_xyz_step, self.args.max_xyz_step)
        rpy_delta = np.clip(action[3:6], -self.args.max_rpy_step, self.args.max_rpy_step)
        lower = self.scene_engine.initial_position - self.args.max_ee_displacement
        upper = self.scene_engine.initial_position + self.args.max_ee_displacement
        # Keep the arm above the table: a tiny floor stops the bounded target
        # from driving the hand through the ground plane.
        target_position = np.clip(position + translation, lower, upper)
        target_position[2] = max(target_position[2], 0.03)
        target_quaternion = _multiply_quaternions(
            quaternion, _rpy_to_quaternion(rpy_delta)
        )
        return target_position, target_quaternion, self._gripper_command(float(action[6]))

    def _resolve_gripper_mode(self) -> str:
        """Pick a gripper convention once, from the first chunk, and keep it.

        Re-deciding per chunk made the convention flip mid-episode, so the
        decision is cached the first time real data is available.  Before any
        chunk arrives, the profile's declared convention is used.
        """
        if self.gripper_mode != "auto":
            return self.gripper_mode
        if self._resolved_gripper_mode is None:
            if self.action_chunk is not None and self.action_chunk[:, 6].min() < 0:
                # Official Franka convention: signed, -1 close / +1 open.
                self._resolved_gripper_mode = "official"
            else:
                self._resolved_gripper_mode = str(self.profile.get("gripper", "official"))
        return self._resolved_gripper_mode

    def _gripper_command(self, value: float) -> bool | None:
        """Map the policy's gripper channel onto open / closed / hold.

        The official Franka reference gates at +/-0.9 and, in between, issues no
        command at all -- the gripper keeps its previous state.  Returning
        ``None`` here means "hold"; the caller preserves the last state.

        ``signed`` / ``unit`` are the older always-commit rules kept for
        comparison against the earlier runs.
        """
        mode = self._resolve_gripper_mode()
        if mode == "official":
            if value >= 0.9:
                return True
            if value <= -0.9:
                return False
            return None
        if mode == "signed":
            return value > 0.0
        return value > 0.5

    # -- episode ------------------------------------------------------------ #

    def initialize(self) -> bool:
        """Build the scene, then connect and validate the policy server."""
        if self.args.no_render:
            problems = []
            if self.args.render_only:
                problems.append("--render-only exists to produce images")
            elif not self.args.dry_run:
                problems.append("a policy profile needs camera images")
            if self.args.export_raw:
                problems.append("--export-raw needs camera images")
            if problems:
                self._log(f"--no-render is incompatible: {'; '.join(problems)}")
                return False
        try:
            self.scene_engine.build()
            self.initial_position = self.scene_engine.initial_position
            self._log(
                f"scene ready, EE at {np.round(self.initial_position, 4).tolist()}, "
                f"cube at {np.round(self.scene_engine.cube_position(), 4).tolist()}"
            )
            self._write_event(
                "initialized",
                ee_initial=self.initial_position.tolist(),
                cube_initial=self.scene_engine.cube_position().tolist(),
                cube_color=list(self.args.cube_color),
                instruction=self.args.instruction,
            )
            # Record framing quality while the arm is still at the ready pose --
            # measured after the episode the gripper usually occludes the cube,
            # which says nothing about the framing.
            if not self.args.no_render:
                preview = self.scene_engine.render_all()
                self.initial_visibility = {
                    name: self.scene_engine.cube_visible_pixels(frame)
                    for name, frame in preview.items()
                }
                self._log(f"cube visible px at start: {self.initial_visibility}")

            if self.args.dry_run or self.args.render_only:
                reason = "render-only" if self.args.render_only else "dry run"
                self._log(f"{reason}: policy server is not contacted")
                self._write_event("dry_run", chunk=int(self.profile["chunk"]))
                return True

            self.client = StarVLAPolicyClient(self.args.policy_host, self.args.policy_port)
            metadata = self.client.metadata
            reported_chunk = metadata.get("action_chunk_size")
            if reported_chunk != self.profile["chunk"]:
                raise RuntimeError(
                    f"Profile '{self.args.profile}' expects chunk {self.profile['chunk']}, "
                    f"server reports {reported_chunk}: {metadata}"
                )
            available = metadata.get("available_unnorm_keys", [])
            if self.profile["unnorm_key"] not in available:
                raise RuntimeError(
                    f"unnorm_key '{self.profile['unnorm_key']}' unavailable; server has {available}"
                )
            self._log(f"connected: chunk={reported_chunk} unnorm_key={self.profile['unnorm_key']}")
            self._write_event("connected", metadata={
                k: v for k, v in metadata.items() if k != "available_unnorm_keys"
            }, available_unnorm_keys=list(available))
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            self._log("initialization FAILED")
            return False

    def run(self) -> int:
        """Execute the requested number of episodes; return a process exit code."""
        if not self.initialize():
            self.shutdown()
            return 1

        if self.args.render_only:
            self.render_preview()
            self.shutdown()
            return 0

        raw_chunks = 0
        try:
            for episode in range(self.args.episodes):
                # Reset every episode including the first, otherwise episode 0
                # silently keeps the un-randomised default cube position.
                self.reset_episode(episode)
                raw_chunks += self.run_episode(episode)
        except KeyboardInterrupt:
            self._log("interrupted")
        except Exception:
            import traceback

            traceback.print_exc()
            self._write_event("error", detail="see stderr")
            self.shutdown()
            return 1

        self.shutdown()
        self.summarize(raw_chunks)
        return 0

    def sample_cube_position(self, episode: int) -> np.ndarray:
        """Sample a reachable cube position for this episode.

        The box is deliberately modest.  The arm has to reach it from the fixed
        ready pose and the scripted state machine has to close its fingers on
        it; a wider box would mostly produce unreachable episodes that look
        like policy failures rather than useful coverage.
        """
        rng = np.random.default_rng(int(self.args.seed) + 100000 + episode)
        base = np.asarray(self.args.cube_pos, dtype=np.float64)
        span = np.asarray(self.args.cube_range, dtype=np.float64)
        position = base + rng.uniform(-span, span)
        position[2] = base[2]  # stay on the table surface
        return position

    def reset_episode(self, episode: int) -> None:
        """``env.reset()``: place the cube and return the arm to the ready pose."""
        self._log(f"--- episode {episode} reset ---")
        # Domain randomisation: a scripted state machine that always sees the
        # same cube teaches a policy nothing, so the cube is resampled per
        # episode inside a reachable box.  The scene reset must happen first so
        # the ready pose is settled with the cube already in place.
        if self.args.randomize_cube:
            cube_pos = self.sample_cube_position(episode)
            self.scene_engine.set_cube_position(cube_pos)
            self._log(f"randomised cube -> {np.round(cube_pos, 4).tolist()}")
        self.scene_engine.reset()
        self.action_chunk = None
        self.action_index = 0
        self.step_count = 0
        self.gripper_open = True
        self.scripted_phase = None
        self.scripted_phase_chunks = 0
        self.scripted_target = None
        self._write_event(
            "reset",
            episode=episode,
            ee_initial=self.scene_engine.ee_position().tolist(),
            cube_initial=self.scene_engine.cube_position().tolist(),
        )

    def run_episode(self, episode: int) -> int:
        """Run one episode's worth of steps; return how many chunks were pulled."""
        raw_chunks = 0
        recording = self.raw_export_dir is not None
        if recording:
            self.record_frames = {
                "base_view": [], "ego_view": [], "state": [], "action": [], "cube_position": []
            }
        while self.step_count < self.args.steps:
            # RL only needs low-dimensional state, and rendering two cameras per
            # step is the dominant cost, so it is skippable.
            frames = {} if self.args.no_render else self.scene_engine.render_all()
            state = self._state()
            if self.action_chunk is None or self.action_index >= len(self.action_chunk):
                self._request_chunk(frames, state)
                raw_chunks += 1

            assert self.action_chunk is not None
            action = self.action_chunk[self.action_index]
            if recording:
                # (observation, action) pair for this step: the images and state
                # were captured before the action was applied.
                self.record_frames["base_view"].append(frames["primary"])
                if "wrist" in frames:
                    self.record_frames["ego_view"].append(frames["wrist"])
                self.record_frames["state"].append(self.observation_state())
                self.record_frames["action"].append(np.asarray(action, dtype=np.float32))
                self.record_frames["cube_position"].append(self.scene_engine.cube_position())
            position = self.scene_engine.ee_position()
            quaternion = self.scene_engine.ee_quaternion()
            target_position, target_quaternion, gripper_cmd = self._bounded_target(
                action, position, quaternion
            )
            # Official behaviour: a command between -0.9 and +0.9 holds the
            # gripper where it is instead of forcing a state.
            if gripper_cmd is None:
                gripper_open = self.gripper_open
            else:
                gripper_open = gripper_cmd
                self.gripper_open = gripper_open
            ik_error = self.scene_engine.command(
                target_position, target_quaternion, gripper_open
            )
            self.scene_engine.scene.step()

            if self.args.control_period > 0:
                time.sleep(self.args.control_period)

            self.action_index += 1
            self.step_count += 1
            self._write_event(
                "action",
                episode=episode,
                raw_action=action.tolist(),
                state=state.reshape(-1).tolist(),
                target_position=target_position.tolist(),
                gripper_cmd=None if gripper_cmd is None else bool(gripper_cmd),
                gripper_open=gripper_open,
                ee_position=self.scene_engine.ee_position().tolist(),
                cube_position=self.scene_engine.cube_position().tolist(),
                ik_error=ik_error,
            )
            if self.step_count % 16 == 0:
                cube = self.scene_engine.cube_position()
                self._log(
                    f"step {self.step_count}/{self.args.steps} "
                    f"ee={np.round(self.scene_engine.ee_position(), 3).tolist()} "
                    f"cube={np.round(cube, 3).tolist()} ik_err={ik_error:.4f}"
                    )
        if recording:
            self._save_episode_raw(episode)
        return raw_chunks

    def _save_episode_raw(self, episode: int) -> None:
        """Write one episode of (images, state, action) to an .npz for export.

        Keeping the simulator out of the LeRobot-writing business means the
        export format can be iterated on without re-running the physics.
        """
        assert self.raw_export_dir is not None
        assert self.record_frames is not None
        frames = self.record_frames
        cubes = np.stack(frames["cube_position"])
        # Only successful grasps are useful for behaviour cloning: a policy
        # trained on failures learns to fail.
        lift = float(cubes[-1][2] - cubes[0][2])
        if self.args.only_success and lift < self.args.success_lift:
            self._log(f"episode {episode} REJECTED (lift {lift:+.4f} m < {self.args.success_lift} m)")
            self.record_frames = None
            return
        path = self.raw_export_dir / f"episode_{episode:06d}.npz"
        base = np.stack(frames["base_view"])
        ego = np.stack(frames["ego_view"]) if frames["ego_view"] else np.zeros((0,), np.uint8)
        np.savez_compressed(
            path,
            base_view=base,
            ego_view=ego,
            state=np.stack(frames["state"]),
            action=np.stack(frames["action"]),
            cube_position=cubes,
            task=np.array(self.args.instruction),
            fps=np.array(self.args.video_fps),
        )
        self._log(
            f"raw episode -> {path.name}: {len(frames['action'])} steps, "
            f"base_view={base.shape}, lift={lift:+.4f} m"
        )
        self.record_frames = None

    def render_preview(self) -> None:
        """Save one frame per camera plus the cube visibility metric.

        This is how the framing is tuned: build, settle the ready pose, render,
        and measure.  Skipping the policy makes each iteration about a minute.
        """
        from PIL import Image

        frames = self.scene_engine.render_all()
        for name, frame in frames.items():
            path = self.output_dir / f"preview-{name}.png"
            Image.fromarray(frame).save(path)
            pixels = self.scene_engine.cube_visible_pixels(frame)
            self._log(
                f"camera '{name}': {frame.shape} -> {path.name}, "
                f"cube covers {pixels} px ({100.0 * pixels / frame[..., 0].size:.2f}% of frame)"
            )

    def shutdown(self) -> None:
        """Stop recording, close transport, release simulator resources."""
        if self.args.record_video:
            for name, camera in self.scene_engine.cams.items():
                suffix = "" if name == "primary" else f"-{name}"
                target = self.video_file.with_name(
                    f"{self.video_file.stem}{suffix}{self.video_file.suffix}"
                )
                try:
                    camera.stop_recording(
                        save_to_filename=str(target), fps=self.args.video_fps
                    )
                    self._log(f"video ({name}) -> {target}")
                except Exception:
                    pass
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        self._write_event("shutdown")

    def summarize(self, raw_chunks: int) -> None:
        """Read back the JSONL log and print the plumbing evidence."""
        actions, cubes, ees, iks = [], [], [], []
        with self.log_file.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("event") != "action":
                    continue
                actions.append(record["raw_action"])
                cubes.append(record["cube_position"])
                ees.append(record["ee_position"])
                iks.append(record["ik_error"])
        if not actions:
            self._log("no action records were written")
            return

        actions = np.asarray(actions)
        cubes = np.asarray(cubes)
        ees = np.asarray(ees)
        assert self.scene_engine.initial_position is not None
        start = self.scene_engine.initial_position
        cube_start = cubes[0]

        path = float(np.linalg.norm(np.diff(ees, axis=0), axis=1).sum()) if len(ees) > 1 else 0.0
        cube_travel = float(np.linalg.norm(cubes - cube_start, axis=1).max())
        visibility = {}
        if not self.args.no_render:
            final_frames = self.scene_engine.render_all()
            visibility = {
                name: self.scene_engine.cube_visible_pixels(frame)
                for name, frame in final_frames.items()
            }
        lines = [
            "",
            "=" * 72,
            "FRANKA-CUBE / StarVLA closed-loop evidence",
            "=" * 72,
            f"profile            : {self.args.profile} ({self.profile['description']})",
            f"unnorm_key         : {self.profile['unnorm_key']}",
            f"cameras            : {self.profile['cameras']} (send_state={self.profile['send_state']})",
            f"instruction        : {self.args.instruction}",
            f"action space       : {self.args.action_space} "
            f"(server-side unnormalization = {self.args.action_space == 'raw'})",
            f"response key       : {self.last_response_key}",
            f"gripper mode       : {self.gripper_mode} -> effective "
            f"{self._resolve_gripper_mode() if self.action_chunk is not None else 'n/a'}",
            f"episodes           : {self.args.episodes} x {self.args.steps} steps",
            f"steps executed     : {len(actions)} in {raw_chunks} policy chunks "
            f"(chunk={self.profile['chunk']})",
            f"action shape       : (1,{self.profile['chunk']},7) x{len(actions)}, "
            f"finite={bool(np.isfinite(actions).all())}",
            f"action range       : [{actions.min():.5f}, {actions.max():.5f}]",
            f"gripper channel    : min={actions[:, 6].min():.3f} max={actions[:, 6].max():.3f} "
            f"(mode={self.gripper_mode})",
            f"EE start           : {np.round(start, 4).tolist()}",
            f"EE end             : {np.round(ees[-1], 4).tolist()}",
            f"EE total travel    : {np.linalg.norm(ees[-1] - start):.4f} m net, {path:.4f} m path",
            f"EE max displacement: {float(np.linalg.norm(ees - start, axis=1).max()):.4f} m",
            f"IK residual        : mean={float(np.mean(iks)):.5f} m max={float(np.max(iks)):.5f} m",
            f"cube start         : {np.round(cube_start, 4).tolist()}",
            f"cube end           : {np.round(cubes[-1], 4).tolist()}",
            f"cube max travel    : {cube_travel:.5f} m",
            f"cube moved?        : {'YES' if cube_travel > 0.002 else 'no (not touched)'}",
            f"cube lift (z)      : {float(cubes[-1][2] - cube_start[2]):+.4f} m",
            f"grasp succeeded?   : "
            f"{'YES - cube is off the ground' if float(cubes[-1][2] - cube_start[2]) > 0.03 else 'no'}",
            f"scripted phase     : {self.scripted_phase or 'n/a (policy run)'}",
            f"cube visible px    : start "
            + ", ".join(f"{n}={p}" for n, p in (self.initial_visibility or {}).items())
            + " | end "
            + ", ".join(f"{n}={p}" for n, p in visibility.items())
            + "  (of 50176)",
            f"log                : {self.log_file}",
            f"video              : {self.video_file}",
            "=" * 72,
        ]
        summary = "\n".join(lines)
        print(summary, flush=True)
        (self.output_dir / "RESULT-SUMMARY.txt").write_text(summary, encoding="utf-8")


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(POLICY_PROFILES), default="bridge")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=6694)
    parser.add_argument("--instruction", default="grasp the red cube")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", default="recordings/franka_cube")
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--control-period", type=float, default=0.0)
    parser.add_argument("--max-xyz-step", type=float, default=0.01)
    parser.add_argument("--max-rpy-step", type=float, default=0.05)
    parser.add_argument("--max-ee-displacement", type=float, default=0.45)
    parser.add_argument("--gripper-mode", choices=("auto", "official", "signed", "unit"), default="auto")
    parser.add_argument(
        "--action-space",
        choices=("raw", "normalized"),
        default="raw",
        help=(
            "raw = the server already unnormalized (this deployment); "
            "normalized = the server returned [-1,1] and we invert it here "
            "(the Franka README flow). Applying both would corrupt the scale."
        ),
    )
    parser.add_argument(
        "--action-stats",
        default="",
        help="dataset_statistics.json, required with --action-space normalized",
    )
    parser.add_argument("--unnorm-key-override", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--export-raw",
        default="",
        help="directory for per-episode (images, state, action) .npz dumps for LeRobot export",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help=(
            "skip camera rendering; only low-dimensional state is produced. "
            "This is the RL training mode -- rendering two 224x224 cameras per "
            "step dominates the cost. Incompatible with policy profiles and "
            "with --export-raw."
        ),
    )
    parser.add_argument("--record-video", action="store_true", default=True)
    parser.add_argument("--no-record-video", dest="record_video", action="store_false")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--cube-pos", type=float, nargs=3, default=(0.50, 0.0, 0.025))
    parser.add_argument(
        "--randomize-cube",
        action="store_true",
        help="resample the cube position each episode (domain randomisation)",
    )
    parser.add_argument(
        "--cube-range",
        type=float,
        nargs=3,
        default=(0.07, 0.06, 0.0),
        help="half-extents of the cube sampling box around --cube-pos",
    )
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="discard episodes that did not actually lift the cube",
    )
    parser.add_argument(
        "--success-lift",
        type=float,
        default=0.03,
        help="cube rise in metres that counts as a successful grasp",
    )
    parser.add_argument("--cube-size", type=float, default=0.05)
    parser.add_argument("--cube-color", type=float, nargs=3, default=(0.9, 0.1, 0.1))
    # Start the hand just above the cube rather than a third of a metre up: a
    # narrow framing cannot hold both a 5 cm cube and a distant gripper, and
    # seeing both is what the policy actually needs.
    parser.add_argument("--ready-pose", type=float, nargs=3, default=(0.50, 0.0, 0.22))
    parser.add_argument("--ready-steps", type=int, default=150)
    # Framing matters: at the original wide setup (pos 1.25,-0.55,0.85 / fov 45)
    # the 5 cm cube covered only ~11 px across in a 224 px frame.
    parser.add_argument("--cam-pos", type=float, nargs=3, default=(0.26, -0.36, 0.22))
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=(0.50, 0.0, 0.13))
    parser.add_argument("--cam-fov", type=float, default=40.0)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="build the scene, render one frame per camera, then exit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip the policy server and drive the scene with scripted actions",
    )
    parser.add_argument(
        "--pre-grasp-height",
        type=float,
        default=0.09,
        help="dry-run: fingertip height above the cube before descending",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.15,
        help="dry-run: how far to lift after closing the gripper",
    )
    parser.add_argument(
        "--reach-tolerance",
        type=float,
        default=0.012,
        help="dry-run: fingertip distance from target that counts as reached",
    )
    parser.add_argument(
        "--phase-timeout",
        type=int,
        default=30,
        help=(
            "dry-run: max chunks a phase may take before advancing anyway. "
            "The actuator tracks only ~0.0008 m/step here, so descending 9 cm "
            "takes ~110 steps (~14 chunks of 8); a tight timeout truncates the "
            "lift before it finishes."
        ),
    )
    parser.add_argument(
        "--close-chunks",
        type=int,
        default=3,
        help="dry-run: chunks to hold position while the gripper closes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return FrankaCubeStarvlaRunner(args).run()


if __name__ == "__main__":
    sys.exit(main())
