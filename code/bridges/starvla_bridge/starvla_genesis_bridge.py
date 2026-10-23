"""Drive the Genesis satellite manipulator from a remote StarVLA policy server.

The script is intentionally independent from the existing simulation tasks. It
uses the scene camera as a Bridge-compatible RGB observation, supplies the
8-value Bridge proprioceptive state, caches each 16-step policy response, and
applies bounded Cartesian deltas through the simulator's existing IK control.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import websockets.sync.client

try:
    import msgpack
except ModuleNotFoundError:
    # The isolated Genesis venv has pip but no standalone msgpack wheel.
    from pip._vendor import msgpack


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
from controllers.backend import EmptyBackend
from robots.satellite_manipulator import SatelliteManipulator
from tasks.task import Task, TaskStatus


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


def _as_vector(value: Any, size: int) -> np.ndarray:
    """Convert one simulator tensor or array into a single fixed-size vector."""
    array = np.asarray(torch.as_tensor(value).detach().cpu(), dtype=np.float64)
    if array.ndim > 1:
        array = array[0]
    array = array.reshape(-1)
    if array.size != size:
        raise ValueError(f"Expected {size} values, got {array.shape}")
    return array


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


class StarVLAGenesisBridge(Task):
    """Run the Starlink manipulator using bounded Bridge action chunks."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(task_name="starvla_genesis_bridge")
        self.args = args
        self.max_steps = args.steps
        self.starlink_manipulator = SatelliteManipulator(
            name="franka_merge", sensors=[], backends=[EmptyBackend()]
        )
        self.client: StarVLAPolicyClient | None = None
        self.action_chunk: np.ndarray | None = None
        self.action_index = 0
        self.initial_position: np.ndarray | None = None
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "bridge-run.jsonl"
        self.video_file = self.output_dir / "bridge-run.mp4"

    def initialize(self) -> bool:
        """Start Genesis, establish policy transport, and acquire the initial pose."""
        try:
            self.gsim.start()
            self.starlink_manipulator.initialize()
            for _ in range(self.args.warmup_steps):
                self.starlink_manipulator.step()
                self.gsim.step()
            self.starlink_manipulator.update_state()
            self.initial_position = _as_vector(
                self.starlink_manipulator.ee_global_position, 3
            )
            self.client = StarVLAPolicyClient(self.args.policy_host, self.args.policy_port)
            metadata = self.client.metadata
            if metadata.get("action_chunk_size") != 16:
                raise RuntimeError(f"Expected 16 policy actions, got metadata {metadata}")
            if "oxe_bridge" not in metadata.get("available_unnorm_keys", []):
                raise RuntimeError(f"Bridge unnormalization is unavailable: {metadata}")
            self.status = TaskStatus.RUNNING
            self.logger.info("Connected to StarVLA with metadata=%s", metadata)
            self._write_event("initialized", metadata=metadata)
            return True
        except Exception:
            self.logger.exception("Bridge initialization failed")
            self.status = TaskStatus.FAILED
            return False

    def _observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.starlink_manipulator.update_state()
        rgb, *_ = self.gsim.cam.render(rgb=True, force_render=True)
        position = _as_vector(self.starlink_manipulator.ee_global_position, 3)
        quaternion = _as_vector(self.starlink_manipulator.ee_global_quaternion, 4)
        rpy = _quaternion_to_rpy(quaternion)
        self.starlink_manipulator.get_gripper_value()
        gripper = float(np.asarray(self.starlink_manipulator.gripper_state).reshape(-1)[0])
        bridge_state = np.array(
            [*position, *rpy, 0.0, float(np.clip(gripper, 0.0, 1.0))],
            dtype=np.float32,
        ).reshape(1, 8)
        return _to_uint8_rgb(rgb), bridge_state, position, quaternion

    @staticmethod
    def _extract_actions(response: dict[str, Any]) -> np.ndarray:
        payload = response.get("data", response)
        actions = np.asarray(payload["actions"], dtype=np.float64)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1:] != (16, 7):
            raise RuntimeError(f"Expected [1, 16, 7] action tensor, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("StarVLA returned non-finite actions")
        return actions[0]

    def _request_chunk(self, image: np.ndarray, state: np.ndarray) -> None:
        if self.client is None:
            raise RuntimeError("Policy client is not initialized")
        example: dict[str, Any] = {"image": [image], "lang": self.args.instruction}
        if not self.args.omit_state:
            example["state"] = state
        response = self.client.predict_action(
            {
                "examples": [example],
                "unnorm_key": "oxe_bridge",
                "do_sample": False,
                "use_ddim": True,
                "num_ddim_steps": self.args.num_ddim_steps,
            }
        )
        self.action_chunk = self._extract_actions(response)
        self.action_index = 0
        self.logger.info(
            "Received action chunk range=[%.5f, %.5f]",
            float(self.action_chunk.min()),
            float(self.action_chunk.max()),
        )

    def _bounded_target(
        self, action: np.ndarray, position: np.ndarray, quaternion: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if self.initial_position is None:
            raise RuntimeError("Initial end-effector position is unavailable")
        translation = np.clip(
            action[:3], -self.args.max_xyz_step, self.args.max_xyz_step
        )
        rpy_delta = np.clip(action[3:6], -self.args.max_rpy_step, self.args.max_rpy_step)
        lower = self.initial_position - self.args.max_ee_displacement
        upper = self.initial_position + self.args.max_ee_displacement
        target_position = np.clip(position + translation, lower, upper)
        target_quaternion = _multiply_quaternions(quaternion, _rpy_to_quaternion(rpy_delta))
        return target_position, target_quaternion, float(np.clip(action[6], 0.0, 1.0))

    def _write_event(self, event: str, **fields: Any) -> None:
        record = {"event": event, "step": self.step_count, **fields}
        with self.log_file.open("a", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=True)
            stream.write("\n")

    def step(self, dt: float | None = None) -> bool:
        """Request a new chunk when needed and execute exactly one bounded action."""
        if self.status != TaskStatus.RUNNING:
            return False
        try:
            image, state, position, quaternion = self._observation()
            if self.action_chunk is None or self.action_index >= len(self.action_chunk):
                self._request_chunk(image, state)
            assert self.action_chunk is not None
            raw_action = self.action_chunk[self.action_index]
            target_position, target_quaternion, gripper = self._bounded_target(
                raw_action, position, quaternion
            )
            if not self.starlink_manipulator.control_joints(target_position, target_quaternion):
                raise RuntimeError("Genesis IK rejected the bounded policy target")
            if not self.starlink_manipulator.control_gripper(gripper >= 0.5):
                raise RuntimeError("Genesis gripper controller rejected the policy command")
            self.starlink_manipulator.step()
            self.gsim.step()
            self._write_event(
                "action",
                raw_action=raw_action.tolist(),
                state=state.reshape(-1).tolist(),
                target_position=target_position.tolist(),
                target_quaternion=target_quaternion.tolist(),
                gripper=gripper,
            )
            self.action_index += 1
            self.step_count += 1
            if self.step_count % 16 == 0:
                self.logger.info("Completed %d/%d bridge steps", self.step_count, self.max_steps)
            if self.step_count >= self.max_steps:
                self.status = TaskStatus.COMPLETED
                return False
            if self.args.control_period > 0:
                time.sleep(self.args.control_period)
            return True
        except Exception:
            self.logger.exception("Bridge step failed")
            self.status = TaskStatus.FAILED
            return False

    def reset(self) -> bool:
        """The proof-of-motion bridge has one fixed episode and does not reset."""
        return False

    def reward(self) -> torch.Tensor:
        """The bridge verifies transport and motion, not a task reward."""
        return torch.zeros((), dtype=self.datatype, device=self.device)

    def stop(self) -> bool:
        """Persist camera evidence, close transport, and release simulator resources."""
        try:
            if self.args.record_video and self.gsim.cam is not None:
                if getattr(self.gsim.cam, "_in_recording", False):
                    self.gsim.cam.stop_recording(
                        save_to_filename=str(self.video_file), fps=self.args.video_fps
                    )
                self.gsim.camera_enable_recording = False
            if self.client is not None:
                self.client.close()
            self.gsim.stop()
            self._write_event("stopped", status=self.status.value)
            return True
        except Exception:
            logging.getLogger(__name__).exception("Bridge cleanup failed")
            return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-host", default="10.110.10.12")
    parser.add_argument("--policy-port", type=int, default=6694)
    parser.add_argument("--instruction", default="move the robot arm safely")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-xyz-step", type=float, default=0.01)
    parser.add_argument("--max-rpy-step", type=float, default=0.05)
    parser.add_argument("--max-ee-displacement", type=float, default=0.45)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--omit-state", action="store_true")
    parser.add_argument("--no-video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SRC_DIR / "recordings" / "starvla_genesis_bridge",
    )
    return parser.parse_args()


def _configure_scene() -> None:
    """Enable a compact scene camera before Task creates GenesisSim."""
    SCENE_SETTINGS["show_viewer"] = True
    TPV_CAM_SETTINGS["enable_recording"] = True
    TPV_CAM_SETTINGS["episodes_per_video"] = 1_000_000
    TPV_CAM_SETTINGS["camera"].update({"res": (224, 224), "GUI": False})


def main() -> int:
    """Run one evidence-producing StarVLA-controlled simulator episode."""
    args = _parse_args()
    _configure_scene()
    task = StarVLAGenesisBridge(args)
    if not task.initialize():
        return 1
    try:
        while task.step():
            pass
    except KeyboardInterrupt:
        task.status = TaskStatus.CANCELLED
    finally:
        task.stop()
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
