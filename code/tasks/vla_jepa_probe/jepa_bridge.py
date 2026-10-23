#!/usr/bin/env python3
"""Closed-loop bridge: VLA-JEPA policy drives the Genesis satellite manipulator.

Chain: Genesis observation -> NPZ request -> 127.0.0.1:9012 -> action chunk
       -> bounded Cartesian delta -> patched batched IK -> gripper -> sim step.

Action semantics
----------------
VLA-JEPA-LIBERO emits a (7,7) chunk: 7 steps x [dx,dy,dz,droll,dpitch,dyaw,grip].
That is the LIBERO/robosuite OSC_POSE convention, where the env multiplies the
raw action by a per-axis scale. Those scales are exposed here as --pos-scale and
--rot-scale (LIBERO's usual values are 0.05 m and 0.5 rad), and --calibrate
measures what actually materialises so the numbers do not have to be guessed.

Gripper: LIBERO uses -1 = close, +1 = open; Genesis control_gripper takes
1 = open, 0 = close. So g = (a + 1) / 2.

The single formal metric is min_pinch_center_error_m < 0.025.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--jepa-host", default="127.0.0.1")
    p.add_argument("--jepa-port", type=int, default=9012)
    p.add_argument("--instruction", default="put the moka pot on the stove")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--object-size", type=float, default=0.18)
    p.add_argument("--object-thickness", type=float, default=0.04)
    p.add_argument("--object-forward-x", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=128, help="Control steps (one chunk row each).")
    p.add_argument("--action-settle-steps", type=int, default=5,
                   help="Physics frames held per action row; the IK moves only ~0.03 cm/frame.")
    p.add_argument("--pos-scale", type=float, default=0.05, help="LIBERO position scale (m).")
    p.add_argument("--rot-scale", type=float, default=0.5, help="LIBERO rotation scale (rad).")
    p.add_argument("--max-xyz-step", type=float, default=0.01)
    p.add_argument("--max-rpy-step", type=float, default=0.05)
    p.add_argument("--success-threshold", type=float, default=0.025)
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--calibrate", action="store_true",
                   help="Measure the real response to a commanded 1 cm delta, then exit.")
    p.add_argument("--calibrate-frames", type=int, default=40)
    p.add_argument("--calibrate-step", type=float, default=0.01)
    p.add_argument("--no-batch-renderer", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- small helpers

def vec(t) -> np.ndarray:
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    return np.asarray(t, dtype=np.float64).reshape(-1)


def to_uint8_rgb(rendered) -> np.ndarray:
    import torch
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()
    img = np.asarray(rendered)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"unexpected image shape {img.shape}")
    if img.dtype != np.uint8:
        if img.max(initial=0.0) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def sensor_image(sensor, label: str, log):
    """capture() must come first: latest_rgb alone is a stale, all-zero buffer."""
    if sensor is None:
        return None
    try:
        try:
            sensor.capture()
        except TypeError:
            sensor.capture(None)
        out = getattr(sensor, "latest_rgb", None)
        if out is None:
            inner = getattr(sensor, "_batch_cam", None)
            if inner is None:
                return None
            out = inner.render(rgb=True, force_render=True)
        return to_uint8_rgb(out)
    except Exception as exc:  # noqa: BLE001
        log(f"    {label} capture FAILED {type(exc).__name__}: {str(exc)[:70]}")
        return None


def rpy_to_quaternion(rpy) -> np.ndarray:
    r, p, y = [float(v) / 2.0 for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def multiply_quaternions(left, right) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    q = np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else q


class JepaClient:
    """Minimal client for the loopback NPZ inference server."""

    def __init__(self, host: str, port: int, protocol_version: int = 1, log=print):
        self.host, self.port, self.pv, self.log = host, port, protocol_version, log
        self.metadata = self._get_json("/health")

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def infer(self, fields: dict, timeout: float = 180.0) -> tuple[np.ndarray, float]:
        buf = io.BytesIO()
        np.savez(buf, **fields)  # uncompressed: server rejects zip-deflate
        body = buf.getvalue()
        req = urllib.request.Request(
            f"http://{self.host}:{self.port}/infer", data=body,
            headers={"Content-Type": "application/x-npz", "Content-Length": str(len(body))},
            method="POST")
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t
        return np.asarray(out["action_chunk"], dtype=np.float64), dt

    def request_fields(self, scene_img, wrist_img, eef_pos, eef_quat_wxyz, grip2, instruction):
        q = np.asarray(eef_quat_wxyz, dtype=np.float64).reshape(-1)[:4]
        return {
            "protocol_version": np.asarray(self.pv, dtype=np.int64),
            "request_id": np.asarray(uuid.uuid4().hex),
            "instruction": np.asarray(instruction),
            "agentview_image": np.ascontiguousarray(scene_img, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ascontiguousarray(wrist_img, dtype=np.uint8),
            "robot0_eef_pos": np.asarray(eef_pos, dtype=np.float32).reshape(3),
            # robosuite convention is (x, y, z, w); Genesis gives (w, x, y, z)
            "robot0_eef_quat": np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(grip2, dtype=np.float32).reshape(2),
        }


# ------------------------------------------------------------------ scene boot

def boot(args, log):
    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        raise SystemExit(f"{tree_src} is not a directory")
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA

    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    res = (args.image_size, args.image_size)
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"res": res, "GUI": False})
    if isinstance(FRANKA_S_Q_CAMERA.get("camera"), dict):
        FRANKA_S_Q_CAMERA["camera"]["res"] = res
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    base_cfg = FRANKA_S_Q_CAMERA.get("base_camera")
    if isinstance(base_cfg, dict):
        base_cfg["enable_recording"] = False
        if isinstance(base_cfg.get("camera"), dict):
            base_cfg["camera"]["res"] = res

    import genesis as gs
    import torch
    from controllers.backend import EmptyBackend
    from envs.genesis_env import GenesisSim
    from robots.satellite_manipulator import SatelliteManipulator

    if not args.no_batch_renderer:
        SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)

    sim = GenesisSim()
    manip = SatelliteManipulator(name="franka_merge", sensors=[], backends=[EmptyBackend()])
    target = sim.scene.add_entity(
        gs.morphs.Box(size=(args.object_size, args.object_thickness, args.object_thickness),
                      pos=(0.0, 0.0, 0.0), collision=True),
        material=gs.materials.Rigid(),
    )
    sim.start()
    manip.initialize()
    n_envs = int(sim.num_envs)

    st = {"prev": None, "logged": False}
    mjc = float(manip.params["ik_params"]["max_joint_change"])

    def control_joints(position, quaternion):
        pos = np.repeat(np.asarray(vec(position)).reshape(1, 3), n_envs, axis=0)
        quat = np.repeat(np.asarray(vec(quaternion)).reshape(1, 4), n_envs, axis=0)
        pt = torch.as_tensor(pos, dtype=manip.datatype, device=manip.device)
        qt = torch.as_tensor(quat, dtype=manip.datatype, device=manip.device)
        qpos = manip.robot.inverse_kinematics(link=manip.end_effector, pos=pt, quat=qt)
        if not st["logged"]:
            log(f"IK ok: qpos{tuple(qpos.shape)} n_envs={n_envs}")
            st["logged"] = True
        if not torch.isfinite(qpos).all():
            return False
        q = qpos[0] if qpos.dim() > 1 else qpos
        idx = manip.motors_dof if q.numel() == manip.robot.n_dofs else manip.motors_qs
        mq = q[idx]
        prev = st["prev"]
        if prev is not None and prev.numel() == mq.numel():
            d = mq - prev
            nm = torch.linalg.vector_norm(d)
            if float(nm) > mjc:
                mq = prev + d * (mjc / float(nm))
        st["prev"] = mq.detach().clone()
        return manip.control_joint_pos(mq)

    manip.control_joints = control_joints

    def step_once():
        manip.step()
        sim.step()

    def pinch_center() -> np.ndarray:
        pts = [vec(l.get_pos()) for l in manip.robot.links if "Finger" in l.name]
        return np.mean(np.asarray(pts, dtype=np.float64), axis=0) if len(pts) >= 2 else np.zeros(3)

    def gripper_qpos() -> np.ndarray:
        q = vec(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
        return np.resize(q, 2)[:2] if q.size < 2 else q[:2]

    manip.control_gripper(1.0)
    for _ in range(args.settle_steps):
        step_once()
    manip.update_state()
    return sim, manip, target, step_once, pinch_center, gripper_qpos


# ---------------------------------------------------------------- calibrate

def calibrate(args, log, manip, step_once, pinch_center):
    """Command a fixed Cartesian delta per axis and measure what actually happens."""
    axes = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}
    q0 = vec(manip.ee_global_quaternion)
    rows = []
    for name, ax in axes.items():
        for sign in (+1.0, -1.0):
            manip.update_state()
            p_before = pinch_center()
            for _ in range(args.calibrate_frames):
                goal = vec(manip.ee_global_position) + ax * sign * args.calibrate_step
                manip.control_joints(goal, q0)
                step_once()
            manip.update_state()
            p_after = pinch_center()
            d = p_after - p_before
            rows.append({"axis": name, "sign": sign, "commanded_m": args.calibrate_step,
                         "displacement_m": d.tolist(), "norm_m": float(np.linalg.norm(d))})
            log(f"  cmd {sign:+.0f}{name} {args.calibrate_step*1000:.0f}mm -> pinch moved "
                f"{np.round(d*1000, 2).tolist()} mm")
    mat = np.zeros((3, 3))
    for r in rows:
        idx = list(axes).index(r["axis"])
        mat[:, idx] += np.asarray(r["displacement_m"]) * r["sign"] / args.calibrate_step / 2.0
    gains = np.diag(mat).tolist()
    out = {"rows": rows, "matrix_real_over_commanded": mat.tolist(),
           "diagonal_gains": gains,
           "condition_number": float(np.linalg.cond(mat)) if np.all(np.isfinite(mat)) else None}
    log(f"  diagonal gains (real m per commanded m): {[round(g,4) for g in gains]}")
    log(f"  condition number: {out['condition_number']}")
    return out


# ---------------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_fh = (args.output_dir / "bridge.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[jepa-bridge] {msg}", flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    sim, manip, target, step_once, pinch_center, gripper_qpos = boot(args, log)

    scene_cam = getattr(manip, "rl_base_camera", None)
    wrist_cam = getattr(manip, "rl_wrist_camera", None)
    log(f"cameras: agentview={type(scene_cam).__name__} wrist={type(wrist_cam).__name__}")

    if args.calibrate:
        result = calibrate(args, log, manip, step_once, pinch_center)
        (args.output_dir / "calibration.json").write_text(json.dumps(result, indent=2),
                                                          encoding="utf-8")
        log("wrote calibration.json")
        log_fh.close()
        return 0

    client = JepaClient(args.jepa_host, args.jepa_port, log=log)
    log(f"JEPA metadata: {json.dumps(client.metadata)}")

    # place the target clear of the jaws
    pinch_start = pinch_center()
    # The box is object_size long along x. Placing it only a few cm ahead makes
    # the gripper spawn INSIDE it, and in zero gravity the overlap resolution
    # flings it away: we measured 10.5 cm of drift for a 6 cm request, which
    # also destroys the "target" reference the metric is measured against.
    clearance = args.object_size / 2.0 + 0.04
    forward = max(args.object_forward_x, clearance)
    if forward > args.object_forward_x:
        log(f"object_forward_x raised {args.object_forward_x:.3f} -> {forward:.3f} m "
            f"so the {args.object_size:.2f} m box clears the jaws")
    target.set_pos(pinch_start + np.array([forward, 0.0, 0.0]))
    for _ in range(20):
        step_once()
    target_pos = vec(target.get_pos())
    log(f"pinch_start={np.round(pinch_start,4).tolist()} target={np.round(target_pos,4).tolist()} "
        f"gap={float(np.linalg.norm(target_pos - pinch_start)):.4f} m")

    chunk = None
    chunk_idx = 0
    errors = []
    rows = []

    for step in range(args.steps):
        manip.update_state()
        ee_pos = vec(manip.ee_global_position)
        ee_quat = vec(manip.ee_global_quaternion)
        pinch = pinch_center()
        err = float(np.linalg.norm(target_pos - pinch))

        if chunk is None or chunk_idx >= len(chunk):
            img_scene = sensor_image(scene_cam, "agentview", log)
            img_wrist = sensor_image(wrist_cam, "wrist", log)
            if img_scene is None or img_wrist is None:
                log("no images; aborting")
                break
            fields = client.request_fields(img_scene, img_wrist, ee_pos, ee_quat,
                                          gripper_qpos(), args.instruction)
            chunk, dt = client.infer(fields)
            chunk_idx = 0
            log(f"step {step}: chunk {chunk.shape} in {dt:.2f}s")

        a = chunk[chunk_idx]
        raw = np.asarray(a, dtype=np.float64).reshape(-1)[:7]

        d_pos = np.clip(raw[:3] * args.pos_scale, -args.max_xyz_step, args.max_xyz_step)
        d_rpy = np.clip(raw[3:6] * args.rot_scale, -args.max_rpy_step, args.max_rpy_step)
        grasp_cmd = float((raw[6] + 1.0) / 2.0)  # LIBERO -1..+1 -> Genesis 0..1

        goal_pos = ee_pos + d_pos
        goal_quat = multiply_quaternions(ee_quat, rpy_to_quaternion(d_rpy))

        for _ in range(max(1, args.action_settle_steps)):
            manip.control_joints(goal_pos, goal_quat)
            manip.control_gripper(grasp_cmd)
            step_once()

        manip.update_state()
        pinch2 = pinch_center()
        err2 = float(np.linalg.norm(target_pos - pinch2))
        errors.append(err2)
        to_t = target_pos - pinch2
        u = to_t / (np.linalg.norm(to_t) + 1e-12)
        cos_cmd = float(np.dot(u, d_pos / (np.linalg.norm(d_pos) + 1e-12))) if np.linalg.norm(d_pos) > 1e-9 else None
        rows.append({
            "step": step, "chunk_idx": chunk_idx,
            "raw_action": raw.tolist(),
            "commanded_delta_m": d_pos.tolist(),
            "commanded_drpy_rad": d_rpy.tolist(),
            "gripper_command": grasp_cmd,
            "ee_pos": ee_pos.tolist(), "pinch_center": pinch2.tolist(),
            "target": target_pos.tolist(),
            "pinch_center_error_m": err2,
            "cos_cmd_vs_target": cos_cmd,
        })
        if step % 16 == 0 or step == args.steps - 1:
            log(f"step {step:3d}: pinch_err={err2:.4f} m grip={grasp_cmd:.2f} "
                f"|d|={np.linalg.norm(d_pos)*1000:.2f}mm cos={cos_cmd if cos_cmd is None else round(cos_cmd,3)}")
        chunk_idx += 1

    min_err = min(errors) if errors else None
    summary = {
        "steps_run": len(errors),
        "success_threshold_m": args.success_threshold,
        "initial_error_m": errors[0] if errors else None,
        "min_pinch_center_error_m": min_err,
        "final_error_m": errors[-1] if errors else None,
        "argmin_step": int(np.argmin(errors)) if errors else None,
        "aligned_for_grasp": bool(min_err is not None and min_err < args.success_threshold),
        "pos_scale": args.pos_scale, "rot_scale": args.rot_scale,
        "action_settle_steps": args.action_settle_steps,
        "jepa_metadata": client.metadata,
    }
    if rows:
        costs = [r["cos_cmd_vs_target"] for r in rows if r["cos_cmd_vs_target"] is not None]
        summary["mean_cos_cmd_vs_target"] = float(np.mean(costs)) if costs else None
        dpos = [np.linalg.norm(r["commanded_delta_m"]) for r in rows]
        summary["mean_commanded_delta_m"] = float(np.mean(dpos))
        summary["mean_gripper_command"] = float(np.mean([r["gripper_command"] for r in rows]))

    (args.output_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2),
                                                         encoding="utf-8")
    (args.output_dir / "bridge_steps.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    log("=== SUMMARY ===")
    log(json.dumps(summary, indent=2))
    log(f"VERDICT: {'ALIGNED' if summary['aligned_for_grasp'] else 'NOT ALIGNED'} "
        f"(min={min_err}, threshold={args.success_threshold})")
    log_fh.close()
    try:
        manip.stop(); sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0 if summary["aligned_for_grasp"] else 1


if __name__ == "__main__":
    sys.exit(main())
