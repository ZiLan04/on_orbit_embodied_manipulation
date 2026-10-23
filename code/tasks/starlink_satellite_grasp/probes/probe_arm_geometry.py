#!/usr/bin/env python3
"""Probe the satellite manipulator's geometry and reachable workspace.

Why: a scripted grasp needs real numbers -- where the tool centre point (TCP)
starts, how far the arm can actually reach, and what the gripper opening looks
like. Guessing those wastes runs.

This script does NOT grasp anything. It:
  1. brings up the scene + franka_merge manipulator, headless
  2. reports base pose, TCP pose, joint values, gripper qpos
  3. runs a reachability scan: for a set of candidate TCP targets around the
     start pose, drive the IK for N steps and record the residual error
  4. writes everything to <output-dir>/arm_geometry.json

Run on the policy host:
    env -u DISPLAY PYOPENGL_PLATFORM=egl MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      /data/anaconda3/envs/Spaceman_env/bin/python probe_arm_geometry.py \
        --tree /data/Yanhan/genesis/spaceman_core \
        --output-dir /data/Yanhan/genesis/runs/<ts>-arm-probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--steps-per-target", type=int, default=60,
                   help="IK steps per candidate target (max_joint_change=0.05 rad/step).")
    p.add_argument("--settle-steps", type=int, default=20)
    return p.parse_args()


def vec(t) -> list[float]:
    """Torch/ndarray -> flat python list of floats."""
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    a = np.asarray(t, dtype=np.float64).reshape(-1)
    return [float(x) for x in a]


def quat_to_rpy(q):
    """[qw,qx,qy,qz] -> roll,pitch,yaw (radians)."""
    import math
    qw, qx, qy, qz = [float(x) for x in np.asarray(q).reshape(-1)[:4]]
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return [roll, pitch, yaw]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[probe] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    # Headless: same override the eval script uses.
    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})
    print(f"[probe] headless: show_viewer={SCENE_SETTINGS.get('show_viewer')}")

    import genesis as gs

    from controllers.backend import EmptyBackend
    from envs.genesis_env import GenesisSim
    from robots.satellite_manipulator import SatelliteManipulator

    report: dict = {}

    print("[probe] constructing manipulator (adds robot entity to the scene)")
    sim = GenesisSim()
    manip = SatelliteManipulator(name="franka_merge", sensors=[], backends=[EmptyBackend()])

    print("[probe] starting sim (builds scene)")
    sim.start()
    print("[probe] initializing manipulator")
    manip.initialize()

    # --- settle ---
    manip.control_gripper(1.0)  # open
    for _ in range(args.settle_steps):
        manip.step()
        sim.step()
    manip.update_state()

    ee_pos = vec(manip.ee_global_position)
    ee_quat = vec(manip.ee_global_quaternion)
    finger_qpos = vec(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
    motor_qpos = vec(manip.robot.get_qpos(qs_idx_local=manip.motors_qs))

    # The arm sits on a free-floating base; its pose lives on the EE state's parent
    # link (this is what SatelliteManipulator.initialize() uses for the PID setpoint).
    base_pos, base_quat = [], []
    try:
        parent = manip.ee_state.link_parent_global_state
        base_pos = vec(parent.position)
        base_quat = vec(parent.orient)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] could not read base pose: {exc!r}")

    report["arm"] = {
        "base_position_m": base_pos,
        "base_quaternion_wxyz": base_quat,
        "ee_position_m": ee_pos,
        "ee_quaternion_wxyz": ee_quat,
        "ee_rpy_rad": quat_to_rpy(ee_quat),
        "finger_qpos_open": finger_qpos,
        "motor_qpos": motor_qpos,
        "num_motors": int(len(manip.motors_dof)),
        "num_fingers": int(len(manip.fingers_dof)),
        "finger_open_param": [float(x) for x in np.asarray(manip.params["finger_open"]).reshape(-1)],
        "finger_close_param": [float(x) for x in np.asarray(manip.params["finger_close"]).reshape(-1)],
        "gripper_force_scale": float(manip.gripper_force_scale),
        "gripper_state": float(manip.gripper_state),
    }
    print(f"[probe] TCP start position = {[round(x, 4) for x in ee_pos]}")
    print(f"[probe] TCP start quat(wxyz) = {[round(x, 4) for x in ee_quat]}")
    print(f"[probe] TCP start rpy (rad)  = {[round(x, 4) for x in quat_to_rpy(ee_quat)]}")
    print(f"[probe] finger qpos (open)   = {[round(x, 5) for x in finger_qpos]}")
    print(f"[probe] motor qpos           = {[round(x, 5) for x in motor_qpos]}")

    # --- gripper closed reading ---
    manip.control_gripper(0.0)
    for _ in range(args.settle_steps):
        manip.step()
        sim.step()
    closed_qpos = vec(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
    report["arm"]["finger_qpos_closed"] = closed_qpos
    report["arm"]["gripper_state_after_close"] = float(manip.gripper_state)
    print(f"[probe] finger qpos (closed) = {[round(x, 5) for x in closed_qpos]}")
    print(f"[probe] gripper_state after close = {manip.gripper_state}")
    manip.control_gripper(1.0)  # back to open
    for _ in range(args.settle_steps):
        manip.step()
        sim.step()

    # --- reachability scan around the start pose ---
    # Directions in world frame; we probe several distances along each.
    p0 = np.asarray(ee_pos, dtype=np.float64)
    q0 = np.asarray(ee_quat, dtype=np.float64)
    directions = {
        "x+": [1.0, 0.0, 0.0],
        "x-": [-1.0, 0.0, 0.0],
        "y+": [0.0, 1.0, 0.0],
        "y-": [0.0, -1.0, 0.0],
        "z+": [0.0, 0.0, 1.0],
        "z-": [0.0, 0.0, -1.0],
    }
    distances = [0.05, 0.10, 0.15, 0.20, 0.30]
    scan = []
    print(f"[probe] reachability scan: {len(directions)} directions x {len(distances)} distances "
          f"x {args.steps_per_target} steps")

    for dname, dvec in directions.items():
        for dist in distances:
            target = p0 + np.asarray(dvec, dtype=np.float64) * dist
            ok_any = True
            try:
                for _ in range(args.steps_per_target):
                    ok = manip.control_joints(target, q0)
                    if not ok:
                        ok_any = False
                        break
                    manip.step()
                    sim.step()
                manip.update_state()
                reached = np.asarray(vec(manip.ee_global_position), dtype=np.float64)
                err = float(np.linalg.norm(reached - target))
            except Exception as exc:  # noqa: BLE001
                err = float("nan")
                ok_any = False
                reached = np.zeros(3)
            scan.append({
                "direction": dname,
                "distance_m": dist,
                "target": [float(x) for x in target],
                "reached": [float(x) for x in reached],
                "residual_m": err,
                "control_ok": bool(ok_any),
            })
            print(f"[probe]   {dname} {dist:.2f}m -> residual {err:.5f} m  ok={ok_any}")

    report["reachability_scan"] = scan
    good = [s for s in scan if np.isfinite(s["residual_m"]) and s["residual_m"] < 0.01]
    report["reachable_count"] = len(good)
    report["reachable"] = [
        {"direction": s["direction"], "distance_m": s["distance_m"], "residual_m": s["residual_m"]}
        for s in good
    ]
    print(f"[probe] reachable targets (residual < 1cm): {len(good)} / {len(scan)}")

    report["tree"] = str(args.tree)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = args.output_dir / "arm_geometry.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out}")

    try:
        manip.stop()
        sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
