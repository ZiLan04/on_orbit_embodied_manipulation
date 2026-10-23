#!/usr/bin/env python3
"""Scripted (non-learned) grasp oracle for the satellite manipulator.

Goal
----
Answer one question with no policy involved: **can this scene, arm, gripper and
IK actually grasp an object at all?**

If a hand-written trajectory cannot grasp it, no learned policy ever will, so
this is the sanity floor for everything else.

Why it is built this way
------------------------
* Gravity is (0,0,0) and there is no ground plane in this scene, so "lift it off
  the table" is not a usable success test. Instead: close the fingers on the
  target, then pull the tool away and check whether the target came with it.
* `Manipulator.control_joints` is broken for the single-env case (it indexes a
  batched IK result as if it were 1-D). The StarVLA bridge launchers already
  work around this; we reuse exactly their adapter.
* Object placement is *measured*, not guessed: we scan candidate offsets around
  the start pose, keep the ones the IK can actually reach, and drop the target
  at the best one.

Outputs (in --output-dir):
    grasp_report.json   per-candidate scan + the chosen grasp result
    grasp_trace.jsonl   one record per simulation step of the chosen attempt
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
    p.add_argument("--object-size", type=float, default=0.18,
                   help="Edge length of the cubic target, metres. Must be near the "
                        "gripper's closed aperture (~0.17 m measured) or the fingers "
                        "close past it and never touch.")
    p.add_argument("--settle-steps", type=int, default=60)
    p.add_argument("--scan-ik-steps", type=int, default=500,
                   help="The IK adapter bounds joint change to 0.05 rad/step, so the "
                        "tool moves only ~0.03 cm/step; large values are expected.")
    p.add_argument("--ik-steps", type=int, default=1200)
    p.add_argument("--close-steps", type=int, default=80)
    p.add_argument("--pull-steps", type=int, default=120)
    p.add_argument("--pre-grasp-distance", type=float, default=0.25)
    p.add_argument("--pull-distance", type=float, default=0.10)
    p.add_argument("--reach-tolerance", type=float, default=0.02)
    p.add_argument("--grasp-hold-ratio-min", type=float, default=0.6,
                   help="Target must follow at least this fraction of the tool's pull.")
    p.add_argument("--grasp-hold-ratio-max", type=float, default=1.4,
                   help="...and no more than this. A target that moves much further than "
                        "the tool was knocked away, not grasped.")
    p.add_argument("--relative-slip-tolerance", type=float, default=0.02,
                   help="Max change in tool->target offset during the pull, metres.")
    p.add_argument("--finger-block-margin", type=float, default=0.004,
                   help="Fingers must stop this far short of their free-closing position, "
                        "otherwise they closed through empty space.")
    p.add_argument("--no-batch-renderer", action="store_true")
    return p.parse_args()


def vec(t) -> np.ndarray:
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    return np.asarray(t, dtype=np.float64).reshape(-1)


def install_batched_ik(manip, logger, n_envs: int) -> None:
    """Adapter for the single-env case.

    Genesis 0.4.3's `inverse_kinematics` asserts `pos.shape[0] == scene.n_envs`,
    so targets are explicitly batched even with one environment. The StarVLA
    bridge launchers hit the same wall; see
    starvla_bridge/run_starvla_genesis_bridge_7d_v4..v6.py.
    """
    import torch
    state = {"prev": None, "logged": False}
    max_joint_change = float(manip.params["ik_params"]["max_joint_change"])

    def control_joints(position, quaternion):
        pos = np.asarray(vec(position), dtype=np.float64).reshape(1, 3)
        quat = np.asarray(vec(quaternion), dtype=np.float64).reshape(1, 4)
        pos = np.repeat(pos, n_envs, axis=0)
        quat = np.repeat(quat, n_envs, axis=0)
        pos_t = torch.as_tensor(pos, dtype=manip.datatype, device=manip.device)
        quat_t = torch.as_tensor(quat, dtype=manip.datatype, device=manip.device)

        qpos = manip.robot.inverse_kinematics(
            link=manip.end_effector, pos=pos_t, quat=quat_t
        )
        if not state["logged"]:
            logger(f"IK in {tuple(pos_t.shape)}/{tuple(quat_t.shape)} -> "
                   f"qpos{tuple(qpos.shape)} (n_envs={n_envs}, "
                   f"n_dofs={manip.robot.n_dofs}, n_qs={manip.robot.n_qs})")
            state["logged"] = True
        if not torch.isfinite(qpos).all():
            logger("IK returned a non-finite joint solution")
            return False

        q = qpos[0] if qpos.dim() > 1 else qpos
        # Index by whichever ordering the returned vector actually uses.
        idx = manip.motors_dof if q.numel() == manip.robot.n_dofs else manip.motors_qs
        motor_qpos = q[idx]
        prev = state["prev"]
        if prev is not None and prev.numel() == motor_qpos.numel():
            delta = motor_qpos - prev
            norm = torch.linalg.vector_norm(delta)
            if float(norm) > max_joint_change:
                motor_qpos = prev + delta * (max_joint_change / float(norm))
        state["prev"] = motor_qpos.detach().clone()
        return manip.control_joint_pos(motor_qpos)

    manip.control_joints = control_joints


def finger_link_positions(manip) -> dict:
    """Origins of the finger links, for measuring the real jaw aperture."""
    out = {}
    try:
        for link in manip.robot.links:
            if "Finger" in link.name or "finger" in link.name:
                out[link.name] = vec(link.get_pos()).tolist()
    except Exception:  # noqa: BLE001
        pass
    return out


def max_pairwise(points: list[list[float]]) -> float:
    if len(points) < 2:
        return float("nan")
    a = np.asarray(points, dtype=np.float64)
    best = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            best = max(best, float(np.linalg.norm(a[i] - a[j])))
    return best


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "grasp_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[grasp] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    import genesis as gs
    import torch
    from controllers.backend import EmptyBackend
    from envs.genesis_env import GenesisSim
    from robots.satellite_manipulator import SatelliteManipulator

    if not args.no_batch_renderer:
        SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
        print("[grasp] forced BatchRenderer")

    trace_file = trace_path.open("a", encoding="utf-8")

    def trace(record: dict) -> None:
        trace_file.write(json.dumps(record) + "\n")
        trace_file.flush()

    def log(msg: str) -> None:
        print(f"[grasp] {msg}", flush=True)

    report: dict = {"args": {k: str(v) for k, v in vars(args).items()}}

    # ---------- bring up ----------
    log("constructing manipulator")
    sim = GenesisSim()
    manip = SatelliteManipulator(name="franka_merge", sensors=[], backends=[EmptyBackend()])

    log(f"adding target box (size={args.object_size} m) before scene build")
    target = sim.scene.add_entity(
        gs.morphs.Box(size=(args.object_size,) * 3, pos=(0.0, 0.0, 0.0), collision=True),
        material=gs.materials.Rigid(),
    )

    log("starting sim")
    sim.start()
    log("initializing manipulator")
    manip.initialize()
    install_batched_ik(manip, log, int(sim.num_envs))

    def step_once():
        manip.step()
        sim.step()

    def settle(n: int):
        for _ in range(n):
            step_once()

    def read_gripper() -> np.ndarray:
        return vec(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))

    def move_tool(position, quaternion, steps: int):
        """Drive the tool to a world-frame pose; return (reached_pos, ok_any)."""
        ok_any = False
        last_err = None
        for _ in range(steps):
            try:
                ok = manip.control_joints(position, quaternion)
            except Exception as exc:  # noqa: BLE001
                log(f"control_joints raised: {exc!r}")
                return np.array(vec(manip.ee_global_position)), False
            if ok:
                ok_any = True
            step_once()
            manip.update_state()
            reached = np.array(vec(manip.ee_global_position))
            last_err = float(np.linalg.norm(reached - np.asarray(position)))
        manip.update_state()
        return np.array(vec(manip.ee_global_position)), ok_any

    # ---------- settle + read start pose ----------
    manip.control_gripper(1.0)
    settle(args.settle_steps)
    manip.update_state()

    p0 = np.array(vec(manip.ee_global_position))
    q0 = np.array(vec(manip.ee_global_quaternion))
    report["start"] = {
        "tcp_position_m": p0.tolist(),
        "tcp_quaternion_wxyz": q0.tolist(),
        "finger_qpos_open": read_gripper().tolist(),
    }
    log(f"TCP start = {np.round(p0, 4).tolist()}")

    # ---------- gripper travel + real aperture ----------
    fingers_open = finger_link_positions(manip)
    manip.control_gripper(0.0)
    settle(args.settle_steps)
    closed = read_gripper()
    fingers_closed = finger_link_positions(manip)
    manip.control_gripper(1.0)
    settle(args.settle_steps)

    ap_open = max_pairwise(list(fingers_open.values()))
    ap_closed = max_pairwise(list(fingers_closed.values()))

    # The TCP is the gripper BODY (2F-Body_Link), not the point between the jaws.
    # Dropping the object on the TCP puts it beside/behind the fingers, which is
    # why the fingers closed through empty space. Measure the jaw-centre offset
    # from the TCP and aim with that instead. The commanded orientation is q0
    # throughout, so this world-frame offset stays valid.
    jaw_offset = None
    if len(fingers_open) >= 2:
        jaw_center = np.mean(np.asarray(list(fingers_open.values()), dtype=np.float64), axis=0)
        jaw_offset = jaw_center - p0
    report["jaw"] = {
        "finger_positions_open": fingers_open,
        "jaw_center_offset_from_tcp_m": (jaw_offset.tolist() if jaw_offset is not None else None),
    }
    if jaw_offset is not None:
        log(f"jaw centre offset from TCP = {np.round(jaw_offset, 4).tolist()} m")

    report["gripper"] = {
        "open_qpos": report["start"]["finger_qpos_open"],
        "closed_qpos": closed.tolist(),
        "travel_rad": (closed - np.array(report["start"]["finger_qpos_open"])).tolist(),
        "finger_links": sorted(fingers_open.keys()),
        "aperture_open_m": ap_open,
        "aperture_closed_m": ap_closed,
        "aperture_travel_m": (ap_open - ap_closed) if np.isfinite(ap_open) and np.isfinite(ap_closed) else None,
        "object_size_m": args.object_size,
    }
    log(f"gripper qpos open={np.round(report['start']['finger_qpos_open'], 5).tolist()} "
        f"closed={np.round(closed, 5).tolist()}")
    log(f"gripper aperture open={ap_open:.4f} m closed={ap_closed:.4f} m "
        f"(travel {ap_open - ap_closed:.4f} m) vs object {args.object_size} m")

    # ---------- scan reachability around the start pose ----------
    directions = {
        "x+": np.array([1.0, 0.0, 0.0]),
        "x-": np.array([-1.0, 0.0, 0.0]),
        "y+": np.array([0.0, 1.0, 0.0]),
        "y-": np.array([0.0, -1.0, 0.0]),
        "z+": np.array([0.0, 0.0, 1.0]),
        "z-": np.array([0.0, 0.0, -1.0]),
    }
    distances = [0.06, 0.10, 0.15]
    scan = []
    log(f"reachability scan: {len(directions) * len(distances)} candidates "
        f"x {args.scan_ik_steps} IK steps")
    for dname, dvec in directions.items():
        for dist in distances:
            target_pos = p0 + dvec * dist
            reached, ok = move_tool(target_pos, q0, args.scan_ik_steps)
            residual = float(np.linalg.norm(reached - target_pos))
            scan.append({
                "direction": dname, "distance_m": dist,
                "target": target_pos.tolist(), "reached": reached.tolist(),
                "residual_m": residual, "control_ok": bool(ok),
            })
            log(f"  {dname} {dist:.2f}m residual={residual:.4f} m ok={ok}")
    report["scan"] = scan

    reachable = [s for s in scan if np.isfinite(s["residual_m"]) and s["residual_m"] < args.reach_tolerance]
    report["reachable_count"] = len(reachable)
    log(f"reachable: {len(reachable)} / {len(scan)} (tol={args.reach_tolerance} m)")

    if not scan:
        report["verdict"] = "no_candidates"
        (args.output_dir / "grasp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        trace_file.close()
        log("VERDICT: scan produced no candidates")
        return 1

    # Even if nothing met the tolerance, attempt the best candidate anyway: a
    # partial approach still tells us whether the gripper can close on the object.
    best = min(reachable or scan, key=lambda s: s["residual_m"])
    report["used_fallback_best"] = bool(not reachable)
    chosen = np.array(best["target"])
    approach_dir = (chosen - p0)
    norm = float(np.linalg.norm(approach_dir))
    approach_dir = approach_dir / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    report["chosen"] = {"direction": best["direction"], "distance_m": best["distance_m"],
                        "residual_m": best["residual_m"], "position": chosen.tolist(),
                        "approach_dir": approach_dir.tolist()}
    log(f"chosen target {np.round(chosen, 4).tolist()} ({best['direction']} {best['distance_m']}m)")

    # ---------- place the object and run the grasp ----------
    target.set_pos(chosen)
    settle(20)
    obj_before = np.array(vec(target.get_pos()))
    report["object_requested_at"] = chosen.tolist()
    report["object_placed_at"] = obj_before.tolist()
    report["object_placement_drift_m"] = float(np.linalg.norm(obj_before - chosen))
    # Aim at where the object ACTUALLY is, not where we asked it to be.
    chosen = obj_before.copy()
    log(f"object placed at {np.round(obj_before, 4).tolist()} "
        f"(drift {report['object_placement_drift_m']:.4f} m from request)")

    # Aim the JAW CENTRE at the object, not the TCP.
    object_centre = obj_before.copy()
    if jaw_offset is not None:
        grasp_tcp_target = object_centre - jaw_offset
    else:
        grasp_tcp_target = object_centre.copy()
    report["grasp_tcp_target"] = grasp_tcp_target.tolist()
    log(f"object centre {np.round(object_centre, 4).tolist()} -> "
        f"TCP target {np.round(grasp_tcp_target, 4).tolist()}")

    pre_grasp = grasp_tcp_target - approach_dir * args.pre_grasp_distance

    log("phase: approach (pre-grasp)")
    reached, ok = move_tool(pre_grasp, q0, args.ik_steps)
    log(f"  pre-grasp residual={np.linalg.norm(reached - pre_grasp):.4f} m ok={ok}")

    log("phase: descend to grasp pose")
    reached, ok = move_tool(grasp_tcp_target, q0, args.ik_steps)
    grasp_residual = float(np.linalg.norm(reached - grasp_tcp_target))
    report["grasp_residual_m"] = grasp_residual
    report["grasp_reached_object"] = bool(grasp_residual < args.reach_tolerance)
    # Where the jaw centre actually ended up, relative to the object.
    try:
        jc = np.mean(np.asarray(list(finger_link_positions(manip).values()), dtype=np.float64), axis=0)
        report["jaw_centre_at_grasp_m"] = jc.tolist()
        report["jaw_centre_to_object_m"] = float(np.linalg.norm(jc - object_centre))
    except Exception:  # noqa: BLE001
        pass
    log(f"  grasp residual={grasp_residual:.4f} m "
        f"(jaw-centre to object {report.get('jaw_centre_to_object_m')})")

    object_at_grasp = np.array(vec(target.get_pos()))
    report["object_at_grasp_m"] = object_at_grasp.tolist()
    report["jaw_centre_to_object_at_grasp_m"] = float(
        np.linalg.norm(np.array(list(report.get("jaw_centre_at_grasp_m", object_at_grasp)))
                       - object_at_grasp)
    ) if report.get("jaw_centre_at_grasp_m") else None

    log("phase: close gripper")
    for i in range(args.close_steps):
        manip.control_gripper(0.0)
        step_once()
        if i % 20 == 0:
            trace({"phase": "close", "i": i, "finger_qpos": read_gripper().tolist(),
                   "object": vec(target.get_pos()).tolist()})
    closed_on_object = read_gripper()
    report["finger_qpos_closed_on_object"] = closed_on_object.tolist()
    log(f"  fingers on object: {np.round(closed_on_object, 5).tolist()}")

    # Did the object get pushed away DURING the close (rather than carried during
    # the pull)? Without this split we cannot tell "squeezed out" from "grasped".
    object_after_close = np.array(vec(target.get_pos()))
    report["object_after_close_m"] = object_after_close.tolist()
    report["object_push_during_close_m"] = float(
        np.linalg.norm(object_after_close - object_at_grasp))
    log(f"  object pushed during close: {report['object_push_during_close_m']:.4f} m")

    obj_at_close = np.array(vec(target.get_pos()))
    ee_at_close = np.array(vec(manip.ee_global_position))

    log("phase: pull away")
    pull_target = grasp_tcp_target + approach_dir * args.pull_distance
    for i in range(args.pull_steps):
        manip.control_joints(pull_target, q0)
        step_once()
        manip.update_state()
        if i % 20 == 0:
            trace({"phase": "pull", "i": i,
                   "ee": vec(manip.ee_global_position).tolist(),
                   "object": vec(target.get_pos()).tolist()})
    manip.update_state()

    obj_after = np.array(vec(target.get_pos()))
    ee_after = np.array(vec(manip.ee_global_position))

    ee_disp = float(np.linalg.norm(ee_after - ee_at_close))
    obj_disp = float(np.linalg.norm(obj_after - obj_at_close))
    ratio = obj_disp / ee_disp if ee_disp > 1e-6 else float("inf")

    # A grasp is only real if all three hold. Any one alone is satisfiable by
    # simply knocking the object around, which is what happens without contact.
    rel_before = obj_at_close - ee_at_close
    rel_after = obj_after - ee_after
    slip = float(np.linalg.norm(rel_after - rel_before))
    free_closed = float(np.mean(report["gripper"]["closed_qpos"]))
    on_object = float(np.mean(closed_on_object))
    fingers_blocked = on_object < (free_closed - args.finger_block_margin)

    checks = {
        "fingers_blocked_by_object": fingers_blocked,
        "moves_with_tool": args.grasp_hold_ratio_min <= ratio <= args.grasp_hold_ratio_max,
        "no_relative_slip": slip < args.relative_slip_tolerance,
    }
    held = all(checks.values())

    report["pull"] = {
        "ee_before": ee_at_close.tolist(), "ee_after": ee_after.tolist(),
        "object_before": obj_at_close.tolist(), "object_after": obj_after.tolist(),
        "ee_displacement_m": ee_disp, "object_displacement_m": obj_disp,
        "displacement_ratio": ratio,
        "relative_slip_m": slip,
        "finger_qpos_free_closed": free_closed,
        "finger_qpos_on_object": on_object,
    }
    report["grasp_checks"] = checks
    report["grasp_held"] = held
    report["verdict"] = "grasp_held" if held else "grasp_failed"
    log(f"  tool {ee_disp:.4f} m, object {obj_disp:.4f} m, ratio {ratio:.2f}")
    log(f"  fingers free-closed {free_closed:.5f} vs on-object {on_object:.5f} "
        f"-> blocked={fingers_blocked}")
    log(f"  tool->object offset changed by {slip:.4f} m -> slip_ok={checks['no_relative_slip']}")
    for name, ok in checks.items():
        log(f"  check {name}: {'PASS' if ok else 'FAIL'}")
    log(f"VERDICT: {'GRASP HELD' if held else 'GRASP FAILED'}")

    trace_file.close()
    (args.output_dir / "grasp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {args.output_dir / 'grasp_report.json'}")

    try:
        manip.stop()
        sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
