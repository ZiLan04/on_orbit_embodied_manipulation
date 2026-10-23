#!/usr/bin/env python3
"""What is the *closest* the gripper can physically get to the object?

Why this exists
---------------
Every controller-based attempt to answer this has been inconclusive:

  * the 20 shipped stage-2 PPO checkpoints plateau at ~0.25-0.30 m (0/10 success);
  * a hand-written IK servo got the reference link to the mandated setpoint C at
    step 416 (so C is reachable) but still left the pinch centre 0.352 m from the
    object;
  * sweeping `stage2_base_distance` over 0.5-1.1 m never beat 0.211 m, and 1.1 m
    was the best of the six.

Those are all statements about *controllers*, and a bad controller cannot prove
unreachability. This script removes the controller from the question entirely: it
samples joint configurations, runs pure Genesis `forward_kinematics`, and reports
the best distance any configuration achieves. No policy, no servo, no episode
budget, no dynamics.

It answers two questions with the same scan:

  1. reachable-object   min ||pinch_centre - object_grasp_point||   (target 0.025)
  2. reachable-setpoint min pose_position_error                     (target 0.055)
                        max pose_quat_similarity                     (target 0.95)

(2) is the honest version of "is the task feasible": if no joint configuration can
satisfy the criterion, no amount of training will.

Genesis allows one scene per process, so this takes a single --base-distance.

Usage
-----
    python reach_scan.py --tree /data/Yanhan/genesis/spaceman_core \
        --base-distance 1.1 --num-envs 512 --batches 200 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-reach-scan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TARGET_PINCH = 0.025
TARGET_POSE = 0.055
TARGET_QUAT = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--base-distance", type=float, default=1.1)
    p.add_argument("--virtual-grasp-height", type=float, default=0.08)
    p.add_argument("--num-envs", type=int, default=512,
                   help="Joint configurations sampled per batch.")
    p.add_argument("--batches", type=int, default=200,
                   help="Total samples = num_envs * batches.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ik-samples", type=int, default=8,
                   help="How many round-robin IK seeds to descend from (Genesis IK is "
                        "a local method, so several starts help find good basins).")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def find_task(obj, depth: int = 0):
    if depth > 6 or obj is None:
        return None
    if hasattr(obj, "target_grasp_pose") and hasattr(obj, "starlink_manipulator"):
        return obj
    for attr in ("venv", "env", "task", "unwrapped"):
        if hasattr(obj, attr):
            found = find_task(getattr(obj, attr), depth + 1)
            if found is not None:
                return found
    return None


def to_np(t):
    return t.detach().float().cpu().numpy()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[scan] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    import genesis as gs
    import torch
    SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)

    def log(msg):
        print(f"[scan] {msg}", flush=True)

    from tasks.RL_task_parallel_stage2 import make_stage2_vec_env
    env = make_stage2_vec_env(
        num_envs=args.num_envs, enable_visual_obs=False,
        enable_target_relative_obs=None,
        stage2_base_distance=args.base_distance,
        virtual_grasp_height=args.virtual_grasp_height,
        target_position_noise=False)
    task = find_task(env)
    if task is None:
        log("FATAL: could not locate the task")
        return 3
    task.max_steps = 10 ** 9
    env.reset()          # builds the scene; starlink_manipulator is None before this

    manip = task.starlink_manipulator
    robot = manip.robot
    n = task.num_envs
    device = task.device
    dtype = task.datatype
    log(f"task={type(task).__name__} num_envs={n} n_qs={robot.n_qs} n_dofs={robot.n_dofs}")
    log(f"motors_qs={list(manip.motors_qs)}  joint_delta_scale={task.joint_delta_scale}")

    finger_names = list(task.grasp_finger_link_names)
    ref_name = task.reference_yaw_link_name
    names = finger_names + [ref_name]
    # get_links_pos wants LOCAL link indices. Passing global `.idx` values returns a
    # (n, 0, 3) tensor and then trips a CUDA device-side assert on the next op.
    local_idx = [robot.get_link(nm).idx_local for nm in names]
    const_pos = task.reference_yaw_link_xyz.detach()
    const_quat = task.reference_yaw_link_quat.detach()
    log(f"links: finger_a={names[0].split('.')[-1]} finger_b={names[1].split('.')[-1]} "
        f"ref={ref_name.split('.')[-1]}")

    target = task.target_grasp_pose[:, :3].detach()          # (n,3), constant (no noise)
    target0 = to_np(target[0])
    log(f"object grasp point (env0) = {np.round(target0, 5).tolist()}")
    log(f"mandated setpoint C        = {np.round(to_np(const_pos), 5).tolist()}")
    log(f"||C - object||             = {np.linalg.norm(to_np(const_pos) - target0):.5f} m")

    envs_idx = torch.arange(n, dtype=torch.long, device=device)
    qs_idx = list(manip.motors_qs)
    orig_qpos = robot.get_qpos().detach().clone()

    def evaluate(qpos_arm):
        """Pure kinematic evaluation of a batch of arm configurations.

        Uses `set_qpos`, not `forward_kinematics`: the latter accepts the call and
        returns, but does NOT move the links (verified -- link positions were
        bit-identical before and after). `set_qpos` defaults to `skip_forward=False`
        and does update them, so it is the usable FK path here.

        Positions only: reading link QUATERNIONS after updating kinematics raises
        "CUDA error: device-side assert triggered" in this Genesis build, so
        orientation is measured by `ik_reachability.py` instead.
        """
        robot.set_qpos(qpos_arm, qs_idx_local=qs_idx, envs_idx=envs_idx,
                       zero_velocity=True)
        pos = robot.get_links_pos(links_idx_local=local_idx, envs_idx=envs_idx)
        pos = pos.detach()
        finger_a = pos[:, 0, :]
        finger_b = pos[:, 1, :]
        ref_pos = pos[:, 2, :]
        pinch = 0.5 * (finger_a + finger_b)
        d_obj = torch.norm(pinch - target, dim=-1)
        d_pose = torch.norm(ref_pos - const_pos.view(1, 3), dim=-1)
        qsim = torch.full_like(d_pose, float("nan"))
        return d_obj, d_pose, qsim, pinch, ref_pos

    # ---- sanity check: FK at the untouched configuration must reproduce the link
    # positions the task reports, otherwise the scan is meaningless.
    #
    # Positions only. Reading link QUATERNIONS after `forward_kinematics` raises
    # "CUDA error: device-side assert triggered" in this Genesis build (both
    # `get_links_quat` and `Link.get_quat`), so quaternion similarity is not part
    # of this scan -- `ik_reachability.py` measures orientation instead.
    pinch_before = to_np(robot.get_links_pos(links_idx_local=local_idx[:2],
                                             envs_idx=envs_idx))[0]
    pinch0 = 0.5 * (pinch_before[0] + pinch_before[1])
    ref0 = to_np(robot.get_links_pos(links_idx_local=local_idx[2:3],
                                     envs_idx=envs_idx))[0][0]
    arm_now = robot.get_qpos(qs_idx_local=qs_idx).detach()
    d_obj_fk, d_pose_fk, _, pinch_fk, ref_fk = evaluate(arm_now)
    dp = float(np.linalg.norm(to_np(pinch_fk[0]) - pinch0))
    dr = float(np.linalg.norm(to_np(ref_fk[0]) - ref0))
    log(f"sanity: FK pinch vs pre-FK pinch   delta {dp:.6f} m "
        f"({np.round(pinch0, 5).tolist()})")
    log(f"sanity: FK ref   vs pre-FK ref     delta {dr:.6f} m")
    log(f"sanity: FK pinch->object {float(d_obj_fk[0]):.5f} m, "
        f"pose_err {float(d_pose_fk[0]):.5f} m (expected ~0.7959 / ~0.7462)")
    if dp > 1e-3 or dr > 1e-3:
        log("FATAL: forward_kinematics does not reproduce the task's own link "
            "positions; the scan would be meaningless. Aborting.")
        return 4

    best = {
        "min_pinch_to_object_m": float("inf"),
        "min_pose_error_m": float("inf"),
        "max_quat_similarity": -float("inf"),
        "argmin_pinch_qpos": None,
        "argmin_pose_qpos": None,
        "argmax_quat_qpos": None,
        "argmin_pinch_point": None,
        "argmin_pose_point": None,
    }
    n_target_pinch = 0
    n_target_pose = 0
    n_target_both = 0
    total = 0
    # Basin breadth: how much of joint space is "close enough" on each criterion?
    # This is what decides whether a reward is learnable by exploration at all.
    THRESHOLDS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
    counts_pinch = {t: 0 for t in THRESHOLDS}
    counts_pose = {t: 0 for t in THRESHOLDS}
    # Reservoir-free: keep a bounded random subsample of each error to report
    # percentiles without storing millions of floats.
    samples = {"pinch": [], "pose": []}
    max_keep = 200_000

    def absorb(d_obj, d_pose, qsim, pinch, arm):
        nonlocal n_target_pinch, n_target_pose, n_target_both, total
        total += d_obj.numel()
        n_target_pinch += int((d_obj <= TARGET_PINCH).sum().item())
        n_target_pose += int((d_pose <= TARGET_POSE).sum().item())
        n_target_both += int(((d_pose <= TARGET_POSE) & (qsim >= TARGET_QUAT)).sum().item())
        for t in THRESHOLDS:
            counts_pinch[t] += int((d_obj <= t).sum().item())
            counts_pose[t] += int((d_pose <= t).sum().item())
        for key, val in (("pinch", d_obj), ("pose", d_pose)):
            room = max_keep - len(samples[key])
            if room > 0:
                v = val[:room]
                samples[key].append(to_np(v))
        i = int(torch.argmin(d_obj).item())
        if float(d_obj[i]) < best["min_pinch_to_object_m"]:
            best["min_pinch_to_object_m"] = float(d_obj[i])
            best["argmin_pinch_qpos"] = to_np(arm[i]).tolist()
            best["argmin_pinch_point"] = to_np(pinch[i]).tolist()
        i = int(torch.argmin(d_pose).item())
        if float(d_pose[i]) < best["min_pose_error_m"]:
            best["min_pose_error_m"] = float(d_pose[i])
            best["argmin_pose_qpos"] = to_np(arm[i]).tolist()
            best["argmin_pose_point"] = to_np(pinch[i]).tolist()
        i = int(torch.argmax(qsim).item())
        if float(qsim[i]) > best["max_quat_similarity"]:
            best["max_quat_similarity"] = float(qsim[i])
            best["argmax_quat_qpos"] = to_np(arm[i]).tolist()
    rng = np.random.default_rng(args.seed)
    lo, hi = -np.pi, np.pi
    t0 = time.perf_counter()

    log(f"scanning {n * args.batches} random joint configurations ...")
    for b in range(args.batches):
        q = torch.as_tensor(rng.uniform(lo, hi, size=(n, 6)), dtype=dtype, device=device)
        d_obj, d_pose, qsim, pinch, _ = evaluate(q)
        absorb(d_obj, d_pose, qsim, pinch, q)
        if b % 25 == 0 or b == args.batches - 1:
            log(f"  batch {b+1}/{args.batches}: best pinch->object "
                f"{best['min_pinch_to_object_m']:.5f}  best pose_err "
                f"{best['min_pose_error_m']:.5f}  max quat_sim "
                f"{best['max_quat_similarity']:.5f}")

    # IK-seeded candidates were dropped: `inverse_kinematics` also trips the same
    # CUDA assert once `forward_kinematics` has been called in the process, and the
    # random scan is the unbiased measurement anyway. `ik_reachability.py` covers
    # the IK-from-a-controller angle separately.

    try:
        robot.set_qpos(orig_qpos, qs_idx_local=None, envs_idx=envs_idx,
                       zero_velocity=True)
    except Exception as exc:  # noqa: BLE001
        log(f"  (could not restore original qpos: {exc!r})")
    elapsed = time.perf_counter() - t0

    summary = {
        "base_distance": args.base_distance,
        "num_envs": args.num_envs,
        "batches": args.batches,
        "ik_samples": args.ik_samples,
        "configurations_evaluated": int(total),
        "seconds": elapsed,
        "object_grasp_point": target0.tolist(),
        "mandated_setpoint_C": to_np(const_pos).tolist(),
        "distance_C_to_object_m": float(np.linalg.norm(to_np(const_pos) - target0)),
        "targets": {"pinch_to_object_m": TARGET_PINCH,
                    "pose_error_m": TARGET_POSE, "quat_similarity": TARGET_QUAT},
        "count_reaching_pinch_target": n_target_pinch,
        "count_reaching_pose_target": n_target_pose,
        "count_reaching_pose_and_quat_target": n_target_both,
        "best": best,
    }
    verdict_lines = []
    if n_target_pinch > 0:
        verdict_lines.append(
            f"OBJECT IS REACHABLE: {n_target_pinch} of {total} configurations put the "
            f"pinch centre within {TARGET_PINCH} m of the grasp point.")
    else:
        verdict_lines.append(
            f"OBJECT IS OUT OF REACH (kinematically): no configuration in {total} "
            f"got the pinch centre within {TARGET_PINCH} m; best was "
            f"{best['min_pinch_to_object_m']:.5f} m "
            f"({best['min_pinch_to_object_m'] / TARGET_PINCH:.1f}x the target).")
    if n_target_pose > 0:
        verdict_lines.append(
            f"TASK SETPOINT IS REACHABLE: {n_target_pose} configurations satisfy "
            f"`is_success()`'s position test (pose_err <= {TARGET_POSE}).")
    else:
        verdict_lines.append(
            f"TASK SETPOINT NOT SATISFIED BY ANY SAMPLE: best pose_err "
            f"{best['min_pose_error_m']:.5f} (need <= {TARGET_POSE}).")
    verdict_lines.append(
        "NOTE: quaternion similarity is NOT measured here -- reading link quats after "
        "forward_kinematics aborts with a CUDA device-side assert in this Genesis "
        "build. Orientation is covered by ik_reachability.py instead.")

    # Since quaternions are unavailable, the honest feasibility statement is about
    # the position half of the criterion only.
    summary["count_reaching_pose_and_quat_target"] = None
    summary["quat_similarity_measured"] = False
    summary["basin_breadth"] = {
        "thresholds_m": list(THRESHOLDS),
        "count_pinch_within": {str(t): counts_pinch[t] for t in THRESHOLDS},
        "count_pose_within": {str(t): counts_pose[t] for t in THRESHOLDS},
        "fraction_pinch_within": {str(t): counts_pinch[t] / total for t in THRESHOLDS},
        "fraction_pose_within": {str(t): counts_pose[t] / total for t in THRESHOLDS},
    }
    for key in ("pinch", "pose"):
        if samples[key]:
            a = np.concatenate(samples[key])
            summary["basin_breadth"][f"{key}_error_percentiles_m"] = {
                p: float(np.percentile(a, p)) for p in (1, 5, 10, 25, 50, 75, 90)
            }
    verdict_lines.append(
        "BASIN BREADTH (decides learnability): fraction of joint space within a "
        "given error -- object-facing pinch error vs the task's own pose_err:")
    for t in THRESHOLDS:
        verdict_lines.append(
            f"  <= {t:.3f} m : pinch {counts_pinch[t]/total:.3e}  "
            f"pose_err {counts_pose[t]/total:.3e}")
    summary["verdict"] = verdict_lines

    (args.output_dir / "reach_scan.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== REACHABLE-SET SCAN =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "best"}, indent=2))
    print("\nbest found:")
    print(json.dumps(best, indent=2))
    print("\nVERDICT:")
    for line in verdict_lines:
        print("  * " + line)
    print(f"\n[scan] wrote {args.output_dir}/reach_scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
