#!/usr/bin/env python3
"""Is the stage-2 success criterion actually *controllable* by the action space?

Motivation
----------
The stage-2 PPO reward and `is_success()` are built from
`pose_position_error` / `pose_quat_similarity` / `reference_link_speed`, and
those are computed by `_compute_reference_pose_metrics()` on the link named

    starlink_space_manipulator_merged.SLDASM9_starlink__space_manipulator_2F-Body_Link

which is the manipulator's *body* link -- the kinematic parent of the yaw
joint, rigidly attached to the satellite root. The action is a 6-vector of
joint-position deltas (`joint_delta_scale = 0.035`), which moves the TCP but
should NOT move the body link at all.

If that reading is right, then:
  * `pose_position_error` is a function of the reset state alone,
  * therefore every action-dependent term in the reward is a *penalty*,
  * therefore "output zero action forever" is optimal, and no amount of steps,
    checkpoints, or hyper-parameters can produce an aligned gripper.

This script tests that directly and cheaply. It drives the exact same env the
trainer uses with a set of deliberately extreme, hand-written action programs
(no learned policy), and compares the resulting metrics. Actions here are far
larger than anything PPO explores, so this is an upper bound on what the
action space can influence.

How to read the result
----------------------
  * `tcp_to_virtual_grasp_error` is the genuinely object-facing quantity. It
    MUST vary wildly across programs -- that is the control that proves the
    actions really are being applied and the sim really responds.
  * If `pose_position_error` / `pose_quat_similarity` / `reference_link_speed`
    stay identical across all programs (down to float noise), the success
    criterion is action-independent and the shipped stage-2 task is degenerate.

Usage
-----
    python stage2_action_control.py \
        --tree /data/Yanhan/genesis/spaceman_core \
        --num-envs 8 --max-steps 200 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-action-control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# info keys published by RL_task_parallel_stage2.py (see eval_stage2_batch.py:40)
SCALAR_KEYS = (
    "tcp_to_virtual_grasp_error",
    "pose_position_error",
    "pose_quat_similarity",
    "reference_link_speed",
    "base_drift_error",
    "target_hold_count",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200,
                   help="Stage-2 training truncates at 120; 200 gives slack.")
    p.add_argument("--target-asset", default="starlink")
    p.add_argument("--stage2-base-distance", type=float, default=1.1)
    p.add_argument("--virtual-grasp-height", type=float, default=0.08)
    p.add_argument("--visual-obs", action="store_true",
                   help="Off by default: cameras are pure overhead for this test.")
    p.add_argument("--target-position-noise", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def build_action_programs(num_envs: int, max_steps: int, rng: np.random.Generator):
    """Return {name: callable(step, num_envs) -> (num_envs, 6) float32 array}.

    Each program is deliberately extreme so that if the criterion *can* be
    moved by joints, it will be.
    """
    progs = {}

    progs["zero"] = lambda s, n: np.zeros((n, 6), dtype=np.float32)

    progs["all_plus"] = lambda s, n: np.ones((n, 6), dtype=np.float32)
    progs["all_minus"] = lambda s, n: -np.ones((n, 6), dtype=np.float32)

    # Move only the wrist-ish joints (last three) to swing the TCP around.
    def _wrist(s, n):
        a = np.zeros((n, 6), dtype=np.float32)
        a[:, 3:] = 1.0
        return a
    progs["wrist_plus"] = _wrist

    def _shoulder(s, n):
        a = np.zeros((n, 6), dtype=np.float32)
        a[:, :3] = 1.0
        return a
    progs["shoulder_plus"] = _shoulder

    # Alternating square wave: keeps the arm sweeping back and forth so it
    # has a chance to pass through the target.
    def _sweep(s, n):
        sign = 1.0 if (s // 10) % 2 == 0 else -1.0
        return np.full((n, 6), sign, dtype=np.float32)
    progs["sweep"] = _sweep

    # Deterministic pseudo-random, fixed per (env, step) so runs are repeatable.
    fixed = rng.uniform(-1.0, 1.0, size=(max_steps, num_envs, 6)).astype(np.float32)
    progs["random"] = lambda s, n: fixed[min(s, max_steps - 1)][:n]

    return progs


def rollout(env, program, max_steps: int, num_envs: int) -> dict:
    env.reset()
    per_step: list[dict] = []
    ever_success = np.zeros(num_envs, dtype=bool)

    for step in range(max_steps):
        actions = program(step, num_envs)
        _, rewards, _, infos = env.step(actions)

        rows = []
        for i in range(num_envs):
            info = infos[i] if i < len(infos) else {}
            row = {"step": step, "env_index": i, "reward": float(rewards[i])}
            for k in SCALAR_KEYS:
                v = info.get(k)
                row[k] = float(v) if v is not None and np.isfinite(float(v)) else None
            row["success"] = bool(info.get("success", False))
            if row["success"]:
                ever_success[i] = True
            rows.append(row)
        per_step.extend(rows)

    agg = {}
    for k in SCALAR_KEYS:
        vals = [r[k] for r in per_step if r[k] is not None]
        if not vals:
            agg[k] = None
            continue
        a = np.asarray(vals, dtype=float)
        agg[k] = {
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
            "std": float(a.std()),
            "final": float(vals[-1]),
        }
    agg["success_any"] = bool(ever_success.any())
    agg["success_env_count"] = int(ever_success.sum())
    agg["reward_sum_mean"] = float(np.sum([r["reward"] for r in per_step]) / max(num_envs, 1))
    return {"aggregate": agg, "per_step": per_step}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[ac] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    import genesis as gs
    SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)

    from tasks.RL_task_parallel_stage2 import make_stage2_vec_env

    kwargs = dict(
        num_envs=args.num_envs,
        target_asset=args.target_asset,
        enable_visual_obs=args.visual_obs,
        enable_target_relative_obs=None,
        stage2_base_distance=args.stage2_base_distance,
        virtual_grasp_height=args.virtual_grasp_height,
        target_position_noise=args.target_position_noise,
    )
    print(f"[ac] env kwargs: {json.dumps({k: str(v) for k, v in kwargs.items()})}")
    env = make_stage2_vec_env(**kwargs)
    print(f"[ac] num_envs={env.num_envs} action_space={env.action_space} "
          f"observation_space={env.observation_space}")

    rng = np.random.default_rng(args.seed)
    progs = build_action_programs(env.num_envs, args.max_steps, rng)

    results = {}
    t0 = time.perf_counter()
    for name, prog in progs.items():
        t_p = time.perf_counter()
        results[name] = rollout(env, prog, args.max_steps, env.num_envs)
        print(f"[ac] {name:<14} done in {time.perf_counter() - t_p:6.1f}s", flush=True)
    total = time.perf_counter() - t0

    # ---------------- comparison table ----------------
    print("\n===== ACTION -> METRIC CONTROLLABILITY =====")
    hdr = f"{'program':<14} {'tcp_err_min':>11} {'tcp_err_final':>13} " \
          f"{'pose_err_min':>12} {'pose_err_final':>14} {'quat_best':>9} " \
          f"{'link_spd_fin':>12} {'succ':>5}"
    print(hdr)
    print("-" * len(hdr))

    def get(name, key, field):
        v = results[name]["aggregate"].get(key)
        return v[field] if v else float("nan")

    for name in progs:
        print(f"{name:<14} {get(name,'tcp_to_virtual_grasp_error','min'):11.4f} "
              f"{get(name,'tcp_to_virtual_grasp_error','final'):13.4f} "
              f"{get(name,'pose_position_error','min'):12.6f} "
              f"{get(name,'pose_position_error','final'):14.6f} "
              f"{get(name,'pose_quat_similarity','max'):9.5f} "
              f"{get(name,'reference_link_speed','final'):12.6f} "
              f"{str(results[name]['aggregate']['success_env_count']):>5}")

    # ---------------- verdict ----------------
    pose_mins = [get(n, "pose_position_error", "min") for n in progs]
    pose_finals = [get(n, "pose_position_error", "final") for n in progs]
    quat_maxs = [get(n, "pose_quat_similarity", "max") for n in progs]
    tcp_mins = [get(n, "tcp_to_virtual_grasp_error", "min") for n in progs]

    pose_spread = max(pose_mins) - min(pose_mins)
    quat_spread = max(quat_maxs) - min(quat_maxs)
    tcp_spread = max(tcp_mins) - min(tcp_mins)

    if tcp_spread < 1e-4:
        verdict = "INCONCLUSIVE: actions did not perturb the TCP either; the test itself failed."
    elif pose_spread < 1e-6 and quat_spread < 1e-9:
        verdict = ("PROVEN: pose_position_error / pose_quat_similarity are EXACTLY "
                   "action-independent. The stage-2 success criterion cannot be "
                   "influenced by the policy, so the task is degenerate.")
    elif pose_spread < 0.005:
        verdict = ("LIKELY: the success criterion barely moves (<5 mm) under "
                   "deliberately extreme actions while the TCP moves metres. "
                   "The reward gives essentially no usable gradient.")
    else:
        verdict = ("REFUTED: the success criterion does respond to joint actions "
                   "(spread %.4f m); the plateau must have another cause." % pose_spread)

    print("\n===== VERDICT =====")
    print(f"tcp_err_min   spread across programs: {tcp_spread:.6f} m  (control: must be large)")
    print(f"pose_err      spread across programs: {pose_spread:.8f} m")
    print(f"quat_sim      spread across programs: {quat_spread:.10f}")
    print(verdict)

    summary = {
        "tree": str(args.tree),
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "env_kwargs": {k: str(v) for k, v in kwargs.items()},
        "total_seconds": total,
        "spreads": {
            "tcp_err_min": tcp_spread,
            "pose_err_min": pose_spread,
            "quat_sim_max": quat_spread,
        },
        "verdict": verdict,
        "per_program": {n: results[n]["aggregate"] for n in progs},
    }
    (args.output_dir / "action_control_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "action_control_per_step.json").write_text(
        json.dumps({n: results[n]["per_step"][:400] for n in progs}, indent=2, default=str),
        encoding="utf-8")
    print(f"\n[ac] wrote {args.output_dir}/action_control_summary.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
