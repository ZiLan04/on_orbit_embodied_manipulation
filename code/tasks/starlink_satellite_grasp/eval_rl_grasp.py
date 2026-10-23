#!/usr/bin/env python3
"""Evaluate an RL grasp policy against the metric we actually care about.

Why this exists
---------------
`eval_stage2_batch.py` only knows the stage-2 family and only reads what that
task publishes in `info`. Two things were missing:

1. It cannot evaluate the *other* shipped family, `ppo_satellite_side_grasp_parallel`
   (`ParallelRLTask`). That family is trained against the **object-facing**
   criterion (`tcp_to_grasp_error <= reach_distance_threshold`, see
   RL_task_parallel.py:825), whereas stage-2 is not. Comparing the two families
   on one metric is the whole point.

2. It reports `tcp_to_virtual_grasp_error`, i.e. the *tool centre point*. The
   downstream requirement is stated on the **pinch centre** (between the jaws),
   and the jaw centre sits ~6.5 cm away from the TCP (measured offset
   [-0.019, -0.047, -0.041] m). So a policy can look "nearly aligned" on the TCP
   metric while the jaws are a whole finger-width away. This script measures the
   pinch centre directly from the two finger link origins.

Both numbers are reported, so the gap between them stays visible.

It never writes inside the vendored spaceman_core tree; output goes to
--output-dir.

Usage
-----
    python eval_rl_grasp.py \
        --tree /data/Yanhan/genesis/spaceman_core \
        --family parallel \
        --model-path .../ppo_satellite_side_grasp_parallel_14000000_steps.zip \
        --num-envs 25 --rounds 2 --max-steps 200 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-parallel-eval/14M
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

FINGER_LINKS = (
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
    "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path,
                   help="Path to the spaceman_core checkout (must contain src/).")
    p.add_argument("--family", required=True, choices=("stage2", "parallel"))
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--train-config", type=Path, default=None,
                   help="Defaults to <model dir>/../train_config.json")
    p.add_argument("--num-envs", type=int, default=25)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--override-max-steps", type=int, default=None,
                   help="Raise the task's own truncation limit (task.max_steps) before "
                        "rolling out. The stage-2 task hardcodes 120; the IK expert needs "
                        "~420 steps to reach the mandated pose, so a policy that merely "
                        "runs out of time looks identical to one that cannot reach it. "
                        "Set this to separate those two cases.")
    p.add_argument("--alignment-threshold", type=float, default=0.025,
                   help="Pinch-centre-to-target success threshold, metres.")
    p.add_argument("--target-position-noise", action="store_true", default=None)
    p.add_argument("--no-target-position-noise", dest="target_position_noise",
                   action="store_false")
    p.add_argument("--stochastic", action="store_true",
                   help="Default is deterministic action selection.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def find_task(obj, depth: int = 0):
    """Locate the RLTask instance behind SB3's wrapper stack.

    VecMonitor -> BatchedGenesis*VecEnv -> gym env -> .task
    """
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


def tensor_stats(xs):
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


class TaskProbe:
    """Reads geometry straight off the live task each step.

    Deliberately reads only leaf tensors and converts them to python floats in
    the same expression -- no live task object is ever stored or serialized.
    """

    def __init__(self, task):
        self.task = task
        self.robot = task.starlink_manipulator.robot

    def _link_pos(self, name):
        link = self.robot.get_link(name)
        p = link.get_pos()
        return p.detach().float().cpu().numpy().reshape(-1, 3)

    def sample(self, n_envs: int) -> list[dict]:
        t = self.task
        tgt = t.target_grasp_pose[:, :3].detach().float().cpu().numpy()
        tcp = t.tcp_pose[:, :3].detach().float().cpu().numpy()

        try:
            fa = self._link_pos(FINGER_LINKS[0])
            fb = self._link_pos(FINGER_LINKS[1])
            pinch = 0.5 * (fa + fb)
            aperture = np.linalg.norm(fa - fb, axis=-1)
        except Exception:  # noqa: BLE001
            pinch = None
            aperture = None

        # stage-2 names its object-facing error differently
        if hasattr(t, "tcp_to_grasp_error") and t.tcp_to_grasp_error is not None:
            tcp_err = t.tcp_to_grasp_error.detach().float().cpu().numpy()
        else:
            tcp_err = t.tcp_to_virtual_grasp_error.detach().float().cpu().numpy()

        def col(name, width=None):
            v = getattr(t, name, None)
            if v is None:
                return [None] * n_envs
            a = v.detach().float().cpu().numpy()
            return a.tolist()

        success = (t.success.detach().cpu().numpy().astype(bool).tolist()
                   if t.success is not None else [False] * n_envs)
        hold = col("target_hold_count")
        align = col("approach_alignment")
        base_err = (col("base_alignment_error") if hasattr(t, "base_alignment_error")
                    else col("base_drift_error"))
        tcp_speed = col("tcp_speed")
        base_speed = col("base_planar_speed")

        out = []
        for i in range(n_envs):
            row = {
                "env_index": i,
                "tcp_to_target_m": float(np.linalg.norm(tcp[i] - tgt[i])),
                "task_tcp_error_m": float(tcp_err[i]),
                "success": bool(success[i]),
                "target_hold_count": int(hold[i]) if hold[i] is not None else None,
                "approach_alignment": float(align[i]) if align[i] is not None else None,
                "base_error_m": float(base_err[i]) if base_err[i] is not None else None,
                "tcp_speed": float(tcp_speed[i]) if tcp_speed[i] is not None else None,
                "base_speed": float(base_speed[i]) if base_speed[i] is not None else None,
            }
            if pinch is not None:
                row["pinch_center_to_target_m"] = float(np.linalg.norm(pinch[i] - tgt[i]))
                row["finger_aperture_m"] = float(aperture[i])
            out.append(row)
        return out


def build_env(args: argparse.Namespace, cfg: dict):
    if args.family == "stage2":
        from tasks.RL_task_parallel_stage2 import make_stage2_vec_env as maker
        kwargs = dict(
            num_envs=args.num_envs,
            target_asset=cfg.get("target_asset", "starlink"),
            env_spacing=cfg.get("env_spacing"),
            n_envs_per_row=cfg.get("n_envs_per_row"),
            center_envs_at_origin=cfg.get("center_envs_at_origin"),
            enable_visual_obs=cfg.get("enable_visual_obs", True),
            image_shape=tuple(cfg.get("image_shape", (3, 84, 84))),
            normalize_images=cfg.get("normalize_images", False),
            enable_target_relative_obs=cfg.get("enable_target_relative_obs"),
            stage2_base_distance=cfg.get("stage2_base_distance", 1.1),
            virtual_grasp_height=cfg.get("stage2_virtual_grasp_height", 0.08),
        )
    else:
        from tasks.RL_task_parallel import make_parallel_vec_env as maker
        kwargs = dict(
            num_envs=args.num_envs,
            target_asset=cfg.get("target_asset", "starlink"),
            env_spacing=cfg.get("env_spacing"),
            n_envs_per_row=cfg.get("n_envs_per_row"),
            center_envs_at_origin=cfg.get("center_envs_at_origin"),
            enable_visual_obs=cfg.get("enable_visual_obs", False),
            image_shape=tuple(cfg.get("image_shape", (3, 84, 84))),
            normalize_images=cfg.get("normalize_images", False),
            target_position_noise=cfg.get("target_position_noise", True),
            target_position_noise_range=tuple(
                cfg.get("target_position_noise_range", (0.03, 0.03, 0.02))),
        )

    noise = args.target_position_noise
    if noise is not None:
        kwargs["target_position_noise"] = noise
    elif "target_position_noise" in cfg and args.family == "stage2":
        kwargs["target_position_noise"] = bool(cfg["target_position_noise"])

    print(f"[eval] family={args.family} env kwargs: "
          f"{json.dumps({k: str(v) for k, v in kwargs.items()})}")
    return maker(**kwargs), kwargs


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[eval] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    import genesis as gs
    SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
    print("[eval] forced SCENE_SETTINGS['renderer'] = BatchRenderer(use_rasterizer=True)")

    cfg_path = args.train_config
    if cfg_path is None:
        cfg_path = args.model_path.resolve().parent.parent / "train_config.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(f"[eval] loaded train_config: {cfg_path}")
    else:
        print(f"[eval] WARNING: no train_config.json at {cfg_path}; using defaults")

    env, env_kwargs = build_env(args, cfg)

    from stable_baselines3 import PPO
    import RL.policies.multimodal_extractor  # noqa: F401  (needed to unpickle)
    model = PPO.load(str(args.model_path), env=env, device=args.device)
    print(f"[eval] loaded model: {args.model_path}")
    print(f"[eval] observation_space: {model.observation_space}")

    task = find_task(env)
    if task is None:
        print("[eval] FATAL: could not locate the RLTask behind the env wrappers")
        return 3
    print(f"[eval] found task: {type(task).__name__}  num_envs={task.num_envs}")
    if args.override_max_steps is not None:
        before = getattr(task, "max_steps", None)
        task.max_steps = int(args.override_max_steps)
        print(f"[eval] override task.max_steps: {before} -> {task.max_steps}")

    # The manipulator is only constructed when the scene is built, which happens
    # on the first reset. Touch the task afterwards, or starlink_manipulator is
    # still None.
    if getattr(task, "starlink_manipulator", None) is None:
        print("[eval] warm-up reset to build the scene before probing the task")
        env.reset()
    probe = TaskProbe(task)

    n = env.num_envs
    all_records: list[dict] = []
    t0 = time.perf_counter()

    for r in range(args.rounds):
        t_round = time.perf_counter()
        obs = env.reset()

        best_pinch = np.full(n, np.inf)
        best_tcp = np.full(n, np.inf)
        # The task's OWN object-facing number. For stage-2 that is
        # `tcp_to_virtual_grasp_error` (= grasp point raised by 8 cm), which is what
        # the shipped 20-checkpoint sweep reported as `min_tcp`. Kept separate so
        # old and new runs can be compared on the identical definition.
        best_task = np.full(n, np.inf)
        ever_success = np.zeros(n, dtype=bool)
        final_pinch = np.full(n, np.nan)
        final_aperture = np.full(n, np.nan)
        ep_return = np.zeros(n)
        grasp_pose_step = np.full(n, -1, dtype=int)
        finished = np.zeros(n, dtype=bool)

        for step in range(args.max_steps):
            actions, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rewards, dones, infos = env.step(actions)
            ep_return += np.asarray(rewards, dtype=float)[:n]
            rows = probe.sample(n)
            # A done env has already been auto-reset by the VecEnv, so its task
            # state now belongs to a fresh episode. Harvest the success flag and
            # then stop touching that env's metrics.
            dones = np.asarray(dones).reshape(-1)[:n].astype(bool)
            for i, row in enumerate(rows):
                if row["success"]:
                    ever_success[i] = True
                if finished[i] or dones[i]:
                    finished[i] = True
                    continue
                if "pinch_center_to_target_m" in row:
                    v = row["pinch_center_to_target_m"]
                    if v < best_pinch[i]:
                        best_pinch[i] = v
                        grasp_pose_step[i] = step
                    final_pinch[i] = v
                    final_aperture[i] = row["finger_aperture_m"]
                best_tcp[i] = min(best_tcp[i], row["tcp_to_target_m"])
                if row.get("task_tcp_error_m") is not None:
                    best_task[i] = min(best_task[i], row["task_tcp_error_m"])
            if finished.all():
                print(f"[eval] round {r+1}: all envs finished at step {step}")
                break

        for i in range(n):
            all_records.append({
                "round": r + 1,
                "env_index": i,
                "success": bool(ever_success[i]),
                "best_pinch_center_error_m": (float(best_pinch[i])
                                              if np.isfinite(best_pinch[i]) else None),
                "final_pinch_center_error_m": (float(final_pinch[i])
                                               if np.isfinite(final_pinch[i]) else None),
                "best_tcp_error_m": (float(best_tcp[i])
                                     if np.isfinite(best_tcp[i]) else None),
                "best_task_tcp_error_m": (float(best_task[i])
                                          if np.isfinite(best_task[i]) else None),
                "step_of_best_pinch": int(grasp_pose_step[i]),
                "final_finger_aperture_m": (float(final_aperture[i])
                                            if np.isfinite(final_aperture[i]) else None),
                "episode_return": float(ep_return[i]),
            })
        n_succ = sum(1 for x in all_records[-n:] if x["success"])
        print(f"[eval] round {r+1}/{args.rounds}: success={n_succ}/{n} "
              f"elapsed={time.perf_counter()-t_round:.1f}s", flush=True)

    total = time.perf_counter() - t0
    thr = args.alignment_threshold
    pinch = [x["best_pinch_center_error_m"] for x in all_records
             if x["best_pinch_center_error_m"] is not None]
    tcp = [x["best_tcp_error_m"] for x in all_records
           if x["best_tcp_error_m"] is not None]
    task_tcp = [x["best_task_tcp_error_m"] for x in all_records
                if x.get("best_task_tcp_error_m") is not None]

    summary = {
        "family": args.family,
        "model_path": str(args.model_path),
        "tree": str(args.tree),
        "num_envs": args.num_envs,
        "rounds": args.rounds,
        "max_steps": args.max_steps,
        "override_max_steps": args.override_max_steps,
        "episodes": len(all_records),
        "alignment_threshold_m": thr,
        "env_kwargs": {k: str(v) for k, v in env_kwargs.items()},
        "task_success_count": sum(1 for x in all_records if x["success"]),
        "task_success_rate": (sum(1 for x in all_records if x["success"]) / len(all_records)
                              if all_records else None),
        "pinch_center_within_threshold_count": sum(1 for v in pinch if v <= thr),
        "pinch_center_within_threshold_rate": (
            sum(1 for v in pinch if v <= thr) / len(pinch) if pinch else None),
        "best_pinch_center_error_m": tensor_stats(pinch),
        "best_tcp_error_m": tensor_stats(tcp),
        "best_task_tcp_error_m": tensor_stats(task_tcp),
        "note_task_tcp_error": ("best_task_tcp_error_m is the task's own "
                                "tcp_to_virtual_grasp_error (= grasp point + 8 cm). "
                                "It is the definition the shipped 20-checkpoint sweep "
                                "reported as min_tcp, so old and new runs compare here."),
        "total_seconds": total,
        "seconds_per_episode": total / len(all_records) if all_records else None,
        "deterministic": not args.stochastic,
    }

    (args.output_dir / "per_episode.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        w.writeheader()
        w.writerows(all_records)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n[eval] wrote {args.output_dir}/summary.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
