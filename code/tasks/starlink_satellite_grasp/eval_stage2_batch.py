#!/usr/bin/env python3
"""Batched stage-2 evaluation for the satellite grasp-pose PPO policies.

Why this exists
---------------
The upstream `src/test/replay_stage2_policy.py` forces `num_envs=1` and, on the
canonical revision (e042566), crashes at reset with:

    Initialization failed: Pos shape torch.Size([1, 3]) does not match (N, 3)

This script takes the other road: it evaluates on the *batched* code path that
training itself uses (`make_stage2_vec_env`, training used num_envs=100), which
sidesteps that broken branch and is far cheaper per episode.

It never writes inside the vendored spaceman_core tree; all output goes to
--output-dir.

Usage
-----
    python eval_stage2_batch.py \
        --tree /data/Yanhan/genesis/spaceman_core \
        --model-path .../checkpoints/ppo_satellite_stage2_arm_40000000_steps.zip \
        --num-envs 25 --rounds 2 --max-steps 200 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-batch-eval/40M
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


# Keys the stage-2 env publishes in `info` (see RL_task_parallel_stage2.py:774).
SCALAR_KEYS = (
    "tcp_to_virtual_grasp_error",
    "pose_position_error",
    "pose_quat_similarity",
    "reference_link_speed",
    "base_drift_error",
    "target_hold_count",
)
BOOL_KEYS = ("success", "target_in_wrist_view")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path,
                   help="Path to the spaceman_core checkout (must contain src/).")
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--train-config", type=Path, default=None,
                   help="Defaults to <model dir>/../train_config.json")
    p.add_argument("--num-envs", type=int, default=25)
    p.add_argument("--rounds", type=int, default=2,
                   help="Episodes collected = num_envs * rounds")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Stage-2 training used 200 steps per episode.")
    p.add_argument("--target-position-noise", action="store_true", default=None)
    p.add_argument("--no-target-position-noise", dest="target_position_noise",
                   action="store_false")
    p.add_argument("--stochastic", action="store_true",
                   help="Default is deterministic action selection.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-batch-renderer", action="store_true",
                   help="Do NOT force BatchRenderer on. By default we enable it; "
                        "see the comment in main().")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def load_train_config(args: argparse.Namespace) -> dict:
    path = args.train_config
    if path is None:
        path = args.model_path.resolve().parent.parent / "train_config.json"
    if not path.exists():
        print(f"[eval] WARNING: no train_config.json at {path}; using defaults")
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    print(f"[eval] loaded train_config: {path}")
    return cfg


def build_env(args: argparse.Namespace, cfg: dict):
    """Reconstruct the training env as faithfully as the config allows."""
    from tasks.RL_task_parallel_stage2 import make_stage2_vec_env

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
    noise = args.target_position_noise
    if noise is None:
        noise = bool(cfg.get("target_position_noise", False))
    kwargs["target_position_noise"] = noise
    if "target_position_noise_range" in cfg:
        kwargs["target_position_noise_range"] = tuple(cfg["target_position_noise_range"])

    print(f"[eval] env kwargs: {json.dumps({k: str(v) for k, v in kwargs.items()}, indent=2)}")
    return make_stage2_vec_env(**kwargs), kwargs


def load_model(args: argparse.Namespace, env):
    from stable_baselines3 import PPO

    # The visual policy uses a custom feature extractor; it must be importable
    # for the pickled policy_kwargs to resolve.
    import RL.policies.multimodal_extractor  # noqa: F401

    model = PPO.load(str(args.model_path), env=env, device=args.device)
    print(f"[eval] loaded model: {args.model_path}")
    print(f"[eval] observation_space: {model.observation_space}")
    print(f"[eval] action_space     : {model.action_space}")
    return model


def rollout_round(env, model, max_steps: int, deterministic: bool):
    """Run one episode per env; return a list of per-env record dicts."""
    obs = env.reset()
    n = env.num_envs

    ever_success = np.zeros(n, dtype=bool)
    first_success = np.full(n, -1, dtype=int)
    min_tcp = np.full(n, np.inf)
    min_pose = np.full(n, np.inf)
    best_quat = np.full(n, -np.inf)
    final_tcp = np.full(n, np.nan)
    final_pose = np.full(n, np.nan)
    final_quat = np.full(n, np.nan)
    wrist_visible = np.zeros(n, dtype=bool)
    base_drift = np.full(n, np.nan)
    ep_return = np.zeros(n)

    for step in range(max_steps):
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)
        for i, info in enumerate(infos):
            if i >= n:
                break
            ep_return[i] += float(rewards[i])
            tcp = info.get("tcp_to_virtual_grasp_error")
            pose = info.get("pose_position_error")
            quat = info.get("pose_quat_similarity")
            if tcp is not None and np.isfinite(tcp):
                min_tcp[i] = min(min_tcp[i], float(tcp))
                final_tcp[i] = float(tcp)
            if pose is not None and np.isfinite(pose):
                min_pose[i] = min(min_pose[i], float(pose))
                final_pose[i] = float(pose)
            if quat is not None and np.isfinite(quat):
                best_quat[i] = max(best_quat[i], float(quat))
                final_quat[i] = float(quat)
            if bool(info.get("success", False)) and not ever_success[i]:
                ever_success[i] = True
                first_success[i] = step + 1
            if bool(info.get("target_in_wrist_view", False)):
                wrist_visible[i] = True
            drift = info.get("base_drift_error")
            if drift is not None and np.isfinite(drift):
                base_drift[i] = float(drift)

    records = []
    for i in range(n):
        records.append({
            "env_index": i,
            "success": bool(ever_success[i]),
            "first_success_step": int(first_success[i]),
            "final_tcp_to_grasp_error": float(final_tcp[i]) if np.isfinite(final_tcp[i]) else None,
            "min_tcp_to_grasp_error": float(min_tcp[i]) if np.isfinite(min_tcp[i]) else None,
            "final_pose_position_error": float(final_pose[i]) if np.isfinite(final_pose[i]) else None,
            "min_pose_position_error": float(min_pose[i]) if np.isfinite(min_pose[i]) else None,
            "final_quat_similarity": float(final_quat[i]) if np.isfinite(final_quat[i]) else None,
            "best_quat_similarity": float(best_quat[i]) if np.isfinite(best_quat[i]) else None,
            "ever_target_in_wrist_view": bool(wrist_visible[i]),
            "base_drift_error": float(base_drift[i]) if np.isfinite(base_drift[i]) else None,
            "episode_return": float(ep_return[i]),
        })
    return records


def summarise(records: list[dict], threshold: float) -> dict:
    n = len(records)
    succ = [r for r in records if r["success"]]
    final_tcp = [r["final_tcp_to_grasp_error"] for r in records
                 if r["final_tcp_to_grasp_error"] is not None]
    min_tcp = [r["min_tcp_to_grasp_error"] for r in records
               if r["min_tcp_to_grasp_error"] is not None]
    pose = [r["final_pose_position_error"] for r in records
            if r["final_pose_position_error"] is not None]
    quat = [r["best_quat_similarity"] for r in records if r["best_quat_similarity"] is not None]
    view = [r for r in records if r["ever_target_in_wrist_view"]]

    def stats(xs):
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

    return {
        "episodes": n,
        "success_threshold_m": threshold,
        "success_count": len(succ),
        "success_rate": (len(succ) / n) if n else None,
        "ever_target_in_wrist_view_count": len(view),
        "ever_target_in_wrist_view_rate": (len(view) / n) if n else None,
        "final_tcp_to_grasp_error_m": stats(final_tcp),
        "min_tcp_to_grasp_error_m": stats(min_tcp),
        "final_pose_position_error_m": stats(pose),
        "best_quat_similarity": stats(quat),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[eval] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    # Genesis must run headless here. The vendored configs default to
    # show_viewer=True, and envs/genesis_env.py passes SCENE_SETTINGS straight
    # into gs.Scene(...), which then tries to open a display and dies with
    # "No display detected" on this host. The upstream replay script does the
    # same override from its --no-viewer flag.
    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    # The canonical configs.py ships with BatchRenderer commented out
    # ("renderer": None). Without it Genesis falls back to Raytracer/Rasterizer,
    # which can only bind a camera to ONE env -- it logs
    #   "Raytracer and Rasterizer requires binding to the camera with a specific
    #    environment index. Defaulting to 'rendered_envs_idx[0]'."
    # and the camera ends up non-batched. The batched RL camera code then calls
    # set_pose(pos=(N, 3)) and dies in genesis/vis/camera.py:637 with
    #   "Pos shape torch.Size([N, 3]) does not match (N, 3)"
    # at env reset. Our own configs.py comment documents that BatchRenderer is
    # what makes the RL cameras render all parallel envs in one batched call.
    # We enable it here rather than editing the vendored tree.
    if not args.no_batch_renderer:
        import genesis as gs
        SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
        print("[eval] forced SCENE_SETTINGS['renderer'] = BatchRenderer(use_rasterizer=True)")

    print(f"[eval] headless override: show_viewer={SCENE_SETTINGS.get('show_viewer')} "
          f"enable_recording={TPV_CAM_SETTINGS.get('enable_recording')}")

    cfg = load_train_config(args)
    env, env_kwargs = build_env(args, cfg)
    model = load_model(args, env)

    threshold = 0.025
    print(f"[eval] running {args.rounds} round(s) x {args.num_envs} envs "
          f"x {args.max_steps} steps")

    all_records: list[dict] = []
    t0 = time.perf_counter()
    for r in range(args.rounds):
        t_round = time.perf_counter()
        recs = rollout_round(env, model, args.max_steps, not args.stochastic)
        for rec in recs:
            rec["round"] = r + 1
        all_records.extend(recs)
        n_succ = sum(1 for x in recs if x["success"])
        print(f"[eval] round {r + 1}/{args.rounds}: episodes={len(recs)} "
              f"success={n_succ} elapsed={time.perf_counter() - t_round:.1f}s", flush=True)

    total = time.perf_counter() - t0
    summary = summarise(all_records, threshold)
    summary["total_seconds"] = total
    summary["seconds_per_episode"] = total / len(all_records) if all_records else None
    summary["num_envs"] = args.num_envs
    summary["rounds"] = args.rounds
    summary["max_steps"] = args.max_steps
    summary["model_path"] = str(args.model_path)
    summary["tree"] = str(args.tree)
    summary["env_kwargs"] = {k: str(v) for k, v in env_kwargs.items()}
    summary["deterministic"] = not args.stochastic

    (args.output_dir / "per_episode.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    keys = list(all_records[0].keys()) if all_records else []
    with (args.output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_records)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n[eval] wrote: {args.output_dir}/summary.json, per_episode.csv")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
