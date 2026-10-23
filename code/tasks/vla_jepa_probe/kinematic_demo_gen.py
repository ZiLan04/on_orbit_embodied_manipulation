#!/usr/bin/env python3
"""Generate an alignment demonstration dataset with a *kinematic* expert.

Why this exists
---------------
`docs/2026-09-15-session-notes.md` §19/§20 establish, with measurements:

  * the shipped stage-2 PPO checkpoints plateau at 0.25-0.30 m and never align;
  * that is not geometry -- a pure-kinematic scan of 6.14M joint configurations
    finds configurations that put the pinch centre **5.5 mm** from the object,
    with 22 of them inside the 2.5 cm target;
  * it is not a fixable-by-training problem either: the target region occupies
    ~1e-6 of joint space (both the object-facing error and the task's own
    `pose_err`), while an episode is only 120 steps.

So rather than train a policy, this produces the data directly. For each reset it
searches joint space *kinematically* (no dynamics, no controller) for a
configuration that aligns the pinch centre with this episode's object, then
replays a joint-space interpolation to it through the env's **real action
interface**, recording what a policy would see.

An earlier version of this script drove the arm with a Cartesian IK servo. That
was abandoned: the servo diverged (the TCP walked from 0.80 m out to 1.24 m), and
it is unnecessary -- the kinematic search is both faster and exact.

Recorded per environment episode (one .npz each):

    state        (T, state_dim)   what a deployed policy observes
    wrist_image  (T, 3, 84, 84)   what a deployed policy observes (uint8)
    action       (T, action_dim)  the expert's action, in [-1, 1]
    reward       (T,)             the env's own reward
    tcp_pos      (T, 3)           privileged: pinch centre, world frame
    target_pos   (T, 3)           privileged: object grasp point, world frame
    tcp_err      (T,)             ||tcp_pos - target_pos||   <- the metric that matters
    pose_err     (T,)             the task's own criterion quantity
    ik_qpos      (6,)             the searched configuration used by the expert
    success_step (T,)             -1, or the step the hold completed

Gripper note: the stage-2 action space has no gripper dimension (the task forces
the gripper open), so these are *alignment* demonstrations, not close-on-object.

Usage
-----
    python kinematic_demo_gen.py --tree /data/Yanhan/genesis/spaceman_core \
        --num-envs 32 --episodes 4 --max-steps 420 \
        --search-samples-per-env 2000 --refine-iters 120 \
        --target-position-noise --success-threshold 0.025 --hold-steps 10 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-demos
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
    p.add_argument("--base-distance", type=float, default=1.1)
    p.add_argument("--virtual-grasp-height", type=float, default=0.08)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=420,
                   help="The IK expert needed ~416 steps at joint_delta_scale=0.035.")
    p.add_argument("--hold-steps", type=int, default=10)
    p.add_argument("--success-threshold", type=float, default=0.025)
    p.add_argument("--search-samples-per-env", type=int, default=2000)
    p.add_argument("--refine-iters", type=int, default=120)
    p.add_argument("--refine-sigma", type=float, default=0.25)
    p.add_argument("--travel-weight", type=float, default=0.01,
                   help="Penalty per radian of joint travel from the reset pose when "
                        "choosing among aligning configurations. The position "
                        "controller tracks ~0.004 rad/step (not the nominal 0.035), so "
                        "a nearer solution is worth a little alignment error.")
    p.add_argument("--visual-obs", dest="visual_obs", action="store_true", default=True)
    p.add_argument("--no-visual-obs", dest="visual_obs", action="store_false")
    p.add_argument("--target-position-noise", dest="tpn", action="store_true", default=True,
                   help="On by default: with it off every replica is bit-identical "
                        "and the dataset has no diversity.")
    p.add_argument("--no-target-position-noise", dest="tpn", action="store_false")
    p.add_argument("--save-failed", dest="save_failed", action="store_true", default=True)
    p.add_argument("--no-save-failed", dest="save_failed", action="store_false")
    p.add_argument("--seed", type=int, default=1234)
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


class KinematicExpert:
    """Finds joint configurations that align the pinch centre, with no dynamics.

    Uses `set_qpos(qs_idx_local=..., skip_forward=False)` + `get_links_pos(...)`.
    Two Genesis 0.4.3 traps are avoided here (both cost real debugging time):

      * `forward_kinematics()` returns happily but does NOT move the links;
        `set_qpos` does.
      * `get_links_pos(links_idx_local=...)` needs LOCAL indices; global `.idx`
        returns an empty selection and then trips a CUDA device-side assert.
    """

    def __init__(self, task, log):
        self.task = task
        self.robot = task.starlink_manipulator.robot
        self.manip = task.starlink_manipulator
        self.n = task.num_envs
        self.device = task.device
        self.dtype = task.datatype
        self.qs_idx = list(self.manip.motors_qs)
        names = list(task.grasp_finger_link_names)
        self.local_idx = [self.robot.get_link(nm).idx_local for nm in names]
        self.log = log
        self.envs_idx = None

    def bind(self):
        import torch
        self.envs_idx = torch.arange(self.n, dtype=torch.long, device=self.device)

    def pinch_error(self, arm_qpos, target):
        """(n,6) arm joints -> (n,) ||pinch - target||. Pure kinematics."""
        import torch
        self.robot.set_qpos(arm_qpos, qs_idx_local=self.qs_idx,
                            envs_idx=self.envs_idx, zero_velocity=True)
        pos = self.robot.get_links_pos(links_idx_local=self.local_idx,
                                       envs_idx=self.envs_idx).detach()
        pinch = 0.5 * (pos[:, 0, :] + pos[:, 1, :])
        return torch.norm(pinch - target, dim=-1), pinch

    def find(self, target, rng, samples_per_env, refine_iters, sigma0,
             q_ref=None, travel_weight=0.0):
        """Search joint space for a configuration that aligns the pinch centre.

        `travel_weight` adds a penalty on how far the configuration is from
        `q_ref` (the reset pose). This matters a lot in practice: the commanded
        joint rate is `joint_delta_scale` = 0.035 rad/step, but the position
        controller only tracks it at ~0.004 rad/step, so a solution 4 rad away needs
        >1000 steps. Among the many aligning configurations, prefer a nearby one.
        """
        import torch
        n = self.n
        best_q = None
        best_score = None
        best_d = None

        def score_of(q):
            d, _ = self.pinch_error(q, target)
            if q_ref is None or travel_weight <= 0.0:
                return d, d
            travel = torch.norm(q - q_ref, dim=-1)
            return d + travel_weight * travel, d

        batches = max(1, samples_per_env // n)
        for _ in range(batches):
            q = torch.as_tensor(rng.uniform(-np.pi, np.pi, size=(n, 6)),
                                dtype=self.dtype, device=self.device)
            s, d = score_of(q)
            if best_score is None:
                best_q, best_score, best_d = q.clone(), s.clone(), d.clone()
            else:
                better = s < best_score
                best_q[better] = q[better]
                best_score[better] = s[better]
                best_d[better] = d[better]
        # Local refinement, vectorised across envs: each env hill-climbs around
        # its own best with a shrinking perturbation.
        for i in range(refine_iters):
            sigma = sigma0 * (1.0 - i / max(refine_iters, 1)) + 1e-3
            cand = best_q + torch.as_tensor(
                rng.normal(0.0, sigma, size=(n, 6)), dtype=self.dtype, device=self.device)
            s, d = score_of(cand)
            better = s < best_score
            best_q[better] = cand[better]
            best_score[better] = s[better]
            best_d[better] = d[better]
        # Report the pinch error of the FINAL chosen configuration.
        final_d, _ = self.pinch_error(best_q, target)
        return best_q, final_d


def extract_obs(obs, n):
    if isinstance(obs, dict):
        state = np.asarray(obs["state"])[:n]
        img = obs.get("wrist_image")
        img = np.asarray(img)[:n] if img is not None else None
        return state, img
    return np.asarray(obs)[:n], None


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[demo] FATAL: {tree_src} is not a directory")
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
        print(f"[demo] {msg}", flush=True)

    from tasks.RL_task_parallel_stage2 import make_stage2_vec_env
    env = make_stage2_vec_env(
        num_envs=args.num_envs, enable_visual_obs=args.visual_obs,
        enable_target_relative_obs=None, stage2_base_distance=args.base_distance,
        virtual_grasp_height=args.virtual_grasp_height,
        target_position_noise=args.tpn)
    task = find_task(env)
    if task is None:
        log("FATAL: could not locate the task")
        return 3
    original_max_steps = getattr(task, "max_steps", None)
    task.max_steps = 10 ** 9            # episodes are cut by this script, not the env

    env.reset()                          # builds the scene (manipulator is None before)
    n = task.num_envs
    log(f"task={type(task).__name__} num_envs={n} action_space={env.action_space} "
        f"observation_space={env.observation_space}")
    log(f"truncation disabled (max_steps {original_max_steps} -> {task.max_steps}); "
        f"target_position_noise={args.tpn}")

    expert = KinematicExpert(task, log)
    expert.bind()
    scale = float(task.joint_delta_scale)
    log(f"joint_delta_scale={scale}; max joint travel per episode = "
        f"{scale * args.max_steps:.2f} rad")

    rng = np.random.default_rng(args.seed)
    manifest = {
        "task": type(task).__name__,
        "num_envs": n,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "hold_steps": args.hold_steps,
        "success_threshold_m": args.success_threshold,
        "visual_obs": args.visual_obs,
        "target_position_noise": args.tpn,
        "search_samples_per_env": args.search_samples_per_env,
        "refine_iters": args.refine_iters,
        "joint_delta_scale": scale,
        "note": ("Kinematic expert: per-reset joint-space search, then joint-space "
                 "interpolation through the env's real action interface. "
                 "state/wrist_image are exactly what a deployed policy sees; "
                 "privileged labels are stored alongside, not inside, the observation."),
        "episodes_meta": [],
    }

    t_start = time.perf_counter()
    for ep in range(args.episodes):
        obs = env.reset()
        state0, img0 = extract_obs(obs, n)

        # ---- search (kinematic, target per env)
        t_search = time.perf_counter()
        target = task.target_grasp_pose[:, :3].detach()
        # The search calls set_qpos, which MOVES the robot. Capture the reset
        # configuration first and put it back before driving, or the recorded
        # trajectory would start already aligned and would not represent what a
        # policy sees.
        arm_reset = task.arm_joint_positions.detach().clone()
        best_q, best_d = expert.find(target, rng, args.search_samples_per_env,
                                     args.refine_iters, args.refine_sigma,
                                     q_ref=arm_reset,
                                     travel_weight=args.travel_weight)
        expert.robot.set_qpos(arm_reset, qs_idx_local=expert.qs_idx,
                              envs_idx=expert.envs_idx, zero_velocity=True)
        search_s = time.perf_counter() - t_search
        travel = to_np(torch.norm(best_q - arm_reset, dim=-1))
        log(f"episode {ep+1}/{args.episodes}: searched {args.search_samples_per_env} "
            f"configs/env in {search_s:.2f}s; best reachable pinch err "
            f"mean {float(best_d.mean()):.5f} max {float(best_d.max()):.5f} m; "
            f"{int((best_d <= args.success_threshold).sum())}/{n} within "
            f"{args.success_threshold}; joint travel mean {float(travel.mean()):.2f} "
            f"max {float(travel.max()):.2f} rad")

        traj = {"state": [state0], "image": ([img0] if img0 is not None else None),
                "action": [], "reward": [], "tcp_pos": [], "target_pos": [],
                "tcp_err": [], "pose_err": []}
        hold = np.zeros(n, dtype=int)
        done = np.zeros(n, dtype=bool)
        first_success = np.full(n, -1, dtype=int)

        for step in range(args.max_steps):
            import torch
            delta = (best_q - task.arm_joint_positions.detach()) / scale
            action = torch.clamp(delta, -1.0, 1.0)
            obs, rewards, _, _ = env.step(to_np(action))
            state, img = extract_obs(obs, n)
            tcp = to_np(task.tcp_pose[:, :3])
            tgt = to_np(task.target_grasp_pose[:, :3])
            err = np.linalg.norm(tcp - tgt, axis=-1)
            pe = to_np(torch.norm(
                task._get_reference_yaw_link_pose()[0].detach()
                - task.reference_yaw_link_xyz.view(1, 3).expand(n, 3), dim=-1))
            traj["state"].append(state)
            if traj["image"] is not None and img is not None:
                traj["image"].append(img)
            traj["action"].append(to_np(action))
            traj["reward"].append(np.asarray(rewards)[:n])
            traj["tcp_pos"].append(tcp)
            traj["target_pos"].append(tgt)
            traj["tcp_err"].append(err)
            traj["pose_err"].append(pe)
            in_thr = err <= args.success_threshold
            hold = np.where(in_thr, hold + 1, 0)
            newly = (hold >= args.hold_steps) & (~done)
            first_success[newly] = step
            done = done | (hold >= args.hold_steps)
            if done.all():
                break

        # Time must end up on axis 0 so that `[:, i]` selects env i's whole series.
        # An earlier version stacked on axis=1, which put env on axis 0 and time on
        # axis 1; `[:, i]` then took one TIME SLICE across all envs, producing
        # 8-step "trajectories" whose joints looked frozen.
        def stack(key, trim=None):
            a = np.stack(traj[key], axis=0)          # (T', n_envs, ...)
            return a[:trim] if trim is not None else a

        n_steps = len(traj["action"])
        round_start = len(manifest["episodes_meta"])
        for i in range(n):
            reached = bool(done[i])
            if not reached and not args.save_failed:
                continue
            arr = {k: stack(k, trim=n_steps if k != "action" and k != "reward" else None)[:, i]
                   for k in ("state", "action", "reward", "tcp_pos", "target_pos",
                             "tcp_err", "pose_err")}
            if traj["image"] is not None:
                arr["wrist_image"] = stack("image", trim=n_steps)[:, i]
            arr["ik_qpos"] = to_np(best_q[i])
            arr["success"] = np.full(arr["tcp_err"].shape, reached, dtype=bool)
            arr["success_step"] = np.full(arr["tcp_err"].shape, int(first_success[i]),
                                          dtype=np.int32)
            fname = f"ep{ep:03d}_env{i:03d}.npz"
            np.savez_compressed(args.output_dir / fname, **arr)
            manifest["episodes_meta"].append({
                "file": fname, "episode": ep, "env_index": i, "steps": n_steps,
                "success": reached,
                "success_step": int(first_success[i]) if reached else None,
                "best_pinch_err_m": float(arr["tcp_err"].min()),
                "final_pinch_err_m": float(arr["tcp_err"][-1]),
                "expert_best_reachable_m": float(best_d[i]),
            })

        this_round = manifest["episodes_meta"][round_start:]
        if this_round:
            log(f"  -> {int(done.sum())}/{n} reached {args.success_threshold} m for "
                f"{args.hold_steps} steps; best achieved "
                f"{min(m['best_pinch_err_m'] for m in this_round):.5f} m; "
                f"median {np.median([m['best_pinch_err_m'] for m in this_round]):.5f} m")

    manifest["total_seconds"] = time.perf_counter() - t_start
    manifest["episodes_written"] = len(manifest["episodes_meta"])
    manifest["success_count"] = sum(1 for m in manifest["episodes_meta"] if m["success"])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n===== KINEMATIC DEMO GEN SUMMARY =====")
    print(f"episodes written : {manifest['episodes_written']}")
    print(f"successes        : {manifest['success_count']}")
    if manifest["episodes_meta"]:
        b = [m["best_pinch_err_m"] for m in manifest["episodes_meta"]]
        e = [m["expert_best_reachable_m"] for m in manifest["episodes_meta"]]
        print(f"best pinch err   : min={np.min(b):.5f} median={np.median(b):.5f} "
              f"max={np.max(b):.5f}")
        print(f"expert reachable : min={np.min(e):.5f} median={np.median(e):.5f} "
              f"max={np.max(e):.5f}")
    print(f"seconds          : {manifest['total_seconds']:.1f}")
    print(f"\n[demo] wrote {args.output_dir}/manifest.json + per-episode .npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
