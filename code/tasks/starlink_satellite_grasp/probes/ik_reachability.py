#!/usr/bin/env python3
"""Can the arm actually *reach* the poses this task demands?

Why this exists
---------------
`stage2_action_control.py` showed the stage-2 criterion is NOT action-independent
(extreme scripted joint programs moved `pose_position_error` by 0.32 m), so the
"degenerate reward" explanation is dead. But every one of those programs, and
every one of the 20 shipped PPO checkpoints, plateaus far short:

    program (deliberately extreme)     pose_err_min   quat_sim_max
    zero                                   0.746          0.884
    all_plus                               0.500          0.884
    shoulder_plus                          0.426          0.884
    best scripted program overall          0.426          0.897
    best PPO checkpoint (40M steps)        0.2508         0.944
    required by is_success()               0.055          0.950

A learned policy beating hand-written extremes and *still* missing by 4.6x, with
quaternion similarity saturating just under its threshold, looks less like
"needs more training" than "outside the feasible set".

This script settles it with the repo's own kinematics instead of guessing. It
drives the env through the **real action interface**, with joint targets from
Genesis `inverse_kinematics`, for as many steps as it likes (truncation is
disabled so episode length cannot be the excuse), and reports the best pose
error / quaternion similarity actually attainable.

Two targets are tested, because they are not the same thing:

  --ik-targets reference   IK the *reference link* to reference_yaw_link_xyz /
                           reference_yaw_link_quat -- exactly the pose
                           `is_success()` demands. If this fails, the shipped
                           success criterion is unreachable and no amount of
                           training will satisfy it.
  --ik-targets object      IK the tool to the object grasp point -- what
                           "align the gripper with the object" actually means,
                           and the metric the downstream work is judged on.

For stage-2 it can also sweep `stage2_base_distance`, the single knob that
decides whether the object is inside the arm's workspace at all.

Outputs (in --output-dir):
    ik_reachability.json    one record per (base_distance, ik_target, env)

Usage
-----
    python ik_reachability.py --tree /data/Yanhan/genesis/spaceman_core \
        --family stage2 --base-distances 1.1 0.9 0.7 0.5 \
        --ik-targets reference object --num-envs 4 --ik-steps 400 \
        --output-dir /data/Yanhan/genesis/runs/<ts>-ik-reach
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
    p.add_argument("--family", default="stage2", choices=("stage2", "parallel"))
    p.add_argument("--base-distances", type=float, nargs="+", default=[1.1],
                   help="stage-2 only: stage2_base_distance values to test.")
    p.add_argument("--ik-targets", nargs="+", default=["reference", "object"],
                   choices=("reference", "object"))
    p.add_argument("--virtual-grasp-height", type=float, default=0.08)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--ik-steps", type=int, default=400,
                   help="Control steps per target; truncation is disabled.")
    p.add_argument("--visual-obs", action="store_true",
                   help="Off by default: cameras are overhead for a kinematics question.")
    p.add_argument("--target-position-noise", action="store_true",
                   help="Off by default so every replica faces the identical target.")
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


class Controller:
    """Turns a world-frame pose for one link into an action on the real interface."""

    def __init__(self, task, family: str, ik_link, log):
        self.task = task
        self.robot = task.starlink_manipulator.robot
        self.manip = task.starlink_manipulator
        self.ik_link = ik_link
        self.n = task.num_envs
        self.action_dim = 8 if family == "parallel" else 6
        self.arm_slice = slice(2, 8) if family == "parallel" else slice(0, 6)
        self.scale = float(task.joint_delta_scale)
        self.log = log
        self._logged = False
        self._logged_shapes = False

    def _extract_arm_qpos(self, qpos):
        import torch
        r = self.robot
        m = self.manip
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        k = qpos.shape[-1]
        # motors_dof / motors_qs belong to the Manipulator wrapper, not to the
        # RigidEntity; n_dofs / n_qs belong to the entity.
        if k == r.n_dofs:
            idx, which = m.motors_dof, "motors_dof"
        elif k == r.n_qs:
            idx, which = m.motors_qs, "motors_qs"
        else:
            raise RuntimeError(f"IK trailing dim {k}; n_dofs={r.n_dofs} n_qs={r.n_qs}")
        if not self._logged:
            self.log(f"  IK qpos {tuple(qpos.shape)} -> {which} ({len(idx)} entries)")
            self._logged = True
        arm = qpos[:, idx]
        if arm.shape[-1] != 6:
            raise RuntimeError(f"expected 6 arm joints, got {arm.shape[-1]}")
        if not self._logged_shapes:
            self.log(f"  arm target joints (env0) = {np.round(to_np(arm[0]), 4).tolist()}")
            self.log(f"  arm current joints (env0) = "
                     f"{np.round(to_np(self.task.arm_joint_positions[0]), 4).tolist()}")
            self._logged_shapes = True
        current = self.task.arm_joint_positions.detach()
        delta = (arm.to(current.dtype) - current) / self.scale
        delta = torch.clamp(delta, -1.0, 1.0)
        action = torch.zeros((self.n, self.action_dim), dtype=current.dtype,
                            device=current.device)
        action[:, self.arm_slice] = delta
        return action

    def action_for(self, target_pos, target_quat, compensate_tcp: bool = False,
                   quat_goal: bool = True):
        import torch
        if compensate_tcp:
            target_pos = self._ee_goal_for_tcp(target_pos)
        pos_t = torch.as_tensor(np.asarray(target_pos, dtype=np.float32),
                                dtype=self.manip.datatype, device=self.manip.device)
        quat_t = None
        if quat_goal:
            quat_t = torch.as_tensor(np.asarray(target_quat, dtype=np.float32),
                                     dtype=self.manip.datatype, device=self.manip.device)
        qpos = self.robot.inverse_kinematics(link=self.ik_link, pos=pos_t, quat=quat_t)
        return self._extract_arm_qpos(qpos)

    def _ee_goal_for_tcp(self, target_pos):
        """Goal for the *reference link* that drives the measured TCP onto target.

        tcp is rigidly attached to the reference link, so shifting the reference
        link by delta shifts the TCP by (very nearly) delta. Commanding

            goal = ref_pos + (target - tcp_pos)

        is therefore a unity-gain servo on the measured error, and it needs no
        knowledge of `tcp_offset_local`.

        Why the reference link and not the EE link: an earlier version IK'd
        `params["end_effector"]` instead and the loop diverged, walking the TCP
        from 0.80 m out to 1.24 m before recovering. The reference-link loop is
        the one that converges monotonically here, so both modes share it.
        """
        ref = to_np(self.task._get_reference_yaw_link_pose()[0])
        tcp = to_np(self.task.tcp_pose[:, :3])
        # Damped (gain 0.5) rather than unity: with unity gain the demanded
        # Cartesian step saturates the clipped action every step and the loop
        # limit-cycles instead of settling.
        return ref + 0.5 * (np.asarray(target_pos, dtype=np.float64) - tcp)


def metrics(task):
    """Leaf quantities only; never returns a live task object."""
    import torch
    t = task
    tgt = t.target_grasp_pose[:, :3].detach()
    tgt_quat = t.target_grasp_pose[:, 3:7].detach()
    tcp = t.tcp_pose[:, :3].detach()
    ref_pos, ref_quat = t._get_reference_yaw_link_pose()
    ref_pos = ref_pos.detach()
    ref_quat = ref_quat.detach()
    const_pos = t.reference_yaw_link_xyz.view(1, 3).expand_as(ref_pos)
    const_quat = t.reference_yaw_link_quat.view(1, 4).expand_as(ref_quat)

    out = {
        "tcp_to_target_m": to_np(torch.norm(tcp - tgt, dim=-1)),
        "pose_position_error_m": to_np(torch.norm(ref_pos - const_pos, dim=-1)),
        "pose_quat_similarity": to_np(torch.abs(torch.sum(ref_quat * const_quat, dim=-1))),
        "target_pos": to_np(tgt),
        "target_quat": to_np(tgt_quat),
        "tcp_pos": to_np(tcp),
        "reference_link_pos": to_np(ref_pos),
        "reference_link_quat": to_np(ref_quat),
        "const_pos": to_np(const_pos),
        "const_quat": to_np(const_quat),
    }
    try:
        names = t.grasp_finger_link_names
        f = [to_np(t.starlink_manipulator.robot.get_link(nm).get_pos()) for nm in names]
        pinch = 0.5 * (f[0] + f[1])
        out["pinch_center_pos"] = pinch
        out["finger_a_pos"] = f[0]
        out["finger_b_pos"] = f[1]
        out["pinch_center_to_target_m"] = np.linalg.norm(pinch - out["target_pos"], axis=-1)
        out["finger_aperture_m"] = np.linalg.norm(f[0] - f[1], axis=-1)
    except Exception:  # noqa: BLE001
        pass
    vg = getattr(t, "tcp_to_virtual_grasp_error", None)
    if vg is not None:
        out["virtual_grasp_error_m"] = to_np(vg)
    return out


def run_target(task, env, controller, kind, steps, log):
    n = task.num_envs
    # Object mode: aim the REFERENCE link at a STATIC world point chosen so that
    # the TCP would land on the grasp point. The body->TCP offset is measured once
    # at reset and then held fixed in the world frame.
    #
    # Why once and not per step: the reference-link loop is only stable when its
    # goal is static (that is what works for the `reference` mode). Re-deriving the
    # goal from the live pose every step makes the setpoint chase the state; two
    # earlier attempts did that and limit-cycled (TCP walking out to 1.24 m, then
    # to 1.37 m). A fixed goal plus a fixed orientation keeps the loop well posed.
    offset0 = None
    quat0 = None
    best = {k: np.full(n, np.inf) for k in
            ("tcp_to_target_m", "pose_position_error_m", "pinch_center_to_target_m",
             "virtual_grasp_error_m")}
    best_quat = np.full(n, -np.inf)
    finished = np.zeros(n, dtype=bool)
    first_done = np.full(n, -1, dtype=int)
    final = None

    m = metrics(task)
    if kind == "reference":
        goal_pos, goal_quat = m["const_pos"], m["const_quat"]
    else:
        offset0 = m["tcp_pos"] - m["reference_link_pos"]   # world-frame, at reset
        quat0 = m["reference_link_quat"]
        goal_pos = m["target_pos"] - offset0
        goal_quat = quat0
    log(f"  [{kind}] goal_pos={np.round(goal_pos[0], 4).tolist()}")
    if kind == "object":
        log(f"  [object] held body->TCP offset {np.round(offset0[0], 4).tolist()} "
            f"(|d|={np.linalg.norm(offset0[0]):.4f} m)")

    for step in range(steps):
        if kind == "reference":
            goal_pos = m["const_pos"]
            goal_quat = m["const_quat"]
        else:
            # Target may move if reset noise is on; the offset and orientation stay
            # fixed, so the goal remains a static point relative to the object.
            goal_pos = m["target_pos"] - offset0
        action = controller.action_for(goal_pos, goal_quat, compensate_tcp=False,
                                       quat_goal=True)
        _, _, dones, _ = env.step(to_np(action))
        dones = np.asarray(dones).reshape(-1)[:n].astype(bool)
        m = metrics(task)

        for i in range(n):
            if finished[i]:
                continue
            if dones[i]:
                finished[i] = True
                first_done[i] = step
                continue
            best["tcp_to_target_m"][i] = min(best["tcp_to_target_m"][i], m["tcp_to_target_m"][i])
            best["pose_position_error_m"][i] = min(
                best["pose_position_error_m"][i], m["pose_position_error_m"][i])
            best_quat[i] = max(best_quat[i], m["pose_quat_similarity"][i])
            for k in ("pinch_center_to_target_m", "virtual_grasp_error_m"):
                if k in m:
                    best[k][i] = min(best[k][i], m[k][i])
        final = m

        if step % 50 == 0 or step == steps - 1:
            log(f"  [{kind}] step {step:4d}: tcp->target "
                f"{np.round(m['tcp_to_target_m'], 4).tolist()} pose_err "
                f"{np.round(m['pose_position_error_m'], 4).tolist()} quat_sim "
                f"{np.round(m['pose_quat_similarity'], 4).tolist()}")
        if finished.all():
            log(f"  [{kind}] all envs done at step {step}")
            break

    return {
        "best": best,
        "best_quat": best_quat,
        "finished": finished,
        "first_done": first_done,
        "final": final,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        print(f"[ik] FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"GUI": False})

    import genesis as gs
    SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)

    def log(msg):
        print(f"[ik] {msg}", flush=True)

    records: list[dict] = []
    distances = args.base_distances if args.family == "stage2" else [None]

    for dist in distances:
        if args.family == "stage2":
            from tasks.RL_task_parallel_stage2 import make_stage2_vec_env
            kwargs = dict(num_envs=args.num_envs, enable_visual_obs=args.visual_obs,
                          enable_target_relative_obs=None,
                          stage2_base_distance=dist,
                          virtual_grasp_height=args.virtual_grasp_height,
                          target_position_noise=args.target_position_noise)
            maker = make_stage2_vec_env
        else:
            from tasks.RL_task_parallel import make_parallel_vec_env
            kwargs = dict(num_envs=args.num_envs, enable_visual_obs=args.visual_obs,
                          target_position_noise=args.target_position_noise)
            maker = make_parallel_vec_env

        log(f"building env: family={args.family} base_distance={dist}")
        t_build = time.perf_counter()
        env = maker(**kwargs)
        task = find_task(env)
        if task is None:
            log("FATAL: could not locate the task")
            return 3
        log(f"env built in {time.perf_counter()-t_build:.1f}s; task={type(task).__name__}")

        original_max_steps = getattr(task, "max_steps", None)
        task.max_steps = 10 ** 9
        log(f"truncation disabled (max_steps {original_max_steps} -> {task.max_steps})")

        for kind in args.ik_targets:
            env.reset()
            # Both modes IK the *reference* (gripper-body) link: that is the loop
            # that converges monotonically in this scene. What differs is the goal.
            ik_link = task.starlink_manipulator.robot.get_link(
                task.reference_yaw_link_name)
            label = "reference_yaw_link"
            log(f"--- target={kind} (IK on {label}, family={args.family}) ---")
            controller = Controller(task, args.family, ik_link, log)
            m0 = metrics(task)
            log(f"  initial: tcp->target {np.round(m0['tcp_to_target_m'], 4).tolist()}  "
                f"pose_err {np.round(m0['pose_position_error_m'], 4).tolist()}  "
                f"quat_sim {np.round(m0['pose_quat_similarity'], 4).tolist()}")
            log(f"  geom env0: target={np.round(m0['target_pos'][0], 4).tolist()} "
                f"tcp={np.round(m0['tcp_pos'][0], 4).tolist()} "
                f"ref_link={np.round(m0['reference_link_pos'][0], 4).tolist()} "
                f"const={np.round(m0['const_pos'][0], 4).tolist()}")
            if "pinch_center_pos" in m0:
                log(f"  geom env0: pinch={np.round(m0['pinch_center_pos'][0], 4).tolist()} "
                    f"finger_a={np.round(m0['finger_a_pos'][0], 4).tolist()} "
                    f"finger_b={np.round(m0['finger_b_pos'][0], 4).tolist()} "
                    f"aperture={m0['finger_aperture_m'][0]:.4f}")
            if "pinch_center_to_target_m" in m0:
                log(f"  initial pinch->target "
                    f"{np.round(m0['pinch_center_to_target_m'], 4).tolist()}")

            # One failing target must not abandon the remaining distances/targets.
            try:
                res = run_target(task, env, controller, kind, args.ik_steps, log)
            except Exception as exc:  # noqa: BLE001
                log(f"  FAILED target={kind} base_distance={dist}: {exc!r}")
                import traceback
                traceback.print_exc()
                continue
            n = task.num_envs
            for i in range(n):
                rec = {
                    "family": args.family,
                    "base_distance": dist,
                    "ik_target": kind,
                    "ik_link": label,
                    "env_index": i,
                    "ik_steps": args.ik_steps,
                    "initial_tcp_to_target_m": float(m0["tcp_to_target_m"][i]),
                    "initial_pose_position_error_m": float(m0["pose_position_error_m"][i]),
                    "initial_pose_quat_similarity": float(m0["pose_quat_similarity"][i]),
                    "best_tcp_to_target_m": float(res["best"]["tcp_to_target_m"][i]),
                    "best_pose_position_error_m": float(res["best"]["pose_position_error_m"][i]),
                    "best_pose_quat_similarity": float(res["best_quat"][i]),
                    "final_tcp_to_target_m": float(res["final"]["tcp_to_target_m"][i]),
                    "final_pose_position_error_m": float(res["final"]["pose_position_error_m"][i]),
                    "final_pose_quat_similarity": float(res["final"]["pose_quat_similarity"][i]),
                    "task_success": bool(res["finished"][i]),
                    "task_success_step": int(res["first_done"][i]),
                    "target_pos": [float(x) for x in m0["target_pos"][i]],
                    "tcp_start_pos": [float(x) for x in m0["tcp_pos"][i]],
                    "reference_link_start_pos": [float(x) for x in
                                                 m0["reference_link_pos"][i]],
                    "const_pos": [float(x) for x in m0["const_pos"][i]],
                }
                for k in ("pinch_center_to_target_m", "virtual_grasp_error_m"):
                    v = res["best"].get(k)
                    if v is not None and np.isfinite(v[i]):
                        rec["best_" + k] = float(v[i])
                rec["meets_success_position"] = bool(
                    res["best"]["pose_position_error_m"][i] <= 0.055)
                rec["meets_success_quat"] = bool(res["best_quat"][i] >= 0.95)
                rec["meets_both"] = bool(rec["meets_success_position"]
                                         and rec["meets_success_quat"])
                rec["meets_pinch_025"] = bool(
                    rec.get("best_pinch_center_to_target_m", float("inf")) <= 0.025)
                records.append(rec)

            log(f"  RESULT [{kind}] base_distance={dist}: best_pose_err "
                f"{np.round(res['best']['pose_position_error_m'], 4).tolist()} best_quat "
                f"{np.round(res['best_quat'], 4).tolist()} success "
                f"{res['finished'].tolist()}")
        env.close()

    (args.output_dir / "ik_reachability.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")

    print("\n===== IK REACHABILITY SUMMARY =====")
    hdr = (f"{'base':>6} {'target':>10} {'env':>3} {'start_tcp':>10} {'best_tcp':>9} "
           f"{'best_pose':>10} {'best_quat':>10} {'pinch':>8} {'pose_ok':>8} "
           f"{'quat_ok':>8} {'SUCCESS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(f"{str(r['base_distance']):>6} {r['ik_target']:>10} {r['env_index']:>3} "
              f"{r['initial_tcp_to_target_m']:10.4f} {r['best_tcp_to_target_m']:9.4f} "
              f"{r['best_pose_position_error_m']:10.4f} "
              f"{r['best_pose_quat_similarity']:10.5f} "
              f"{r.get('best_pinch_center_to_target_m', float('nan')):8.4f} "
              f"{str(r['meets_success_position']):>8} {str(r['meets_success_quat']):>8} "
              f"{str(r['task_success']):>8}")
    print(f"\n[ik] wrote {args.output_dir}/ik_reachability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
