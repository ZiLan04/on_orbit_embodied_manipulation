"""Collect successful Starlink-cube demonstrations with bounded regrasp recovery.

This collector is separate from the baseline Oracle.  It retries a failed grasp
at most a configured number of times, so only successful trajectories are used
as VLA supervision while the raw log retains the failure rate for validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS

SCENE_SETTINGS["show_viewer"] = False
TPV_CAM_SETTINGS["enable_recording"] = True
TPV_CAM_SETTINGS["camera"]["res"] = (224, 224)

from delta_eef_control import DeltaEefLimits
from starlink_cube_delta_eef_env import CubeEnvConfig, StarlinkCubeDeltaEefEnv


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float32)


def observation_state(env: StarlinkCubeDeltaEefEnv) -> np.ndarray:
    eef_pos, eef_quat = env._eef_pose()
    gripper = env.last_gripper_open[0].detach().cpu().numpy().reshape(1).astype(np.float32)
    return np.concatenate((
        eef_pos[0].detach().cpu().numpy().astype(np.float32),
        quat_to_rpy(eef_quat[0].detach().cpu().numpy()),
        gripper,
    ))


def scalar(value: torch.Tensor) -> float:
    """Read a single-environment tensor only at a phase boundary."""
    return float(value[0].detach().cpu())


def attempt_metrics(env: StarlinkCubeDeltaEefEnv, cube_origin: torch.Tensor) -> dict[str, float]:
    cube_pos, _ = env._cube_pose()
    grasp_center = env._grasp_center()
    terms = env.last_reward_terms
    return {
        "distance": scalar(terms["distance"]),
        "lift": scalar(terms["lift"]),
        "hold_count": scalar(terms["hold_count"]),
        "cube_drift": scalar(torch.linalg.vector_norm(cube_pos - cube_origin, dim=-1)),
        "xy_distance": scalar(torch.linalg.vector_norm(cube_pos[:, :2] - grasp_center[:, :2], dim=-1)),
        "grasp_center_z": scalar(grasp_center[:, 2]),
    }


def log_transition(
    events: list[dict[str, object]],
    *,
    step: int,
    phase: str,
    next_phase: str,
    reason: str,
    metrics: dict[str, float],
) -> None:
    event: dict[str, object] = {
        "step": step,
        "from": phase,
        "to": next_phase,
        "reason": reason,
        **metrics,
    }
    events.append(event)
    print(f"phase_transition {json.dumps(event, sort_keys=True)}")


def append_attempt_diagnostic(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--approach-tolerance", type=float, default=0.012)
    parser.add_argument("--max-approach-steps", type=int, default=50)
    parser.add_argument("--close-steps", type=int, default=30)
    parser.add_argument("--lift-verify-steps", type=int, default=45)
    parser.add_argument("--retreat-steps", type=int, default=15)
    parser.add_argument("--regrasp-attempts", type=int, default=2)
    parser.add_argument("--safe-retreat-height", type=float, default=0.080)
    parser.add_argument("--recovery-clearance-tolerance", type=float, default=0.015)
    parser.add_argument("--max-recovery-distance", type=float, default=0.120)
    parser.add_argument("--max-recovery-cube-lift", type=float, default=0.080)
    parser.add_argument("--cube-noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", default="grasp and lift the cube")
    args = parser.parse_args()
    if min(args.episodes, args.max_attempts, args.max_steps, args.max_approach_steps, args.close_steps, args.lift_verify_steps, args.retreat_steps) <= 0:
        raise ValueError("episode and phase lengths must be positive")
    if (
        args.approach_tolerance <= 0
        or args.cube_noise_scale <= 0
        or args.regrasp_attempts < 0
        or args.safe_retreat_height <= 0
        or args.recovery_clearance_tolerance <= 0
        or args.max_recovery_distance <= 0
        or args.max_recovery_cube_lift <= 0
    ):
        raise ValueError("recovery arguments must be non-negative and tolerances positive")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.out / "attempt_diagnostics.jsonl"

    limits = DeltaEefLimits(normalized_actions=True)
    base_noise = CubeEnvConfig().cube_spawn_noise_m
    noise = tuple(value * args.cube_noise_scale for value in base_noise)
    env = StarlinkCubeDeltaEefEnv(
        num_envs=1,
        control_limits=limits,
        config=CubeEnvConfig(cube_spawn_noise_m=noise),
    )
    saved = 0
    try:
        if not env.initialize():
            return 1
        for attempt in range(args.max_attempts):
            if saved >= args.episodes:
                break
            env.reset()
            cube_origin, _ = env._cube_pose()
            cube_origin = cube_origin.detach().clone()
            safe_retreat_z = scalar(cube_origin[:, 2]) + args.safe_retreat_height
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            frames: list[np.ndarray] = []
            cube_positions: list[np.ndarray] = []
            phase = "approach"
            phase_steps = 0
            regrasp_count = 0
            failure_reason = "max_steps"
            transition_events: list[dict[str, object]] = []

            for step in range(args.max_steps):
                rendered = env.gsim.cam.render(rgb=True, force_render=True)
                frame = np.asarray(rendered[0] if isinstance(rendered, tuple) else rendered)
                if frame.ndim == 4:
                    frame = frame[0]

                cube_pos, _ = env._cube_pose()
                grasp_center = env._grasp_center()
                distance = torch.linalg.vector_norm(cube_pos - grasp_center, dim=-1)
                action = torch.zeros((1, 7), dtype=env.datatype, device=env.device)

                if phase in {"approach", "recover_descend"}:
                    action[:, :3] = (cube_pos - grasp_center) / limits.max_xyz_step_m
                    action[:, :3].clamp_(-1.0, 1.0)
                    action[:, 6] = 1.0
                elif phase == "recover_align":
                    # Keep the fingers above the cube while correcting XY.  This
                    # prevents a failed closure from sweeping the cube sideways.
                    action[:, :2] = (cube_pos[:, :2] - grasp_center[:, :2]) / limits.max_xyz_step_m
                    action[:, :2].clamp_(-1.0, 1.0)
                    action[:, 2] = torch.clamp(
                        (safe_retreat_z - grasp_center[:, 2]) / limits.max_xyz_step_m,
                        min=0.0,
                        max=1.0,
                    )
                    action[:, 6] = 1.0
                elif phase == "close":
                    action[:, 6] = -1.0
                elif phase == "lift":
                    action[:, 2] = 1.0
                    action[:, 6] = -1.0
                elif phase == "retreat":
                    action[:, 2] = 1.0
                    action[:, 6] = 1.0
                else:
                    raise RuntimeError(f"unknown recovery phase: {phase}")

                states.append(observation_state(env))
                actions.append(action[0].detach().cpu().numpy().astype(np.float32))
                frames.append(frame.astype(np.uint8, copy=False))
                cube_positions.append(cube_pos[0].detach().cpu().numpy().astype(np.float32))
                if not env.step(action=action):
                    raise RuntimeError("recovery oracle rollout step failed")
                phase_steps += 1
                if bool(env.get_done_mask()[0].item()) or bool(env.get_truncated_mask()[0].item()):
                    break
                metrics = attempt_metrics(env, cube_origin)
                current_distance = metrics["distance"]
                current_lift = metrics["lift"]

                if phase in {"retreat", "recover_align", "recover_descend"} and (
                    current_distance > args.max_recovery_distance
                    or current_lift > args.max_recovery_cube_lift
                ):
                    failure_reason = "cube_out_of_recovery_range"
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="abort",
                        reason=failure_reason,
                        metrics=metrics,
                    )
                    break

                if phase == "approach" and (
                    current_distance <= args.approach_tolerance or phase_steps >= args.max_approach_steps
                ):
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="close",
                        reason="aligned" if current_distance <= args.approach_tolerance else "initial_approach_timeout",
                        metrics=metrics,
                    )
                    phase, phase_steps = "close", 0
                elif phase == "recover_descend" and current_distance <= args.approach_tolerance:
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="close",
                        reason="aligned",
                        metrics=metrics,
                    )
                    phase, phase_steps = "close", 0
                elif phase == "recover_descend" and phase_steps >= args.max_approach_steps:
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="retreat",
                        reason="approach_timeout",
                        metrics=metrics,
                    )
                    phase, phase_steps = "retreat", 0
                elif phase == "close" and phase_steps >= args.close_steps:
                    next_phase = "lift" if current_distance <= 0.035 else "retreat"
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase=next_phase,
                        reason="close_ready" if next_phase == "lift" else "close_misaligned",
                        metrics=metrics,
                    )
                    phase, phase_steps = next_phase, 0
                elif phase == "lift" and phase_steps >= args.lift_verify_steps and (current_distance > 0.035 or current_lift < 0.030):
                    if current_lift > args.max_recovery_cube_lift:
                        failure_reason = "cube_lifted_without_stable_grasp"
                        log_transition(
                            transition_events,
                            step=step + 1,
                            phase=phase,
                            next_phase="abort",
                            reason=failure_reason,
                            metrics=metrics,
                        )
                        break
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="retreat",
                        reason="lift_verification_failed",
                        metrics=metrics,
                    )
                    phase, phase_steps = "retreat", 0
                elif phase == "retreat" and phase_steps >= args.retreat_steps and (
                    metrics["grasp_center_z"] >= safe_retreat_z - args.recovery_clearance_tolerance
                ):
                    if regrasp_count >= args.regrasp_attempts:
                        failure_reason = "regrasp_budget_exhausted"
                        log_transition(
                            transition_events,
                            step=step + 1,
                            phase=phase,
                            next_phase="abort",
                            reason=failure_reason,
                            metrics=metrics,
                        )
                        break
                    regrasp_count += 1
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="recover_align",
                        reason="safe_clearance_reached",
                        metrics=metrics,
                    )
                    phase, phase_steps = "recover_align", 0
                elif phase == "recover_align" and (
                    metrics["xy_distance"] <= args.approach_tolerance
                    and metrics["grasp_center_z"] >= safe_retreat_z - args.recovery_clearance_tolerance
                ):
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="recover_descend",
                        reason="xy_aligned_above_cube",
                        metrics=metrics,
                    )
                    phase, phase_steps = "recover_descend", 0
                elif phase == "recover_align" and phase_steps >= args.max_approach_steps:
                    log_transition(
                        transition_events,
                        step=step + 1,
                        phase=phase,
                        next_phase="retreat",
                        reason="recovery_alignment_timeout",
                        metrics=metrics,
                    )
                    phase, phase_steps = "retreat", 0

            if bool(env.get_done_mask()[0].item()):
                path = args.out / f"episode_{saved:06d}.npz"
                np.savez_compressed(
                    path,
                    state=np.stack(states),
                    action=np.stack(actions),
                    base_view=np.stack(frames),
                    cube_position=np.stack(cube_positions),
                    task=np.asarray(args.task),
                )
                print(
                    f"saved_success episode={saved} steps={len(states)} "
                    f"regrasps={regrasp_count} path={path}"
                )
                append_attempt_diagnostic(
                    diagnostics_path,
                    {
                        "attempt": attempt,
                        "outcome": "success",
                        "steps": len(states),
                        "regrasps": regrasp_count,
                        "final_phase": phase,
                        "final": attempt_metrics(env, cube_origin),
                        "transitions": transition_events,
                    },
                )
                saved += 1
            else:
                diagnostic = attempt_metrics(env, cube_origin)
                print(
                    f"discarded_attempt attempt={attempt} steps={len(states)} "
                    f"regrasps={regrasp_count} reason={failure_reason} diagnostic={diagnostic}"
                )
                append_attempt_diagnostic(
                    diagnostics_path,
                    {
                        "attempt": attempt,
                        "outcome": "failure",
                        "reason": failure_reason,
                        "steps": len(states),
                        "regrasps": regrasp_count,
                        "final_phase": phase,
                        "final": diagnostic,
                        "transitions": transition_events,
                    },
                )
        print(f"collection_complete successes={saved} attempts={attempt + 1}")
        return 0 if saved == args.episodes else 3
    finally:
        camera = getattr(env.gsim, "cam", None)
        if camera is not None and getattr(camera, "_recorded_imgs", None):
            camera.stop_recording(save_to_filename=str(args.out / "collection-preview.mp4"), fps=20)
        env.starlink_manipulator.stop()
        env.gsim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
