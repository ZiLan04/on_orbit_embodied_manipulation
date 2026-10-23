#!/usr/bin/env python3
"""At which target distance does VLA-JEPA's direction stay usable?

The closed loop at a 0.173 m target showed mean cos ~0.15 while the earlier
waypoint pilot at ~0.10 m showed +0.56. Two suspects: the distance itself, and a
scoring artefact -- most action rows are sub-millimetre micro-adjustments whose
direction is noise, and averaging them in dilutes the signal.

So this sweep does two things at once:
  * varies the target distance while holding the arm at the same home pose
  * scores direction ONLY on the big rows (the first row of each chunk), and
    separately reports how the big-row magnitude scales with distance

Record-only: nothing is commanded to the arm from the policy output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jepa_bridge as jb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--jepa-host", default="127.0.0.1")
    p.add_argument("--jepa-port", type=int, default=9012)
    p.add_argument("--instruction", default="put the moka pot on the stove")
    p.add_argument("--distances", type=float, nargs="+",
                   default=[0.08, 0.12, 0.17, 0.22])
    p.add_argument("--repeats", type=int, default=3,
                   help="JEPA draws are stochastic (flow matching); repeat per distance.")
    p.add_argument("--big-step-threshold", type=float, default=0.02,
                   help="Raw |translation| above which a row counts as a real command.")
    p.add_argument("--home-ik-steps", type=int, default=400)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--object-size", type=float, default=0.18)
    p.add_argument("--object-thickness", type=float, default=0.04)
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--no-batch-renderer", action="store_true")
    return p.parse_args()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def cos(a, b):
    ua, ub = unit(a), unit(b)
    return None if ua is None or ub is None else float(np.dot(ua, ub))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_fh = (args.output_dir / "sweep.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[sweep] {msg}", flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    boot_args = argparse.Namespace(
        tree=args.tree, image_size=args.image_size,
        object_size=args.object_size, object_thickness=args.object_thickness,
        no_batch_renderer=args.no_batch_renderer, settle_steps=args.settle_steps,
    )
    sim, manip, target, step_once, pinch_center, gripper_qpos = jb.boot(boot_args, log)
    scene_cam = getattr(manip, "rl_base_camera", None)
    wrist_cam = getattr(manip, "rl_wrist_camera", None)

    client = jb.JepaClient(args.jepa_host, args.jepa_port, log=log)
    log(f"JEPA up: {client.metadata.get('model_id')} on gpu {client.metadata.get('physical_gpu')}")

    manip.update_state()
    home_tcp = jb.vec(manip.ee_global_position).copy()
    home_quat = jb.vec(manip.ee_global_quaternion).copy()
    pinch_start = pinch_center().copy()
    log(f"home_tcp={np.round(home_tcp,4).tolist()} pinch_start={np.round(pinch_start,4).tolist()}")

    def go_home():
        for _ in range(args.home_ik_steps):
            manip.control_joints(home_tcp, home_quat)
            step_once()
        manip.update_state()

    trials = []
    for dist in args.distances:
        go_home()
        pinch_now = pinch_center()
        target.set_pos(pinch_now + np.array([dist, 0.0, 0.0]))
        for _ in range(20):
            step_once()
        manip.update_state()
        obj = jb.vec(target.get_pos())
        pinch_now = pinch_center()
        gap = float(np.linalg.norm(obj - pinch_now))
        to_t = obj - pinch_now
        log(f"--- distance {dist:.3f} m -> actual gap {gap:.4f} m, "
            f"pinch={np.round(pinch_now,4).tolist()} obj={np.round(obj,4).tolist()}")

        for rep in range(args.repeats):
            img_scene = jb.sensor_image(scene_cam, "agentview", log)
            img_wrist = jb.sensor_image(wrist_cam, "wrist", log)
            if img_scene is None or img_wrist is None:
                log("  no images; skipping")
                continue
            fields = client.request_fields(img_scene, img_wrist,
                                           jb.vec(manip.ee_global_position),
                                           jb.vec(manip.ee_global_quaternion),
                                           gripper_qpos(), args.instruction)
            chunk, dt = client.infer(fields)
            mags = np.linalg.norm(chunk[:, :3], axis=1)
            big = mags > args.big_step_threshold
            n_big = int(big.sum())
            big_mean = float(mags[big].mean()) if n_big else None
            # score direction: big rows individually, and their mean
            cs_big = [cos(chunk[i, :3], to_t) for i in range(len(chunk)) if big[i]]
            cs_big = [c for c in cs_big if c is not None]
            cs_all = [cos(chunk[i, :3], to_t) for i in range(len(chunk))]
            cs_all = [c for c in cs_all if c is not None]
            rec = {
                "distance_requested_m": dist, "gap_m": gap, "repeat": rep,
                "n_rows": int(len(chunk)), "n_big_rows": n_big,
                "big_row_magnitudes": mags[big].tolist(),
                "mean_big_row_magnitude": big_mean,
                "cos_big_mean": float(np.mean(cs_big)) if cs_big else None,
                "cos_big_frac_gt_0.3": (float(np.mean([c > 0.3 for c in cs_big]))
                                        if cs_big else None),
                "cos_all_mean": float(np.mean(cs_all)) if cs_all else None,
                "inference_seconds": dt,
            }
            trials.append(rec)
            log(f"  rep{rep}: big={n_big}/{len(chunk)} "
                f"|big|={None if big_mean is None else round(big_mean,4)} "
                f"cos_big={None if rec['cos_big_mean'] is None else round(rec['cos_big_mean'],3)} "
                f"cos_all={None if rec['cos_all_mean'] is None else round(rec['cos_all_mean'],3)}")

    # ---- aggregate per distance ----
    summary = {}
    for dist in args.distances:
        rows = [t for t in trials if t["distance_requested_m"] == dist]
        if not rows:
            continue
        cb = [t["cos_big_mean"] for t in rows if t["cos_big_mean"] is not None]
        ca = [t["cos_all_mean"] for t in rows if t["cos_all_mean"] is not None]
        bm = [t["mean_big_row_magnitude"] for t in rows if t["mean_big_row_magnitude"]]
        summary[f"{dist:.3f}"] = {
            "gap_m": rows[0]["gap_m"],
            "n_big_rows_avg": float(np.mean([t["n_big_rows"] for t in rows])),
            "mean_big_row_magnitude": float(np.mean(bm)) if bm else None,
            "cos_big_mean": float(np.mean(cb)) if cb else None,
            "cos_big_frac_gt_0.3": float(np.mean([t["cos_big_frac_gt_0.3"] for t in rows
                                                 if t["cos_big_frac_gt_0.3"] is not None]))
                                   if cb else None,
            "cos_all_mean": float(np.mean(ca)) if ca else None,
        }

    out = {"trials": trials, "per_distance": summary,
           "big_step_threshold": args.big_step_threshold,
           "repeats": args.repeats}
    (args.output_dir / "distance_sweep.json").write_text(json.dumps(out, indent=2),
                                                         encoding="utf-8")

    log("=== DIRECTION vs TARGET DISTANCE (scored on BIG rows only) ===")
    log(f"{'gap(m)':>8} {'|big|':>8} {'cos_big':>9} {'frac>0.3':>9} {'cos_all':>9}")
    for k, v in summary.items():
        log(f"{v['gap_m']:8.4f} "
            f"{(v['mean_big_row_magnitude'] or 0):8.4f} "
            f"{(v['cos_big_mean'] if v['cos_big_mean'] is not None else float('nan')):9.3f} "
            f"{(v['cos_big_frac_gt_0.3'] if v['cos_big_frac_gt_0.3'] is not None else float('nan')):9.2f} "
            f"{(v['cos_all_mean'] if v['cos_all_mean'] is not None else float('nan')):9.3f}")
    log(f"wrote {args.output_dir / 'distance_sweep.json'}")
    log_fh.close()
    try:
        manip.stop(); sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
