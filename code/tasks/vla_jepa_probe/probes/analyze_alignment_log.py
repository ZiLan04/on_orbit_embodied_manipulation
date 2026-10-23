#!/usr/bin/env python3
"""Analyse alignment-run logs to decide whether the policy's commands are
DIRECTIONALLY INFORMATIVE.

Why this exists
---------------
The oracle already reaches pinch_center_to_object = 1.35 cm on this scene, so
the arm, IK, cameras and control stack can align. When StarVLA then stalls at
~5.4 cm, the open question is whether its commands carry usable direction -- or
whether it is just oscillating.

Pre-compensating uninformative commands with a Jacobian amplifies noise, so this
question must be settled BEFORE building any compensation.

This script only reads bridge-run.jsonl. No simulator, no GPU, no policy server.
Direction is reconstructed from the logged post-clamp targets, which is exactly
what the controller was asked to track:

    applied_i      = target_position[i] - target_position[i-1]
    to_target_i    = placed_target    - target_position[i]
    cos_i          = cos(angle(applied_i, to_target_i))

Also reports the classic "vibrating not moving" signature:
total commanded path length vs net displacement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def flat3(v) -> list:
    """The bridge logs batched arrays ([[x,y,z]]); reduce to a flat triple."""
    while isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        v = v[0]
    return [float(x) for x in v][:3]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def analyse(run_dir: Path) -> dict:
    log = run_dir / "bridge-run.jsonl"
    rec = {"run": run_dir.name, "has_log": log.exists()}
    if not log.exists():
        return rec

    records = load(log)
    placed = None
    for r in records:
        if r.get("event") == "alignment_target_placed":
            placed = flat3(r.get("placed")) if r.get("placed") else None
    actions = [r for r in records if r.get("event") == "action"]

    rec["n_action_records"] = len(actions)
    rec["target_placed"] = placed
    if not actions:
        return rec

    positions = [flat3(r["target_position"]) for r in actions if r.get("target_position")]
    errors = [r.get("pinch_center_error_m") for r in actions
              if r.get("pinch_center_error_m") is not None]
    raw = [r.get("raw_action") for r in actions if r.get("raw_action")]

    if errors:
        rec["initial_error_m"] = errors[0]
        rec["min_error_m"] = min(errors)
        rec["final_error_m"] = errors[-1]
        rec["argmin_step"] = int(errors.index(min(errors)))
        # crude monotonicity: how often the error decreased
        dec = sum(1 for a, b in zip(errors, errors[1:]) if b < a)
        rec["error_decrease_fraction"] = dec / max(1, len(errors) - 1)

    if len(positions) >= 2:
        path = sum(norm(sub(positions[i], positions[i - 1]))
                   for i in range(1, len(positions)))
        net = norm(sub(positions[-1], positions[0]))
        rec["commanded_path_m"] = path
        rec["commanded_net_m"] = net
        rec["path_over_net"] = (path / net) if net > 1e-9 else None

    # The decisive number: is the applied motion aimed at the target?
    if placed and len(positions) >= 2:
        cosines = []
        for i in range(1, len(positions)):
            applied = sub(positions[i], positions[i - 1])
            to_target = sub(placed, positions[i])
            c = cosine(applied, to_target)
            if c is not None:
                cosines.append(c)
        if cosines:
            pos_frac = sum(1 for c in cosines if c > 0) / len(cosines)
            good_frac = sum(1 for c in cosines if c > 0.3) / len(cosines)
            bad_frac = sum(1 for c in cosines if c < 0) / len(cosines)
            rec["direction"] = {
                "n": len(cosines),
                "mean_cos": sum(cosines) / len(cosines),
                "median_cos": percentile(cosines, 50),
                "p10_cos": percentile(cosines, 10),
                "p90_cos": percentile(cosines, 90),
                "fraction_cos_positive": pos_frac,
                "fraction_cos_gt_0.3": good_frac,
                "fraction_cos_negative": bad_frac,
            }

    # How much of the raw action is actually translation?
    if raw:
        import statistics
        tx = [r[0] for r in raw if len(r) >= 3]
        ty = [r[1] for r in raw if len(r) >= 3]
        tz = [r[2] for r in raw if len(r) >= 3]
        rec["raw_translation"] = {
            "n": len(raw),
            "mean_abs_x": statistics.fmean(abs(v) for v in tx) if tx else None,
            "mean_abs_y": statistics.fmean(abs(v) for v in ty) if ty else None,
            "mean_abs_z": statistics.fmean(abs(v) for v in tz) if tz else None,
            "sign_flips_x": sum(1 for a, b in zip(tx, tx[1:]) if a * b < 0),
            "sign_flips_y": sum(1 for a, b in zip(ty, ty[1:]) if a * b < 0),
            "sign_flips_z": sum(1 for a, b in zip(tz, tz[1:]) if a * b < 0),
        }
        if raw[0] and len(raw[0]) >= 7:
            grip = [r[6] for r in raw if len(r) >= 7]
            rec["raw_gripper"] = {"first": grip[0], "last": grip[-1],
                                  "mean": statistics.fmean(grip) if grip else None}
    return rec


def verdict(rec: dict) -> str:
    d = rec.get("direction")
    if not d:
        return "insufficient data"
    if d["fraction_cos_gt_0.3"] < 0.25:
        return ("UNINFORMATIVE: commanded motion rarely points at the target "
                f"(only {d['fraction_cos_gt_0.3']:.0%} of steps have cos>0.3). "
                "Jacobian pre-compensation would amplify noise, not intent.")
    if d["fraction_cos_gt_0.3"] < 0.6:
        return ("WEAK/MIXED: commands are partly directed; a damped compensation "
                "might help, but examine the oscillation first.")
    return ("DIRECTED: commands mostly aim at the target, so the shortfall is more "
            "likely magnitude/tracking than intent.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path,
                    help="Run directories (each with bridge-run.jsonl).")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    results = []
    for run in args.runs:
        rec = analyse(run)
        rec["verdict"] = verdict(rec)
        results.append(rec)

    for rec in results:
        print("=" * 78)
        print(f"RUN  {rec['run']}")
        if not rec.get("has_log") or not rec.get("n_action_records"):
            print("  no usable action records")
            continue
        print(f"  action records      : {rec['n_action_records']}")
        print(f"  target placed       : {[round(v, 4) for v in rec['target_placed']] if rec.get('target_placed') else None}")
        for k in ("initial_error_m", "min_error_m", "final_error_m", "argmin_step",
                  "error_decrease_fraction", "commanded_path_m", "commanded_net_m",
                  "path_over_net"):
            if rec.get(k) is not None:
                v = rec[k]
                print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
        d = rec.get("direction")
        if d:
            print("  --- direction of commanded motion vs direction to target ---")
            for k, v in d.items():
                print(f"    {k:<24}: {v:.4f}" if isinstance(v, float) else f"    {k:<24}: {v}")
        rt = rec.get("raw_translation")
        if rt:
            print(f"  raw translation |x|={rt['mean_abs_x']:.5f} |y|={rt['mean_abs_y']:.5f} "
                  f"|z|={rt['mean_abs_z']:.5f}")
            print(f"  sign flips  x={rt['sign_flips_x']} y={rt['sign_flips_y']} "
                  f"z={rt['sign_flips_z']}  (of {rt['n']} steps)")
        if rec.get("raw_gripper"):
            g = rec["raw_gripper"]
            print(f"  gripper action      : first={g['first']:.3f} last={g['last']:.3f} mean={g['mean']:.3f}")
        print(f"  VERDICT: {rec['verdict']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
