#!/usr/bin/env python3
"""Health check for the StarVLA policy server.

Verifies, without any Genesis involvement:
  1. the server accepts a connection and sends its handshake metadata
  2. a request is accepted and returns a finite action chunk of the expected shape
  3. round-trip latency, so we know what the control loop has to budget for

It uses StarVLA's OWN client (deployment.model_server.tools.websocket_policy_client)
so the wire format is authoritative rather than re-implemented.

Run on the policy host with the repo on PYTHONPATH:

    cd /data/VLA_model/StarVLA/repo
    PYTHONPATH=$PWD /data/anaconda3/envs/starvla_env/bin/python \
        /path/to/starvla_healthcheck.py --host 127.0.0.1 --port 6694
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
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6694)
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--instruction", default="pick up the block")
    p.add_argument("--image-size", type=int, default=224,
                   help="Square RGB observation size. The checkpoint config says 224.")
    p.add_argument("--unnorm-key", default="oxe_bridge",
                   help="Required: the deployed ckpt has default_unnorm_key=None.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    except Exception as exc:  # noqa: BLE001
        print(f"[health] FATAL: cannot import StarVLA's client: {exc!r}")
        print("[health] hint: run with PYTHONPATH pointing at the StarVLA repo root")
        return 2

    print(f"[health] connecting to ws://{args.host}:{args.port} ...")
    t0 = time.perf_counter()
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    connect_seconds = time.perf_counter() - t0

    meta = client.get_server_metadata()
    print(f"[health] connected in {connect_seconds:.3f}s")
    print("[health] ===== handshake metadata =====")
    for key in sorted(meta.keys()):
        print(f"[health]   {key} = {meta[key]!r}")

    expected_chunk = meta.get("action_chunk_size")
    available = meta.get("available_unnorm_keys") or []
    default_key = meta.get("default_unnorm_key")

    # Build a synthetic observation in the exact shape the bridge proved works.
    side = args.image_size
    image = np.zeros((side, side, 3), dtype=np.uint8)
    state = np.zeros((1, 7), dtype=np.float32)

    results = []
    for i in range(args.requests):
        query = {
            "examples": [{
                "image": [image],
                "lang": args.instruction,
                "state": state,
            }],
            "unnorm_key": args.unnorm_key,
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": 10,
        }
        t = time.perf_counter()
        try:
            response = client.predict_action(query)
        except Exception as exc:  # noqa: BLE001
            print(f"[health] request {i + 1} FAILED: {exc!r}")
            results.append({"request": i + 1, "ok": False, "error": repr(exc)})
            continue
        elapsed = time.perf_counter() - t

        payload = response.get("data", response)
        actions = np.asarray(payload.get("actions"), dtype=np.float64)
        finite = bool(np.isfinite(actions).all())
        rec = {
            "request": i + 1,
            "ok": True,
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "all_finite": finite,
            "min": float(actions.min()),
            "max": float(actions.max()),
            "round_trip_seconds": elapsed,
            "response_top_level_keys": sorted(response.keys()),
        }
        results.append(rec)
        print(f"[health] request {i + 1}: shape={actions.shape} finite={finite} "
              f"range=[{actions.min():.5f}, {actions.max():.5f}] "
              f"latency={elapsed:.3f}s")

    client.close()

    first_ok = next((r for r in results if r.get("ok")), None)
    verdict = {
        "server_reachable": True,
        "connect_seconds": connect_seconds,
        "metadata": {k: (v if isinstance(v, (str, int, float, bool, list, type(None))) else repr(v))
                     for k, v in meta.items()},
        "action_chunk_size_from_metadata": expected_chunk,
        "available_unnorm_keys": available,
        "default_unnorm_key": default_key,
        "unnorm_key_used": args.unnorm_key,
        "instruction_used": args.instruction,
        "requests": results,
        "shape_matches_metadata": bool(
            first_ok and expected_chunk is not None
            and first_ok["shape"][:2] == [1, expected_chunk]
        ),
        "warm_request_seconds": (results[-1].get("round_trip_seconds")
                                 if results and results[-1].get("ok") else None),
    }

    print("\n[health] ===== VERDICT =====")
    print(json.dumps(verdict, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "starvla_healthcheck.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")
        print(f"[health] wrote {args.output_dir / 'starvla_healthcheck.json'}")

    return 0 if (first_ok and first_ok["all_finite"]) else 1


if __name__ == "__main__":
    sys.exit(main())
