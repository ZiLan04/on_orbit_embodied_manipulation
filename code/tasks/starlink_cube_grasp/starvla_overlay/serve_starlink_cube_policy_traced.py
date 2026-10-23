"""Serve the Starlink-cube checkpoint with action-space instrumentation.

Identical to serve_starlink_cube_policy.py except that
PolicyNormProcessor.unapply_actions is wrapped so both the model's raw
(normalized) output and the un-normalized command are recorded. Used to decide
whether a near-constant applied action comes from the model or from the
normalization statistics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REGISTRY_DIR = (
    Path(__file__).resolve().parent / "examples" / "StarlinkCube" / "train_files" / "data_registry"
)
sys.path.insert(0, str(REGISTRY_DIR))

from data_config import DATASET_NAMED_MIXTURES as LOCAL_MIXTURES
from data_config import ROBOT_TYPE_CONFIG_MAP as LOCAL_ROBOT_CONFIGS
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.server_policy import build_argparser, main


def register_starlink_cube() -> None:
    DATASET_NAMED_MIXTURES.update(LOCAL_MIXTURES)
    ROBOT_TYPE_CONFIG_MAP.update(LOCAL_ROBOT_CONFIGS)


LOG_PATH = Path(os.environ.get("ACTION_TRACE", "/tmp/action-trace.jsonl"))
_CALLS = {"n": 0}


def instrument() -> None:
    original = PolicyNormProcessor.unapply_actions

    def traced(self, normalized_actions):
        out = original(self, normalized_actions)
        _CALLS["n"] += 1
        try:
            import numpy as np

            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "call": _CALLS["n"],
                            "normalized": np.asarray(normalized_actions).tolist(),
                            "unnormalized": np.asarray(out).tolist(),
                        }
                    )
                    + "\n"
                )
        except Exception as exc:
            print(f"[trace] write failed: {exc}", flush=True)
        return out

    PolicyNormProcessor.unapply_actions = traced
    print(f"[trace] instrumented; logging to {LOG_PATH}", flush=True)


if __name__ == "__main__":
    register_starlink_cube()
    instrument()
    main(build_argparser().parse_args())
