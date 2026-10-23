"""Serve the Starlink-cube checkpoint with its project-local data registry."""

from __future__ import annotations

import sys
from pathlib import Path


REGISTRY_DIR = Path(__file__).resolve().parent / "examples" / "StarlinkCube" / "train_files" / "data_registry"
sys.path.insert(0, str(REGISTRY_DIR))

from data_config import DATASET_NAMED_MIXTURES as LOCAL_MIXTURES
from data_config import ROBOT_TYPE_CONFIG_MAP as LOCAL_ROBOT_CONFIGS
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from deployment.model_server.server_policy import build_argparser, main


def register_starlink_cube() -> None:
    """Expose the training-time config to the generic policy server."""
    DATASET_NAMED_MIXTURES.update(LOCAL_MIXTURES)
    ROBOT_TYPE_CONFIG_MAP.update(LOCAL_ROBOT_CONFIGS)


if __name__ == "__main__":
    register_starlink_cube()
    main(build_argparser().parse_args())
