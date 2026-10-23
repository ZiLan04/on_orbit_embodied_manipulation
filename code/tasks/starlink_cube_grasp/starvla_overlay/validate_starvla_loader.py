"""Validate that the Starlink LeRobot dataset loads through the isolated StarVLA overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def describe(value):
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", type(value).__name__))}
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.overlay))

    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    assert "starlink_cube_vla" in DATASET_NAMED_MIXTURES, "Starlink data mixture was not discovered"
    assert "starlink_cube_delta_eef" in ROBOT_TYPE_CONFIG_MAP, "Starlink robot config was not discovered"
    data_cfg = OmegaConf.create({
        "data_root_dir": str(args.data_root),
        "data_mix": "starlink_cube_vla",
        "video_backend": "torchvision_av",
        "load_all_data_for_training": True,
    })
    dataset = get_vla_dataset(data_cfg=data_cfg)
    sample = dataset[0]
    report = {"dataset_length": len(dataset), "sample": describe(sample)}
    print("starvla_loader_ok", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())