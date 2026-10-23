"""Validate an exported Genesis LeRobot dataset with StarVLA's own dataloader.

This deliberately does not re-implement any loading logic: it instantiates
``LeRobotSingleDataset`` with the official Franka ``DataConfig`` so that a pass
here means StarVLA training can consume the dataset.

Usage:
    python validate_lerobot.py --dataset /path/to/lerobot_dataset
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path("/data/VLA_model/StarVLA/repo")
FRANKA_DATA_CONFIG = REPO / "examples/Franka/train_files/data_registry/data_config.py"


def load_franka_data_config():
    spec = importlib.util.spec_from_file_location("franka_data_config", FRANKA_DATA_CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SingleFrankaRobotiqDeltaEefDataConfig()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    dataset_path = Path(args.dataset).resolve()
    print(f"dataset : {dataset_path}")
    print(f"config  : SingleFrankaRobotiqDeltaEefDataConfig (official)")
    print(f"backend : {args.video_backend}\n")

    config = load_franka_data_config()
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=config.modality_config(),
        embodiment_tag=config.embodiment_tag,
        video_backend=args.video_backend,
        transforms=config.transform(),
        data_cfg={"lerobot_version": "v2.0", "action_mode": "abs"},
    )

    print(f"\nloaded OK")
    print(f"  episodes : {len(dataset.trajectory_ids)} -> {dataset.trajectory_ids.tolist()}")
    print(f"  lengths  : {dataset.trajectory_lengths.tolist()}")
    print(f"  steps    : {len(dataset._all_steps)}")
    print(f"  tasks    : {dataset.tasks.to_dict()}")

    print(f"\nreading {args.samples} samples (decodes video, so this proves the mp4s work):")
    for index in range(min(args.samples, len(dataset))):
        sample = dataset[index]
        print(f"  [{index}] keys: {sorted(sample.keys())}")
        for key in sorted(sample.keys()):
            value = sample[key]
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            if shape is not None:
                print(f"        {key:34s} {tuple(shape)} {dtype}")

    import numpy as np

    sample = dataset[0]
    def report(label: str, value) -> None:
        array = np.asarray(value, dtype=np.float32)
        print(f"  {label:12s} shape {array.shape} "
              f"range [{float(array.min()):+.4f}, {float(array.max()):+.4f}] "
              f"finite={bool(np.isfinite(array).all())}")

    print()
    for key in sorted(sample.keys()):
        value = sample[key]
        if key.startswith("action"):
            report(key, value)
        elif key.startswith("state"):
            report(key, value)
        elif key.startswith("video.") or key.startswith("image"):
            array = np.asarray(value)
            print(f"  {key:12s} shape {array.shape} dtype {array.dtype} "
                  f"mean {float(array.mean()):.2f} min {array.min()} max {array.max()}")
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
