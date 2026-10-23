#!/usr/bin/env python3
"""Run the stock stage-2 PPO trainer with a configurable episode horizon.

Experiment "A": change exactly ONE variable relative to the shipped 40M run
(`run_20260724_214752`, see its train_config.json) -- the episode horizon.

    ParallelStage2ArmTask.max_steps:  120  ->  --max-episode-steps N

Reward, success criterion, observation, network, hyper-parameters, seed, env
count and task code are all left exactly as they were. That is the whole point:
if the 0.25 m plateau is produced by "actuation rate x episode length", raising
only the horizon should move it; if it does not move, the explanation is wrong.

Why a launcher instead of editing the driver
--------------------------------------------
`self.max_steps = 120` is hard-coded in the vendored `spaceman_core` tree. This
script patches the task class **in memory** before the driver imports/builds
anything, so the vendored tree stays byte-identical to the revision it came from
and the experiment is trivially reversible (just don't use this launcher).

Usage
-----
    python run_stage2_horizon.py --tree /data/Yanhan/genesis/spaceman_core \
        --max-episode-steps 600 \
        --stage stage2 --total-timesteps 16000000 --num-envs 100 \
        --save-dir src/outputs/rl/ppo_satellite_stage2_arm_A600 \
        --run-name A600 --checkpoint-freq 20000

Everything not consumed here is forwarded verbatim to the driver's argparse.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--tree", default="/data/Yanhan/genesis/spaceman_core")
    ap.add_argument("--max-episode-steps", type=int, default=None)
    ap.add_argument("--force-batch-renderer", dest="force_batch_renderer",
                    action="store_true", default=True,
                    help="Install BatchRenderer for the batched RL cameras (default on; "
                         "see the headless note below).")
    ap.add_argument("--no-force-batch-renderer", dest="force_batch_renderer",
                    action="store_false")
    ap.add_argument("--disable-camera-recording", dest="disable_camera_recording",
                    action="store_true", default=True)
    ap.add_argument("--no-disable-camera-recording", dest="disable_camera_recording",
                    action="store_false")
    ap.add_argument("-h", "--help", action="store_true")
    known, rest = ap.parse_known_args()

    tree = Path(known.tree).resolve()
    src = tree / "src"
    if not src.is_dir():
        print(f"[A] FATAL: {src} is not a directory", flush=True)
        return 2

    if known.help:
        print(__doc__)
        return 0

    # The driver's default save dirs are relative ("src/outputs/rl/..."), so the
    # working directory must be the tree root, exactly as in the original run.
    os.chdir(tree)
    sys.path.insert(0, str(src))

    if known.max_episode_steps is not None:
        import tasks.RL_task_parallel_stage2 as stage2

        orig_init = stage2.ParallelStage2ArmTask.__init__
        horizon = int(known.max_episode_steps)

        def patched_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            print(f"[A] patched {type(self).__name__}.max_steps: "
                  f"{self.max_steps} -> {horizon}", flush=True)
            self.max_steps = horizon

        stage2.ParallelStage2ArmTask.__init__ = patched_init
        # Sanity: the driver reaches the task through make_stage2_vec_env, which
        # constructs this very class, so the patch is picked up.
        print(f"[A] in-memory patch armed: max_episode_steps={horizon} "
              f"(vendored tree NOT modified)", flush=True)
    else:
        print("[A] WARNING: --max-episode-steps not given; running stock 120", flush=True)

    # ---- headless adaptation (environment, NOT part of experiment A) ----------
    # The stock driver cannot train a *visual-obs* policy headless:
    #   def configure_headless_training(enable_visual_obs):
    #       if enable_visual_obs: return          <-- returns immediately
    #       SCENE_SETTINGS["renderer"] = None ...
    # so with `--enable-visual-obs` (the shipped stage-2 config) nothing disables
    # the viewer, and Genesis dies with "No display detected". We therefore make
    # the same overrides our evaluator already makes, before the driver builds the
    # scene:
    #   * show_viewer=False  -> headless
    #   * renderer=BatchRenderer(use_rasterizer=True) -> the batched RL cameras
    #     need it; without it Genesis binds the camera to a single env and the
    #     batched camera call dies with "Pos shape torch.Size([N,3]) does not
    #     match (N,3)".
    # This does not touch the reward, observation, horizon, or network.
    import genesis as gs
    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    print(f"[A] headless override: show_viewer {SCENE_SETTINGS.get('show_viewer')} "
          f"-> False", flush=True)
    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    if known.force_batch_renderer:
        SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
        print("[A] headless override: renderer -> BatchRenderer(use_rasterizer=True)",
              flush=True)

    # ---- I/O adaptation (environment, NOT part of experiment A) --------------
    # The stock stage-2 path writes a wrist-camera MP4 for every episode chunk
    # ("[RL_IMAGE_RECORDER] Saved RL camera recording to: .../recordings/..."), which
    # is pure disk churn: the policy's observation still comes from the camera and
    # is unaffected. Off by default here.
    if known.disable_camera_recording:
        # NOTE: the module is `config_starlink_manipulator` -- the driver imports
        # `FRANKA_S_Q_CAMERA` from exactly there. The top-level and base_camera
        # `enable_recording` flags are already False in the shipped config, but
        # the `rl_obs` (and `base_camera.rl_obs`) sub-sections are True, which is
        # what produced "[RL_IMAGE_RECORDER] Saved RL camera recording to ...".
        from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA as _CAM
        changed = []
        for _name in ("rl_obs", "base_camera", "wrist_camera", None):
            _s = _CAM if _name is None else _CAM.get(_name)
            if not isinstance(_s, dict):
                continue
            if _s.get("enable_recording"):
                _s["enable_recording"] = False
                changed.append(_name or "<top>")
            _sub = _s.get("rl_obs")
            if isinstance(_sub, dict) and _sub.get("enable_recording"):
                _sub["enable_recording"] = False
                changed.append(("base_camera." if _name else "") + "rl_obs")
        print(f"[A] I/O override: camera recording disabled in {changed}", flush=True)

    sys.argv = [sys.argv[0]] + rest

    from RL import train_satellite_side_grasp_ppo as driver
    driver.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
