"""Measure batched Genesis throughput for the Franka + cube scene.

The single-env no-render rate is only ~50 steps/s, which is far too slow for RL
(1e6 steps would take 5.5 hours).  Genesis is built for parallel environments, so
this measures how throughput scales with ``n_envs`` -- the number that decides
whether RL is feasible at all.

Genesis allows one scene per process, so run this once per n_envs value:

    for N in 1 32 128 256; do
      python bench_parallel.py --n-envs $N
    done
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import genesis as gs

# A plausible mid-workspace arm configuration to hold while stepping.
ARM_TARGET = np.array([0.0, -0.30, 0.0, -2.00, 0.0, 2.00, 0.80])
FINGER_TARGET = 0.04


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--render", action="store_true", help="also render one camera per env")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 100.0, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.5, 0.0, 0.025)),
        material=gs.materials.Rigid(),
    )
    scene.build(n_envs=args.n_envs)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    arm = np.tile(ARM_TARGET, (args.n_envs, 1))
    fingers = np.full((args.n_envs, 2), FINGER_TARGET)

    def one_step() -> None:
        franka.control_dofs_position(arm, motors_dof)
        franka.control_dofs_position(fingers, fingers_dof)
        scene.step()

    for _ in range(args.warmup):
        one_step()
    # synchronise before timing so warm-up work is not counted
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass

    start = time.time()
    for _ in range(args.steps):
        one_step()
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass
    elapsed = time.time() - start

    scene_steps_per_s = args.steps / elapsed
    env_steps_per_s = args.n_envs * scene_steps_per_s
    print(
        f"n_envs={args.n_envs:4d}  steps={args.steps}  wall={elapsed:7.3f}s  "
        f"scene_steps/s={scene_steps_per_s:8.1f}  env_steps/s={env_steps_per_s:10.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
