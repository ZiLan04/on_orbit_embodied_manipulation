# Code Map

> What every file and directory in this repository does, and how the code reaches the GPU
> server where it actually runs.
> Chinese counterpart: `code-map.md`. Standards: `code-standards.en.md`. Structure and
> migration: `README.en.md`.
>
> **Note on two layouts.** This repository is in the target layout (`code/tasks/...`).
> The GPU server still runs the *pre-restructure* layout (`code/starlink_cube_rl/...`).
> §1.3 maps one to the other. Paths below refer to **this repository** unless marked
> otherwise.

---

## 0. Repository tree

```
code/
├── tasks/                             one directory per task
│   ├── starlink_cube_grasp/           ★ current main line (18 files)
│   │   ├── delta_eef_control.py       the 7-D delta-EE action contract
│   │   ├── starlink_cube_delta_eef_env.py   batched Genesis environment
│   │   ├── task_contract.py           reward terms and success criteria
│   │   ├── gymnasium_adapter.py       single-env Gymnasium bridge
│   │   ├── sb3_vecenv_adapter.py      batched SB3 VecEnv bridge
│   │   ├── collect_oracle_rgb_rollouts.py           scripted-oracle collection
│   │   ├── collect_oracle_recovery_rgb_rollouts.py  oracle + recovery states
│   │   ├── collect_ppo_rgb_rollouts.py             PPO-driven collection
│   │   ├── export_lerobot.py          raw dumps -> LeRobot v2.1
│   │   ├── run_batched_ppo_train.py   batched PPO training
│   │   ├── evaluate_batched_ppo.py    deterministic PPO evaluation
│   │   ├── run_starvla_closed_loop_smoke.py  VLA closed-loop evaluation
│   │   ├── run_delta_eef_env_smoke.py        smoke: does the env build and step
│   │   ├── run_delta_eef_oracle_smoke.py     smoke: is the reward achievable
│   │   ├── run_ppo_integration_smoke.py      smoke: single-env SB3 wiring
│   │   ├── run_batched_ppo_integration_smoke.py  smoke: batched SB3 wiring
│   │   ├── run_real_cube_smoke.py     smoke: existing physical grasp task
│   │   ├── test_delta_eef_control.py  unit test: action-control safety, gripper semantics
│   │   ├── probes/
│   │   │   └── starvla_healthcheck.py is the policy server alive, which checkpoint loaded
│   │   └── starvla_overlay/           StarVLA-side entry points and configs
│   │       ├── train_starvla_no_deepspeed.py    single-GPU training entry
│   │       ├── serve_starlink_cube_policy.py    start the policy server
│   │       ├── serve_starlink_cube_policy_traced.py  debug launcher with normalisation logging
│   │       ├── validate_starvla_loader.py       verify our dataset loads via StarVLA's loader
│   │       ├── accelerate_one_gpu.yaml
│   │       ├── accelerate_one_gpu_ddp.yaml
│   │       └── examples/StarlinkCube/train_files/
│   │           ├── data_registry/data_config.py       action keys, slices, normalisation
│   │           ├── starvla_starlink_cube_action_head_10k.yaml         full run, PPO data
│   │           ├── starvla_starlink_cube_oracle_action_head_10k.yaml  full run, oracle data
│   │           ├── starvla_starlink_cube_action_head_smoke_10.yaml    10-step smoke
│   │           ├── starvla_starlink_cube_smoke.yaml                   smoke
│   │           └── starvla_starlink_cube_single.yaml                  single-sample debug
│   │
│   ├── starlink_satellite_grasp/       satellite-side stage-2 grasp pose (5 + 3 probes)
│   │   ├── scripted_grasp.py           scripted (non-learned) grasp oracle
│   │   ├── eval_stage2_batch.py        batched stage-2 evaluation
│   │   ├── eval_rl_grasp.py            score an RL grasp policy properly
│   │   ├── stage2_action_control.py    is the success criterion actually controllable
│   │   ├── run_stage2_horizon.py       stage-2 PPO with configurable horizon
│   │   └── probes/
│   │       ├── ik_reachability.py      can the arm reach the demanded poses
│   │       ├── reach_scan.py           how close can the gripper get to the object
│   │       └── probe_arm_geometry.py   arm geometry and reachable workspace
│   │
│   ├── franka_cube_grasp/              earlier Franka Panda + cube scene (4)
│   │   ├── franka_cube_starvla.py      self-contained scene, scripted state machine, runner
│   │   ├── export_lerobot.py           raw dumps -> LeRobot v2.1 (Franka variant)
│   │   ├── validate_lerobot.py         validate via StarVLA's own dataloader
│   │   ├── bench_parallel.py           batched Genesis throughput benchmark
│   │   └── pylibs/README.md            how to recreate the websockets path shim
│   │
│   └── vla_jepa_probe/                 retired JEPA line, kept for reference (4 + 1 probe)
│       ├── jepa_bridge.py              VLA-JEPA drives the satellite arm
│       ├── jepa_direction_pilot.py     ask the server what it would do here
│       ├── jepa_distance_sweep.py      over what distance does its direction stay usable
│       ├── kinematic_demo_gen.py       alignment demos from a kinematic expert
│       └── probes/analyze_alignment_log.py   are the policy's commands doing anything
│
└── bridges/
    └── starvla_bridge/                 Genesis <-> StarVLA WebSocket bridge (19, NOT ours alone)
        ├── starvla_genesis_bridge.py    the main bridge
        ├── starvla_genesis_bridge_calvin.py  CALVIN-format variant
        ├── run_starvla_genesis_bridge.py          thin launcher
        ├── run_starvla_genesis_bridge_7d.py       launcher, 7-D action variant
        ├── run_starvla_genesis_bridge_7d_v2..v7.py   an override CHAIN, not copies (see below)
        └── README.md, VALIDATION-RESULT.md, RUNBOOK*.md  contract and runbooks

documents/                             binding standards and this map
deploy/deploy.ps1                      one-way sync to the GPU server
.agents/skills/                        two agent skills encoding the pipeline and the method
```

There is deliberately **no `code/utils/`**. Every reusable module currently has exactly
one consumer, so it lives inside that consumer's task directory. Create `utils/` on the day
a second task needs the same code, not before.

### A trap worth knowing: the `_7d_v*` files are a chain

`run_starvla_genesis_bridge_7d_v3.py` imports `..._v2`, `v4` imports `v3`, and so on
through `v7`. They are incremental overrides, **not parallel snapshots**. Deleting any one
of `v2..v6` breaks `v7`. Collapsing them into one file is a real refactor with its own
verification, not a cleanup.

---

## 1. How this code runs

### 1.1 Four implicit path dependencies

`code/` is **not self-contained**. Each of these must be on `PYTHONPATH` or the import
fails at runtime with no hint about why.

**① `spaceman_core/src`** — many files import `configs`, `tasks`, `RL`, `envs`, `robots`,
`controllers`, `utils`; all of those resolve into the upstream simulation tree, which is
**not** in this repository.

**② `code/bridges/starvla_bridge`** — `run_starvla_closed_loop_smoke.py` line 38 does
`from starvla_genesis_bridge import StarVLAPolicyClient`. A cross-directory dependency into
the one directory that another person maintains.

**③ `.../train_files/data_registry`** — `serve_starlink_cube_policy*.py` insert that
directory on `sys.path` and then `import data_config`. The path is hard-coded relative to
the overlay, so `data_config.py` cannot be relocated without editing both launchers.

**④ `code/tasks/franka_cube_grasp/pylibs`** — an environment-level symlink shim that
exposes `websockets` (installed only in `starvla_env`) to the Genesis interpreter. See
`pylibs/README.md`.

A working set:

```bash
export PYTHONPATH=<repo>/code/tasks/starlink_cube_grasp:\
<repo>/code/bridges/starvla_bridge:\
<repo>/code/tasks/franka_cube_grasp/pylibs:\
<repo>/code/tasks/starlink_cube_grasp/starvla_overlay/examples/StarlinkCube/train_files/data_registry:\
<spaceman_core>/src
```

### 1.2 Runtime environment

Two conda environments, complementary by design. **Do not install into either one** to
patch a missing package — that is how you break the other side.

| package | `Spaceman_env` (Genesis 0.4.3) | `starvla_env` (train / serve) |
|---|---|---|
| numpy / torch / msgpack | ✓ | ✓ |
| genesis | ✓ | ✗ |
| gymnasium / stable_baselines3 | ✓ | ✗ |
| pyarrow | ✗ | ✓ |
| websockets | ✗ (use the `pylibs` shim) | ✓ |
| imageio_ffmpeg | ✓ | ✗ |

Headless Genesis always needs:

```bash
env -u DISPLAY PYOPENGL_PLATFORM=egl MUJOCO_GL=egl <python> ...
```

and the scene script itself must set `show_viewer=False`. Pin `CUDA_VISIBLE_DEVICES`
explicitly — the box is 8×RTX 5090 shared by several people.

### 1.3 External dependencies (not in this repository)

| what | where | note |
|---|---|---|
| `spaceman_core/` | GPU server, sibling of `code/` | upstream simulation tree, synced from the Genesis machine. Read-only. Deviations from upstream are documented in its `CANONICAL-DEVIATIONS.md` (URDF mesh paths `package://` → relative, because Genesis cannot resolve ROS URIs) |
| `starVLA/` | GPU server, inside the deployment | ~100 files. The policy service runs at a fixed path, so this is a runtime dependency rather than vendored source |

**One local modification exists in the server's copy of StarVLA.**
`starVLA/dataloader/__init__.py` is patched relative to upstream `3422b9f`: the `DataLoader`
worker count, prefetch factor and pin-memory flag are read from `STARVLA_DATALOADER_*`
environment variables with conservative defaults (`workers=4, prefetch=2, pin_memory=0`)
instead of hard-coded `16 / 4 / True`. Reason: video decoding plus pin-memory prefetch
multiplies host RAM across workers, which is not survivable on a shared machine. **If
StarVLA is ever re-vendored, this patch has to be reapplied** — it is not recorded anywhere
else.

### 1.4 Deploying

`deploy/deploy.ps1` syncs the whole `code/` tree to the server with a per-file mtime guard
that refuses to overwrite anything newer on the remote side. Two things it does not do:

- It does **not** cut the server over to this layout. The server still holds the old
  directories; a deploy today adds `code/tasks/` and `code/bridges/` beside them. The
  cut-over is a manual, reversible step (see `README.en.md`).
- It does **not** sync `code/bridges/starvla_bridge/`. That directory is maintained on the
  server by someone else; the original script threw if you asked it to.

---

## 2. Known functional defect — action contract mismatch

**Outside the scope of any structural change, but it changes how every dataset produced so
far must be read.**

**Symptom.** The PPO-collected 128-episode dataset has near-constant action labels: channels
z, roll and yaw have `min == max == 1.0`, and the inter-episode standard deviation is 2e-4.

**Cause.** The **declared** and **interpreted** action ranges differ by a factor of 100.

| stage | location | content |
|---|---|---|
| action space declared | `sb3_vecenv_adapter.py:25`, `gymnasium_adapter.py:26` | `Box(-1.0, 1.0, 7)` |
| environment default | `starlink_cube_delta_eef_env.py:65` | `DeltaEefLimits()` → `normalized_actions=False` |
| controller | `delta_eef_control.py:67-68` | `clamp(action[:3], -0.01, 0.01)`, `clamp(action[3:6], -0.05, 0.05)` |

Consequence: 99 % of the action range is a dead zone — an output of 0.01 and an output of
1.0 produce the same command. The policy loses all gradient pressure on magnitude and
degenerates into saturated bang-bang control; the recorded labels retain only the sign.

**Three call sites must all be fixed; fixing one alone changes nothing:**

1. `sb3_vecenv_adapter.py:25` — PPO training and evaluation
2. `collect_ppo_rgb_rollouts.py:50` — PPO data collection
3. `run_starvla_closed_loop_smoke.py:83` — **VLA closed-loop evaluation** (the critical one:
   however well the model is trained, its output magnitude is discarded at evaluation time)

**The reference implementation that is already correct** is `collect_oracle_rgb_rollouts.py:66`,
which passes `DeltaEefLimits(normalized_actions=True)` explicitly. That is also why the
oracle dataset has 400–4000× the action diversity of the PPO dataset — same environment,
same cube noise, one parameter different.

**Trap.** `test_delta_eef_control.py` asserts that an action of `1.0` maps to the full step.
That assertion encodes the defect as expected behaviour. Fix the assertion with the
contract, or someone will revert the fix to make the test pass.

---

## 3. Measured results

| policy | result | notes |
|---|---|---|
| PPO, deterministic | 245 / 256 = 95.7 % | batched, deterministic |
| VLA action head, closed loop | 58 / 64 = 90.6 % | 95 % CI [0.84, 0.98] |
| scripted oracle, 1.25× randomisation | 32 / 35 = 91.4 % | the teacher |

Scripted-oracle success versus cube randomisation width:

| width | 1.0× | 1.25× | 1.5× | 2.0× | 4.0× |
|---|---|---|---|---|---|
| success | 96.0 % | 91.4 % | 73.3 % | 55.6 % | 6.7 % |

Wide coverage therefore has to come from RL, not from a hand-written controller.

---

## 4. Verification protocol

Structural changes are checked at four levels. The last one is the decisive one.

| level | check |
|---|---|
| 1 | file contents unchanged (md5 against the reference copy) |
| 2 | `py_compile` under Python 3.10; every module imports with the documented `PYTHONPATH` |
| 3 | a scripted-oracle episode, same seed: `action`, `state`, `cube_position` bit-identical |
| 4 | the full 64-episode closed-loop VLA evaluation, same seed and same policy server: per-episode `success`, `steps` and initial `cube_position` identical |

**Rendered pixels are not reproducible run-to-run**, even with identical code on the same
GPU. A control run of the *same* code twice differs by up to 45/255 on about 0.001 % of
pixel samples. So never use image comparison as the pass criterion — compare
`success` / `steps` / `action` / `state`.

---

## 5. Safety requirements before changing anything

The server tree has no version control beyond this repository. Therefore:

1. Copy before you edit — `backups/`, dated. It is gitignored and it is the local net.
2. Run the task's own `run_*_smoke.py` / `test_*.py` afterwards, not just an import check.
3. If the change touches the data-production path, re-run one existing evaluation and
   require identical per-episode results (§4 level 4).
4. Do not hand-edit the deployed tree on the server — it is a sync target.
5. Do not change `code/bridges/starvla_bridge/` without the owner.
