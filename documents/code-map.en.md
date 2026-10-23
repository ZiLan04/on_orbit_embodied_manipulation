# Code Map — Server Tree

> **Scope**: `10.110.10.12:/data/Yanhan/genesis` only.
> **Excluded**: `runs/` (run outputs), `__pycache__/`.
> Everything below describes what actually exists on the server, verified 2026-10-21.
> Chinese counterpart: `code-map.md`. Handover document: `README.md`. Rules: `code-standards.md`.

---

## 0. Overview

```
/data/Yanhan/genesis/
├── code/                       ← our source code (4 top-level dirs, see §2)
├── spaceman_core/              ← upstream simulation tree, read-only reference, NOT our code (§3)
├── runs/                       ← 167 run directories, outputs (§8 index only)
├── recovered/                  ← artifacts rescued from a deploy-clobber incident (§5)
├── documents/                  ← these docs
├── .agents/skills/             ← 2 agent skills (§6)
├── README.md                   ← authority / deployment-target rules
├── CANONICAL-DEVIATIONS.md     ← deliberate deviations from upstream
└── DEPLOYED.json               ← last deploy record
```

**This tree has no version control.** The only `.git` lives inside `spaceman_core/` (an upstream clone). Files under `code/` have exactly one version — the current one. No `git checkout`, no rollback. Back up before you edit (§7).

Runtime facts:

- 41 self-written Python files under `code/` (excluding the vendored StarVLA subtree)
- Python envs: `/data/anaconda3/envs/Spaceman_env` (Genesis 0.4.3) and `/data/anaconda3/envs/starvla_env` (training / serving)
- All 41 files pass `py_compile`; the core modules import cleanly under the PYTHONPATH in §1.1

---

## 1. Where the code comes from

| Directory | Origin | Updated by `deploy.ps1`? |
|---|---|---|
| `code/genesis_tools/` | local repo `genesis_tools/` via `scp` | ✅ yes |
| `code/starlink_cube_rl/` | local `genesis_tools/rl/starlink_cube/`, **copied by hand and renamed** | ❌ **no** |
| `code/franka_cube/` | local `genesis_tools/tasks/` (Franka part) + `rl/bench_parallel.py`, **reorganized by hand** | ❌ **no** |
| `code/starvla_bridge/` | maintained by another person, on the server | ❌ no (excluded on purpose) |

`deploy.ps1` does exactly one thing: `scp -r <repo>/genesis_tools → code/`, landing at `code/genesis_tools/<relative path>`.

**⚠ Consequence — the deploy target does not match the real layout.**

- The code that actually runs lives in `code/starlink_cube_rl/`
- `deploy.ps1` would write it to `code/genesis_tools/rl/starlink_cube/`, a path that does not exist today
- So running a deploy **does not update the running code**; it creates a duplicate copy beside it. The two would then drift with nobody knowing which one is live.

`DEPLOYED.json` records the last deploy as `2026-09-18T04:22:47Z`, commit `98b7491`, `dirty: true` — **earlier than all the starlink_cube work** (earliest related run is 09-23). That subtree has never gone through `deploy.ps1`; it has been hand-copied.

**Until the deploy path is fixed, editing that code anywhere else does not reach the running pipeline.**

### 1.1 Two implicit dependencies (the main hazard for any restructuring)

**① `code/` is not self-contained — it needs `spaceman_core/src/` on the path**

`code/` contains many bare imports — `configs`, `tasks`, `RL`, `envs`, `robots`, `controllers`, `utils` — all of which resolve into `spaceman_core/src/`. A minimal working environment (verified):

```bash
export PYTHONPATH=/data/Yanhan/genesis/code/starlink_cube_rl:\
/data/Yanhan/genesis/code/starvla_bridge:\
/data/Yanhan/genesis/spaceman_core/src
```

Move `code/` without updating `PYTHONPATH` and every import fails.

**② `run_starvla_closed_loop_smoke.py` depends on `code/starvla_bridge/`**

Line 38: `from starvla_genesis_bridge import StarVLAPolicyClient` — a cross-directory dependency into the one directory that another person maintains and that deploy deliberately never touches. Relocating `starlink_cube_rl/` breaks this import unless the client class is first extracted (e.g. into `utils/`) or the relative layout is preserved.

### 1.2 Runtime environment (measured)

The two conda envs are **complementary by design**:

| Package | Spaceman_env (Genesis) | starvla_env (train / serve) |
|---|---|---|
| numpy / torch / msgpack | ✓ | ✓ |
| genesis | ✓ | ✗ |
| gymnasium / stable_baselines3 | ✓ | ✗ |
| pyarrow | ✗ | ✓ |
| websockets | ✗ (symlink shim) | ✓ |
| imageio_ffmpeg | ✓ | ✗ |

Work around the gaps — **do not install packages into these envs**:

- **websockets on the Genesis side**: `code/franka_cube/pylibs/websockets` is a symlink to `starvla_env`'s site-packages copy. Put `code/franka_cube/pylibs` on `PYTHONPATH`.
- **Video encoding**: run the exporter under `starvla_env` and pass the ffmpeg binary from `Spaceman_env` explicitly.
- **LeRobot export** needs `starvla_env` (pyarrow).

Headless Genesis always needs:

```bash
env -u DISPLAY PYOPENGL_PLATFORM=egl MUJOCO_GL=egl <python> ...
```

and the script itself must set `show_viewer=False`. Pin `CUDA_VISIBLE_DEVICES` explicitly — the box is 8×RTX 5090 shared by several people.

---

## 2. `code/` — our code

```
code/
├── starlink_cube_rl/     ★ current main line: RL env / data collection / PPO / VLA closed loop
├── franka_cube/          earlier Franka + cube scene
├── genesis_tools/        satellite-side tasks, evaluation, health checks, analysis
└── starvla_bridge/       Genesis ↔ StarVLA WebSocket bridge (maintained by another person)
```

### 2.1 `code/genesis_tools/` — satellite-side tasks and tools (14 files)

Synced by deploy; keeps the local relative layout.

```
genesis_tools/
├── analysis/
│   └── analyze_alignment_log.py      Analyse alignment-run logs: are the policy's commands
│                                     actually doing anything?
├── eval/
│   ├── eval_rl_grasp.py              Score an RL grasp policy against the metric that matters
│   │                                 (not the reward curve)
│   ├── eval_stage2_batch.py          Batched eval of satellite stage-2 grasp-pose PPO policies
│   ├── jepa_bridge.py                Closed-loop bridge: VLA-JEPA drives the satellite arm (RETIRED)
│   ├── jepa_direction_pilot.py       Ask the VLA-JEPA server what it would do here (RETIRED)
│   └── jepa_distance_sweep.py        Over what target distance does VLA-JEPA's direction stay
│                                     usable? (RETIRED)
├── healthcheck/
│   ├── ik_reachability.py            Can the arm physically reach the poses the task demands?
│   ├── reach_scan.py                 What is the closest the gripper can get to the object?
│   ├── stage2_action_control.py      Is the stage-2 success criterion actually controllable by
│   │                                 the action (not merely observable)?
│   └── starvla_healthcheck.py        Is the StarVLA policy server alive, and which checkpoint
│                                     did it load?
├── rl/
│   └── run_stage2_horizon.py         Stock stage-2 PPO trainer with a configurable episode horizon
└── tasks/
    ├── kinematic_demo_gen.py         Generate alignment demos from a kinematic expert
    ├── probe_arm_geometry.py         Probe arm geometry and reachable workspace
    └── scripted_grasp.py             Scripted (non-learned) grasp oracle for the satellite arm
```

The three JEPA files are retired: its output was shown to be a constant vector that does not change with the target. See `docs/2026-09-15-session-notes.md`.

### 2.2 `code/starlink_cube_rl/` — ★ current main line

20 own files plus `starvla_overlay/`. **Start reading at `starlink_cube_delta_eef_env.py` and `delta_eef_control.py`.**

```
starlink_cube_rl/
│
├── [action & environment]
│   ├── delta_eef_control.py                ★ the single definition of 7-D delta-EE action semantics
│   ├── starlink_cube_delta_eef_env.py      standalone batched Genesis environment
│   ├── task_contract.py                    reward terms and success criteria
│   ├── gymnasium_adapter.py                single-env Gymnasium bridge
│   └── sb3_vecenv_adapter.py               batched SB3 VecEnv bridge
│
├── [data collection]
│   ├── collect_oracle_rgb_rollouts.py      driven by the scripted oracle
│   ├── collect_oracle_recovery_rgb_rollouts.py  oracle + failure-recovery states
│   ├── collect_ppo_rgb_rollouts.py         driven by the trained PPO policy
│   ├── collect_oracle_rgb_rollouts.py.orig backup from an earlier edit — should move to backups/
│   └── export_lerobot.py                   raw dumps → LeRobot v2.1
│
├── [training & evaluation]
│   ├── run_batched_ppo_train.py            batched PPO launcher, checkpointed
│   ├── evaluate_batched_ppo.py             independent deterministic evaluation
│   └── run_starvla_closed_loop_smoke.py    VLA closed-loop eval (connects to the policy server)
│
├── [smoke tests / unit tests]
│   ├── run_delta_eef_env_smoke.py          does the environment run at all?
│   ├── run_delta_eef_oracle_smoke.py       is the reward criterion physically achievable?
│   ├── run_ppo_integration_smoke.py        single-env SB3 wiring
│   ├── run_batched_ppo_integration_smoke.py batched SB3 wiring
│   ├── run_real_cube_smoke.py              headless smoke of the existing physical grasp task
│   └── test_delta_eef_control.py           action-control safety and gripper semantics — see §4
│
├── [ops]
│   └── run_guarded_starvla.sh              resource-guarded launcher
│
└── starvla_overlay/                        StarVLA-side overlay
    ├── train_starvla_no_deepspeed.py       single-GPU training entry (adapted from upstream)
    ├── serve_starlink_cube_policy.py       start the policy server
    ├── serve_starlink_cube_policy_traced.py ★ debug launcher: instruments the normalisation
    ├── validate_starvla_loader.py          verify our dataset loads through StarVLA's loader
    ├── accelerate_one_gpu_ddp.yaml
    ├── accelerate_one_gpu.yaml
    ├── examples/StarlinkCube/train_files/
    │   ├── data_registry/data_config.py    data registry: which key slices to what, how normalised
    │   ├── starvla_starlink_cube_action_head_10k.yaml          full run (PPO data)
    │   ├── starvla_starlink_cube_oracle_action_head_10k.yaml   full run (oracle data)
    │   ├── starvla_starlink_cube_action_head_smoke_10.yaml     10-step smoke
    │   ├── starvla_starlink_cube_smoke.yaml                    smoke
    │   └── starvla_starlink_cube_single.yaml                   single-sample debugging
    └── starVLA/                            ★ vendored StarVLA framework, 169 files, NOT our code
        ├── config/        8 files    training and deepspeed configs
        ├── dataloader/   37 files    LeRobot / GR00T dataset loading and transforms
        ├── model/       109 files    VLM wrappers, action heads, frameworks
        └── training/     15 files    training loop and trainer utilities
```

What the important files do:

| File | Role |
|---|---|
| `delta_eef_control.py` | **Single source of truth for the action contract.** `DeltaEefLimits` holds four limits (0.01 m and 0.05 rad per step, 0.45 m cumulative displacement) plus a mode flag `normalized_actions` (**default `False`**). `DeltaEefController.targets()` turns an action into a target pose plus a gripper state. ⚠ the default mode disagrees with the policy's action space — see §4 |
| `starlink_cube_delta_eef_env.py` | Standalone batched Genesis environment. Cube spawn randomisation lives in `CubeEnvConfig`: `cube_spawn_offset_m=(0,-0.15,0)`, `cube_spawn_noise_m=(0.02,0.02,0.01)`. Default `control_limits=DeltaEefLimits()` (line 65) |
| `task_contract.py` | Success cascade: `grasp_ready` (closed and within distance) → `stable` (tighter distance and lift ≥ 0.030 m) → `success` (hold for N consecutive steps). Batched on GPU |
| `gymnasium_adapter.py` | Declares `action_space = Box(-1.0, 1.0, 7)` (line 26) |
| `sb3_vecenv_adapter.py` | Declares `Box(-1.0, 1.0, 7)` as well (line 25). Used for 100+ environment batched training |
| `collect_oracle_rgb_rollouts.py` | Oracle-driven collection. Action is proportional: `(cube_pos - grasp_center) / 0.01` clamped to [-1, 1] (lines 92–93), and it **explicitly passes `DeltaEefLimits(normalized_actions=True)`** (line 66). Supports `--cube-noise-scale` and `--seed`. Only successful episodes are kept |
| `collect_ppo_rgb_rollouts.py` | PPO-driven collection. The raw 7-D output of `model.predict(obs, deterministic=True)` is recorded verbatim as the label. **Passes no `control_limits`, so the environment default applies** |
| `run_starvla_closed_loop_smoke.py` | Talks to the policy server, pulls an action chunk, applies it step by step. Line 83 uses `StarlinkCubeDeltaEefEnv(num_envs=1)` — **again the default `control_limits`**. Emits `summary.json`, `actions.jsonl`, `closed-loop.mp4`. Supports `--execution-horizon` and `--seed` |
| `test_delta_eef_control.py` | Two cases: in physical-delta mode an action of 1.0 clamps to +0.01; in normalised mode 0.5 scales to 0.005. ⚠ see §4 |
| `starvla_overlay/serve_starlink_cube_policy_traced.py` | Monkeypatches `PolicyNormProcessor.unapply_actions` to log `{call, normalized, unnormalized}` per call into `action-trace.jsonl`. **This is what localised the defect in §4** |
| `starvla_overlay/examples/.../data_config.py` | Slice and normalisation for the three action keys: `delta_eef_position` and `delta_eef_rotation` use `min_max`, `gripper_close` uses `binary` |
| `starvla_overlay/train_starvla_no_deepspeed.py` | Single-GPU training entry that bypasses deepspeed. `load_pretrained_backbones` uses `strict=True` and raises on any mismatch |

### 2.3 `code/franka_cube/` — earlier Franka + cube scene (4 files)

| File | Role |
|---|---|
| `franka_cube_starvla.py` (59 KB) | Self-contained Genesis Franka Panda + cube scene. One scene per process, headless. Contains a scripted grasp state machine (approach → descend → close → lift → hold) that servos **fingertip** position with world-frame targets frozen at phase entry, a closed-loop episode runner, recording and export. Flags: `--dry-run`, `--export-raw`, `--randomize-cube`, `--only-success`, `--action-space raw\|normalized` |
| `export_lerobot.py` | Raw Genesis dumps → LeRobot v2.1. Writes `meta/{info,modality,episodes,tasks}.json` plus `stats_gr00t.json` |
| `validate_lerobot.py` | Re-validates an exported dataset through **StarVLA's own dataloader** — deliberately not the in-house loader, so the check is not self-referential |
| `bench_parallel.py` | Batched Genesis throughput benchmark. Single env, no render: 427 steps/s. Batched 256 envs: 93,091 env-steps/s |

### 2.4 `code/starvla_bridge/` — Genesis ↔ StarVLA bridge (13 files, another person's code)

| File | Role |
|---|---|
| `starvla_genesis_bridge.py` (17.6 KB) | **The main bridge.** Runs one Genesis episode whose Cartesian targets come from the remote StarVLA policy: build an RGB observation → derive end-effector state → request a 16-step action chunk → apply each step with position/rotation limits → request the next chunk. WebSocket `10.110.10.12:6694` with StarVLA's msgpack+NumPy encoding |
| `starvla_genesis_bridge_calvin.py` | CALVIN-format variant |
| `run_starvla_genesis_bridge.py` | Thin launcher for the main bridge |
| `run_starvla_genesis_bridge_7d.py` … `_7d_v7.py` | **Eight iterative snapshots** of the 7-D action wrapper. v7 is the last. Version numbers in filenames are an anti-pattern (see standards §5.2) |
| `run_starvla_genesis_bridge_calvin.py` | Launcher for the CALVIN variant |
| `README.md`, `VALIDATION-RESULT.md` | Verified interface contract and validation results |
| `RUNBOOK.md` … `RUNBOOK-v6.md` | Six runbook snapshots |
| `GENESIS_STARVLA_REMOTE_DESKTOP_RUNBOOK.md` | Remote-desktop procedure |

⚠ **This directory is outside deploy's sync scope** — `deploy.ps1` refuses to touch it because someone is actively developing it on the server. Do not treat it as a deploy artifact.

---

## 3. `spaceman_core/` — upstream simulation tree (read-only)

Synced from the Genesis machine `10.110.34.88`. **Not our code; do not modify.**

```
spaceman_core/
├── .git                 ← the only version history anywhere in this tree (upstream clone)
├── .codex
├── outputs/rl/
├── scripts/             10 plotting scripts (stage1/stage2 error bands, success time,
│                        trajectories, training reward curves)
└── src/
    ├── assets/          URDF and meshes
    ├── configs/         scene and camera settings
    ├── controllers/
    ├── envs/            Genesis environment wrappers
    ├── outputs/  recordings/
    ├── RL/              upstream PPO implementation
    ├── robots/          arm definitions
    ├── sensors/         cameras
    ├── tasks/           task definitions (incl. stage1 / stage2)
    ├── test/            upstream replay / test scripts
    └── utils/           data_recorder, etc.
```

`CANONICAL-DEVIATIONS.md` records the deliberate deviations from upstream (commit `e042566`). Both are mesh paths rewritten from `package://` to relative paths, because Genesis cannot resolve ROS `package://` URIs. The file is explicit that **only paths were changed, not material colours or names**, since a vision-based policy is sensitive to target appearance; a colleague's uncommitted URDF that also changed materials was deliberately not adopted.

---

## 4. ⚠ Known functional defect — action contract mismatch

**This is not part of any restructuring work, but it affects the interpretation of every dataset produced so far, so read it before changing anything.**

**Symptom**: the PPO-collected 128-episode dataset (`runs/20260924-starlink-cube-vla-batch1`) has near-constant action labels — channels z, roll and yaw have `min == max == 1.0`, and the inter-episode standard deviation is 2e-4.

**Cause**: the **declared** and **interpreted** action ranges differ by a factor of 100.

| Stage | Location | Content |
|---|---|---|
| Action space declared | `sb3_vecenv_adapter.py:25`, `gymnasium_adapter.py:26` | `Box(-1.0, 1.0, 7)` |
| Environment default | `starlink_cube_delta_eef_env.py:65` | `DeltaEefLimits()` → `normalized_actions=False` |
| Controller | `delta_eef_control.py:67-68` | `clamp(action[:3], -0.01, 0.01)`, `clamp(action[3:6], -0.05, 0.05)` |

Consequence: 99% of the action range is a dead zone — an output of 0.01 and an output of 1.0 produce the same command. The policy loses any gradient pressure on magnitude and degenerates into saturated bang-bang control; the labels retain only the sign.

**Three call sites must all be fixed; fixing one alone changes nothing:**

1. `sb3_vecenv_adapter.py:25` — PPO training and evaluation
2. `collect_ppo_rgb_rollouts.py:50` — PPO data collection
3. `run_starvla_closed_loop_smoke.py:83` — **VLA closed-loop evaluation** (the critical one: however well the model is trained, its output magnitude is discarded at evaluation time)

**The reference implementation that is already correct**: `collect_oracle_rgb_rollouts.py:66` passes `DeltaEefLimits(normalized_actions=True)` explicitly. That is also why the new oracle dataset (`runs/20261016-145416-phase1-oracle-128`) has 400–4000× the action diversity of the old one — same environment, same cube noise, one parameter different.

**⚠ Trap**: `test_delta_eef_control.py` asserts that an action of 1.0 maps to the full step. That assertion encodes the defect as expected behaviour. Fix the assertion together with the contract, or someone will revert the fix to make the test pass.

---

## 5. `recovered/` — artifacts from an incident

```
recovered/
├── recovered_structure.txt
├── run_starvla_genesis_bridge_7d_v2.cpython-310.pyc
├── run_starvla_genesis_bridge_7d_v6.cpython-310.pyc
└── starvla_genesis_bridge.cpython-310.pyc
```

On 2026-09-17 a deploy overwrote the `starvla_bridge/` directory that someone else was actively developing on the server. Only the `.pyc` files in `__pycache__` were recoverable; these are what was rescued. **Do not delete.** The consequence is documented in the header of `deploy.ps1`, and the two hard rules that came out of it are: `starvla_bridge` is never synced, and remote files newer than local ones are refused rather than overwritten.

---

## 6. `.agents/skills/` — agent skills

| Directory | Content |
|---|---|
| `starlink-cube-vla-pipeline/SKILL.md` | 250 lines. End-to-end pipeline (data → LeRobot → finetune → closed-loop eval), the complementary gaps between the two conda envs, and 20 verified pitfalls |
| `sim-driven-policy-iteration/SKILL.md` | 241 lines. Methodology for improving a policy from simulation feedback: confidence intervals, failure stratification, one change at a time, paired re-evaluation on identical seeds |

A third copy lives at `/home/student/.codex/skills/`. Both are byte-identical to the source in the local repo (verified by md5).

---

## 7. Safety requirements before changing anything

This tree has **no git** outside `spaceman_core/`. Therefore:

1. **Copy before you edit.** Put the copy under `code/backups/` (or `/data/Yanhan/genesis/backups/`) with a date in the filename. This is the only rollback mechanism that exists.
2. **Run the task's own `run_*_smoke.py` afterwards**, not just an import check.
3. **If you touch anything on the data-production path**, re-run one existing evaluation and require **identical per-episode results**. This is genuinely verifiable here: evaluation is fully deterministic under a fixed seed (verified — step-cap 128 vs 256 produced identical results across all 64 episodes, episode by episode). Do not substitute "looks about the same".
4. **Do not hand-edit `code/genesis_tools/` on the server** — the next deploy overwrites it. Note that, per §1, that overwrite does *not* reach `code/starlink_cube_rl/`; the two directories behave differently, so keep them straight.

---

## 8. `runs/` — where to look first (index only, not expanded)

167 run directories, named `<YYYYMMDD>-<slug>-<version>`. Start with these:

| Directory | What it is |
|---|---|
| `20260923-starlink-cube-ppo-overnight` | PPO training; `evaluation-final.json` = 245/256 = 95.7% |
| `20260924-starlink-cube-vla-batch1` | The old 128-episode dataset (PPO-collected; degenerate actions) |
| `20260928-starlink-cube-starvla-action-head-10k-v2` | VLA action-head training plus four closed-loop evals; `closed-loop-randomized-64-v3` is the 58/64 = 90.6% run |
| `20260929-starlink-cube-oracle-coverage{2x,1p5x}-v1`, `...-recovery-coverage1p25x-v1` | The 1.25× / 1.5× / 2.0× points of the oracle randomisation sweep |
| `20261016-140642-cap-ablation` | Paired step-cap experiment, 128 vs 256 |
| `20261016-142930-action-trace` | 74 instrumented server calls; `action-trace.jsonl` |
| `20261016-143810-phase1-noise-sweep` | The two extremes of the sweep, 1.0× (24/25) and 4.0× (4/60) |
| `20261016-145416-phase1-oracle-128` | The new 128-episode high-diversity dataset |

Typical contents of a run directory: `launch.txt` (exact command), `*.log`, `policy-server.log`, `summary.json` / `evaluation-final.json` (with a per-episode `episode_summaries` list), `actions.jsonl`, `closed-loop.mp4`, `raw/*.npz`, `lerobot/`, `dataset_statistics.json`, `checkpoints/`, `tensorboard/`.
