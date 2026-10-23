---
name: starlink-cube-vla-pipeline
description: Use when collecting robot manipulation data in Genesis, exporting it to LeRobot for StarVLA, finetuning a VLA action head, or running a closed-loop policy evaluation — specifically for the starlink / franka_merge arm and its cube task. Covers the verified data-format contract, the two-environment split, the verified entry points, and the traps that silently corrupt results.
---

# starlink cube VLA pipeline

End-to-end pipeline that has been run successfully: **Genesis simulation → RGB
rollout collection → LeRobot v2.1 → StarVLA action-head finetuning → closed-loop
evaluation**. Measured closed-loop results:

| Policy | Result | Notes |
|---|---|---|
| VLA action head (`...-action-head-10k-v2`) | **58/64 = 90.6%**, 95% CI [0.84, 0.98] | `closed-loop-randomized-64-v3` |
| PPO (deterministic) | **245/256 = 95.7%** | `...-ppo-overnight/evaluation-final.json` |
| Scripted oracle (the teacher) | **32/35 = 91.4%** | on a 1.25x-coverage validation set |

**Read the run id carefully before quoting a number.** Several similarly named
runs exist with very different results — an earlier `oracle-action-head-10k-v1`
variant scored 6/16 = 37.5%, and quoting that as the project's result is wrong.
Always confirm the run directory, the checkpoint path in its `summary.json`
metadata, and the episode count before comparing anything.

**This skill is guidance, not a script.** Paths and flags below were verified at
the time of writing. Before trusting a command, confirm the file still exists and
read the header of the script you are about to run.

## Where everything lives

All work is on `10.110.10.12`, user `student`, under `/data/Yanhan/genesis/`:

| Path | Role |
|---|---|
| `code/starlink_cube_rl/` | The pipeline itself (env, adapters, collectors, trainers, evaluator) |
| `code/franka_cube/` | Older Franka+cube scene: scripted grasp, LeRobot exporter, validator |
| `code/starvla_bridge/` | Original StarVLA↔Genesis closed-loop bridge (older) |
| `runs/` | Every run's outputs, named `<YYYYMMDD>-<slug>-<version>` |
| `runs/guarded-jobs/` | Logs from guarded/background job runs |
| `spaceman_core/` | Simulation tree — **read-only reference, never modify** |
| `README.md`, `CANONICAL-DEVIATIONS.md` | Deliberate local deviations from upstream; read before assuming a file matches canonical |

`/data/Yanhan/genesis/README.md` states the authority rule: this directory is the
**deployment target and run output**, not the source of truth. Canonical code lives
in a local git repo and is synced here one-way. Do not hand-edit deployed code as a
substitute for fixing the source.

The starlink arm's URDF is a **deliberately modified** copy — mesh paths were
rewritten from `package://` to relative because Genesis cannot resolve ROS URIs.
This is documented in `CANONICAL-DEVIATIONS.md`. Preserve it.

## Environment split — do not install packages to "fix" this

Two conda environments with **complementary** contents:

| | pyarrow | imageio_ffmpeg | websockets |
|---|---|---|---|
| `/data/anaconda3/envs/Spaceman_env` (Genesis, 0.4.3) | ✗ | ✓ | ✗ |
| `/data/anaconda3/envs/starvla_env` (StarVLA, training, serving) | ✓ | ✗ | ✓ |

Resolve gaps with a path or a binary, never with an install:

- **websockets on the Genesis side**: a symlink shim exists at
  `code/franka_cube/pylibs/websockets`. Put `code/franka_cube/pylibs` on
  `PYTHONPATH` — it exposes only that one package, so `starvla_env`'s
  numpy/torch do not leak in.
- **Video encoding**: run the exporter under `starvla_env` and pass the ffmpeg
  binary from `Spaceman_env` explicitly:
  `/data/anaconda3/envs/Spaceman_env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`

Headless Genesis always needs:
```bash
env -u DISPLAY PYOPENGL_PLATFORM=egl MUJOCO_GL=egl <python> ...
```
and the script itself must set `show_viewer=False`. Pin `CUDA_VISIBLE_DEVICES`
explicitly — the box has 8×RTX 5090 shared by several people.

## The five stages

### 1. Collect RGB rollouts

Two collectors exist; pick by what should drive the arm.

| Collector | Driver | Gives |
|---|---|---|
| `collect_oracle_rgb_rollouts.py` | scripted oracle | guaranteed-successful trajectories |
| `collect_oracle_recovery_rgb_rollouts.py` | oracle + recovery states | adds off-nominal states the policy must recover from |
| `collect_ppo_rgb_rollouts.py` | trained PPO policy | on-policy trajectories, needs filtering |

**Only successful trajectories are useful for behavior cloning.** Filter at
collection time. A grasp is successful when the cube rises more than ~0.03 m
relative to the episode start. A stronger check: during the hold, the distance
between the end effector and the cube must stay constant — in verified successful
grasps it held at **0.094 m**; if it opens up, the grasp slipped.

### 2. Export to LeRobot v2.1

`code/starlink_cube_rl/export_lerobot.py` (and the older, validated
`code/franka_cube/export_lerobot.py`).

The loader is `starVLA/dataloader/gr00t_lerobot/datasets.py`; it wants this exact
layout (it calls the layout "v2.0" internally, `info.json` reports `v2.1`):

```
<dataset>/
├── meta/info.json        data_path + video_path templates, chunks_size, fps, features
├── meta/modality.json    state/action/video keys -> parquet column (original_key) + start/end slice
├── meta/episodes.jsonl   {"episode_index", "tasks", "length"}
├── meta/tasks.jsonl      {"task_index", "task"}
├── data/chunk-XXX/episode_XXXXXX.parquet     must contain a timestamp column (loader asserts it)
└── videos/chunk-XXX/<original_key>/episode_XXXXXX.mp4
```

- `features[original_key]` needs `shape` plus `names` containing
  `height`/`width`/`channel`, and an `info.video.*` block with fps and channels.
- Videos are decoded with `video_backend=torchvision_av`; encode real H.264 mp4.
- **Do not write `meta/stats_gr00t.json`** — the loader computes it from the
  parquet files and caches it.
- `meta/modality.json` sub-key names must match the DataConfig's
  `state_keys`/`action_keys` or the loader's integrity check fails.

### 3. Register a DataConfig

Training configs are overlays under
`code/starlink_cube_rl/starvla_overlay/`, e.g.
`examples/StarlinkCube/train_files/`. Keep new files additive; do not edit the
shipped `examples/Franka/` or `examples/calvin/` templates.

A working StarlinkCube action-head config (verified):
```yaml
name: QwenGR00T
base_vlm: /data/VLA_model/StarVLA/repo/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
action_dim: 7
state_dim: 7
action_horizon: 8
data_mix: starlink_cube_vla
action_type: delta_ee
max_train_steps: 10000
freeze_modules: qwen_vl_interface      # VLM frozen: this trains the action head only
```

### 4. Finetune

`starvla_overlay/train_starvla_no_deepspeed.py` (single-GPU path) or the
accelerate configs in `starvla_overlay/`. Checkpoints land in
`runs/<run-id>/checkpoints/steps_N_pytorch_model.pt` plus `final_model/`.

**Loading a pretrained checkpoint is `strict=True`.** `load_pretrained_backbones`
in `starVLA/model/framework/base_framework.py` logs missing/unexpected keys and
then **raises**; the `reload_modules` argument is accepted but unused. So a
changed `action_dim` cannot be warm-started without writing a partial loader
(drop the mismatched keys, then `strict=False`). **Prefer keeping `action_dim`
identical to the checkpoint you start from.**

### 5. Closed-loop evaluation

Entry points: `run_starvla_closed_loop_smoke.py` (client) plus a policy server.
The server is the StarVLA websocket server; launch it detached — a short-lived
ssh session kills background children:

```bash
cd /data/VLA_model/StarVLA/repo
setsid nohup env CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD \
  /data/anaconda3/envs/starvla_env/bin/python deployment/model_server/server_policy.py \
  --ckpt_path <checkpoint.pt> --port <port> --use_bf16 \
  > run.log 2>&1 < /dev/null &
# ready when the log shows: websockets.server: server listening on 0.0.0.0:<port>
```

Each evaluation run writes: `summary.json` (episodes / successes / success_rate /
per-episode outcomes), `actions.jsonl`, `closed-loop.mp4`, `evaluator.log`,
`policy-server.log`.

**A 16-episode evaluation cannot support an improvement claim.** See the
`sim-driven-policy-iteration` skill before drawing any conclusion from one.

## Traps that silently corrupt results

These were all hit for real.

**Unnormalization applied twice.** The official Franka README describes a server
that returns *normalized* actions and leaves unnormalization to the client. The
server deployed here already unnormalizes server-side
(`PolicyNormProcessor.unapply_actions`). Doing both inverts the transform twice
and destroys the action scale. Decide which side owns it and enforce it.

**Response key differs by deployment.** The README documents
`data.normalized_actions`; this deployment answers `data.actions`. Try
`normalized_actions`, `actions`, then `action` — do not hard-code one.

**State width.** The Franka checkpoint's `state_encoder` takes `(1, 7)`. The
server metadata advertises an eighth pad field; sending it fails with a
`1x8 -> 7x1024` matrix error. Omit it.

**Gripper sign.** starlink's `control_gripper` takes `[0, 1]` with **1 = open**.
StarVLA's Franka convention is `[-1, +1]` with **+1 = open**. Same direction, but
write the mapping down explicitly rather than assuming it.

**The starlink arm has 6 revolute joints, not 7.** Franka is a 7-DoF *redundant*
arm; starlink is 6-DoF and *non-redundant*. End-effector poses that Franka reaches
comfortably may be singular or unreachable for starlink, and joint-limit
asymmetries differ. Log IK failure rates and watch for singularities.

**Action semantics are metres and radians, `min_max` normalized.** Position deltas
in metres, orientation deltas in radians. Do not carry over a normalized ±1
interpretation from another dataset.

**Genesis specifics:**
- One scene per process — a parameter sweep needs one process per value.
- `link.get_pos()` / `get_quat()` / `get_dofs_position()` return CUDA tensors;
  `.detach().cpu()` before numpy.
- `get_links_pos` needs **local** indices (`.idx_local`); global `.idx` yields
  `(n,0,3)` and then a device-side assert. Reading link quaternions right after a
  kinematics update can trip the same assert.
- `forward_kinematics()` may return without moving links; drive the pose with
  `set_qpos(..., zero_forward=False)` semantics instead.
- The MJCF neutral pose folds the arm upward, far from the table. Servo to a
  reachable ready pose before starting an episode.
- The actuator tracks only about **0.0008 m/step**. Travel is slow by design;
  budget steps accordingly and do not "fix" it with higher PD gains — a policy
  trained that way would not transfer.

**Throughput.** A clean stepping loop reaches ~427 env-steps/s single-env, but a
loop that synchronizes to host 3–4 times per step (reading positions, quaternions,
gripper width) drops to ~50 — an **8× loss**. Keep the step path on GPU tensors.
Rendering costs only ~2.2×; batching is what scales (256 envs ≈ 93,000
env-steps/s).

**Background jobs and ssh.** Do as much as possible per ssh connection — one
connection per file has tripped the server's rate limiter, locking out all ports
for ~12 minutes. When piping a script over ssh, base64-encode it to avoid quote
and CR mangling, and remember that variables inside a quoted `<<'EOF'` heredoc do
not expand — redefine them in the inner script.

**Read-only directories.** Never write under `/data/pengyuxin`,
`/data/jingguohao`, or `/data/adammaytoussi`. `/data/VLA_model` is shared;
additive writes are fine and the `StarVLA/` subtree is ours.

**Release the GPU.** Stop policy servers when done, confirm the port is free, and
confirm `nvidia-smi` shows the memory released.

## Verification gates

Do not advance a stage until its gate passes.

| After | Gate |
|---|---|
| Export | `code/franka_cube/validate_lerobot.py` loads the dataset with the official `LeRobotSingleDataset` and reports action/image shapes |
| DataConfig | the loader accepts the config and reads the dataset |
| Training | a 1-step smoke run completes before a full run; loss decreases; a checkpoint is written |
| Evaluation | `summary.json` exists with an episode count large enough to be meaningful |
