# documents/ — Standards and Handover

> This directory holds the **currently binding** technical standards. Read it before creating any file, directory, function or configuration.
>
> | File | Content |
> |---|---|
> | `README.en.md` | This file: what the directory is for, the target structure, the migration map |
> | `code-standards.en.md` | The rules for writing code and creating files |
> | `code-map.en.md` | What every existing file and directory does (server tree) |
>
> Chinese versions of all three exist alongside. `documents/` (binding rules, updated) and `docs/` (historical investigation, append-only) are different things — do not merge them.

---

## 1. Where to start

In this order. Twenty minutes gets you productive.

1. **`code-map.en.md` §0 and §1** — the shape of the tree, and how code gets here: which directories a deploy updates and which it does not. §1 contains a trap: the directory that actually runs is *not* the one `deploy.ps1` writes to.
2. **`code-map.en.md` §2.2** — the 20 files of the current main line, one line each.
3. **`code-map.en.md` §1.1** — two implicit dependencies. Both will break silently during any restructuring if you have not read them.
4. **`code-map.en.md` §4** — a known functional defect in the action contract. It is *outside* the scope of the restructure, but it changes how every dataset produced so far must be read.
5. **`code-standards.en.md` §10** — the three things to do before touching anything.

---

## 2. Target structure

The target form of `/data/Yanhan/genesis`, implementing the rules in `code-standards.en.md`.

```
/data/Yanhan/genesis/
│
├── documents/                       ← binding standards (this directory)
│
├── code/                            ← our own source only
│   │
│   ├── tasks/                       ← one directory per task
│   │   ├── starlink_cube_grasp/     ★ current main line: RL env + collection + PPO + VLA loop
│   │   │   ├── env_starlink_cube_delta_eef.py
│   │   │   ├── collect_oracle_rollouts.py
│   │   │   ├── collect_ppo_rollouts.py
│   │   │   ├── train_ppo_batched.py
│   │   │   ├── eval_ppo_deterministic.py
│   │   │   ├── eval_starvla_closed_loop.py
│   │   │   ├── serve_starlink_cube_policy.py
│   │   │   ├── smoke_env.py  smoke_reward_oracle.py  smoke_ppo_batched.py  …
│   │   │   ├── test_action_control.py  test_task_contract.py
│   │   │   └── config/
│   │   │       ├── env.yaml
│   │   │       ├── ppo.yaml
│   │   │       ├── action_head.yaml
│   │   │       └── data_registry.py
│   │   │
│   │   ├── starlink_satellite_grasp/   ← satellite-side stage-2 grasp pose
│   │   ├── franka_cube_grasp/          ← earlier Franka + cube scene
│   │   └── vla_jepa_probe/             ← retired, kept for reference
│   │
│   ├── utils/                       ← cross-task code (ON DEMAND — absent today, see note)
│   │   ├── control/                 ← action contract and controllers
│   │   ├── data/                    ← LeRobot export and validation
│   │   ├── sim/                     ← shared Genesis concerns: cameras, headless init, batched wrappers
│   │   ├── geometry/                ← poses, quaternions, RPY, IK helpers
│   │   └── io/                      ← files, logs, video
│   │
│   ├── third_party/                 ← vendored third-party (ON DEMAND — absent today)
│   │   └── starvla_overlay/starVLA/
│   │
│   ├── bridges/                     ← shared-ownership boundary
│   │   └── starvla_bridge/          ← maintained by another person; deploy never touches it
│   │
│   └── backups/                     ← copies taken before edits
│
├── spaceman_core/                   ← upstream simulation tree, read-only, NOT inside code/
├── runs/                            ← run outputs
├── recovered/                       ← incident artifacts, do not delete
├── README.md
├── CANONICAL-DEVIATIONS.md
└── DEPLOYED.json
```

Three structural invariants:

1. `spaceman_core/` always sits **outside** `code/`. It already does; keep it that way.
2. No third-party source inside `code/`. StarVLA is currently an external dependency (§1.4),
   so there is **no `third_party/` directory yet** — create it only if something ever has to
   be vendored.
3. `utils/` is created **on demand**, not up front: the day a second task needs the same
   code. Until then there is no `utils/` at all, and every module lives in its task.

---

## 3. Migration map — server cut-over plan, **not** a repo to-do list

> **Status. This repository is already in the target layout of §2.** What follows is a
> *proposal that was written before the move* and then partly not followed, because two of
> the moves turned out to be wrong. It is kept for one reason only: it is the sketch for the
> still-pending **server** cut-over, where the pre-restructure layout
> (`code/starlink_cube_rl/`, `code/genesis_tools/`, `code/franka_cube/`) is what actually
> runs today.
>
> Read the two boxes below first, then treat the tables as historical intent.

### What the repo actually did

| done | |
|---|---|
| one directory per task under `code/tasks/` | 4 tasks |
| `code/bridges/starvla_bridge/` in its own place | it is a shared boundary |
| probes moved out of a `utils/` that should not have existed | into `tasks/*/probes/` |
| filenames left unchanged | behaviour neutrality was the priority; see §4 |
| no `third_party/`, no `config/` | see the two boxes below |

### What was deliberately rejected, and why

**`utils/control/delta_eef_control.py` — rejected.** Five modules
(`delta_eef_control`, `task_contract`, `starlink_cube_delta_eef_env`, `gymnasium_adapter`,
`sb3_vecenv_adapter`) are imported as *flat top-level modules* by 16 files in the same
directory. Moving any one of them to `utils/` breaks those imports unless the new path also
goes on `PYTHONPATH`. And by our own rule, `utils/` only exists once a second task needs the
code — today each has exactly one consumer. So they stay together in the task directory.

**`config/` for the StarVLA YAMLs — rejected.** `serve_starlink_cube_policy*.py` hard-codes
`<overlay>/examples/StarlinkCube/train_files/data_registry` on `sys.path`. That layout is a
StarVLA convention, and the whole `starvla_overlay/` tree mirrors the framework's own
structure. Relocating the YAMLs would break the convention and both launchers. Rule §7 of the
standards applies to configs *we* load; these load through someone else's path convention.

**Deleting `run_starvla_genesis_bridge_7d_v2..v7.py` — impossible as written.** They are a
dependency *chain* (`v3` imports `v2`, `v4` imports `v3`, … `v7` imports `v6`), not parallel
snapshots. Deleting `v2..v6` breaks `v7`. Flattening them is a refactor with its own
verification.

---

### 3.1 `code/starlink_cube_rl/` → `code/tasks/starlink_cube_grasp/` (historical proposal)

| Now | Target | |
|---|---|---|
| `delta_eef_control.py` | `utils/control/delta_eef_control.py` | ✅ shared by the RL and VLA sides |
| `task_contract.py` | stays in the task directory, or `utils/control/` | ⚠ currently serves one task |
| `starlink_cube_delta_eef_env.py` | `env_starlink_cube_delta_eef.py` | ✅ |
| `gymnasium_adapter.py` | `utils/sim/adapter_gymnasium.py` | ✅ |
| `sb3_vecenv_adapter.py` | `utils/sim/adapter_sb3_vecenv.py` | ✅ |
| `collect_oracle_rgb_rollouts.py`<br>`collect_oracle_recovery_rgb_rollouts.py`<br>`collect_ppo_rgb_rollouts.py` | `collect_oracle_rollouts.py`<br>`collect_oracle_recovery_rollouts.py`<br>`collect_ppo_rollouts.py` | ✅ |
| `export_lerobot.py` | `utils/data/export_lerobot.py` | ⚠ a same-named file exists in `code/franka_cube/`; diff them before merging |
| `run_batched_ppo_train.py` | `train_ppo_batched.py` | ✅ |
| `evaluate_batched_ppo.py` | `eval_ppo_deterministic.py` | ✅ |
| `run_starvla_closed_loop_smoke.py` | `eval_starvla_closed_loop.py` | ✅ |
| `run_delta_eef_env_smoke.py`<br>`run_delta_eef_oracle_smoke.py`<br>`run_ppo_integration_smoke.py`<br>`run_batched_ppo_integration_smoke.py`<br>`run_real_cube_smoke.py` | `smoke_env.py`<br>`smoke_reward_oracle.py`<br>`smoke_ppo_integration.py`<br>`smoke_ppo_batched.py`<br>`smoke_real_cube.py` | ✅ |
| `test_delta_eef_control.py` | `test_action_control.py` | ⚠ the assertion has to change with it — see `code-map.en.md` §4 |
| `run_guarded_starvla.sh` | stays in the task directory | ✅ |
| `collect_oracle_rgb_rollouts.py.orig` | **delete**; history belongs in `backups/` | ✅ |
| `starlink_cube_rl/starvla_overlay/starVLA/**` (169 files) | `code/third_party/starvla_overlay/starVLA/` | ⚠ every import and every launcher must change with it |
| `starvla_overlay/train_starvla_no_deepspeed.py`<br>`serve_starlink_cube_policy.py`<br>`serve_starlink_cube_policy_traced.py`<br>`validate_starvla_loader.py` | stays in the task directory | ⚠ must still be able to import starVLA from `third_party/` |
| `starvla_overlay/examples/.../data_config.py` | `tasks/starlink_cube_grasp/config/data_registry.py` | ⚠ StarVLA has path expectations; verify the dataset still loads |
| `starvla_overlay/examples/.../*.yaml` (5 files) | `tasks/starlink_cube_grasp/config/` | ⚠ same as above |
| `starvla_overlay/accelerate_one_gpu*.yaml` | task `config/`, or `utils/sim/` | ⚠ |

**Before moving anything here, read `code-map.en.md` §1.1** — two dependencies cross this boundary and will break.

### 3.2 `code/genesis_tools/` → split

| Now | Target | |
|---|---|---|
| `tasks/scripted_grasp.py` | `tasks/starlink_satellite_grasp/oracle_scripted_grasp.py` | ✅ |
| `tasks/kinematic_demo_gen.py` | `tasks/vla_jepa_probe/` or a dedicated alignment task | ⚠ ownership needs confirming |
| `tasks/probe_arm_geometry.py` | `utils/geometry/probe_arm_workspace.py` | ✅ |
| `healthcheck/ik_reachability.py`<br>`healthcheck/reach_scan.py` | `utils/geometry/` | ✅ |
| `healthcheck/stage2_action_control.py` | `tasks/starlink_satellite_grasp/` | ⚠ |
| `healthcheck/starvla_healthcheck.py` | `utils/io/` (it probes a service) | ⚠ |
| `eval/eval_stage2_batch.py` | `tasks/starlink_satellite_grasp/eval_stage2_batched.py` | ✅ |
| `eval/eval_rl_grasp.py` | `tasks/starlink_satellite_grasp/` | ⚠ confirm which task it scores |
| `eval/jepa_*.py` (3 files) | `tasks/vla_jepa_probe/` | ✅ retired, kept for reference |
| `analysis/analyze_alignment_log.py` | `tasks/vla_jepa_probe/` or `utils/io/` | ⚠ |
| `rl/run_stage2_horizon.py` | `tasks/starlink_satellite_grasp/train_stage2_horizon.py` | ✅ |

### 3.3 `code/franka_cube/` → `code/tasks/franka_cube_grasp/`

| Now | Target | |
|---|---|---|
| `franka_cube_starvla.py` | `scene_franka_cube_starvla.py` | ✅ |
| `export_lerobot.py`, `validate_lerobot.py` | `utils/data/`, merged with the copies from §3.1 into one of each | ⚠ diff first |
| `bench_parallel.py` | `utils/sim/bench_parallel.py` | ✅ |

### 3.4 `code/starvla_bridge/` → `code/bridges/starvla_bridge/`

⚠ **Move only. Do not restructure.**

This directory is maintained by someone else on the server, and `deploy.ps1` deliberately excludes it. Moving it already affects them — agree with them first.

Separately, `run_starvla_genesis_bridge_7d_v2..v7.py` are **not** eight snapshots to delete —
they are a chain of imports (`v3` imports `v2`, … `v7` imports `v6`). Collapsing them is a
refactor that needs its own verification; it is not a cleanup. See `code-map.en.md` §0.

---

## 4. Server cut-over (still pending — deliberate)

The server runs the pre-restructure layout and is **kept that way** until the cut-over is
explicitly approved. When it happens, it is one reversible sequence:

```
Step 0   tar the whole server code/ into backups/   (no git there — this is the only fallback)
Step 1   rename it aside: code -> code_<date>_old   (keep it, do not delete)
Step 2   place the new tree at code/ and fix PYTHONPATH per code-map.en.md §1.1
Step 3   fix nothing else -- deploy.ps1 already targets the new layout
Step 4   re-run the four-level verification (code-map.en.md §4)
         pass criterion: 58/64 and failures at episodes [4, 29, 37, 46, 49, 59]
Step 5   only then remove code_<date>_old
```

**Level 4 is the real check, not a formality.** The closed-loop evaluation is deterministic
under a fixed seed: the same seed set produced identical per-episode `success`, `steps` and
initial `cube_position` from both the old and the new layout. Use exactly that comparison;
do not substitute "looks about the same".

**StarVLA is the one thing that cannot simply be moved.** It lives inside the old tree on the
server. Since we are not vendoring it, after the cut-over either leave that subtree where it
is and put its parent on `PYTHONPATH`, or move it to a stable dedicated path and update the
documented `PYTHONPATH`. Decide this at cut-over time, not before.

---

## 5. Keeping these documents honest

- Rules change → `code-standards.en.md` (and its Chinese counterpart)
- Structure changes → this file: update the target structure and mark completed migrations
- Files added or removed → `code-map.en.md`

**These three must stay in sync with the code.** When the documented structure and the real structure disagree, the documentation is what is broken, and fixing it takes priority over the next refactor step.
