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
│   ├── utils/                       ← cross-task code, split by purpose
│   │   ├── control/                 ← action contract and controllers
│   │   ├── data/                    ← LeRobot export and validation
│   │   ├── sim/                     ← shared Genesis concerns: cameras, headless init, batched wrappers
│   │   ├── geometry/                ← poses, quaternions, RPY, IK helpers
│   │   └── io/                      ← files, logs, video
│   │
│   ├── third_party/                 ← vendored third-party, not our source
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
2. No upstream code inside `code/`. Vendored frameworks go to `third_party/`.
3. Every task directory carries its own `config/`; anything shared goes in `utils/`.

---

## 3. Migration map

✅ = mechanical move, low risk. ⚠ = needs a judgement call; **confirm before moving**.

### 3.1 `code/starlink_cube_rl/` → `code/tasks/starlink_cube_grasp/`

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

Separately, the eight `run_starvla_genesis_bridge_7d_v*.py` snapshots should be collapsed. Per `code-standards.en.md` §5.2, version numbers do not belong in file names; keep the history in `backups/`.

---

## 4. Order of execution

Do the moves that need no judgement first. **Every step must be independently revertible.**

```
Step 0   Full backup: tar the whole code/ into backups/code-YYYYMMDD.tar.gz
         (there is no git here — this is the only fallback)
Step 1   Create the skeleton: code/tasks/ code/utils/ code/third_party/ code/bridges/ code/backups/
Step 2   Move utils/ (low risk, shared; check imports as you go)
Step 3   Move tasks/starlink_cube_grasp/ (the main line)
         then run the smoke tests, then re-run one existing evaluation from runs/
         and require identical per-episode results
Step 4   Move the vendored framework to third_party/
         (imports and every launcher must change in the same step)
Step 5   Move the remaining task directories (smoke test after each)
Step 6   starvla_bridge — agree with the other person first
Step 7   Update code-map.en.md, and update the deploy path to match
```

**Step 3's "identical per-episode results" is a real check, not a formality.** Evaluation here is fully deterministic under a fixed seed — measured: the same seed set at step cap 128 and at step cap 256 produced identical results across all 64 episodes, episode by episode. Do not substitute "looks about the same".

**Do not forget the deploy path.** Once the structure moves, `deploy.ps1`'s `Dirs` parameter and its target path must change in the same commit — otherwise the next deploy recreates the old layout beside the new one.

---

## 5. Keeping these documents honest

- Rules change → `code-standards.en.md` (and its Chinese counterpart)
- Structure changes → this file: update the target structure and mark completed migrations
- Files added or removed → `code-map.en.md`

**These three must stay in sync with the code.** When the documented structure and the real structure disagree, the documentation is what is broken, and fixing it takes priority over the next refactor step.
