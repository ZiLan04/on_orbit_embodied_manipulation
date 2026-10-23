# Code Standards

> Single source of truth for how code in `/data/Yanhan/genesis` is laid out, named and documented.
> **Read this before creating any file, directory, function or config.**
> Violations in new work are grounds for sending it back.

Version 1 · 2026-10-21 · Companion documents: `README.md` (target structure and migration map), `code-map.en.md` (what every existing file does).

---

## 0. The one rule above all others: behaviour must not change

This is a **structural** refactor. Directories, file names, function names and call flow may change. **Observable behaviour may not.** Same input, same output. Same command, same numbers.

Every change passes three gates. A change that skips one of them is not finished:

| Gate | Requirement |
|---|---|
| **1. A copy exists** | The original was copied aside before you touched it (§3) |
| **2. It still runs** | The task's own `smoke_*` / `test_*` passes (§9) |
| **3. It is documented** | `documents/code-map.en.md` reflects the result (§6) |

---

## 1. One directory per task, all under `tasks/`

Every experiment or task gets **its own subdirectory under `tasks/`**. The task's scene, collection, training, evaluation and configuration all live inside that one directory. Nothing about a task lives anywhere else.

```
code/tasks/
├── starlink_cube_grasp/          # one task = one directory
│   ├── env_starlink_cube_delta_eef.py
│   ├── collect_oracle_rollouts.py
│   ├── collect_ppo_rollouts.py
│   ├── train_ppo_batched.py
│   ├── eval_ppo_deterministic.py
│   ├── eval_starvla_closed_loop.py
│   └── config/
├── starlink_satellite_grasp/     # satellite-side stage-2 grasp pose
├── franka_cube_grasp/
└── vla_jepa_probe/               # retired, kept for reference
```

**How to decide whether a file belongs in `tasks/`**: does it exist *for one specific task*?

- Yes → the subdirectory for that task
- No, several tasks need it → `utils/` (§4)

**Not allowed**: task code sitting in the repository root, or collected into a `misc/` or `tmp/` folder.

**Why**: the single biggest cost right now is that nobody can tell which files belong together. `code/starlink_cube_rl/` is the current main line and is 20 files that only make sense as a set; the directory boundary is what makes that visible.

---

## 2. `code/` contains only our own source

`code/` holds **code we wrote and maintain**. Anything synced from upstream stays out.

Three categories are not ours, and each has a designated home:

| Category | Where it goes | Example already in this tree |
|---|---|---|
| Upstream simulation tree, read-only | A sibling directory **outside** `code/` | `spaceman_core/` — already correct, keep it that way |
| Vendored third-party framework | `code/third_party/` | `starlink_cube_rl/starvla_overlay/starVLA/` — 169 files currently mixed into the task directory |
| Upstream reference implementation we copied for comparison | `reference/` inside the task directory, keep the `_reference` suffix | `parallel_rl_reference.py`, `real_cube_grasp_reference.py` |

**The test**: if upstream releases a new version, do we *follow* it or *rewrite* it?

- Follow → not our source, keep it out of `code/`
- Rewrite / maintain ourselves → it is our source

**Why**: when something breaks, the first question is always "is this our bug or theirs?". A directory boundary answers it in one second.

---

## 3. `backups/` holds copies taken before edits

Before modifying any file that currently works, **copy it to `backups/` first**, then edit. Name the copy with a date.

```
code/backups/
├── delta_eef_control.py.20261021.bak
└── collect_ppo_rgb_rollouts.py.20261021.bak
```

Organise by task subdirectory; prune anything older than six months.

**This is a hard rule, not a suggestion.** `/data/Yanhan/genesis` has **no version control** — the only `.git` in the tree is inside `spaceman_core/`, which is an upstream clone. Every file under `code/` has exactly one version: the current one. There is no `git checkout`, no history, no rollback.

`backups/` is therefore the *only* safety net that exists, and it is why §0 lists it as gate 1.

---

## 4. `utils/` holds cross-task code, split by purpose

Code used by more than one task goes in `utils/`. **Split by purpose, not by file type** — no `utils/py/`, no `utils/helpers/`.

```
code/utils/
├── control/     actions and controllers
│   └── delta_eef_control.py
├── data/        LeRobot import, export and validation
│   ├── export_lerobot.py
│   └── validate_lerobot.py
├── sim/         shared Genesis concerns: cameras, headless init, batched wrappers
├── geometry/    poses, quaternions, RPY, IK helpers
└── io/          files, logs, video
```

**How to decide**: the moment a second task needs it, move it. **Do not move it in advance** — "shared code" with exactly one user is premature abstraction and makes the tree harder to read, not easier.

**When you move something**: rename within the directory, then grep the whole tree for the old path and fix every import and every document reference. A move that leaves a stale import behind is worse than no move.

---

## 5. Naming

### 5.1 Functions and variables: `<what it does>_<what it operates on>_<what it uses>`

Three informative segments. The name alone should tell a reader what the thing does, what it takes, and what mechanism it uses — without opening the file.

```python
# Good
def normalize_action_minmax_torch(action, stats): ...
def clamp_delta_eef_metre(delta_xyz, limit_m): ...
def compute_success_hold_gpu(grasp_ready, lift, hold_count): ...
def sample_cube_pose_uniform(config, num_envs): ...

# Bad — no information about inputs or mechanism
def process(data): ...
def helper(x, y): ...
def do_thing(): ...
```

Use the third segment for the tool or framework: `_torch`, `_numpy`, `_gpu`, `_yaml`.

Aim for clarity, not length. If the name needs more than three segments to be honest, the function is probably doing more than one thing.

Keep the existing prefixes — do not rename for the sake of this rule:

- `test_` — tests (pytest depends on it)
- `run_` — directly executable entry points
- `serve_` — long-running service processes

### 5.2 Files: `<task>_<what the file is for>`

Task first, purpose second, all lowercase with underscores.

```
starlink_cube_env_delta_eef.py            # task + this is the environment
starlink_cube_collect_oracle_rollouts.py  # task + collects oracle trajectories
starlink_cube_eval_closed_loop.py         # task + closed-loop evaluation
starlink_cube_config_ppo.yaml             # task + PPO configuration

franka_cube_grasp_collect_rollouts.py
```

**Not allowed**: version numbers in file names. `run_starvla_genesis_bridge_7d_v2.py` through `_v7.py` are exactly the anti-pattern — eight near-identical files and no way to tell which is current without reading all of them. Keep history in `backups/`.

**Not allowed**: `.orig`, `.bak`, `_old`, `_new`, `_final`, `_copy` in the source tree. Those suffixes belong in `backups/` and nowhere else. (There is one right now: `collect_oracle_rgb_rollouts.py.orig`.)

### 5.3 Directories: lowercase, underscores, nouns

`starlink_cube/`, `delta_eef/`, `third_party/`. Avoid abbreviations unless they are genuinely standard in the field: `io`, `rl`, `eef`, `sim`.

---

## 6. New files and features must be explained

Creating a file, module, public function or config option requires **an explanation, not just code**.

### 6.1 Every file starts with a docstring

```python
"""One line on what this file does, and why it exists separately.

If it is not obvious, say why it cannot be merged into the neighbouring file.
"""
```

The existing code already does this well — all 41 self-written files under `code/` have one. Keep it up.

### 6.2 Register the file in `documents/code-map.en.md`

Add a row under the right section: what it is called, what it does in one line, which task it belongs to. **Unregistered means undelivered.**

### 6.3 Explain the "why", not just the "what", in three specific cases

1. **Counter-intuitive parameter values** — e.g. `cube_spawn_noise_m=(0.02, 0.02, 0.01)`. Say how that magnitude was chosen, and what breaks if it changes.
2. **Overriding a default** — e.g. "this call passes `normalized_actions=True` explicitly because the environment default disagrees with the policy's action space". That one comment would have saved several days.
3. **A bug that was fixed** — record the symptom, the cause, and how you verified the fix, in the code next to the fix.

Good models already in the tree: the header of `deploy.ps1` (lines 8–20) and `delta_eef_control.py` (lines 65–66).

---

## 7. Many parameters → a config file. Many config files → a directory. But only when warranted.

**The rule:**

- More than three magic numbers in one file, or one value that must stay consistent in several places → extract to config
- More than three config files → create a `config/` directory
- **Do not manufacture config for its own sake.** One caller, changed once — inline is clearer than a YAML file

```
code/tasks/starlink_cube_grasp/
├── config/
│   ├── env.yaml          # cube size, spawn offset and noise, episode cap
│   ├── ppo.yaml          # env count, learning rate, total steps
│   ├── action_head.yaml  # backbone, action head, training steps
│   └── eval.yaml         # episode count, seed, randomisation width
└── ...
```

**A config must be able to answer "which experiment was this?"** Write its purpose at the top of the file, and — this matters — **copy the config actually used into `runs/<run-id>/` when you run**. The existing code already does this. Keep doing it.

**The reverse requirement**: never write the same value in both code and config. The config is authoritative; code reads it and does not carry a duplicate default.

**One exemption.** Config files that a third-party framework loads **by its own path convention** stay where that framework expects them. `starvla_overlay/examples/StarlinkCube/train_files/*.yaml` is the current case: `serve_starlink_cube_policy*.py` hard-codes `.../examples/StarlinkCube/train_files/data_registry` on `sys.path`, and the whole `starvla_overlay/` tree mirrors StarVLA's structure. Relocating those files would break the convention and both launchers. This rule governs configs **we** load.

---

## 8. `documents/` is the single source of truth for standards

```
documents/
├── README.md             what this directory is, target structure, migration map
├── code-standards.md     this file (Chinese)
├── code-standards.en.md  this file (English)
├── code-map.md           what every file does (Chinese)
└── code-map.en.md        what every file does (English)
```

**Rules:**

- New technical standards, directory conventions and naming conventions go **only** here. Not in `docs/`, not in a README, not in a chat log.
- `docs/` holds historical investigation and design notes (append-only, never rewritten). `documents/` holds the **currently binding rules** and is updated as the refactor proceeds. They are different things; do not merge them.
- **Before an agent creates anything, it reads `code-standards.en.md`. After creating anything, it updates `code-map.en.md`.** This is what makes the convention survive past the person who set it up.

---

## 9. Refactor safety procedure

For structural-only work, in this order:

```
1. BACK UP     copy the files you will touch into backups/, with a date
2. RESTRUCTURE move, rename, split — do not change logic in the same step
3. SMOKE       run the task's run_*_smoke.py (now smoke_*.py) and its test_*.py
4. COMPARE     if the change touches the data-production path, re-run one existing
               evaluation and require identical per-episode results
5. REGISTER    update documents/code-map.en.md
6. DEPLOY      deploy.ps1 -DryRun first; confirm no "remote file is newer" warnings
               before a real deploy
```

**Step 4 is the one that gets skipped and the one that matters.**

It is genuinely verifiable here, which is why it is a requirement rather than a nice-to-have: **evaluation in this project is fully deterministic under a fixed seed.** Measured — running the identical seed set at step cap 128 and again at step cap 256 produced identical results across all 64 episodes, episode by episode, with the same six failures consuming the full budget both times.

So "identical per-episode results" is a real check. Do not replace it with "looks about the same".

Also: keep each step independently revertible. Do not accumulate five moves into one change.

---

## 10. Three things to do before anything else

In risk order:

**1. Take one full backup.** This tree has no version control; every byte under `code/` exists once.

```bash
cd /data/Yanhan/genesis && tar czf code/backups/code-$(date +%Y%m%d).tar.gz code/
```

This is the only fallback for every move that follows.

**2. Fix the deploy path mismatch.** `deploy.ps1` writes `genesis_tools/` to `code/genesis_tools/`, but the code that actually runs is in `code/starlink_cube_rl/`. Running a deploy today does not update the live code — it creates a duplicate beside it, and the two then drift. Before the restructure, at minimum make the deploy target and the live location agree. (Details in `code-map.en.md` §1.)

**3. The action-contract defect** (`code-map.en.md` §4). This is a known **functional** bug, not a structural one. It must be fixed and verified **separately** from the refactor, because it changes numerical behaviour and therefore invalidates the step-4 comparison above.

---

## 11. Change log

| Date | Change | Why |
|---|---|---|
| 2026-10-21 | Standards v1 established | Handing structural optimisation to a colleague; conventions need to be explicit and machine-followable |
| 2026-10-21 | §3 rewritten from "the repo is not under git" to "the deployment tree is not under git" | Development happens on the server, so the standards target the server tree |
