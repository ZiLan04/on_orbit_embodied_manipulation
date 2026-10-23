# on_orbit_embodied_manipulation

RL-assisted VLA self-learning for data-scarce on-orbit robotic arm satellite manipulation.

A closed loop that turns a simulated robot arm into a self-feeding data factory for a
vision-language-action model:

```
        Genesis simulation
                │
    ┌───────────┴───────────┐
    │                       │
scripted oracle         trained PPO policy          ← two interchangeable teachers
    │                       │
    └───────────┬───────────┘
                │  successful episodes only
                ▼
        raw RGB rollouts  (state / action / base_view / task)
                │
        LeRobot v2.1 export
                │
    StarVLA action-head finetuning  (VLM backbone frozen)
                │
    closed-loop evaluation in Genesis  (fixed seed set, per-episode jsonl + video)
                │
    failure diagnosis → one intervention → paired re-evaluation
                │
                └──────────── back to the top
```

The point of the loop is that every step is measured, and every change is attributable.
Evaluation is fully deterministic under a fixed seed, so a change either reproduces the
baseline episode-for-episode or it does not.

---

## Repository layout

```
code/
├── tasks/                        one directory per task
│   ├── starlink_cube_grasp/      ★ main line: RL env, data collection, PPO, VLA closed loop
│   ├── starlink_satellite_grasp/ satellite-side stage-2 grasp pose
│   ├── franka_cube_grasp/        earlier Franka Panda + cube scene
│   └── vla_jepa_probe/           retired JEPA line, kept for reference
├── utils/                        cross-task helpers, split by purpose
│   ├── geometry/                 reachability and IK probes
│   └── io/                       service probes and log analysis
└── bridges/
    └── starvla_bridge/           Genesis ↔ StarVLA WebSocket bridge

documents/                        binding standards; read before adding code
docs/                             historical investigation and design notes
deploy/                           one-way sync to the GPU server
.agents/skills/                   agent skills that encode this pipeline
```

**Start with `documents/code-map.en.md`** — it documents every file, how code reaches
the server, and the two implicit path dependencies that will break any restructuring
that ignores them.

---

## Quick start

Two conda environments, deliberately complementary. Do not install into either one to
patch a missing package.

| package | `Spaceman_env` (Genesis) | `starvla_env` (train / serve) |
|---|---|---|
| genesis, gymnasium, stable_baselines3 | ✓ | ✗ |
| pyarrow, websockets | ✗ | ✓ |
| imageio_ffmpeg | ✓ | ✗ |

```bash
export PYTHONPATH=<repo>/code/tasks/starlink_cube_grasp:\
<repo>/code/bridges/starvla_bridge:\
<path-to-spaceman_core>/src
```

Headless Genesis:

```bash
env -u DISPLAY PYOPENGL_USE_EGL=1 PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python ...
```

and the scene script must set `show_viewer=False`. Pin `CUDA_VISIBLE_DEVICES` explicitly.

### External dependency: StarVLA

The vendored StarVLA framework (`starVLA/`, ~100 files) is **not** in this repository.
The policy service runs on the GPU server at a fixed path and is treated as part of the
deployment environment. Add its parent directory to `PYTHONPATH`:

```
<starVLA parent directory containing the starVLA/ package>
```

The same applies to `spaceman_core/`, the upstream simulation tree.

---

## Measured results

| Policy | Result | Notes |
|---|---|---|
| PPO, deterministic | **245 / 256 = 95.7 %** | batched, deterministic evaluation |
| VLA action head, closed loop | **58 / 64 = 90.6 %** | 95 % CI [0.84, 0.98] |
| Scripted oracle, 1.25× randomisation | **32 / 35 = 91.4 %** | the teacher |

Scripted-oracle success versus cube randomisation width — this is why wide coverage has
to come from RL rather than from a hand-written controller:

| width | 1.0× | 1.25× | 1.5× | 2.0× | 4.0× |
|---|---|---|---|---|---|
| success | 96.0 % | 91.4 % | 73.3 % | 55.6 % | 6.7 % |

---

## Known issue

There is a **unit mismatch in the action contract**. The policy's action space is
declared as `Box(-1, 1, 7)`, while the controller interprets the same numbers as
physical deltas and clamps them to ±0.01 m / ±0.05 rad. 99 % of the action range is
therefore a dead zone; the policy degenerates into saturated bang-bang control and the
recorded labels retain only the sign. Any dataset produced through that path has
near-constant action labels.

`documents/code-map.en.md` §4 has the three call sites, the reference implementation
that is already correct, and the test assertion that currently encodes the defect as
expected behaviour. **Fix and verify this separately from any structural change.**

---

## Contributing

`documents/code-standards.en.md` is the binding rule set: one directory per task,
`utils/` only for genuinely shared code, `<what it does>_<what it operates on>_<what it
uses>` for functions, `<task>_<purpose>` for files, three magic numbers or more means a
config file. Behaviour must not change in a structural refactor, and the check for that
is identical per-episode evaluation results, not visual inspection.

## License

Apache 2.0 — see `LICENSE`.
