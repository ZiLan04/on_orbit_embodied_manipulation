# documents/ — 规范与交接文档

> 这个目录放**当前有效**的技术规范。新建任何文件、目录、函数、配置之前先读这里。
>
> | 文件 | 内容 |
> |---|---|
> | `README.md` | 本文：目录用途、目标结构、旧→新迁移映射 |
> | `code-standards.md` | 写代码与新建文件时必须遵守的规则 |
> | `code-map.md` | **服务器上每个文件/目录是干什么的**（交接用，重构后要同步更新） |
>
> `documents/`（规则，会更新）与 `docs/`（历史调研，只增不改）是两回事，不要混。

---

## 一、接手先读什么

按这个顺序，20 分钟能上手：

1. **`code-map.md` §0 和 §1** — 全貌，以及"代码是怎么到这里来的、哪些会随 deploy 更新、哪些不会"。**§1 里有一条会咬人的**：正在跑的 `code/starlink_cube_rl/` 不受 `deploy.ps1` 管理。
2. **`code-map.md` §2.2** — 当前主线的 20 个文件逐个作用。
3. **`code-map.md` §4** — 一个已知功能缺陷（动作契约不一致）。**不属于重构范围**，但影响所有已产出数据的解读。
4. **`code-standards.md` §10** — 动手前必须先做的三件事。

---

## 二、目标目录结构

对应 `code-standards.md` 的规则。**这是服务器 `/data/Yanhan/genesis` 的目标形态**（因为实际开发在服务器上）。

```
/data/Yanhan/genesis/
│
├── documents/                       ← 规范（本文所在）
│
├── code/                            ← 只放我们自己写的源码
│   │
│   ├── tasks/                       ← 一个任务一个目录
│   │   ├── starlink_cube_grasp/     ★ 当前主线：RL 环境 + 采集 + PPO + VLA 闭环
│   │   │   ├── env_starlink_cube_delta_eef.py
│   │   │   ├── collect_oracle_rollouts.py
│   │   │   ├── collect_ppo_rollouts.py
│   │   │   ├── train_ppo_batched.py
│   │   │   ├── eval_ppo_deterministic.py
│   │   │   ├── eval_starvla_closed_loop.py
│   │   │   ├── serve_starlink_cube_policy.py
│   │   │   ├── smoke_env.py  smoke_reward_oracle.py  smoke_ppo_batched.py …
│   │   │   ├── test_action_control.py  test_task_contract.py
│   │   │   └── config/
│   │   │       ├── env.yaml
│   │   │       ├── ppo.yaml
│   │   │       ├── action_head.yaml
│   │   │       └── data_registry.py
│   │   │
│   │   ├── starlink_satellite_grasp/   ← 卫星侧 stage-2 抓取位姿
│   │   ├── franka_cube_grasp/          ← 早期 Franka + cube 场景
│   │   └── vla_jepa_probe/             ← JEPA 线（已停用，留档）
│   │
│   ├── utils/                       ← 跨任务共享，按用途分子目录
│   │   ├── control/                 ← 动作契约与控制器
│   │   ├── data/                    ← LeRobot 导出与校验
│   │   ├── sim/                     ← Genesis 共性：相机、headless 初始化、批量封装
│   │   ├── geometry/                ← 位姿 / 四元数 / RPY / IK 辅助
│   │   └── io/                      ← 文件、日志、录像
│   │
│   ├── third_party/                 ← vendored 第三方，不算我们的源码
│   │   └── starvla_overlay/starVLA/
│   │
│   ├── bridges/                     ← 与他人共管的边界
│   │   └── starvla_bridge/          ← 由服务器侧的人维护，deploy 不覆盖
│   │
│   └── backups/                     ← 改动前的复制件
│
├── spaceman_core/                   ← 上游仿真树，只读参照，**不在 code/ 内**
├── runs/                            ← 运行输出
├── recovered/                       ← 事故恢复物，不要删
├── README.md
├── CANONICAL-DEVIATIONS.md
└── DEPLOYED.json
```

**三条结构性要求**：

1. `spaceman_core/` 永远在 `code/` 外面（现在已经是，保持）
2. `code/` 里不出现上游代码，vendored 的放 `third_party/`
3. 每个任务目录自带 `config/`；跨任务的放 `utils/`

---

## 三、旧 → 新 迁移映射（提案）

标 ✅ 是机械搬动、低风险；标 ⚠ 需要判断，**动手前先确认**。

### 3.1 `code/starlink_cube_rl/` → `code/tasks/starlink_cube_grasp/`

| 现在 | 目标 | |
|---|---|---|
| `delta_eef_control.py` | `utils/control/delta_eef_control.py` | ✅ 被 RL 和 VLA 两侧共用 |
| `task_contract.py` | 留在任务目录，或 `utils/control/` | ⚠ 目前只服务一个任务 |
| `starlink_cube_delta_eef_env.py` | `env_starlink_cube_delta_eef.py` | ✅ |
| `gymnasium_adapter.py` | `utils/sim/adapter_gymnasium.py` | ✅ |
| `sb3_vecenv_adapter.py` | `utils/sim/adapter_sb3_vecenv.py` | ✅ |
| `collect_oracle_rgb_rollouts.py`<br>`collect_oracle_recovery_rgb_rollouts.py`<br>`collect_ppo_rgb_rollouts.py` | `collect_oracle_rollouts.py`<br>`collect_oracle_recovery_rollouts.py`<br>`collect_ppo_rollouts.py` | ✅ |
| `export_lerobot.py` | `utils/data/export_lerobot.py` | ⚠ 与 `code/franka_cube/export_lerobot.py` 同名，需先确认是否同一份 |
| `run_batched_ppo_train.py` | `train_ppo_batched.py` | ✅ |
| `evaluate_batched_ppo.py` | `eval_ppo_deterministic.py` | ✅ |
| `run_starvla_closed_loop_smoke.py` | `eval_starvla_closed_loop.py` | ✅ |
| `run_delta_eef_env_smoke.py`<br>`run_delta_eef_oracle_smoke.py`<br>`run_ppo_integration_smoke.py`<br>`run_batched_ppo_integration_smoke.py`<br>`run_real_cube_smoke.py` | `smoke_env.py`<br>`smoke_reward_oracle.py`<br>`smoke_ppo_integration.py`<br>`smoke_ppo_batched.py`<br>`smoke_real_cube.py` | ✅ |
| `test_delta_eef_control.py` | `test_action_control.py`（⚠ 断言也要一起改） | ⚠ 见 `code-map.md` §4 |
| `run_guarded_starvla.sh` | 任务目录下保留 | ✅ |
| `collect_oracle_rgb_rollouts.py.orig` | **删除**，历史进 `backups/` | ✅ |
| `starlink_cube_rl/starvla_overlay/starVLA/**`（169 文件） | `code/third_party/starvla_overlay/starVLA/` | ⚠ 要同时改所有 import 路径和启动脚本 |
| `starvla_overlay/train_starvla_no_deepspeed.py`<br>`serve_starlink_cube_policy.py`<br>`serve_starlink_cube_policy_traced.py`<br>`validate_starvla_loader.py` | 任务目录下保留 | ⚠ 它们要能 import 到 third_party 里的 starVLA |
| `starvla_overlay/examples/.../data_config.py` | `tasks/starlink_cube_grasp/config/data_registry.py` | ⚠ StarVLA 对路径有约定，改前先验证能加载 |
| `starvla_overlay/examples/.../*.yaml`（5 个） | `tasks/starlink_cube_grasp/config/` | ⚠ 同上 |
| `starvla_overlay/accelerate_one_gpu*.yaml` | `tasks/starlink_cube_grasp/config/` 或 `utils/sim/` | ⚠ |

### 3.2 `code/genesis_tools/` → 拆分

| 现在 | 目标 | |
|---|---|---|
| `tasks/scripted_grasp.py` | `tasks/starlink_satellite_grasp/oracle_scripted_grasp.py` | ✅ |
| `tasks/kinematic_demo_gen.py` | `tasks/vla_jepa_probe/` 或独立对齐任务 | ⚠ 需确认归属 |
| `tasks/probe_arm_geometry.py` | `utils/geometry/probe_arm_workspace.py` | ✅ |
| `healthcheck/ik_reachability.py`<br>`healthcheck/reach_scan.py` | `utils/geometry/` | ✅ |
| `healthcheck/stage2_action_control.py` | `tasks/starlink_satellite_grasp/` | ⚠ |
| `healthcheck/starvla_healthcheck.py` | `utils/io/`（服务探针） | ⚠ |
| `eval/eval_stage2_batch.py` | `tasks/starlink_satellite_grasp/eval_stage2_batched.py` | ✅ |
| `eval/eval_rl_grasp.py` | `tasks/starlink_satellite_grasp/` | ⚠ 确认评的是哪个任务 |
| `eval/jepa_*.py`（3 个） | `tasks/vla_jepa_probe/` | ✅ |
| `analysis/analyze_alignment_log.py` | `tasks/vla_jepa_probe/` 或 `utils/io/` | ⚠ |
| `rl/run_stage2_horizon.py` | `tasks/starlink_satellite_grasp/train_stage2_horizon.py` | ✅ |

### 3.3 `code/franka_cube/` → `code/tasks/franka_cube_grasp/`

| 现在 | 目标 | |
|---|---|---|
| `franka_cube_starvla.py` | `scene_franka_cube_starvla.py` | ✅ |
| `export_lerobot.py` / `validate_lerobot.py` | `utils/data/`，与 3.1 那份合并成一份 | ⚠ 先 diff 确认差异 |
| `bench_parallel.py` | `utils/sim/bench_parallel.py` | ✅ |

### 3.4 `code/starvla_bridge/` → `code/bridges/starvla_bridge/`

⚠ **建议只做搬动，不做重构。**

因为这个目录**由服务器侧的人维护**，`deploy.ps1` 明确排除它。搬动本身已经会影响到对方。建议先和对方确认再动。

另外 `run_starvla_genesis_bridge_7d_v2..v7.py` 这八个版本快照应当清理——按规范 §5.2，版本号不进文件名。历史用 `backups/` 保留。

---

## 四、迁移执行顺序

先做不依赖判断的，再做需要决定的。**每一步都要能独立回退。**

```
第 0 步  全量备份：tar 一份整个 code/ 到 backups/code-YYYYMMDD.tar.gz
         （服务器这棵树没有 git，这是唯一的退路）
第 1 步  建目录骨架：code/tasks/ code/utils/ code/third_party/ code/backups/
第 2 步  搬 utils/（低风险，被多任务共用；搬完逐个 import 检查）
第 3 步  搬 tasks/starlink_cube_grasp/（当前主线；搬完跑 smoke，
         再拿 runs/ 里一次既有评测复跑，逐回合结果必须一致）
第 4 步  搬第三方 vendored 到 third_party/（要同时改 import 和启动脚本）
第 5 步  搬其余任务目录（每个搬完跑该任务的 smoke）
第 6 步  starvla_bridge（先和对方确认）
第 7 步  更新 code-map.md，确认 deploy 路径也要跟着改
```

**第 3 步的"逐回合结果必须一致"是可以真正验证的判据**：这个项目的评测在同种子下完全确定性（已验证：步数上限 128 vs 256 两次跑，64 回合逐回合结果完全一致）。不要用"看起来差不多"代替。

**别忘了 deploy 路径**：结构搬完之后，`deploy.ps1` 的目标路径和 `Dirs` 参数必须同步改，否则下次 deploy 又会在旧位置造一份。

---

## 五、这份文档本身怎么维护

- 规范变了 → 改 `code-standards.md`
- 结构变了 → 改本文的"目标结构"和"迁移映射"，已完成的项标 ✅
- 文件增删 → 改 `code-map.md`
- **三者必须与代码同步。**文档描述的结构和实际结构不一致时，视为文档坏了，优先修文档
