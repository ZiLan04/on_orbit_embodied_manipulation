# 服务器代码树功能说明（交接用）

> 范围：**`10.110.10.12:/data/Yanhan/genesis`**。
> **不含** `runs/`（跑出来的结果）、`__pycache__/`。
> 本文只描述服务器上实际存在的东西。
>
> 生成时间：2026-10-16。文件数与内容以当时实测为准。

---

## 0. 全貌

```
/data/Yanhan/genesis/
├── code/                 ← 我们写的代码（4 个目录，见 §2）
├── spaceman_core/        ← 上游仿真树，只读参照，不是我们的代码（见 §3）
├── runs/                 ← 167 个运行目录，结果（本文不展开）
├── recovered/            ← 2026-09-17 部署覆盖事故的恢复物（见 §5）
├── .agents/skills/       ← 2 个 skill（见 §6）
├── README.md             ← 权威源/部署目标的分工约定
├── CANONICAL-DEVIATIONS.md ← 相对上游的故意改动清单
└── DEPLOYED.json         ← 上次部署记录
```

**这个目录树没有 git。**唯一的 `.git` 在 `spaceman_core/` 里（那是上游克隆）。所以 `code/` 下的代码**没有任何版本历史、没有回滚手段**——改动前必须自己备份（见 §7）。

---

## 1. 代码从哪来（必读）

| 目录 | 来源 | 是否随 deploy 更新 |
|---|---|---|
| `code/genesis_tools/` | 本地仓库 `genesis_tools/` 经 `deploy/deploy.ps1` scp 同步 | ✅ 是 |
| `code/starlink_cube_rl/` | 本地 `genesis_tools/rl/starlink_cube/` **人工复制过来并改名** | ❌ **否** |
| `code/franka_cube/` | 本地 `genesis_tools/tasks/` 的 Franka 部分 + `rl/bench_parallel.py` **人工重组** | ❌ **否** |
| `code/starvla_bridge/` | 服务器侧的人自己维护 | ❌ 否（deploy 明确排除） |

`deploy.ps1` 只做一件事：`scp -r <repo>/genesis_tools  →  code/`，落点是 `code/genesis_tools/<原相对路径>`。

**⚠ 关键后果**：`deploy.ps1` 的落点与服务器实际布局不一致。

- 正在跑的主线代码在 `code/starlink_cube_rl/`
- 而 deploy 会把它写到 `code/genesis_tools/rl/starlink_cube/`
- 也就是说：**跑一次 deploy，不会更新 `code/starlink_cube_rl/`，反而会在 `code/genesis_tools/rl/starlink_cube/` 造一份重复副本。**

**所以现在本地改 `starlink_cube` 相关文件，不会自动到达服务器。**必须人工复制。这是接手时最要知道的一件事。

`DEPLOYED.json` 记录的上次部署是 `2026-09-18T04:22:47Z`、commit `98b7491`、`dirty: true`——**早于 starlink_cube 全部工作**（最早的相关 run 是 09-23）。可以确认这个子目录从未走过 deploy。

实测：本地 `genesis_tools/` 的 47 个文件里，**42 个在服务器上能按内容找到**（分布在上面三个目录），**5 个找不到**：

| 本地文件 | 服务器情况 |
|---|---|
| `rl/starlink_cube/collect_oracle_rgb_rollouts.py` | 存在但内容不同——**服务器版更新**：多了 `--cube-noise-scale` 和 `--seed` |
| `rl/starlink_cube/run_starvla_closed_loop_smoke.py` | 存在但内容不同——**服务器版更新**：多了 `--execution-horizon` 和 `--seed` |
| `rl/starlink_cube/parallel_rl_reference.py` | 不存在（全树 grep 零引用，不影响运行） |
| `rl/starlink_cube/real_cube_grasp_reference.py` | 不存在（同上） |
| `rl/starlink_cube/test_task_contract.py` | 不存在（少一份测试，不影响运行） |

**⚠ 版本方向：服务器超前于本地。**

上面前两条不是"服务器缺东西"，是**本地落后**。如果按老思路"本地 → 服务器"同步，会把 `--cube-noise-scale` / `--seed` / `--execution-horizon` 这三个功能**回退掉**。所以在 `starlink_cube_rl` 这一块，**别再假设本地是最新的**。

### 1.1 两条隐式依赖（重构时最容易踩）

**① 依赖 `spaceman_core/src/`**

`code/` 里大量 `import configs` / `import tasks` / `import RL` / `import envs` / `import robots` / `import controllers` / `import utils`，全部来自 `spaceman_core/src/`。也就是说——

**`code/` 不能独立运行**，必须把 `/data/Yanhan/genesis/spaceman_core/src` 加进 `PYTHONPATH`。

正确的最小可运行环境（实测通过）：

```bash
export PYTHONPATH=/data/Yanhan/genesis/code/starlink_cube_rl:\
/data/Yanhan/genesis/code/starvla_bridge:\
/data/Yanhan/genesis/spaceman_core/src
```

重构时如果只搬 `code/` 而不改 `PYTHONPATH`，全部会 import 失败。

**② `run_starvla_closed_loop_smoke.py` 依赖 `code/starvla_bridge/`**

该文件第 38 行 `from starvla_genesis_bridge import StarVLAPolicyClient`——**跨目录依赖**，而 `starvla_bridge/` 恰好是"他人维护、deploy 不同步"的那个目录。

所以重构时搬动 `starlink_cube_rl/` 会**直接弄断这条依赖**。要么先把这个客户端类复制/提取到 `utils/`，要么保持两者的相对位置不变。

### 1.2 运行环境完整性（实测）

两个 conda 环境是**互补的**，缺口是有意为之：

| 包 | Spaceman_env（Genesis） | starvla_env（训练/服务） |
|---|---|---|
| numpy / torch / msgpack | ✓ | ✓ |
| genesis | ✓ | ✗ |
| gymnasium / stable_baselines3 | ✓ | ✗ |
| pyarrow | ✗ | ✓ |
| websockets | ✗（有 symlink shim） | ✓ |
| imageio_ffmpeg | ✓ | ✗ |

绕开缺口的方式（**不要装包**）：

- `websockets` 在 Genesis 侧：`code/franka_cube/pylibs/websockets` 是指向 `starvla_env` site-packages 的符号链接，把 `code/franka_cube/pylibs` 放进 `PYTHONPATH` 即可
- 视频编码：用 starvla_env 跑导出，并显式传 Spaceman_env 里的 ffmpeg 二进制路径
- 导出 LeRobot 要用 starvla_env（需要 pyarrow）

已验证：41 个自有 `.py` 全部 `py_compile` 通过；`delta_eef_control` / `task_contract` / `gymnasium_adapter` / `sb3_vecenv_adapter` 在 Spaceman_env 下全部 import 成功；vendored `starVLA` 及其 `dataloader` / `model.framework` 在 starvla_env 下全部 import 成功。

---

## 2. `code/` — 我们的代码

```
code/
├── starlink_cube_rl/     ★ 当前主线：RL 环境 / 数据采集 / PPO / VLA 闭环
├── franka_cube/          早期 Franka + cube 场景
├── genesis_tools/        卫星侧任务、评测、健康检查、分析
└── starvla_bridge/       Genesis ↔ StarVLA 的 WebSocket 桥（他人维护）
```

### 2.1 `code/genesis_tools/` — 卫星侧任务与工具（14 个文件）

由 deploy 同步，保持本地相对路径。

```
genesis_tools/
├── analysis/
│   └── analyze_alignment_log.py        分析对齐运行日志，判断策略指令是否真在起作用
├── eval/
│   ├── eval_rl_grasp.py                按真正关心的指标评 RL 抓取策略（不看 reward 曲线）
│   ├── eval_stage2_batch.py            卫星侧 stage-2 抓取位姿 PPO 的批量评测
│   ├── jepa_bridge.py                  VLA-JEPA 驱动卫星臂的闭环桥（已停用）
│   ├── jepa_direction_pilot.py         问 VLA-JEPA 服务"它在场景里会怎么做"（已停用）
│   └── jepa_distance_sweep.py          VLA-JEPA 的方向在多远距离内还可信（已停用）
├── healthcheck/
│   ├── ik_reachability.py              机械臂够得到任务要求的位姿吗
│   ├── reach_scan.py                   夹爪最近能离物体多近
│   ├── stage2_action_control.py        stage-2 的成功判据真的能被动作控制吗
│   └── starvla_healthcheck.py          StarVLA 策略服务活着吗、加载的是哪个 ckpt
├── rl/
│   └── run_stage2_horizon.py           原版 stage-2 PPO 训练器，带可配 episode horizon
└── tasks/
    ├── kinematic_demo_gen.py           用运动学专家生成对齐用的演示数据
    ├── probe_arm_geometry.py           探机械臂几何与可达工作空间
    └── scripted_grasp.py               卫星机械臂的脚本抓取 oracle（非学习）
```

JEPA 那三个已停用，原因见 §8 的调研文档：它的输出是不随目标变化的常数向量。

### 2.2 `code/starlink_cube_rl/` — ★ 当前主线（20 个自有文件 + `starvla_overlay/`）

**最近所有工作都在这里。读代码从 `starlink_cube_delta_eef_env.py` 和 `delta_eef_control.py` 开始。**

```
starlink_cube_rl/
│
├── [动作与环境]
│   ├── delta_eef_control.py                ★ 7 维 delta-EE 动作的唯一契约定义（3.5 KB）
│   ├── starlink_cube_delta_eef_env.py      独立批量 Genesis 环境（14.6 KB）
│   ├── task_contract.py                    奖励项与成功判据（4.8 KB）
│   ├── gymnasium_adapter.py                单环境 Gymnasium 桥（2.7 KB）
│   └── sb3_vecenv_adapter.py               批量 SB3 VecEnv 桥（5.7 KB）
│
├── [数据采集]
│   ├── collect_oracle_rgb_rollouts.py      脚本专家驱动采集（5.6 KB）
│   ├── collect_oracle_recovery_rgb_rollouts.py  脚本专家 + 失败恢复状态（服务器独有）
│   ├── collect_ppo_rgb_rollouts.py         PPO 策略驱动采集（4.1 KB）
│   ├── collect_oracle_rgb_rollouts.py.orig 某次改动前的备份（应移入 backups/）
│   └── export_lerobot.py                   原始 dump → LeRobot v2.1
│
├── [训练与评测]
│   ├── run_batched_ppo_train.py            批量 PPO 训练启动器，带 checkpoint 续训
│   ├── evaluate_batched_ppo.py             独立确定性评测
│   └── run_starvla_closed_loop_smoke.py    VLA 闭环评测（连策略服务，取 8 步 chunk）
│
├── [冒烟 / 单元测试]
│   ├── run_delta_eef_env_smoke.py          环境本身能跑起来吗
│   ├── run_delta_eef_oracle_smoke.py       奖励判据物理上可行吗
│   ├── run_ppo_integration_smoke.py        单环境 SB3 接线
│   ├── run_batched_ppo_integration_smoke.py 批量 SB3 接线
│   ├── run_real_cube_smoke.py              既有物理抓取任务的 headless 冒烟
│   └── test_delta_eef_control.py           动作控制安全性与夹爪语义 ⚠ 见 §4
│
├── [运维]
│   └── run_guarded_starvla.sh              带资源守卫启动（服务器独有）
│
└── starvla_overlay/                        StarVLA 侧的覆盖层
    ├── train_starvla_no_deepspeed.py       单卡训练入口（20.5 KB，改自官方脚本）
    ├── serve_starlink_cube_policy.py       起策略服务
    ├── serve_starlink_cube_policy_traced.py  ★ 调试启动器：插桩记录归一化前后（服务器独有）
    ├── validate_starvla_loader.py          验证本任务数据集能过 StarVLA 的 loader
    ├── accelerate_one_gpu_ddp.yaml
    ├── accelerate_one_gpu.yaml
    ├── examples/StarlinkCube/train_files/
    │   ├── data_registry/data_config.py    数据注册：哪个 key 切片成什么、怎么归一化
    │   ├── starvla_starlink_cube_action_head_10k.yaml          正式训练（PPO 数据）
    │   ├── starvla_starlink_cube_oracle_action_head_10k.yaml   正式训练（oracle 数据）
    │   ├── starvla_starlink_cube_action_head_smoke_10.yaml     10 步冒烟
    │   ├── starvla_starlink_cube_smoke.yaml                    冒烟
    │   └── starvla_starlink_cube_single.yaml                   单样本调试
    └── starVLA/                            ★ vendored StarVLA 框架，169 个文件，不是我们的代码
        ├── config/        8 个   训练与 deepspeed 配置
        ├── dataloader/   37 个   LeRobot / GR00T 数据集加载与 transform
        ├── model/       109 个   VLM 封装、各 action head、framework
        └── training/     15 个   训练主循环与 trainer 工具
```

主要文件的作用：

| 文件 | 作用 |
|---|---|
| `delta_eef_control.py` | **动作契约的唯一来源。**`DeltaEefLimits` 定义四个上限（每步 0.01 m / 0.05 rad、累计 0.45 m）和一个模式开关 `normalized_actions`（**默认 `False`**）。`DeltaEefController.targets()` 把动作翻译成目标位姿 + 夹爪状态。⚠ 当前默认模式与策略动作空间不一致，见 §4 |
| `starlink_cube_delta_eef_env.py` | 独立批量 Genesis 环境。方块出生位姿随机化在 `CubeEnvConfig`：`cube_spawn_offset_m=(0,-0.15,0)`、`cube_spawn_noise_m=(0.02,0.02,0.01)`。默认 `control_limits=DeltaEefLimits()`（第 65 行） |
| `task_contract.py` | 成功判据：`grasp_ready`（闭合且距离达标）→ `stable`（更严阈值且抬升 ≥ 0.030 m）→ `success`（连续 hold 若干步）。GPU 批量计算 |
| `gymnasium_adapter.py` | 声明 `action_space = Box(-1.0, 1.0, 7)`（第 26 行） |
| `sb3_vecenv_adapter.py` | 同样声明 `Box(-1.0, 1.0, 7)`（第 25 行）。100+ 环境批量训练用这个 |
| `collect_oracle_rgb_rollouts.py` | **服务器版比本地多 `--noise` 支持**：`CubeEnvConfig(cube_spawn_noise_m=noise)`（第 68 行）。动作是比例控制 `(cube_pos - grasp_center) / 0.01` 再 clamp 到 [-1,1]（第 92–93 行），并显式传 `DeltaEefLimits(normalized_actions=True)`（第 66 行）。只保存成功回合 |
| `collect_ppo_rgb_rollouts.py` | `model.predict(obs, deterministic=True)` 的原始 7 维输出直接当标签记录。**没传 `control_limits`，用环境默认值** |
| `run_starvla_closed_loop_smoke.py` | 连策略服务取动作 chunk 逐帧下发。第 83 行 `StarlinkCubeDeltaEefEnv(num_envs=1)` **同样用默认 `control_limits`**。产出 `summary.json` / `actions.jsonl` / `closed-loop.mp4` |
| `test_delta_eef_control.py` | 两个用例：物理增量模式下动作 1.0 被截到 +0.01；归一化模式下 0.5 缩放到 0.005。⚠ 见 §4 |
| `starvla_overlay/serve_starlink_cube_policy_traced.py` | monkeypatch `PolicyNormProcessor.unapply_actions`，把每次调用的 `{call, normalized, unnormalized}` 写进 `action-trace.jsonl`。**定位 §4 那个缺陷时就靠它** |
| `starvla_overlay/examples/.../data_config.py` | 三个 action key 的切片与归一化方式：`delta_eef_position` / `delta_eef_rotation` 用 `min_max`，`gripper_close` 用 `binary` |
| `starvla_overlay/train_starvla_no_deepspeed.py` | 绕开 deepspeed 的单卡训练入口。`load_pretrained_backbones` 用 `strict=True`，权重对不上直接抛错 |

### 2.3 `code/franka_cube/` — 早期 Franka + cube 场景（4 个文件）

| 文件 | 作用 |
|---|---|
| `franka_cube_starvla.py` (59 KB) | 独立的 Genesis Franka Panda + cube 场景。一个进程一个 scene，headless。内含脚本抓取状态机（approach→descend→close→lift→hold，伺服**指尖**位置、目标点世界系冻结）、闭环 episode runner、录像与导出。支持 `--dry-run` / `--export-raw` / `--randomize-cube` / `--only-success` / `--action-space raw\|normalized` |
| `export_lerobot.py` | 原始 Genesis dump → LeRobot v2.1。写 `meta/{info,modality,episodes,tasks}.json` 和 `stats_gr00t.json` |
| `validate_lerobot.py` | 用 **StarVLA 自己的 dataloader** 回验导出的数据集能否读进来（故意不复用自研 loader，避免"自己验自己"） |
| `bench_parallel.py` | 批量 Genesis 吞吐基准。单环境无渲染 427 步/秒；批量 256 环境 93,091 env-steps/秒 |

### 2.4 `code/starvla_bridge/` — Genesis ↔ StarVLA 的桥（13 个文件，他人维护）

| 文件 | 作用 |
|---|---|
| `starvla_genesis_bridge.py` (17.6 KB) | **主桥**。跑一个 Genesis episode，Cartesian 目标由远端 StarVLA 策略给出：造 RGB 观测 → 推末端状态 → 请求 16 步动作 chunk → 带位置/旋转限幅逐步执行 → 再请求下一 chunk。WebSocket `10.110.10.12:6694` + msgpack+NumPy 编码 |
| `starvla_genesis_bridge_calvin.py` | CALVIN 格式变体 |
| `run_starvla_genesis_bridge.py` | 主桥的薄启动器 |
| `run_starvla_genesis_bridge_7d.py` … `_7d_v7.py` | 7 维动作版的**八个迭代快照**。v7 是最后一版。文件名带版本号是反面例子（见规范 §5.2） |
| `run_starvla_genesis_bridge_calvin.py` | CALVIN 变体的启动器 |
| `README.md` / `VALIDATION-RESULT.md` | 已验证的接口契约与验证结果 |
| `RUNBOOK.md` … `RUNBOOK-v6.md` | 六版操作手册快照 |
| `GENESIS_STARVLA_REMOTE_DESKTOP_RUNBOOK.md` | 远程桌面跑法 |

⚠ **这个目录不在 deploy 的同步范围内**（`deploy.ps1` 对它会直接报错退出），因为服务器上有人在持续开发它。不要把它当部署产物。

---

## 3. `spaceman_core/` — 上游仿真树（只读）

从 Genesis 机 `10.110.34.88` 同步过来，**不是我们的代码，不要改**。

```
spaceman_core/
├── .git                 ← 唯一有版本历史的地方（上游克隆）
├── .codex
├── outputs/rl/
├── scripts/             10 个绘图脚本（stage1/stage2 的误差带、成功率、轨迹、reward 曲线）
└── src/
    ├── assets/          URDF 与 mesh
    ├── configs/         场景与相机配置
    ├── controllers/
    ├── envs/            Genesis 环境封装
    ├── outputs/  recordings/
    ├── RL/              上游 PPO 实现
    ├── robots/          机械臂定义
    ├── sensors/         相机
    ├── tasks/           各任务定义（含 stage1 / stage2）
    ├── test/            上游的 replay / 测试脚本
    └── utils/           data_recorder 等
```

**`CANONICAL-DEVIATIONS.md` 记录了相对上游（commit `e042566`）的故意改动**，两条都是 mesh 路径从 `package://` 改成相对路径——因为 Genesis 认不了 ROS 的 `package://` URI。文件里明确写了：**只改路径，不改材质颜色和名字**，因为视觉策略对目标外观敏感；pengyuxin 那份未提交的 URDF 同时改了材质，有意不采纳。

---

## 4. ⚠ 已知功能缺陷：动作契约不一致

**这条不属于重构范围，但它影响所有已产出数据的解读，动手前必须知道。**

**现象**：PPO 采集的 128 回合数据集（`runs/20260924-starlink-cube-vla-batch1`）动作标签退化成近似常数——z / roll / yaw 三通道 `min == max == 1.0`，跨回合 std 2e-4。

**原因**：动作空间的**声明区间**与**解释区间**相差 100 倍。

| 环节 | 位置 | 内容 |
|---|---|---|
| 动作空间声明 | `sb3_vecenv_adapter.py:25`、`gymnasium_adapter.py:26` | `Box(-1.0, 1.0, 7)` |
| 环境默认 | `starlink_cube_delta_eef_env.py:65` | `DeltaEefLimits()` → `normalized_actions=False` |
| 控制器 | `delta_eef_control.py:67-68` | `clamp(action[:3], -0.01, 0.01)`、`clamp(action[3:6], -0.05, 0.05)` |

结果：动作区间 99% 是死区，输出 0.01 和 1.0 施加相同指令；策略失去梯度压力后退化为饱和的开关控制；标签只剩符号位，没有幅度信息。

**三个受影响的调用点（都要改，只改一处无效）**：

1. `sb3_vecenv_adapter.py:25` — PPO 训练/评测
2. `collect_ppo_rgb_rollouts.py:50` — PPO 数据采集
3. `run_starvla_closed_loop_smoke.py:83` — **VLA 闭环评测**（最关键：模型训得再准，评测时输出幅度照样被丢掉）

**已正确的参照实现**：`collect_oracle_rgb_rollouts.py:66` 显式传 `DeltaEefLimits(normalized_actions=True)`。这也是新 oracle 数据（`runs/20261016-145416-phase1-oracle-128`）动作多样性比旧数据高 400~4000 倍的原因——同一个环境、同一份方块噪声，只差这一个参数。

**⚠ 陷阱**：`test_delta_eef_control.py` 的断言把"动作 1.0 → 走满步长"当成预期行为。修约定时必须同时改这个断言，否则会出现"为了让测试通过而回滚修复"。

---

## 5. `recovered/` — 事故恢复物

```
recovered/
├── recovered_structure.txt
├── run_starvla_genesis_bridge_7d_v2.cpython-310.pyc
├── run_starvla_genesis_bridge_7d_v6.cpython-310.pyc
└── starvla_genesis_bridge.cpython-310.pyc
```

2026-09-17 一次 `deploy` 把服务器上别人正在开发的 `starvla_bridge/` 整个覆盖掉，只剩 `__pycache__` 里的 `.pyc` 可以恢复，抢救出来的就是这几个。**不要删。**这件事的直接后果写在 `deploy.ps1` 头部注释里。

---

## 6. `.agents/skills/` — 给 agent 用的 skill

| 目录 | 内容 |
|---|---|
| `starlink-cube-vla-pipeline/SKILL.md` | 250 行。数据 → LeRobot → 微调 → 闭环评测全流程、两个 conda 环境的互补缺口、20 条实测踩坑记录 |
| `sim-driven-policy-iteration/SKILL.md` | 241 行。从仿真评测结果定位并改进策略的方法论：置信区间、失败分层、单变量改动、同种子配对复测 |

另有一份装在 `/home/student/.codex/skills/`。两份内容与本地仓库的源逐字一致（md5 校验过）。

---

## 7. 改动前的安全要求

服务器这棵树**没有 git**，最外层的 `.git` 在 `spaceman_core/` 里（上游的）。所以：

1. **改任何能工作的文件之前，先复制一份**到 `code/<任务>/backups/` 或 `/data/Yanhan/genesis/backups/`，文件名带日期。
2. **改完跑对应的 `run_*_smoke.py`**，不要只看 import 成功。
3. **如果要动与数据产出相关的代码**，拿一次既有 run 复跑，**逐回合结果必须一致**——这个项目的评测在同种子下是完全确定性的（已验证：步数上限 128 vs 256 两次跑，64 回合逐回合结果完全一致），所以这是可以真正验证的判据。
4. **不要在服务器上手工改 `code/genesis_tools/`**——它会被下一次 `deploy.ps1` 覆盖（不过按 §1，它覆盖不到 `starlink_cube_rl/`，所以那两个目录的行为不同，别混为一谈）。

---

## 8. `runs/` 里值得先看的目录（不展开，只给索引）

167 个运行目录，命名 `<YYYYMMDD>-<slug>-<version>`。要看结论就从这几个入手：

| 目录 | 是什么 |
|---|---|
| `20260923-starlink-cube-ppo-overnight` | PPO 训练：`evaluation-final.json` = 245/256 = 95.7% |
| `20260924-starlink-cube-vla-batch1` | 旧 128 回合数据集（PPO 采的，动作退化） |
| `20260928-starlink-cube-starvla-action-head-10k-v2` | VLA 动作头训练 + 四个闭环评测；`closed-loop-randomized-64-v3` 就是 58/64 = 90.6% |
| `20260929-starlink-cube-oracle-coverage{2x,1p5x}-v1`、`...-recovery-coverage1p25x-v1` | oracle 随机化 sweep 的 1.25× / 1.5× / 2.0× 三各点 |
| `20261016-140642-cap-ablation` | 步数上限 128 vs 256 配对实验 |
| `20261016-142930-action-trace` | 服务端插桩 74 次调用，`action-trace.jsonl` |
| `20261016-143810-phase1-noise-sweep` | 1.0×（24/25）与 4.0×（4/60）两端 |
| `20261016-145416-phase1-oracle-128` | 新 128 回合高多样性数据集 |

单个 run 目录里的典型产物：`launch.txt`（启动命令）、`*.log`、`policy-server.log`、`summary.json` / `evaluation-final.json`、`actions.jsonl`、`closed-loop.mp4`、`raw/*.npz`、`lerobot/`、`dataset_statistics.json`、`checkpoints/`、`tensorboard/`。
