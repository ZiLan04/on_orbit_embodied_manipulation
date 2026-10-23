# 代码树功能说明

> 本仓库每个文件/目录是干什么的，以及代码怎么到达真正运行它的 GPU 服务器。
> 英文版：`code-map.en.md`。规范：`code-standards.md`。结构与迁移：`README.md`。
>
> **两套布局的说明。** 本仓库已经是目标布局（`code/tasks/...`）；GPU 服务器上跑的仍是
> **重构前**的布局（`code/starlink_cube_rl/...`）。两者的对应关系见 §1.3。除特别标注外，
> 下文路径都指**本仓库**。

---

## 0. 仓库树

```
code/
├── tasks/                             一个任务一个目录
│   ├── starlink_cube_grasp/           ★ 当前主线（18 个文件）
│   │   ├── delta_eef_control.py       7 维 delta-EE 动作契约
│   │   ├── starlink_cube_delta_eef_env.py   批量 Genesis 环境
│   │   ├── task_contract.py           奖励项与成功判据
│   │   ├── gymnasium_adapter.py       单环境 Gymnasium 桥
│   │   ├── sb3_vecenv_adapter.py      批量 SB3 VecEnv 桥
│   │   ├── collect_oracle_rgb_rollouts.py           脚本专家采集
│   │   ├── collect_oracle_recovery_rgb_rollouts.py  脚本专家 + 失败恢复状态
│   │   ├── collect_ppo_rgb_rollouts.py              PPO 策略采集
│   │   ├── export_lerobot.py          原始 dump → LeRobot v2.1
│   │   ├── run_batched_ppo_train.py   批量 PPO 训练
│   │   ├── evaluate_batched_ppo.py    确定性 PPO 评测
│   │   ├── run_starvla_closed_loop_smoke.py  VLA 闭环评测
│   │   ├── run_delta_eef_env_smoke.py        冒烟：环境能否构建并推进
│   │   ├── run_delta_eef_oracle_smoke.py     冒烟：奖励判据物理上可行吗
│   │   ├── run_ppo_integration_smoke.py      冒烟：单环境 SB3 接线
│   │   ├── run_batched_ppo_integration_smoke.py  冒烟：批量 SB3 接线
│   │   ├── run_real_cube_smoke.py     冒烟：既有物理抓取任务
│   │   ├── test_delta_eef_control.py  单元测试：动作控制安全性与夹爪语义
│   │   ├── probes/
│   │   │   └── starvla_healthcheck.py 策略服务活着吗、加载了哪个 checkpoint
│   │   └── starvla_overlay/           StarVLA 侧入口与配置
│   │       ├── train_starvla_no_deepspeed.py    单卡训练入口
│   │       ├── serve_starlink_cube_policy.py    起策略服务
│   │       ├── serve_starlink_cube_policy_traced.py  带归一化日志的调试启动器
│   │       ├── validate_starvla_loader.py       用 StarVLA 的 loader 回验数据集
│   │       ├── accelerate_one_gpu.yaml
│   │       ├── accelerate_one_gpu_ddp.yaml
│   │       └── examples/StarlinkCube/train_files/
│   │           ├── data_registry/data_config.py       动作 key、切片、归一化方式
│   │           ├── starvla_starlink_cube_action_head_10k.yaml         正式训练，PPO 数据
│   │           ├── starvla_starlink_cube_oracle_action_head_10k.yaml  正式训练，oracle 数据
│   │           ├── starvla_starlink_cube_action_head_smoke_10.yaml    10 步冒烟
│   │           ├── starvla_starlink_cube_smoke.yaml                   冒烟
│   │           └── starvla_starlink_cube_single.yaml                  单样本调试
│   │
│   ├── starlink_satellite_grasp/      卫星侧 stage-2 抓取位姿（5 + 3 probes）
│   │   ├── scripted_grasp.py           脚本抓取 oracle（非学习）
│   │   ├── eval_stage2_batch.py        批量 stage-2 评测
│   │   ├── eval_rl_grasp.py            按真正关心的指标评 RL 抓取策略
│   │   ├── stage2_action_control.py    成功判据真的能被动作控制吗
│   │   ├── run_stage2_horizon.py       带可配 horizon 的 stage-2 PPO
│   │   └── probes/
│   │       ├── ik_reachability.py      机械臂够得到任务要求的位姿吗
│   │       ├── reach_scan.py           夹爪最近能离物体多近
│   │       └── probe_arm_geometry.py   机械臂几何与可达工作空间
│   │
│   ├── franka_cube_grasp/              早期 Franka Panda + cube 场景（4）
│   │   ├── franka_cube_starvla.py      独立场景、脚本状态机、闭环 runner
│   │   ├── export_lerobot.py           原始 dump → LeRobot v2.1（Franka 版）
│   │   ├── validate_lerobot.py         用 StarVLA 自己的 dataloader 回验
│   │   ├── bench_parallel.py           批量 Genesis 吞吐基准
│   │   └── pylibs/README.md            websockets 路径 shim 怎么重建
│   │
│   └── vla_jepa_probe/                 已停用的 JEPA 线，留档（4 + 1 probes）
│       ├── jepa_bridge.py              VLA-JEPA 驱动卫星机械臂
│       ├── jepa_direction_pilot.py     问服务"它在场景里会怎么做"
│       ├── jepa_distance_sweep.py      多远的距离内它的方向还可信
│       ├── kinematic_demo_gen.py       用运动学专家生成对齐演示
│       └── probes/analyze_alignment_log.py  策略指令到底有没有在起作用
│
└── bridges/
    └── starvla_bridge/                 Genesis ↔ StarVLA WebSocket 桥（19，非我们独有）
        ├── starvla_genesis_bridge.py    主桥
        ├── starvla_genesis_bridge_calvin.py  CALVIN 格式变体
        ├── run_starvla_genesis_bridge.py          薄启动器
        ├── run_starvla_genesis_bridge_7d.py       启动器，7 维动作版
        ├── run_starvla_genesis_bridge_7d_v2..v7.py  是覆盖链，不是副本（见下）
        └── README.md、VALIDATION-RESULT.md、RUNBOOK*.md  契约与操作手册

documents/                             现行规范与本文
deploy/deploy.ps1                      单向同步到 GPU 服务器
.agents/skills/                        两个 agent skill，编码整条管线与改进方法
```

**故意没有 `code/utils/`。**目前每个可复用模块都只有一个调用方，所以它留在那个任务的
目录里。等第二个任务真正需要同一份代码那天再建 `utils/`，不要提前。

### 一个坑：`_7d_v*` 是一条链

`run_starvla_genesis_bridge_7d_v3.py` import 了 `..._v2`，`v4` import `v3`，一直到 `v7`。
它们是**逐层叠加的覆盖，不是平行快照**。删掉 `v2..v6` 里任何一个都会打断 `v7`。把它们
合并成一个文件是一次真重构，需要单独验证，不是"清理"。

---

## 1. 这套代码怎么跑起来

### 1.1 四条隐式路径依赖

`code/` **不是自包含的**。下面每一条都必须在 `PYTHONPATH` 上，否则运行时会 import 失败，
而且报错不会告诉你原因。

**① `spaceman_core/src`** —— 很多文件 `import configs` / `tasks` / `RL` / `envs` /
`robots` / `controllers` / `utils`，这些全部落在上游仿真树里，而它**不在本仓库**。

**② `code/bridges/starvla_bridge`** —— `run_starvla_closed_loop_smoke.py` 第 38 行
`from starvla_genesis_bridge import StarVLAPolicyClient`。跨目录依赖，而且依赖的是**由别人
维护**的那个目录。

**③ `.../train_files/data_registry`** —— `serve_starlink_cube_policy*.py` 把这个目录插进
`sys.path` 再 `import data_config`。路径是相对 overlay 硬编码的，所以 `data_config.py`
不能搬走，除非同时改两个启动器。

**④ `code/tasks/franka_cube_grasp/pylibs`** —— 环境级的符号链接 shim，把只装在
`starvla_env` 里的 `websockets` 暴露给 Genesis 侧解释器。见 `pylibs/README.md`。

一套可用的设置：

```bash
export PYTHONPATH=<repo>/code/tasks/starlink_cube_grasp:\
<repo>/code/bridges/starvla_bridge:\
<repo>/code/tasks/franka_cube_grasp/pylibs:\
<repo>/code/tasks/starlink_cube_grasp/starvla_overlay/examples/StarlinkCube/train_files/data_registry:\
<spaceman_core>/src
```

### 1.2 运行环境

两个 conda 环境，**互补是设计如此**。**不要往任何一个里装包**去补缺失——那正是弄坏另一边
的方式。

| 包 | `Spaceman_env`（Genesis 0.4.3） | `starvla_env`（训练/服务） |
|---|---|---|
| numpy / torch / msgpack | ✓ | ✓ |
| genesis | ✓ | ✗ |
| gymnasium / stable_baselines3 | ✓ | ✗ |
| pyarrow | ✗ | ✓ |
| websockets | ✗（用 `pylibs` shim） | ✓ |
| imageio_ffmpeg | ✓ | ✗ |

无头 Genesis 永远需要：

```bash
env -u DISPLAY PYOPENGL_PLATFORM=egl MUJOCO_GL=egl <python> ...
```

并且场景脚本里必须 `show_viewer=False`。显式钉死 `CUDA_VISIBLE_DEVICES`——这台机器是
8×RTX 5090，多人共用。

### 1.3 外部依赖（不在本仓库）

| 是什么 | 在哪 | 说明 |
|---|---|---|
| `spaceman_core/` | GPU 服务器，与 `code/` 平级 | 上游仿真树，从 Genesis 机同步。只读。相对上游的改动记在它的 `CANONICAL-DEVIATIONS.md`（URDF mesh 路径 `package://` → 相对路径，因为 Genesis 认不了 ROS URI） |
| `starVLA/` | GPU 服务器，部署环境内 | 约 100 个文件。策略服务跑在固定路径上，所以它是运行时依赖，不是 vendored 源码 |

**服务器上那份 StarVLA 有一处本地修改。**`starVLA/dataloader/__init__.py` 相对上游
`3422b9f` 打过补丁：`DataLoader` 的 worker 数、prefetch、pin-memory 改为从
`STARVLA_DATALOADER_*` 环境变量读取，默认值保守（`workers=4, prefetch=2, pin_memory=0`），
替换掉硬编码的 `16 / 4 / True`。理由是视频解码 + pin-memory 预取会在多 worker 下成倍放大
主机内存，共享机器上撑不住。**如果以后重新 vendor StarVLA，这个补丁必须重新打上**——它没
记录在别处。

### 1.4 部署

`deploy/deploy.ps1` 把整个 `code/` 树同步到服务器，带逐文件 mtime 守卫，远端更新则拒绝
覆盖。它**不做**两件事：

- **不**负责把服务器切到本布局。服务器上仍是旧目录；现在跑一次 deploy 只会把
  `code/tasks/` 和 `code/bridges/` **加在**旧目录旁边。切换是一次手工、可回退的操作
  （见 `README.md`）。
- **不**同步 `code/bridges/starvla_bridge/`。那个目录由服务器侧的人维护。

---

## 2. 已知功能缺陷：动作契约不一致

**不属于任何结构调整的范围，但它改变了此前所有数据集该怎么读。**

**现象**：PPO 采集的 128 回合数据集动作标签近似常数——z、roll、yaw 三通道
`min == max == 1.0`，跨回合标准差 2e-4。

**原因**：动作的**声明区间**与**解释区间**相差 100 倍。

| 环节 | 位置 | 内容 |
|---|---|---|
| 动作空间声明 | `sb3_vecenv_adapter.py:25`、`gymnasium_adapter.py:26` | `Box(-1.0, 1.0, 7)` |
| 环境默认 | `starlink_cube_delta_eef_env.py:65` | `DeltaEefLimits()` → `normalized_actions=False` |
| 控制器 | `delta_eef_control.py:67-68` | `clamp(action[:3], -0.01, 0.01)`、`clamp(action[3:6], -0.05, 0.05)` |

后果：动作区间 99% 是死区——输出 0.01 和输出 1.0 产生同一条指令。策略在幅度上失去梯度
压力，退化成饱和的开关控制；记录下来的标签只剩符号位。

**三处调用点必须一起改，只改一处等于没改：**

1. `sb3_vecenv_adapter.py:25` —— PPO 训练与评测
2. `collect_ppo_rgb_rollouts.py:50` —— PPO 数据采集
3. `run_starvla_closed_loop_smoke.py:83` —— **VLA 闭环评测**（最关键的一处：模型训得再准，
   评测时它的输出幅度照样被丢掉）

**已经正确的参照实现**是 `collect_oracle_rgb_rollouts.py:66`，它显式传
`DeltaEefLimits(normalized_actions=True)`。这也是 oracle 数据集的动作多样性比 PPO 数据高
400~4000 倍的原因——同一个环境、同一份方块噪声，只差这一个参数。

**陷阱**：`test_delta_eef_control.py` 断言"动作 1.0 → 走满步长"。那条断言把缺陷写成了预期
行为。修契约时必须同时改这条断言，否则会有人为了让测试通过而把修复回滚掉。

---

## 3. 实测结果

| 策略 | 结果 | 说明 |
|---|---|---|
| PPO 确定性评测 | 245 / 256 = 95.7% | 批量、确定性 |
| VLA 动作头闭环 | 58 / 64 = 90.6% | 95% CI [0.84, 0.98] |
| 脚本专家，1.25× 随机化 | 32 / 35 = 91.4% | 教师 |

脚本专家成功率随方块随机化加宽：

| 倍数 | 1.0× | 1.25× | 1.5× | 2.0× | 4.0× |
|---|---|---|---|---|---|
| 成功率 | 96.0% | 91.4% | 73.3% | 55.6% | 6.7% |

所以宽覆盖必须由 RL 产出，手写控制器做不到。

---

## 4. 验证协议

结构调整按四层验证，最后一层是决定性的。

| 层级 | 检查 |
|---|---|
| 1 | 文件内容未变（与参考副本比 md5） |
| 2 | Python 3.10 下 `py_compile` 全过；每个模块在文档给出的 `PYTHONPATH` 下能 import |
| 3 | 同种子的一个脚本专家回合：`action`、`state`、`cube_position` 逐比特相同 |
| 4 | 完整 64 回合闭环 VLA 评测，同种子、同一策略服务：逐回合 `success`、`steps`、初始 `cube_position` 相同 |

**渲染像素本身不可逐次复现**，即使同一份代码在同一张卡上跑两次，也会有约 0.001% 的像素
采样相差最多 45/255。所以**绝不能用图像比对当通过判据**——要比 `success` / `steps` /
`action` / `state`。

---

## 5. 改动前的安全要求

服务器那棵树除本仓库外没有版本控制，因此：

1. 改之前先复制到 `backups/`，带日期。它被 gitignore，是本机的安全网。
2. 改完跑该任务自己的 `run_*_smoke.py` / `test_*.py`，不要只看 import 成功。
3. 若改到数据产出路径，重跑一次既有评测，要求逐回合结果一致（§4 第 4 层）。
4. 不要手工改服务器上的部署树——那是同步目标。
5. 不要在没有所有者同意的情况下改 `code/bridges/starvla_bridge/`。
