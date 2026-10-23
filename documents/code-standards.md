# 代码规范

> 本文是这个仓库的唯一规范来源。**新建任何文件、目录、函数、配置之前先读这一份。**
> 由人和 agent 共同遵守。不符合本规范的新增内容，评审时直接退回。
>
> 版本：v1 · 2026-10-16

---

## 0. 总则

**重构的边界：功能不变。**目录、文件名、函数名可以变，流程可以变，但重构前后的**行为必须一致**——同样的输入产出同样的结果，同样的命令跑出同样的数。

每条改动都要通过三道闸：

1. **有备份**（见 §3）
2. **有冒烟测试跑通**（见 §6）
3. **`documents/code-map.md` 同步更新**（见 §6）

任何一条没做，不算完成。

---

## 1. 任务与实验放 `tasks/`

**规则**：`tasks/` 目录下，**一个任务一个子目录**。该任务的场景、采集、训练、评测、配置全部放在这个子目录里，不散落。

```
tasks/
├── starlink_cube_grasp/          # 一个任务 = 一个目录
│   ├── env_starlink_cube_delta_eef.py
│   ├── collect_oracle_rollouts.py
│   ├── collect_ppo_rollouts.py
│   ├── train_ppo_batched.py
│   ├── eval_ppo_deterministic.py
│   ├── eval_starvla_closed_loop.py
│   └── config/
├── starlink_satellite_grasp/     # 卫星侧 stage-2
├── franka_cube_grasp/
└── vla_jepa_probe/
```

**判断一个文件该不该进 `tasks/`**：它是不是"为某个具体任务而存在"？
- 是 → 进对应任务的子目录
- 否，多个任务都要用 → 进 `utils/`（见 §4）

**禁止**：把任务代码放在仓库根目录，或者用一个 `misc/`、`tmp/` 之类的地方收纳。

---

## 2. `code/` 只放我们自己写的源码

**规则**：`code/`（或未来的源码根目录）**只包含我们新写的源码**。上游同步来的东西不放在里面。

不属于"我们的源码"的三类，各自有归属：

| 类别 | 摆放位置 | 现有例子 |
|---|---|---|
| 上游仿真树，只读参照 | 与 `code/` **平级**的独立目录 | `spaceman_core/`（服务器上已经是这个结构，保持） |
| 第三方框架的 vendored 副本 | `third_party/` 或交付时由部署脚本拉取 | `starlink_cube_rl/starvla_overlay/starVLA/**` 约 90 个文件，目前混在任务目录里，应移出 |
| 上游参考实现（我们抄来对照，不改） | 任务目录下的 `reference/`，文件名保留 `_reference` 后缀 | `parallel_rl_reference.py`、`real_cube_grasp_reference.py` |

**判断标准**：这个文件如果上游升级，我们是"跟着改"还是"重写"？
- 跟着改 → 不是我们的源码，不放 `code/`
- 重写/自己维护 → 是我们的源码

---

## 3. `backups/` 存放复制出来的文件

**规则**：要修改一个已经能工作的文件之前，**先复制一份到 `backups/`**，然后才动手。备份文件名带时间戳和后缀说明。

```
backups/
├── delta_eef_control.py.20261016.bak        # 改动作契约前的备份
└── collect_ppo_rgb_rollouts.py.20261016.bak
```

**为什么是硬规则**：服务器 `/data/Yanhan/genesis` **没有 git**（唯一的 `.git` 在 `spaceman_core/` 里，那是上游克隆）。`code/` 下几十个文件的**全部历史只有一个版本**——当前这份。改错就找不回来，没有 `git checkout` 可用。

所以 `backups/` 在这个项目里不是可选项，它是唯一的版本安全网。建议位置：

```
/data/Yanhan/genesis/code/backups/
```

按任务分子目录，文件带日期。整个目录体积不大，定期清掉超过半年的。

---

## 4. `utils/` 放跨任务共享代码，按用途分子目录

**规则**：被两个以上任务用到的代码放 `utils/`。**不按文件类型分（不搞 `utils/py/`），按用途分。**

```
utils/
├── control/          动作与控制器
│   └── delta_eef_control.py
├── data/             LeRobot 导入导出与校验
│   ├── export_lerobot.py
│   └── validate_lerobot.py
├── sim/              Genesis 场景共性（相机、headless 初始化、批量封装）
├── geometry/         位姿、四元数、RPY、IK 辅助
└── io/               文件、日志、录像
```

**判断标准**：一个函数/模块被第二个任务用到的那一刻，就该搬到 `utils/`。**不要提前搬**——只有一个使用者的"共享代码"是过度设计。

**搬动时**：同目录改名（`git mv` 或先复制到 `backups/`），然后全仓库搜旧路径，改所有 import 和文档引用。

---

## 5. 命名规范

### 5.1 代码：`功能_有什么变量_用什么元素`

即 `<做什么>_<对什么>_<用什么手段>`。三段都要有信息量，能自解释，不用打开文件就能猜出作用。

```python
# 好
def normalize_action_minmax_torch(action, stats): ...
def clamp_delta_eef_metre(delta_xyz, limit_m): ...
def compute_success_hold_gpu(grasp_ready, lift, hold_count): ...
def sample_cube_pose_uniform(config, num_envs): ...

# 差（看不出变量和手段）
def process(data): ...
def helper(x, y): ...
def do_thing(): ...
```

工具/框架作为"元素"段：`_torch` / `_numpy` / `_gpu` / `_yaml`。

保留既有前缀约定，不要因为这条规则去改它们：
- `test_*` — 测试（pytest 要认）
- `run_*` — 可直接执行的入口脚本
- `serve_*` — 起服务的进程

### 5.2 文件：`任务_文件是用来干什么的`

即 `<任务>_<用途>.<ext>`。任务名在前，用途在后，全小写下划线。

```
starlink_cube_env_delta_eef.py           # 任务 + 这是个环境
starlink_cube_collect_oracle_rollouts.py # 任务 + 用途是采 oracle 轨迹
starlink_cube_eval_closed_loop.py        # 任务 + 用途是闭环评测
starlink_cube_config_ppo.yaml            # 任务 + 用途是 PPO 配置

franka_cube_grasp_collect_rollouts.py
```

**禁止**：`v1/v2/v3/v4/v5/v6/v7` 这种版本号堆在文件名上（现有 `run_starvla_genesis_bridge_7d_v2..v7.py` 就是反例）。要留历史用 git 或 `backups/`，不要用文件名。

**禁止**：`.orig`、`.bak`、`_old`、`_new`、`_final`、`_copy` 这类后缀在源码目录里出现——它们只允许出现在 `backups/`。

### 5.3 目录：小写 + 下划线，名词

`starlink_cube/`、`delta_eef/`、`third_party/`。不用缩写除非是公认缩写（`io`、`rl`、`eef`）。

---

## 6. 新增文件或功能必须写清"为什么"

**规则**：新建任何文件、模块、公开函数、配置项，**必须同时给出定义或解释**，不能只写代码。

### 6.1 文件头必须有 docstring

```python
"""一句话说清这个文件做什么，以及为什么要单独存在。

若不显然，说明它为什么不能合并进相邻文件。
"""
```

现有代码基本做到了（`genesis_tools/` 下 41 个 .py 都有 docstring），**继续保持**。

### 6.2 新增文件要登记在 `documents/code-map.md`

在对应分组下加一行，写清：文件名、作用一句话、属于哪个任务。**不登记等于没交付。**

### 6.3 非显然的决定要写"为什么"，不只是"是什么"

三个必须留说明的场景：

1. **反直觉的参数值** —— 例如 `cube_spawn_noise_m=(0.02, 0.02, 0.01)`，要说明这个量级是怎么定的
2. **绕开某个默认行为** —— 例如"这里显式传 `normalized_actions=True`，因为环境默认值与策略动作空间不一致"
3. **修掉的坑** —— 在代码注释里写明症状、原因、以及怎么验证修好了

写得好的现成例子：`delta_eef_control.py:65-66` 和 `deploy.ps1:8-20` 都留了决策来龙去脉。

---

## 7. 参数多就上 config，config 多就上目录

**规则**：

- **一个文件的魔数超过 3 个**，或者同一组参数要在多处保持一致 → 抽成 config
- **config 超过 3 个** → 建 `config/` 目录
- **但不要为了有 config 而造 config。**只有一个调用方、参数只改一次的，直接写在代码里更清楚

```
tasks/starlink_cube_grasp/
├── config/
│   ├── env.yaml          # 环境：方块尺寸、出生偏移与噪声、episode 上限
│   ├── ppo.yaml          # PPO：环境数、学习率、总步数
│   ├── action_head.yaml  # VLA：骨干、动作头、训练步数
│   └── eval.yaml         # 评测：回合数、种子、随机化宽度
└── ...
```

**config 必须能回答"这份配置对应哪次实验"**：在文件头写清用途，并且 —— 重要 —— **跑实验时把实际用的 config 副本存进 `runs/<run-id>/`**（现有代码已经是这么做的，保持）。

**反向要求**：同一个值不要在代码和 config 里各写一遍。以 config 为准，代码里读 config，不写默认常量。

**一个豁免**：由第三方框架**按自己的路径约定**加载的配置文件，留在框架要求的位置。
当前案例是 `starvla_overlay/examples/StarlinkCube/train_files/*.yaml`——`serve_starlink_cube_policy*.py`
把 `.../examples/StarlinkCube/train_files/data_registry` 硬编码进 `sys.path`，整个
`starvla_overlay/` 都在镜像 StarVLA 的结构。搬走这些文件会破坏约定和两个启动器。
本条规则管的是**我们自己**加载的配置。

---

## 8. `documents/` 是规范的唯一来源

```
documents/
├── README.md           # 本目录是什么、目标目录结构、迁移映射
├── code-standards.md   # 本文：写代码和新建文件的规则
└── code-map.md         # 每个文件是干什么的（交接用）
```

**规则**：

- 新的技术规范、目录约定、命名约定，**只写进 `documents/`**，不要散落在 `docs/`、README、聊天记录里
- `docs/` 保留历史调研与方案记录（只增不改）；`documents/` 是**当前有效的规则**（会随重构更新）
- **agent 新建任何东西之前，先读 `documents/code-standards.md`；建完更新 `documents/code-map.md`**

---

## 9. 重构的安全流程（每个任务走一遍）

只改结构、不改功能时，按这个顺序：

```
1. 备份     把要动的文件复制进 backups/（带时间戳）
2. 改结构   只移动/改名/拆文件，不改逻辑
3. 冒烟     跑该任务目录下的 run_*_smoke.py，以及与它相关的 test_*.py
4. 比对     如果动了数据产出相关的代码，拿一次既有 run 复跑，逐回合结果必须一致
5. 登记     更新 documents/code-map.md
6. 部署     .\deploy\deploy.ps1 -DryRun 先看，确认没有"远端更新"的文件再正式 deploy
```

**第 4 步是最容易被跳过、也最容易出事的**。这个项目的评测在同种子下是完全确定性的（已验证：步数上限 128 vs 256 两次跑逐回合结果完全一致），所以"逐回合结果必须一致"是一个**可以真正验证**的判据，不要用"看起来差不多"代替。

---

## 10. 当前必须先处理的三件事（2026-10-16）

按风险排序，交接时请先做：

1. **全量备份一次。**服务器这棵树没有 git，现在 `code/` 下所有文件只有唯一一份、没有历史。
   ```bash
   cd /data/Yanhan/genesis && tar czf code/backups/code-$(date +%Y%m%d).tar.gz code/
   ```
   这是后面所有搬动的唯一退路。

2. **修 `deploy.ps1` 的路径错配。**它现在把 `genesis_tools/` 写到 `code/genesis_tools/`，但**正在跑的代码在 `code/starlink_cube_rl/`**。结果是：跑一次 deploy 不会更新主线代码，反而会造一份重复副本。在结构重构完成之前，至少要让两者的落点一致。（详见 `code-map.md` §1）

3. **动作契约不一致**（`code-map.md` §4）。这是已知的**功能缺陷**，不属于重构，但它影响所有已产出数据的解读。必须和结构重构**分开做、单独验证**。

---

## 11. 变更记录

| 日期 | 改了什么 | 为什么 |
|---|---|---|
| 2026-10-16 | 建立本规范 v1 | 交给同事做结构优化，需要统一的目录/命名/文档约定 |
| 2026-10-16 | §3 从"本地仓库无 git 追踪"改为"服务器部署树无 git" | 实际开发在服务器上，规范应针对服务器 |
