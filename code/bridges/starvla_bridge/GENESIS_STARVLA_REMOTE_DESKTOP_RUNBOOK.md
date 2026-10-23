# Genesis - StarVLA 远程桌面复现实操手册

本文用于复现已经验证的 Genesis 到 StarVLA 闭环：Genesis 渲染仿真场景并读取机械臂状态，将图像、状态和语言指令发送给 StarVLA；StarVLA 返回 16 步动作块；桥接程序将动作限幅、调用 IK、下发关节和夹爪控制，然后推进仿真。

## 开始前的注意事项

- 在 Genesis 机器的远程桌面会话中使用显示器 `:1`，不要使用 `:0`。
- Python 环境必须是 `/home/robotarm/genesis043_venv`，不要修改或使用旧的 `genesis_env`。
- 策略服务在另一台 GPU 机器上运行。整个 Genesis 运行期间，启动服务的 SSH 终端必须保持打开，关闭终端会结束策略服务。
- 不要把登录凭据写入命令历史、脚本或本文档。只在交互式密码提示时输入。
- 工作区边界和单次关节变化限制是安全保护，调整动作幅度时必须保留这些保护。

## 1. 打开策略服务终端

在 Genesis 远程桌面中打开一个终端，连接策略服务器：

```bash
ssh student@10.110.10.12
```

登录后，在该终端启动 StarVLA 策略服务：

```bash
cd /data/VLA_model/StarVLA/repo
CKPT=playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/VLA_model/StarVLA/repo \
  /data/anaconda3/envs/starvla_env/bin/python \
  deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 6694 --use_bf16
```

看到 `server listening on 0.0.0.0:6694` 后，说明服务就绪。此终端保持运行，不要关闭。

## 2. 打开 Genesis 运行终端

在 Genesis 桌面中另开一个终端，先检查策略端口是否可达：

```bash
python3 -c "import socket; s=socket.create_connection(('10.110.10.12', 6694), 5); print('策略端口可达'); s.close()"
```

输出 `策略端口可达` 后，启动更清晰、动作更明显的演示：

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v7.py \
  --instruction "move the robot arm through a large, smooth arc" \
  --steps 200 \
  --translation-scale 3 \
  --rotation-scale 1.5 \
  --max-xyz-step 0.03 \
  --max-rpy-step 0.08 \
  --video-resolution 512 \
  --video-fps 25 \
  --output-dir recordings/starvla_genesis_bridge/large-motion-512
```

上述 instruction 会与当前相机图像和机械臂状态一起发送给 StarVLA。`--translation-scale 3` 让策略输出的平移增量更明显；随后仍要经过每步位移限幅、末端工作区限制、数值有限性检查、批量 IK 和关节变化限制。

## 3. 观察和中止运行

- Genesis 查看器会显示在远程桌面会话的显示器 `:1` 上。
- Genesis 终端会打印收到的每个 16 步动作块，以及每完成 16 步的进度。
- 如需提前停止，在 Genesis 终端按 `Ctrl+C`。程序会关闭 WebSocket 并在清理阶段写出录像。
- 正常完成 200 步时，进程退出码为 `0`。

## 4. 查看录像和日志

运行结束后，证据文件位于指定输出目录：

```bash
cd /home/robotarm/spaceman_core/src
ls -lh recordings/starvla_genesis_bridge/large-motion-512
```

- `bridge-run.mp4`：仿真相机录制的视频。
- `bridge-run.jsonl`：初始化记录、每一步执行的动作记录和结束记录。动作记录包括原始 StarVLA 动作、状态、经安全限制后的目标位置、目标四元数和夹爪目标。

用以下命令确认已完成 200 步：

```bash
python3 -c "import json; p='recordings/starvla_genesis_bridge/large-motion-512/bridge-run.jsonl'; r=[json.loads(x) for x in open(p)]; a=[x for x in r if x['event']=='action']; print('动作记录数:', len(a)); print('最后末端目标:', a[-1]['target_position'])"
```

## 5. 常用调整项

| 目的 | 调整方式 |
| --- | --- |
| 位移更明显 | 小幅提高 `--translation-scale`，同时明确设置 `--max-xyz-step` 作为每步上限。建议从 `3` 和 `0.03` 开始。 |
| 更高录像清晰度 | 使用 `--video-resolution 640`。Genesis 机器仅有 4 GB 显存，渲染会更慢且 MP4 更大。 |
| 更长录像 | 提高 `--steps`，并使用新的 `--output-dir`，避免覆盖已有证据。 |
| 更换语言指令 | 替换 `--instruction`。它会改变策略输入，但不会绕过安全限幅，也不保证完成任务。 |
| 加快仿真 | 减小 `--control-period`，但应保留足够的渲染和控制稳定时间。 |

## 6. 常见问题排查

| 现象 | 检查方法 |
| --- | --- |
| 连接被拒绝或超时 | 确认策略服务终端仍在运行且监听 `6694` 端口，再执行端口连通性检查。 |
| `No display detected` | 必须以 `XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1` 启动 Genesis。 |
| Genesis 使用了错误版本 | 确认命令调用 `/home/robotarm/genesis043_venv/bin/python`，而不是 `genesis_env`。 |
| 策略端提示 unnormalization 错误 | 请求必须携带 `unnorm_key="oxe_bridge"`；提供的启动器已固定处理。 |
| 中断后没有录像 | 按 `Ctrl+C` 后等待清理完成，不要强制杀掉进程。 |
| GPU 显存紧张或显示很慢 | 回退到 512 分辨率，关闭其他 GPU 程序，或减少步数。 |

## 7. 正常关闭

Genesis 退出后，回到策略服务器终端按 `Ctrl+C`，确认策略服务停止后再关闭终端，以释放共享 GPU。

## 该流程验证了什么

该流程验证了跨机通信、观测格式、语言指令传递、16 步动作块调度、动作安全控制、IK、夹爪控制、仿真推进、结构化日志和录像写入。它不证明这个 Bridge 预训练策略能够在当前场景完成任意语义任务。若要评价任务跟随能力，需要放置与训练数据相近的物体和相机视角，并定义可度量的具体任务。
