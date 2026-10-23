# Genesis - StarVLA Bridge Runbook

## Delivered files

The Genesis host now has an isolated directory:

```text
/home/robotarm/spaceman_core/src/starvla_bridge/
  starvla_genesis_bridge.py
  run_starvla_genesis_bridge_7d_v2.py
  RUNBOOK.md
```

`starvla_genesis_bridge.py` owns camera capture, policy chunks, action bounds, Cartesian IK, JSONL logs, and MP4 output. `run_starvla_genesis_bridge_7d_v2.py` supplies the two host/checkpoint-specific compatibility corrections: the older WebSocket client's unsupported keepalive arguments and the checkpoint's 7-value state encoder.

## Verified policy contract

| Item | Observed value |
| --- | --- |
| Transport | WebSocket + StarVLA MsgPack/NumPy encoding |
| Endpoint | `10.110.10.12:6694` |
| Unnormalization key | `oxe_bridge` |
| Request image | one `uint8` HWC RGB image inside `image: [image]` |
| Request state | `float32 (1, 7)`: end-effector `xyz`, `rpy`, normalized gripper |
| Request language | `lang: <instruction>` |
| Response envelope | `data.actions` |
| Action tensor | finite `float` `(1, 16, 7)` |
| Action meaning | `[dx, dy, dz, droll, dpitch, dyaw, open_gripper]` |

The checkpoint metadata exposes eight state labels including a `pad` slot, but the loaded state encoder requires 7 values. A live `(1, 8)` request produced `mat1 and mat2 shapes cannot be multiplied (1x8 and 7x1024)`; a live `(1, 7)` request completed and returned actions in `[-0.03493022546172142, 1.0]`.

## Start the StarVLA service

On the policy host, keep this terminal open for the simulator run:

```bash
cd /data/VLA_model/StarVLA/repo
CKPT=playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/VLA_model/StarVLA/repo \
  /data/anaconda3/envs/starvla_env/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 6694 --use_bf16
```

Wait for `server listening on 0.0.0.0:6694`. The policy host clears child processes created by a short-lived remote SSH command, so use a retained terminal session. Stop the service with `Ctrl-C` after the run to release its shared GPU.

## Run the 200-step episode

On the Genesis host:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v2.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

Evidence is written to `recordings/starvla_genesis_bridge/bridge-run.jsonl` and `bridge-run.mp4` below the Genesis source tree.

## Safety limits

- Each translation increment is clipped to `0.01 m` by default.
- Each RPY increment is clipped to `0.05 rad` by default.
- Cartesian targets are limited to `0.45 m` from the initialized end-effector position.
- A non-finite action, unexpected tensor shape, failed IK result, or policy error stops the simulation.
- The bridge makes one policy call per 16 simulator steps, then consumes the local action chunk.

## Validation record

1. Confirmed the policy starts when the command runs from the StarVLA repository working directory.
2. Confirmed a live Genesis-host WebSocket request reaches the server and returns the expected finite `(1, 16, 7)` action chunk.
3. Confirmed Genesis 0.4.3 creates the camera and viewer under `DISPLAY=:1`, initializes the satellite manipulator, and creates an MP4 recording.
4. Corrected the CUDA gripper-state conversion before the full 200-step run.
