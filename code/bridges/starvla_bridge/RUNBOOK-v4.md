# Genesis - StarVLA Bridge Runbook

## Acceptance command

Keep the existing StarVLA policy process running on port `6694` from the StarVLA repository directory. Then run this exact command on the Genesis host:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v4.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

The policy server command is:

```bash
cd /data/VLA_model/StarVLA/repo
CKPT=playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/VLA_model/StarVLA/repo \
  /data/anaconda3/envs/starvla_env/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 6694 --use_bf16
```

## Closed-loop behavior

1. Render a `224x224` Genesis scene RGB image and read end-effector `xyz`, `rpy`, and gripper state.
2. Send the image, a `(1, 7)` state, instruction, and `unnorm_key="oxe_bridge"` through the live WebSocket.
3. Receive `data.actions` with shape `(1, 16, 7)` and cache its 16 actions.
4. Clamp Cartesian increments to `0.01 m`, RPY increments to `0.05 rad`, and workspace drift to `0.45 m` from the initial end-effector position.
5. Solve IK using the existing `SmoothIKSolver`, extract the six motor qpos values from the solver's `[env, qpos]` result, and issue them through the existing direct joint controller.
6. Command the gripper, advance Genesis, log the command, and repeat.

MP4 and JSONL evidence are placed in `recordings/starvla_genesis_bridge/`.

## Compatibility facts discovered during integration

- The Genesis WebSocket client rejects the newer keepalive options; the launcher retries without them.
- The loaded Bridge checkpoint accepts `(1, 7)`, not the metadata's padded 8-value state. The request state is `xyz + rpy + normalized_gripper`.
- Genesis 0.4.3 IK needs batched end-effector inputs, `(1, 3)` and `(1, 4)`.
- Its existing `control_joints` helper indexes the batched qpos result as though it were one-dimensional. The v4 launcher preserves its IK solver but extracts `qpos[:, motors_qs]` before calling `control_joint_pos`.

No existing simulator or StarVLA source files are changed.
