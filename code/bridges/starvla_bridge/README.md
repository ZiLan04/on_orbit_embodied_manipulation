# Genesis - StarVLA bridge

## Goal

`starvla_genesis_bridge.py` runs a single Genesis episode whose Cartesian arm targets are supplied by the remote StarVLA Bridge policy. It creates a scene RGB observation, derives the end-effector state, requests one 16-step action chunk, applies each action with Cartesian and rotational limits, then requests the next chunk.

The bridge is isolated in `spaceman_core/src/starvla_bridge/`; existing `spaceman_core` files and the StarVLA repository remain unchanged.

## Verified interface

- Transport: WebSocket on `10.110.10.12:6694`, with StarVLA's MsgPack plus NumPy-array encoding.
- Request: `examples=[{"image": [rgb_uint8_hwc], "lang": instruction, "state": state}]`, `unnorm_key="oxe_bridge"`, `do_sample=False`, `use_ddim=True`, and `num_ddim_steps=10`.
- Live response: `{"status", "ok", "type", "request_id", "data": {"actions": ndarray}}`.
- Live action shape: `(1, 16, 7)` with finite values. The seven values are `[dx, dy, dz, droll, dpitch, dyaw, open_gripper]`.
- State: the checkpoint's `state_encoder` accepts `(1, 7)`, specifically `xyz + rpy + normalized_gripper`. Its metadata lists an eighth pad field, but sending that field causes a `1x8` to `7x1024` matrix error. `run_starvla_genesis_bridge_7d.py` removes this pad before the request.

## Start the policy server

Run this from a terminal on the policy host and keep the terminal open during the Genesis episode:

```bash
cd /data/VLA_model/StarVLA/repo
CKPT=playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/VLA_model/StarVLA/repo \
  /data/anaconda3/envs/starvla_env/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 6694 --use_bf16
```

The required readiness message is `server listening on 0.0.0.0:6694`. The current remote SSH session clears background children, so do not rely on `nohup` from a short-lived session. Stop the server with `Ctrl-C` after testing to release the shared GPU.

## Run the Genesis bridge

On the Genesis host:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

The default command records `recordings/starvla_genesis_bridge/bridge-run.jsonl` and `bridge-run.mp4`. The JSONL file records every bounded command, observed state, target position, target quaternion, and gripper command.

Useful overrides:

```bash
# Start with a shorter evidence run.
... run_starvla_genesis_bridge_7d.py --steps 32

# Tighten safety limits while retaining the same policy interface.
... run_starvla_genesis_bridge_7d.py --max-xyz-step 0.005 --max-rpy-step 0.03
```

## Safety behavior

- Translation action components are clipped to `0.01 m` by default.
- Rotation components are clipped to `0.05 rad` by default.
- The end effector remains within `0.45 m` of its initialized position.
- A failed IK command, non-finite action, unexpected output shape, or server error stops the run instead of advancing the simulator.
- A policy call occurs once per 16 simulated control steps; the remaining actions are cached locally.

## Progress record

1. Confirmed the existing server starts from the repository working directory and publishes the expected metadata.
2. Confirmed Genesis can connect to the policy host through the actual WebSocket client.
3. Sent a real `224x224` zero-image plus `(1, 7)` state request from Genesis. The server returned a finite `(1, 16, 7)` Bridge action chunk, with observed range `[-0.03493022546172142, 1.0]`.
4. Added the isolated simulator bridge and runtime compatibility launcher. The remaining validation is the 200-step Genesis visual run, which exercises camera initialization, IK, and recording together.
