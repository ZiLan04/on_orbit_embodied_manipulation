# Genesis - StarVLA Bridge Runbook

## Run command

Start the existing policy service on the policy host from its repository directory:

```bash
cd /data/VLA_model/StarVLA/repo
CKPT=playground/Pretrained_models/StarVLA/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/VLA_model/StarVLA/repo \
  /data/anaconda3/envs/starvla_env/bin/python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port 6694 --use_bf16
```

Wait for `server listening on 0.0.0.0:6694`, then run this on the Genesis host:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v3.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

The bridge writes `recordings/starvla_genesis_bridge/bridge-run.jsonl` and `bridge-run.mp4`.

## What is verified

- Genesis successfully connects to the live StarVLA WebSocket service.
- A live request with one `224x224` RGB image and a `(1, 7)` end-effector state receives finite `(1, 16, 7)` Bridge actions.
- The action chunk is decoded as `[dx, dy, dz, droll, dpitch, dyaw, open_gripper]` and cached for 16 simulator steps.
- Translation is limited to `0.01 m`, rotation to `0.05 rad`, and motion to `0.45 m` from the initial end-effector position.
- The launcher adapts the Genesis host's WebSocket library, the checkpoint's 7-value state encoder, GPU-resident gripper state, and Genesis 0.4.3's batch-shaped IK arguments.

## Implementation notes

The bridge lives entirely in `/home/robotarm/spaceman_core/src/starvla_bridge/`. It does not modify the simulator core or StarVLA source. The server must receive `unnorm_key="oxe_bridge"`; the metadata has no default key.

The checkpoint advertises eight state labels but its loaded encoder accepts seven values. The request state is `xyz + rpy + normalized_gripper`; the metadata-only pad slot is omitted. The Genesis IK accepts one-environment targets only when position and quaternion carry leading dimensions `(1, 3)` and `(1, 4)`.

## Debug progress

1. Policy startup, cross-machine WebSocket handshake, server response envelope, and 16-step action tensor were verified with a live request.
2. Genesis scene, camera, viewer, robot initialization, and MP4 recording were verified under `DISPLAY=:1`.
3. The policy action reached Genesis IK during the two-step smoke test. The v3 launcher adds the required batch dimension before the acceptance run.
