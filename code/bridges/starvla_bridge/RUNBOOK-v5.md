# Genesis - StarVLA Bridge Runbook

## Final command

On the Genesis host, after the policy server reports that it is listening on port `6694`:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v5.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

## Closed-loop pipeline

```text
Genesis scene RGB + end-effector xyz/RPY/gripper
  -> WebSocket request to StarVLA with oxe_bridge statistics
  -> finite (1, 16, 7) action chunk
  -> per-action Cartesian and rotational limits
  -> SmoothIKSolver -> six motor joint qpos -> Genesis step
  -> next camera observation
```

The bridge is isolated in `starvla_bridge/`. It records `bridge-run.jsonl` and `bridge-run.mp4` under `recordings/starvla_genesis_bridge/`.

## Runtime-specific adapters

- StarVLA's Bridge checkpoint accepts `state.shape == (1, 7)`: `xyz + rpy + normalized_gripper`.
- The Genesis host's WebSocket library retries without unsupported keepalive arguments.
- Genesis 0.4.3 IK receives `[1, 3]` position and `[1, 4]` quaternion CUDA tensors.
- The batched IK output is indexed as `qpos[:, motors_qs]` before calling the existing direct joint controller.

All adapters are in the launcher; `spaceman_core` and StarVLA source files are unchanged.
