# Genesis - StarVLA Bridge Runbook

## Run the validated bridge

With the existing StarVLA policy server listening on `10.110.10.12:6694`, run on the Genesis host:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v6.py \
  --instruction "move the robot arm safely" \
  --steps 200
```

Evidence files are `recordings/starvla_genesis_bridge/bridge-run.jsonl` and `bridge-run.mp4`.

## Pipeline

1. Render a `224x224` scene image and read the end-effector `xyz`, `rpy`, and normalized gripper state.
2. Request `oxe_bridge` actions from StarVLA through WebSocket with `(1, 7)` state.
3. Cache each finite `(1, 16, 7)` action chunk.
4. Clamp Cartesian action components and compose a bounded end-effector target.
5. Invoke the Genesis 0.4.3 batched IK API, select the six motor qpos entries, limit the total joint-target change, and use the existing direct joint controller.
6. Command the gripper, advance the simulator, write JSONL, and repeat.

## Integration-specific facts

- `oxe_bridge` has no default unnormalization key and must be supplied on every request.
- The live checkpoint accepts `xyz + rpy + gripper` state `(1, 7)`, not the metadata-only padded 8-value layout.
- The Genesis host's WebSocket library needs keepalive parameters omitted.
- Genesis 0.4.3 requires batched tensor end-effector IK inputs. Its local smooth wrapper cannot handle those batch dimensions, so the launcher uses the underlying batched Genesis IK and applies the same maximum joint-change protection before the existing direct controller.

No core simulator or StarVLA source file is modified.
