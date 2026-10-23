# Genesis - StarVLA Bridge Acceptance Result

## Result

The Genesis simulation completed a 200-step closed-loop episode driven by the remote StarVLA Bridge policy. The policy server was stopped after the run released its GPU.

| Check | Evidence |
| --- | --- |
| Cross-machine policy access | Genesis connected to the WebSocket service at port `6694` and received server metadata. |
| Policy output | Every received chunk had the verified Bridge layout `(1, 16, 7)` and finite values. |
| Continuous run | The process exited with status `0` after 200 simulator steps. |
| Action log | `bridge-run.jsonl` has 202 records: initialization, 200 action records, and shutdown. |
| Numeric safety | All raw actions and target positions were finite. |
| Motion command evidence | Target positions reached 2.166 cm from the initial target and accumulated 41.933 cm of Cartesian path. |
| Video evidence | Genesis saved `bridge-run.mp4`; size is 76 KB. |

## Artifacts

```text
/home/robotarm/spaceman_core/src/recordings/starvla_genesis_bridge/acceptance-200/
  bridge-run.jsonl   109 KB
  bridge-run.mp4      76 KB
```

## Reproduction

Start the existing policy server from `/data/VLA_model/StarVLA/repo`, wait for `server listening on 0.0.0.0:6694`, then run on Genesis:

```bash
cd /home/robotarm/spaceman_core/src
XAUTHORITY=/home/robotarm/.Xauthority DISPLAY=:1 \
PYTHONPATH=/home/robotarm/spaceman_core/src \
  /home/robotarm/genesis043_venv/bin/python \
  starvla_bridge/run_starvla_genesis_bridge_7d_v6.py \
  --instruction "move the robot arm safely" \
  --steps 200 \
  --output-dir recordings/starvla_genesis_bridge/acceptance-200
```

## Important compatibility facts

- Pass `unnorm_key="oxe_bridge"` for every policy request.
- The loaded checkpoint accepts `(1, 7)` state: end-effector `xyz`, `rpy`, and normalized gripper. The metadata's additional pad label must not be sent.
- The Genesis host's WebSocket client requires keepalive options to be omitted.
- Genesis 0.4.3 uses batched tensor IK inputs. The launcher sends these to the underlying Genesis IK API, extracts the six arm motor targets, and applies a maximum joint-target change before the existing direct joint controller.
