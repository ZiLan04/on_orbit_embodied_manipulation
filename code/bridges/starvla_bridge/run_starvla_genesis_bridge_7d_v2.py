"""Run the Genesis bridge using the verified StarVLA transport and state layout."""

from __future__ import annotations

import numpy as np
import torch
import websockets.sync.client


_connect = websockets.sync.client.connect


def _compatible_connect(*args, **kwargs):
    """Support the host sync client, which rejects newer ping options."""
    try:
        return _connect(*args, **kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        kwargs.pop("ping_interval", None)
        kwargs.pop("ping_timeout", None)
        return _connect(*args, **kwargs)


websockets.sync.client.connect = _compatible_connect

from starvla_genesis_bridge import _as_vector, _quaternion_to_rpy, _to_uint8_rgb
from starvla_genesis_bridge import StarVLAGenesisBridge, main


def _bridge_observation(self):
    """Produce the checkpoint's 7-value state without converting CUDA tensors implicitly."""
    self.starlink_manipulator.update_state()
    rgb, *_ = self.gsim.cam.render(rgb=True, force_render=True)
    position = _as_vector(self.starlink_manipulator.ee_global_position, 3)
    quaternion = _as_vector(self.starlink_manipulator.ee_global_quaternion, 4)
    rpy = _quaternion_to_rpy(quaternion)
    self.starlink_manipulator.get_gripper_value()
    gripper = float(
        torch.as_tensor(self.starlink_manipulator.gripper_state)
        .detach()
        .cpu()
        .reshape(-1)[0]
    )
    state = np.array(
        [*position, *rpy, float(np.clip(gripper, 0.0, 1.0))], dtype=np.float32
    ).reshape(1, 7)
    return _to_uint8_rgb(rgb), state, position, quaternion


StarVLAGenesisBridge._observation = _bridge_observation


if __name__ == "__main__":
    raise SystemExit(main())
