"""Run the Genesis bridge with the StarVLA checkpoint's verified 7-value state."""

from __future__ import annotations

import numpy as np
import websockets.sync.client


_connect = websockets.sync.client.connect


def _compatible_connect(*args, **kwargs):
    """Support the host's sync WebSocket client, which rejects ping options."""
    try:
        return _connect(*args, **kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        kwargs.pop("ping_interval", None)
        kwargs.pop("ping_timeout", None)
        return _connect(*args, **kwargs)


websockets.sync.client.connect = _compatible_connect

from starvla_genesis_bridge import StarVLAGenesisBridge, main


_observation = StarVLAGenesisBridge._observation


def _bridge_observation(self):
    """Drop the all-zero pad slot because this checkpoint encodes state_dim=7."""
    image, state, position, quaternion = _observation(self)
    return image, np.concatenate((state[:, :6], state[:, 7:]), axis=1), position, quaternion


StarVLAGenesisBridge._observation = _bridge_observation


if __name__ == "__main__":
    raise SystemExit(main())
