"""Launch the StarVLA Genesis bridge with the installed WebSocket compatibility shim."""

from __future__ import annotations

import websockets.sync.client


_connect = websockets.sync.client.connect


def _compatible_connect(*args, **kwargs):
    """Retry without keepalive options on the Genesis host's older sync client."""
    try:
        return _connect(*args, **kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        kwargs.pop("ping_interval", None)
        kwargs.pop("ping_timeout", None)
        return _connect(*args, **kwargs)


websockets.sync.client.connect = _compatible_connect

from starvla_genesis_bridge import main


if __name__ == "__main__":
    raise SystemExit(main())
