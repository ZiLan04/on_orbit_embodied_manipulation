# pylibs/ — path shim, not source

This directory exists so the Genesis-side interpreter (`Spaceman_env`) can import a
package that only the StarVLA-side interpreter (`starvla_env`) has installed.

On the server, the single entry here is a **symlink**:

```
pylibs/websockets -> /data/anaconda3/envs/starvla_env/lib/python3.10/site-packages/websockets
```

It is not committed, because a symlink into an absolute interpreter path is only
meaningful on that one machine. Recreate it during setup:

```bash
ln -s /data/anaconda3/envs/starvla_env/lib/python3.10/site-packages/websockets \
      /data/Yanhan/genesis/code/franka_cube/pylibs/websockets
```

Then put this directory on `PYTHONPATH` when running the Genesis side:

```bash
export PYTHONPATH=/data/Yanhan/genesis/code/franka_cube/pylibs:$PYTHONPATH
```

## Why a shim instead of an install

`Spaceman_env` and `starvla_env` are deliberately complementary (see
`documents/code-map.en.md` §1.2). Installing into either one to "fix" a missing
package tends to pull a conflicting numpy/torch and break the other side. A path
shim exposes exactly one package and nothing else, so no other package leaks in.
