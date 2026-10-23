#!/usr/bin/env python3
"""Pilot: ask the VLA-JEPA server what IT would do in the Genesis scene.

Record-only. The arm is never driven by JEPA's output, so the unknown action
semantics cannot cause harm or corrupt the test. We only need the DIRECTION of
its commanded displacement, compared against the known direction to the target.

Why this is the cheapest decisive experiment
--------------------------------------------
Bridge (real WidowX photos) produced commands statistically indistinguishable
from random directions here: mean cos ~0, path/net 8-51x. VLA-JEPA-LIBERO is
sim-trained (MuJoCo/Franka), so the render-domain gap is smaller -- but it has
its own gap (LIBERO is a tabletop with gravity; this scene is a zero-g floating
satellite). Deciding "does this checkpoint understand this scene" tells us
whether ANY off-the-shelf VLA can work here, or whether we must fine-tune.

The metric needs no knowledge of the action scale, only its direction, so
several candidate interpretations of the 7-vector are scored and reported.

--introspect boots the scene, reports camera accessors, and exits.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tree", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--introspect", action="store_true")
    p.add_argument("--jepa-host", default="127.0.0.1")
    p.add_argument("--jepa-port", type=int, default=9012)
    p.add_argument("--instruction", default="put the moka pot on the stove")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--object-size", type=float, default=0.18)
    p.add_argument("--object-thickness", type=float, default=0.04)
    p.add_argument("--object-forward-x", type=float, default=0.06,
                   help="Where to put the target, relative to the initial pinch centre.")
    p.add_argument("--waypoints", type=int, default=5)
    p.add_argument("--ik-steps", type=int, default=500)
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--protocol-version", type=int, default=1)
    p.add_argument("--no-batch-renderer", action="store_true")
    p.add_argument("--skip-jepa", action="store_true",
                   help="Collect observations only; do not contact the server.")
    return p.parse_args()


def vec(t) -> np.ndarray:
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    return np.asarray(t, dtype=np.float64).reshape(-1)


def to_uint8_rgb(rendered) -> np.ndarray:
    """Genesis returns CUDA tensors; convert, then normalise to HWC uint8."""
    import torch
    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()
    img = np.asarray(rendered)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"unexpected image shape {img.shape}")
    if img.dtype != np.uint8:
        if img.max(initial=0.0) <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def sensor_image(sensor, label: str, log):
    """capture() FIRST, then latest_rgb.

    latest_rgb is just a buffer: read without a preceding capture and you get a
    correctly-shaped all-zero image, which silently passes shape checks.
    """
    attempts = []
    if hasattr(sensor, "capture"):
        def _cap_then_read(s=sensor):
            try:
                s.capture()
            except TypeError:
                s.capture(None)
            return getattr(s, "latest_rgb", None)
        attempts.append(("capture->latest_rgb", _cap_then_read))
    inner = getattr(sensor, "_batch_cam", None)
    if inner is not None:
        attempts.append(("_batch_cam.render", lambda: inner.render(rgb=True, force_render=True)))
    if hasattr(sensor, "latest_rgb"):
        attempts.append(("latest_rgb(no capture)", lambda: sensor.latest_rgb))
    for name, fn in attempts:
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001
            log(f"    {label}.{name}: FAILED {type(exc).__name__}: {str(exc)[:70]}")
            continue
        if out is None:
            log(f"    {label}.{name}: returned None")
            continue
        try:
            arr = to_uint8_rgb(out)
            log(f"    {label}.{name}: OK shape={arr.shape} dtype={arr.dtype} mean={arr.mean():.2f}")
            return arr
        except Exception as exc:  # noqa: BLE001
            hint = f" tuple[{len(out)}]" if isinstance(out, (tuple, list)) else ""
            log(f"    {label}.{name}: unusable {type(out).__name__}{hint} ({exc})")
    return None


def encode_npz(fields: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **fields)  # uncompressed: the server rejects zip-deflate
    return buf.getvalue()


def post_infer(host: str, port: int, body: bytes, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"http://{host}:{port}/infer",
        data=body,
        headers={"Content-Type": "application/x-npz", "Content-Length": str(len(body))},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def cos(a, b) -> float | None:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_fh = (args.output_dir / "pilot.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        print(f"[jepa-pilot] {msg}", flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    tree_src = args.tree.resolve() / "src"
    if not tree_src.is_dir():
        log(f"FATAL: {tree_src} is not a directory")
        return 2
    sys.path.insert(0, str(tree_src))

    from configs.configs import SCENE_SETTINGS, TPV_CAM_SETTINGS
    from configs.config_starlink_manipulator import FRANKA_S_Q_CAMERA

    SCENE_SETTINGS["show_viewer"] = False
    TPV_CAM_SETTINGS["enable_recording"] = False
    res = (args.image_size, args.image_size)
    if isinstance(TPV_CAM_SETTINGS.get("camera"), dict):
        TPV_CAM_SETTINGS["camera"].update({"res": res, "GUI": False})
    if isinstance(FRANKA_S_Q_CAMERA.get("camera"), dict):
        FRANKA_S_Q_CAMERA["camera"]["res"] = res
    FRANKA_S_Q_CAMERA["enable_recording"] = False
    for key in ("base_camera",):
        sub = FRANKA_S_Q_CAMERA.get(key)
        if isinstance(sub, dict):
            sub["enable_recording"] = False
            if isinstance(sub.get("camera"), dict):
                sub["camera"]["res"] = res

    import genesis as gs
    import torch
    from controllers.backend import EmptyBackend
    from envs.genesis_env import GenesisSim
    from robots.satellite_manipulator import SatelliteManipulator

    if not args.no_batch_renderer:
        SCENE_SETTINGS["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
        log("forced BatchRenderer")

    log("booting scene (robot first, then our target box, then build)")
    sim = GenesisSim()
    manip = SatelliteManipulator(name="franka_merge", sensors=[], backends=[EmptyBackend()])
    target_entity = sim.scene.add_entity(
        gs.morphs.Box(size=(args.object_size, args.object_thickness, args.object_thickness),
                      pos=(0.0, 0.0, 0.0), collision=True),
        material=gs.materials.Rigid(),
    )
    sim.start()
    manip.initialize()
    n_envs = int(sim.num_envs)

    # --- keep the joint-change bound while satisfying the batched IK contract ---
    st = {"prev": None, "logged": False}
    mjc = float(manip.params["ik_params"]["max_joint_change"])

    def control_joints(position, quaternion):
        pos = np.repeat(np.asarray(vec(position)).reshape(1, 3), n_envs, axis=0)
        quat = np.repeat(np.asarray(vec(quaternion)).reshape(1, 4), n_envs, axis=0)
        pt = torch.as_tensor(pos, dtype=manip.datatype, device=manip.device)
        qt = torch.as_tensor(quat, dtype=manip.datatype, device=manip.device)
        qpos = manip.robot.inverse_kinematics(link=manip.end_effector, pos=pt, quat=qt)
        if not st["logged"]:
            log(f"IK ok: in {tuple(pt.shape)} -> qpos{tuple(qpos.shape)} n_envs={n_envs}")
            st["logged"] = True
        if not torch.isfinite(qpos).all():
            return False
        q = qpos[0] if qpos.dim() > 1 else qpos
        idx = manip.motors_dof if q.numel() == manip.robot.n_dofs else manip.motors_qs
        mq = q[idx]
        prev = st["prev"]
        if prev is not None and prev.numel() == mq.numel():
            d = mq - prev
            nm = torch.linalg.vector_norm(d)
            if float(nm) > mjc:
                mq = prev + d * (mjc / float(nm))
        st["prev"] = mq.detach().clone()
        return manip.control_joint_pos(mq)

    manip.control_joints = control_joints

    def step_once():
        manip.step()
        sim.step()

    def pinch_center() -> np.ndarray:
        pts = [vec(l.get_pos()) for l in manip.robot.links if "Finger" in l.name]
        return np.mean(np.asarray(pts, dtype=np.float64), axis=0) if len(pts) >= 2 else np.zeros(3)

    def gripper_qpos() -> np.ndarray:
        q = vec(manip.robot.get_qpos(qs_idx_local=manip.fingers_qs))
        if q.size < 2:
            q = np.resize(q, 2)
        return q[:2]

    manip.control_gripper(1.0)
    for _ in range(args.settle_steps):
        step_once()
    manip.update_state()

    sim_cam = getattr(sim, "cam", None)
    wrist = getattr(manip, "rl_wrist_camera", None)
    base = getattr(manip, "rl_base_camera", None)

    if args.introspect:
        info = {"sensors": {}, "pinch_center_m": pinch_center().tolist(),
                "ee_position_m": vec(manip.ee_global_position).tolist(), "num_envs": n_envs}
        log("probing image accessors:")
        for label, s in (("sim.cam", sim_cam), ("rl_wrist_camera", wrist), ("rl_base_camera", base)):
            if s is None:
                log(f"  {label}: absent")
                continue
            arr = sensor_image(s, label, log)
            info[f"image_ok::{label}"] = None if arr is None else list(arr.shape)
            if arr is not None:
                info[f"image_mean::{label}"] = float(arr.mean())
        (args.output_dir / "introspect.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log(f"wrote {args.output_dir / 'introspect.json'}")
        log_fh.close()
        try:
            manip.stop(); sim.stop()
        except Exception:  # noqa: BLE001
            pass
        return 0

    # ---------------- target placement ----------------
    pinch_start = pinch_center()
    target_pos = pinch_start + np.array([args.object_forward_x, 0.0, 0.0])
    target_entity.set_pos(target_pos)
    for _ in range(20):
        step_once()
    placed = vec(target_entity.get_pos())
    log(f"pinch_start={np.round(pinch_start, 4).tolist()} target_placed={np.round(placed, 4).tolist()}")

    # ---------------- waypoint sweep ----------------
    q0 = vec(manip.ee_global_quaternion)
    records = []
    for wi in range(args.waypoints):
        t = wi / max(1, args.waypoints - 1)
        pinch_goal = pinch_start + (placed - pinch_start) * t
        # command the TCP so the pinch centre lands on the goal (jaw offset applied)
        jaw_off = pinch_center() - vec(manip.ee_global_position)
        tcp_goal = pinch_goal - jaw_off
        for _ in range(args.ik_steps):
            manip.control_joints(tcp_goal, q0)
            step_once()
        manip.update_state()
        pinch_now = pinch_center()
        to_target = placed - pinch_now
        rec = {
            "waypoint": wi,
            "t": t,
            "pinch_center_m": pinch_now.tolist(),
            "ee_position_m": vec(manip.ee_global_position).tolist(),
            "ee_quaternion_wxyz": vec(manip.ee_global_quaternion).tolist(),
            "target_m": placed.tolist(),
            "to_target_m": to_target.tolist(),
            "pinch_center_to_target_m": float(np.linalg.norm(to_target)),
            "gripper_qpos": gripper_qpos().tolist(),
        }
        # agentview = the base-mounted camera looking at the workspace (the closest
        # analogue to LIBERO's fixed third-person 'agentview'); wrist = eye-in-hand.
        img_scene = sensor_image(base, "rl_base_camera(agentview)", log) if base is not None else None
        img_wrist = sensor_image(wrist, "rl_wrist_camera", log) if wrist is not None else None
        rec["image_shapes"] = {
            "scene": None if img_scene is None else list(img_scene.shape),
            "wrist": None if img_wrist is None else list(img_wrist.shape),
        }
        rec["image_mean"] = {
            "scene": None if img_scene is None else float(img_scene.mean()),
            "wrist": None if img_wrist is None else float(img_wrist.mean()),
        }
        if img_scene is not None and float(img_scene.mean()) == 0.0:
            log(f"  waypoint {wi}: WARNING agentview is all-black")
        if img_scene is not None and img_wrist is not None and not args.skip_jepa:
            ts = time.perf_counter()
            try:
                body = encode_npz({
                    "protocol_version": np.asarray(args.protocol_version, dtype=np.int64),
                    "request_id": np.asarray(uuid.uuid4().hex),
                    "instruction": np.asarray(args.instruction),
                    "agentview_image": img_scene.astype(np.uint8),
                    "robot0_eye_in_hand_image": img_wrist.astype(np.uint8),
                    "robot0_eef_pos": vec(manip.ee_global_position).astype(np.float32),
                    "robot0_eef_quat": vec(manip.ee_global_quaternion).astype(np.float32),
                    "robot0_gripper_qpos": gripper_qpos().astype(np.float32),
                })
                resp = post_infer(args.jepa_host, args.jepa_port, body)
                chunk = np.asarray(resp.get("action_chunk"), dtype=np.float64)
                rec["jepa_ok"] = True
                rec["jepa_seconds"] = time.perf_counter() - ts
                rec["action_chunk"] = chunk.tolist()
                rec["action_shape"] = list(chunk.shape)
                log(f"  waypoint {wi}: chunk {chunk.shape} in {rec['jepa_seconds']:.2f}s")
            except Exception as exc:  # noqa: BLE001
                rec["jepa_ok"] = False
                rec["jepa_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  waypoint {wi}: JEPA FAILED {rec['jepa_error']}")
        records.append(rec)
        log(f"  waypoint {wi}: pinch_to_target={rec['pinch_center_to_target_m']:.4f} m "
            f"images scene={rec['image_shapes']['scene']} wrist={rec['image_shapes']['wrist']}")

    # ---------------- direction analysis over candidate interpretations ----------------
    analysis = {}
    if not args.skip_jepa and any(r.get("jepa_ok") for r in records):
        def score(name, fn):
            vals = []
            for r in records:
                if not r.get("jepa_ok"):
                    continue
                a = fn(np.asarray(r["action_chunk"], dtype=np.float64), r)
                c = cos(a, r["to_target_m"])
                if c is not None:
                    vals.append(c)
            if vals:
                analysis[name] = {
                    "n": len(vals), "mean_cos": float(np.mean(vals)),
                    "median_cos": float(np.median(vals)),
                    "fraction_cos_positive": float(np.mean([v > 0 for v in vals])),
                    "fraction_cos_gt_0.3": float(np.mean([v > 0.3 for v in vals])),
                }

        score("delta::row0_xyz", lambda a, r: a[0, :3])
        score("delta::neg_row0_xyz", lambda a, r: -a[0, :3])
        score("delta::mean_xyz", lambda a, r: a[:, :3].mean(axis=0))
        score("absolute::row0_minus_pinch",
              lambda a, r: a[0, :3] - np.asarray(r["pinch_center_m"]))
        score("absolute::row0_minus_ee",
              lambda a, r: a[0, :3] - np.asarray(r["ee_position_m"]))
        score("absolute::last_minus_first",
              lambda a, r: a[-1, :3] - a[0, :3])

    out = {"args": {k: str(v) for k, v in vars(args).items()},
           "pinch_start_m": pinch_start.tolist(),
           "target_placed_m": placed.tolist(),
           "records": records,
           "direction_analysis": analysis}
    (args.output_dir / "jepa_pilot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.output_dir / "jepa_pilot_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")

    log("=== DIRECTION ANALYSIS (cos between candidate action interpretation and direction to target) ===")
    for name, st_ in analysis.items():
        log(f"  {name:<34} mean={st_['mean_cos']:+.3f} med={st_['median_cos']:+.3f} "
            f"frac>0={st_['fraction_cos_positive']:.0%} frac>0.3={st_['fraction_cos_gt_0.3']:.0%}")
    if not analysis:
        log("  (no successful JEPA queries)")
    log(f"wrote {args.output_dir / 'jepa_pilot.json'}")
    log_fh.close()
    try:
        manip.stop(); sim.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
