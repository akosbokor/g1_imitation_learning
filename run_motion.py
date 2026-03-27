"""
G1 Boxing — Motion Streamer
===========================
Loads the bundled punching03.pkl motion, converts it to TWIST mimic observations,
and streams them into Redis at 50 Hz so the robot policy tracks the motion.

Run AFTER starting run_robot.py (which also starts Redis internally).

Usage:
    python3 run_motion.py           # streams punching03 once
    python3 run_motion.py --loop    # loops continuously
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from shm_store import SharedMemStore

HERE        = os.path.dirname(os.path.abspath(__file__))
MOTION_PATH = os.path.join(HERE, "assets", "motions", "punching03.pkl")

sys.path.insert(0, HERE)
from data_utils.params import DEFAULT_MIMIC_OBS

CONTROL_DT = 0.02   # 50 Hz


# ── Quaternion helpers ────────────────────────────────────────────────────────

def quat_to_rpy(q):
    """wxyz → roll, pitch, yaw"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (wxyz convention)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


def quat_mul(qa, qb):
    wa, xa, ya, za = qa
    wb, xb, yb, zb = qb
    return np.array([
        wa*wb - xa*xb - ya*yb - za*zb,
        wa*xb + xa*wb + ya*zb - za*yb,
        wa*yb - xa*zb + ya*wb + za*xb,
        wa*zb + xa*yb - ya*xb + za*wb,
    ])


# ── Motion processing ─────────────────────────────────────────────────────────

def build_mimic_stream(pkl_path, target_fps=50.0):
    """
    Load .pkl motion, resample to target_fps, compute 33-dim mimic obs per frame.

    mimic_obs layout (33 dims):
      [height(1), roll(1), pitch(1), yaw(1), root_vel_local(3), ang_vel_z(1), dof_pos_25(25)]
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    src_fps  = float(data["fps"])
    root_pos = np.array(data["root_pos"], dtype=np.float32)   # (N, 3)
    root_rot = np.array(data["root_rot"], dtype=np.float32)   # (N, 4) xyzw
    dof_pos  = np.array(data["dof_pos"],  dtype=np.float32)   # (N, 23)
    N = root_pos.shape[0]

    # Resample to target_fps
    src_t = np.arange(N) / src_fps
    tgt_t = np.arange(0, src_t[-1], 1.0 / target_fps)

    def interp(arr):
        return np.stack([np.interp(tgt_t, src_t, arr[:, i]) for i in range(arr.shape[1])], axis=1)

    rp = interp(root_pos).astype(np.float32)
    rr = interp(root_rot).astype(np.float32)
    dp = interp(dof_pos).astype(np.float32)

    # pkl quaternions are xyzw → convert to wxyz
    rr = rr[:, [3, 0, 1, 2]]
    rr /= np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8

    M  = rp.shape[0]
    dt = 1.0 / target_fps

    # Root linear velocity (world frame → finite diff)
    rv = np.zeros((M, 3), np.float32)
    rv[:-1] = (rp[1:] - rp[:-1]) / dt
    rv[-1]  = rv[-2]

    # Root angular velocity (body frame, from quaternion differences)
    av = np.zeros((M, 3), np.float32)
    for i in range(M - 1):
        q1_conj = np.array([rr[i, 0], -rr[i, 1], -rr[i, 2], -rr[i, 3]])
        dq = quat_mul(q1_conj, rr[i + 1])
        if dq[0] < 0:
            dq = -dq
        av[i] = 2.0 * dq[1:] / dt
    av[-1] = av[-2]

    # Reference yaw at frame 0 (so motion always starts facing forward)
    _, _, yaw_0 = quat_to_rpy(rr[0])

    stream = []
    for i in range(M):
        roll, pitch, yaw = quat_to_rpy(rr[i])
        yaw = (yaw - yaw_0 + np.pi) % (2 * np.pi) - np.pi

        rv_local = quat_rotate_inverse(rr[i], rv[i])
        av_z     = av[i, 2]

        # Expand 23-DOF → 25-DOF (insert wrist zeros at indices 19 and 24)
        dp25 = np.zeros(25, np.float32)
        body_ids = [j for j in range(25) if j not in [19, 24]]
        dp25[body_ids] = dp[i]

        obs = np.concatenate([
            [rp[i, 2]],         # height (1)
            [roll, pitch, yaw], # orientation (3)
            rv_local,           # root linear vel local (3)
            [av_z],             # yaw rate (1)
            dp25,               # joint positions (25)
        ]).astype(np.float32)   # total: 33

        stream.append(obs)

    return stream


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Loop motion continuously")
    parser.add_argument("--motion", default=MOTION_PATH,
                        help="Path to .pkl motion file (default: bundled punching03.pkl)")
    args = parser.parse_args()

    r = SharedMemStore()

    print(f"Loading motion: {args.motion}")
    stream = build_mimic_stream(args.motion)
    duration = len(stream) * CONTROL_DT
    print(f"Motion: {len(stream)} frames  ({duration:.1f}s at 50 Hz)")

    default_mimic = DEFAULT_MIMIC_OBS["g1"].tolist()

    # Write init pose for Reference State Initialization
    # Layout: [height, roll, pitch, yaw, vel(3), av_z, dof_25(25)]
    first = stream[0]
    r.set("init_pose_g1", json.dumps({
        "height": float(first[0]),
        "dof_25": first[8:].tolist(),
    }))
    print(f"Init pose: height={first[0]:.3f}  joints_max={float(np.max(np.abs(first[8:]))):.3f} rad")

    # Warm-up: hold first frame for 3 s so policy history fills up
    warmup_steps = int(3.0 / CONTROL_DT)
    print(f"Warm-up ({warmup_steps} steps, 3 s)...")
    for _ in range(warmup_steps):
        r.set("action_mimic_g1", json.dumps(stream[0].tolist()))
        time.sleep(CONTROL_DT)

    print("Streaming boxing motion... (Ctrl+C to stop)")
    try:
        while True:
            for obs in stream:
                t0 = time.time()
                r.set("action_mimic_g1", json.dumps(obs.tolist()))
                elapsed = time.time() - t0
                rem = CONTROL_DT - elapsed
                if rem > 0:
                    time.sleep(rem)
            if not args.loop:
                break
            print("  [loop restart]")
    except KeyboardInterrupt:
        pass

    # Smooth return to default pose
    print("Returning to default pose...")
    last = stream[-1].copy()
    default = np.array(default_mimic, dtype=np.float32)
    for i in range(50):
        alpha = (i + 1) / 50.0
        blended = last + (default - last) * alpha
        r.set("action_mimic_g1", json.dumps(blended.tolist()))
        time.sleep(CONTROL_DT)

    r.set("action_mimic_g1", json.dumps(default_mimic))
    print("Done.")


if __name__ == "__main__":
    main()
