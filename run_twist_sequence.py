"""
TWIST motion sequence streamer with smooth transitions.

Plays a list of motions back-to-back, blending linearly between them.

Usage — inline:
    python3 run_twist_sequence.py \\
        --motions accad/A5___pick_up_box.pkl cmu/07_01.pkl kit/punching03.pkl

Usage — YAML config:
    python3 run_twist_sequence.py --sequence configs/twist_demo_sequence.yaml

    python3 run_twist_sequence.py --sequence configs/twist_demo_sequence.yaml --loop

YAML format example (configs/twist_demo_sequence.yaml):
    transition: 1.0   # default blend duration in seconds
    motions:
      - path: accad/A5___pick_up_box.pkl
      - path: accad/A1___Stand.pkl
        hold_last: 2.0          # hold last frame N extra seconds
      - path: cmu/07_01.pkl
      - path: eyes_japan/karate_08_jab_yokoyama.pkl
        repeat: 2               # play motion N times
        max_roll: 0.4
        max_pitch: 0.4
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import redis

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

TWIST   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TWIST")
DATASET = os.path.join(TWIST, "assets/twist_dataset/home/yanjieze/projects/"
                               "g1_wbc/humanoid-motion-imitation/track_dataset/amass-dev-g1")
sys.path.insert(0, os.path.join(TWIST, "deploy_real"))
from data_utils.params import DEFAULT_MIMIC_OBS

CONTROL_DT = 0.02   # 50 Hz


def quat_to_rpy(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def quat_rotate_inverse(q, v):
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


def load_motion(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_mimic_stream(data, target_fps=50.0,
                       max_roll=None, max_pitch=None, max_av_z=None):
    src_fps  = float(data["fps"])
    root_pos = np.array(data["root_pos"], dtype=np.float32)
    root_rot = np.array(data["root_rot"], dtype=np.float32)
    dof_pos  = np.array(data["dof_pos"],  dtype=np.float32)
    N = root_pos.shape[0]

    src_times = np.arange(N) / src_fps
    duration  = src_times[-1]
    tgt_times = np.arange(0, duration, 1.0 / target_fps)

    def interp(arr):
        return np.array([
            np.interp(tgt_times, src_times, arr[:, i])
            for i in range(arr.shape[1])
        ]).T

    rp = interp(root_pos)
    rr = interp(root_rot)
    dp = interp(dof_pos)

    rr = rr[:, [3, 0, 1, 2]]  # xyzw → wxyz
    rr /= np.linalg.norm(rr, axis=1, keepdims=True) + 1e-8

    M  = rp.shape[0]
    dt = 1.0 / target_fps

    rv = np.zeros((M, 3), np.float32)
    rv[:-1] = (rp[1:] - rp[:-1]) / dt
    rv[-1]  = rv[-2]

    av = np.zeros((M, 3), np.float32)
    for i in range(M - 1):
        q1_conj = np.array([rr[i,0], -rr[i,1], -rr[i,2], -rr[i,3]])
        dq = quat_mul(q1_conj, rr[i + 1])
        if dq[0] < 0:
            dq = -dq
        av[i] = 2.0 * dq[1:] / dt
    av[-1] = av[-2]

    _, _, yaw_0 = quat_to_rpy(rr[0])

    stream = []
    for i in range(M):
        roll, pitch, yaw = quat_to_rpy(rr[i])
        yaw = (yaw - yaw_0 + np.pi) % (2 * np.pi) - np.pi
        if max_roll  is not None: roll  = float(np.clip(roll,  -max_roll,  max_roll))
        if max_pitch is not None: pitch = float(np.clip(pitch, -max_pitch, max_pitch))
        rv_local = quat_rotate_inverse(rr[i], rv[i])
        av_z = av[i, 2]
        if max_av_z is not None: av_z = float(np.clip(av_z, -max_av_z, max_av_z))

        dp25 = np.zeros(25, np.float32)
        other_ids = [j for j in range(25) if j not in [19, 24]]
        dp25[other_ids] = dp[i]

        obs = np.concatenate([
            [rp[i, 2]], [roll, pitch, yaw], rv_local, [av_z], dp25,
        ]).astype(np.float32)
        stream.append(obs)

    return stream


def blend(frame_a, frame_b, steps):
    """Yield `steps` cosine-eased frames from frame_a to frame_b.

    Cosine easing (ease-in-out) produces smooth acceleration and deceleration
    at both ends of the transition, avoiding abrupt velocity changes.
    """
    a, b = np.array(frame_a), np.array(frame_b)
    for i in range(steps):
        t = (i + 1) / steps
        alpha = 0.5 * (1.0 - np.cos(np.pi * t))  # ease-in-out
        yield a + (b - a) * alpha


def stream_frame(r, obs, hand_zeros):
    t0 = time.time()
    r.set("action_mimic_g1", json.dumps(obs.tolist()))
    r.set("action_hand_g1",  hand_zeros)
    elapsed = time.time() - t0
    sleep_t = CONTROL_DT - elapsed
    if sleep_t > 0:
        time.sleep(sleep_t)


# ── Motion entry dataclass ───────────────────────────────────────────────────

class MotionEntry:
    """One entry in the sequence."""
    def __init__(self, path, repeat=1, hold_last=0.0, max_duration=None,
                 max_roll=None, max_pitch=None, max_av_z=None,
                 transition=None):
        self.path         = path
        self.repeat       = int(repeat)
        self.hold_last    = float(hold_last)   # seconds to freeze on last frame
        self.max_duration = max_duration       # truncate stream to N seconds if set
        self.max_roll     = max_roll
        self.max_pitch    = max_pitch
        self.max_av_z     = max_av_z
        self.transition   = transition         # override global transition if set


def load_sequence_yaml(yaml_path):
    """Parse a YAML sequence config. Returns (entries, default_transition)."""
    if not _YAML_AVAILABLE:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    default_transition = float(cfg.get("transition", 1.0))
    entries = []
    for item in cfg["motions"]:
        e = MotionEntry(
            path         = item["path"],
            repeat       = item.get("repeat",       1),
            hold_last    = item.get("hold_last",    0.0),
            max_duration = item.get("max_duration", None),
            max_roll     = item.get("max_roll",     None),
            max_pitch    = item.get("max_pitch",    None),
            max_av_z     = item.get("max_av_z",     None),
            transition   = item.get("transition",   None),
        )
        entries.append(e)
    return entries, default_transition


def resolve_path(motion_path):
    """Return absolute pkl path, trying dataset-relative first."""
    candidate = os.path.join(DATASET, motion_path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(motion_path):
        return motion_path
    print(f"Motion not found: {motion_path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--motions",  nargs="+",
                       help="Ordered motion paths (dataset-relative or absolute)")
    group.add_argument("--sequence", metavar="YAML",
                       help="YAML sequence config file (supports per-motion options)")
    parser.add_argument("--loop",       action="store_true", help="Loop the whole sequence")
    parser.add_argument("--transition", type=float, default=None,
                        help="Blend duration between motions in seconds (default: 1.0 or from YAML)")
    args = parser.parse_args()

    # Build entry list
    if args.sequence:
        entries, default_transition = load_sequence_yaml(args.sequence)
        if args.transition is not None:
            default_transition = args.transition
    else:
        default_transition = args.transition if args.transition is not None else 1.0
        entries = [MotionEntry(p) for p in args.motions]

    transition_steps = max(1, int(default_transition / CONTROL_DT))

    # Load streams
    streams = []
    for e in entries:
        pkl_path = resolve_path(e.path)
        print(f"Loading: {e.path}")
        data   = load_motion(pkl_path)
        stream = build_mimic_stream(data, target_fps=1.0 / CONTROL_DT,
                                    max_roll=e.max_roll, max_pitch=e.max_pitch,
                                    max_av_z=e.max_av_z)
        if e.max_duration is not None:
            max_frames = max(1, int(e.max_duration / CONTROL_DT))
            stream = stream[:max_frames]
        print(f"  {len(stream)} frames ({len(stream)*CONTROL_DT:.1f}s)"
              f"  ×{e.repeat} repeat"
              + (f"  +{e.hold_last:.1f}s hold" if e.hold_last > 0 else ""))
        streams.append(stream)

    r = redis.Redis(host="localhost", port=6379)
    try:
        r.ping()
    except Exception:
        print("ERROR: Redis not running. Start run_twist_sim.py first.")
        sys.exit(1)

    hand_zeros  = json.dumps([0.0] * 14)
    default_obs = DEFAULT_MIMIC_OBS["g1"].tolist()

    # RSI: init robot to first frame of first motion
    first_obs   = streams[0][0]
    init_height = float(first_obs[0])
    init_dof25  = first_obs[8:].tolist()
    r.set("init_pose_g1", json.dumps({"height": init_height, "dof_25": init_dof25}))
    print(f"Init pose seeded: height={init_height:.3f}")

    # 3s warmup on first frame
    warmup_steps = int(3.0 / CONTROL_DT)
    print(f"Warming up ({warmup_steps} steps, 3s) ...")
    for _ in range(warmup_steps):
        stream_frame(r, streams[0][0], hand_zeros)

    print("Streaming sequence ... (Ctrl+C to stop)")
    try:
        while True:
            for idx, (entry, stream) in enumerate(zip(entries, streams)):
                t_steps = int((entry.transition or default_transition) / CONTROL_DT)
                t_steps = max(1, t_steps)

                label = entry.path
                print(f"  [{idx+1}/{len(entries)}] {label}"
                      + (f"  ×{entry.repeat}" if entry.repeat > 1 else ""))

                for rep in range(entry.repeat):
                    for obs in stream:
                        stream_frame(r, obs, hand_zeros)

                # Hold last frame
                if entry.hold_last > 0:
                    hold_steps = int(entry.hold_last / CONTROL_DT)
                    print(f"    holding last frame {entry.hold_last:.1f}s ...")
                    for _ in range(hold_steps):
                        stream_frame(r, stream[-1], hand_zeros)

                # Blend to next motion
                next_idx = idx + 1
                if next_idx < len(entries):
                    next_first = streams[next_idx][0]
                    blend_label = entries[next_idx].path
                    t_label = entry.transition or default_transition
                    print(f"    blending → {blend_label}  ({t_label:.1f}s)")
                    for blended in blend(stream[-1], next_first, t_steps):
                        stream_frame(r, blended, hand_zeros)

            if not args.loop:
                break
            print("  [loop]")
            # Blend end → start for seamless loop
            for blended in blend(streams[-1][-1], streams[0][0], transition_steps):
                stream_frame(r, blended, hand_zeros)

    except KeyboardInterrupt:
        pass

    # Smooth return to default
    last = np.array(streams[-1][-1])
    default = np.array(default_obs)
    for i in range(50):
        alpha = (i + 1) / 50.0
        interp_obs = last + (default - last) * alpha
        r.set("action_mimic_g1", json.dumps(interp_obs.tolist()))
        time.sleep(CONTROL_DT)
    r.set("action_mimic_g1", json.dumps(default_obs))
    print("Done — returned to default pose.")


if __name__ == "__main__":
    main()
