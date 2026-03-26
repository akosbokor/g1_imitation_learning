"""
G1 Boxing — Real Robot Controller
===================================
Runs the TWIST low-level policy on the real Unitree G1 robot.
Reads mimic reference from Redis (written by run_motion.py) and
sends joint position commands to the robot at 50 Hz.

Startup sequence (use the wireless remote):
  1. Run this script  →  robot enters zero-torque mode
  2. Press START      →  robot moves to default standing pose (2 s)
  3. Press A          →  robot holds default pose, policy loop begins
  4. Press SELECT     →  exit, robot returns to zero-torque

Usage:
    python3 run_robot.py --net eno1          # change to your network interface
    python3 run_robot.py --net eno1 --cpu    # force CPU inference
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np
import redis
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler
from robot_control.config import Config
from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.common.remote_controller import KeyMap

# ── Paths ─────────────────────────────────────────────────────────────────────
POLICY_PATH = os.path.join(HERE, "assets", "twist_general_motion_tracker.pt")
CONFIG_PATH = os.path.join(HERE, "robot_control", "configs", "g1.yaml")
REDIS_BIN   = "redis-server"   # assumes redis-server is on PATH

# ── Constants (must match training) ───────────────────────────────────────────
NUM_ACTIONS  = 23
ACTION_SCALE = 0.5
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
ANKLE_IDX     = [4, 5, 10, 11]
HISTORY_LEN   = 10

MIMIC_WRIST_IDS = [27, 32]
MIMIC_BODY_IDS  = [i for i in range(33) if i not in MIMIC_WRIST_IDS]  # 31 dims

N_MIMIC  = 31
N_PROPRIO = 74   # ang_vel(3)+rpy2(2)+dof_pos(23)+dof_vel(23)+last_action(23)
N_FULL   = N_MIMIC + N_PROPRIO   # 105
OBS_DIM  = N_FULL * (HISTORY_LEN + 1)  # 1155


def start_redis():
    proc = subprocess.Popen(
        [REDIS_BIN, "--port", "6379", "--daemonize", "no", "--loglevel", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Wait until Redis accepts connections
    r = redis.Redis()
    for _ in range(20):
        try:
            r.ping()
            break
        except Exception:
            time.sleep(0.3)
    else:
        print("ERROR: Redis did not start in time.")
        sys.exit(1)
    return proc


def seed_redis(r: redis.Redis):
    default_mimic = DEFAULT_MIMIC_OBS["g1"]
    r.set("action_mimic_g1", json.dumps(default_mimic.tolist()))
    r.set("state_body_g1",   json.dumps(np.zeros(105).tolist()))
    print(f"Redis seeded with default pose.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net",    default="eno1",
                        help="Network interface connected to G1 (e.g. eno1, eth0)")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU inference")
    parser.add_argument("--policy", default=POLICY_PATH, help="Path to policy .pt")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Policy : {args.policy}")
    print(f"Device : {device}")
    print(f"Network: {args.net}")

    # Start Redis
    redis_proc = start_redis()
    r = redis.Redis()
    seed_redis(r)

    # Load policy
    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()
    print("Policy loaded.")

    # Init robot
    config = Config(CONFIG_PATH)
    env    = G1RealWorldEnv(net=args.net, config=config)

    default_dof_pos = np.concatenate([config.default_angles, config.arm_waist_target])

    # History buffer
    history = deque(maxlen=HISTORY_LEN)
    for _ in range(HISTORY_LEN):
        history.append(np.zeros(N_FULL, dtype=np.float32))
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # ── Startup sequence ──────────────────────────────────────────────────────
    print("\n[1/3] Zero-torque mode — press START on remote to continue...")
    env.zero_torque_state()

    print("[2/3] Moving to default pose (2 s)...")
    env.move_to_default_pos()

    print("[3/3] Holding default pose — press A to start policy, SELECT to exit.")
    env.default_pos_state()

    print("\n✓ Policy loop running.  Press SELECT to stop.\n")

    try:
        while True:
            t_start = time.time()

            if env.remote_controller.button[KeyMap.select] == 1:
                print("SELECT pressed — exiting.")
                break

            # Robot state
            dof_pos, dof_vel, quat, ang_vel = env.get_robot_state()
            rpy = quatToEuler(quat)

            obs_dof_vel = dof_vel.copy()
            obs_dof_vel[ANKLE_IDX] = 0.0

            obs_proprio = np.concatenate([
                ang_vel * ANG_VEL_SCALE,
                rpy[:2],
                dof_pos - default_dof_pos,
                obs_dof_vel * DOF_VEL_SCALE,
                last_action,
            ]).astype(np.float32)   # 74 dims

            # Mimic reference from Redis
            raw = r.get("action_mimic_g1")
            if raw is None:
                print("WARNING: no mimic obs in Redis, using default")
                mimic_full = DEFAULT_MIMIC_OBS["g1"]
            else:
                mimic_full = np.array(json.loads(raw), dtype=np.float32)

            mimic_31     = mimic_full[MIMIC_BODY_IDS]          # 31 dims
            wrist_target = mimic_full[[27, 32]]                 # left/right wrist

            obs_full = np.concatenate([mimic_31, obs_proprio])  # 105 dims
            obs_hist = np.array(history, dtype=np.float32).flatten()  # 1050 dims
            obs_buf  = np.concatenate([obs_full, obs_hist])            # 1155 dims
            history.append(obs_full.copy())

            obs_t = torch.from_numpy(obs_buf).float().unsqueeze(0).to(device)
            with torch.no_grad():
                raw_action = policy(obs_t).cpu().numpy().squeeze()

            last_action = raw_action.copy()
            raw_action  = np.clip(raw_action, -10.0, 10.0)
            target_dof_pos = default_dof_pos + raw_action * ACTION_SCALE

            env.send_robot_action(
                target_dof_pos,
                left_wrist_roll=float(wrist_target[0]),
                right_wrist_roll=float(wrist_target[1]),
            )

            # Send proprio state to Redis (for monitoring)
            r.set("state_body_g1", json.dumps(obs_proprio.tolist()))

            elapsed = time.time() - t_start
            if elapsed < config.control_dt:
                time.sleep(config.control_dt - elapsed)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    finally:
        print("Entering zero-torque mode...")
        env.zero_torque_state()
        redis_proc.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
