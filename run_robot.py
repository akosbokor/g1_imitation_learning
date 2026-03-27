"""
G1 Boxing — Real Robot Controller
===================================
Runs the TWIST low-level policy on the real Unitree G1 robot.
Reads mimic reference from shared memory (written by run_motion.py) and
sends joint position commands to the robot at 50 Hz.

Startup sequence (use the wireless remote):
  1. Run run_robot.py  →  robot enters zero-torque mode
  2. Run run_motion.py →  loads motion, waits for robot
  3. Press START       →  robot moves to boxing guard pose (2 s)
  4. Press A           →  RSI history pre-fill, policy loop begins
  5. Press SELECT      →  exit, robot returns to zero-torque

Usage:
    python3 run_robot.py --net eth0          # change to your network interface
    python3 run_robot.py --net eth0 --cpu    # force CPU inference
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from shm_store import SharedMemStore
from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import quatToEuler
from robot_control.config import Config
from robot_control.g1_wrapper import G1RealWorldEnv
from robot_control.common.remote_controller import KeyMap

# ── Paths ─────────────────────────────────────────────────────────────────────
POLICY_PATH = os.path.join(HERE, "assets", "twist_general_motion_tracker.pt")
CONFIG_PATH = os.path.join(HERE, "robot_control", "configs", "g1.yaml")

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


def seed_shm(r: SharedMemStore):
    default_mimic = DEFAULT_MIMIC_OBS["g1"]
    r.set("action_mimic_g1", json.dumps(default_mimic.tolist()))
    r.set("state_body_g1",   json.dumps(np.zeros(105).tolist()))
    print("Shared memory seeded with default pose.")


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

    r = SharedMemStore()
    r.set("policy_running", "0")   # signal run_motion.py that we are alive
    seed_shm(r)

    # Load policy
    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()
    print("Policy loaded.")

    # Init robot
    config = Config(CONFIG_PATH)
    env    = G1RealWorldEnv(net=args.net, config=config)

    # Hardware target: robot physically moves here (boxing guard = motion frame 0)
    hardware_dof_pos = np.concatenate([config.default_angles, config.arm_waist_target])

    # Policy observation default: must match TWIST training (standing pose).
    # The policy was trained with (dof_pos - standing_pose) as the arm deviation term.
    # Using boxing-guard here would give zero deviation and confuse the policy.
    TRAINING_ARM_WAIST = np.array([0.0, 0.0, 0.0,       # waist
                                    0.0, 0.4, 0.0, 1.2,  # left arm
                                    0.0, -0.4, 0.0, 1.2], dtype=np.float32)  # right arm
    default_dof_pos = np.concatenate([config.default_angles, TRAINING_ARM_WAIST])

    # Steady-state action needed to hold the boxing guard from the training default
    steady_state_action = np.clip(
        (hardware_dof_pos - default_dof_pos) / ACTION_SCALE, -10.0, 10.0
    ).astype(np.float32)

    # History buffer (pre-filled later via RSI)
    history = deque(maxlen=HISTORY_LEN)
    for _ in range(HISTORY_LEN):
        history.append(np.zeros(N_FULL, dtype=np.float32))
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # ── Startup sequence ──────────────────────────────────────────────────────
    print("\n[1/3] Zero-torque mode — press START on remote to continue...")
    env.zero_torque_state()

    print("[2/3] Moving to boxing guard pose (2 s)...")
    env.move_to_default_pos()

    print("[3/3] Holding guard pose — press A to start policy, SELECT to exit.")
    env.default_pos_state()

    # Wait for run_motion.py to provide init_pose_g1 (written on startup, before warm-up)
    print("Waiting for run_motion.py init pose...")
    t_wait = time.time()
    while r.get("init_pose_g1") is None:
        if time.time() - t_wait > 30:
            print("WARNING: init_pose_g1 not received after 30 s, using zero history.")
            break
        time.sleep(0.05)

    # RSI: pre-fill history so the policy starts with a coherent 10-step context.
    # Uses the actual robot state at boxing guard + the steady-state action,
    # matching what the policy would have seen in training at motion frame 0.
    if r.get("init_pose_g1") is not None:
        last_action = steady_state_action.copy()
        dof_pos_init, dof_vel_init, quat_init, ang_vel_init = env.get_robot_state()
        rpy_init = quatToEuler(quat_init)
        obs_dof_vel_init = dof_vel_init.copy()
        obs_dof_vel_init[ANKLE_IDX] = 0.0
        obs_proprio_init = np.concatenate([
            ang_vel_init * ANG_VEL_SCALE,
            rpy_init[:2],
            dof_pos_init - default_dof_pos,   # deviation from training default (non-zero)
            obs_dof_vel_init * DOF_VEL_SCALE,
            last_action,
        ]).astype(np.float32)
        mimic_init = DEFAULT_MIMIC_OBS["g1"][MIMIC_BODY_IDS]
        obs_full_init = np.concatenate([mimic_init, obs_proprio_init])
        for _ in range(HISTORY_LEN):
            history.append(obs_full_init.copy())
        print("History pre-filled with frame-0 observation (RSI).")

    print("\n✓ Policy loop running.  Press SELECT to stop.\n")
    r.set("policy_running", "1")   # signal run_motion.py to start warm-up

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

            # Mimic reference from shared memory
            raw = r.get("action_mimic_g1")
            if raw is None:
                print("WARNING: no mimic obs in shared memory, using default")
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

            # Send proprio state to shared memory (for monitoring)
            r.set("state_body_g1", json.dumps(obs_proprio.tolist()))

            elapsed = time.time() - t_start
            if elapsed < config.control_dt:
                time.sleep(config.control_dt - elapsed)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
    finally:
        print("Entering zero-torque mode...")
        env.zero_torque_state()
        print("Done.")


if __name__ == "__main__":
    main()
