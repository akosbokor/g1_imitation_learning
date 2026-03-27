# G1 Boxing — Real Robot Deployment

Plug-and-play deployment of the [TWIST](https://github.com/YanjieZe/TWIST) whole-body
imitation policy on the Unitree G1 robot, pre-loaded with a boxing motion (`punching03.pkl`).

The pretrained policy (`twist_general_motion_tracker.pt`) tracks the boxing reference motion
streamed from `run_motion.py` and sends 23-DOF joint position commands to the real robot at
50 Hz.

---

## Requirements

- Ubuntu 20.04 / 22.04
- Python 3.8+
- Unitree G1 robot connected via Ethernet (e.g. `eth0`)
- [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2py) installed
- **No Redis server required** — shared memory via `/tmp/g1_shm/` (see `shm_store.py`)

```bash
pip install -r requirements.txt
```

---

## Directory layout

```
g1-boxing/
├── assets/
│   ├── twist_general_motion_tracker.pt   # pretrained TWIST policy (JIT)
│   └── motions/
│       └── punching03.pkl                # boxing reference motion (33 fps, 385 frames)
├── robot_control/                        # G1 low-level interface (unitree_sdk2py)
│   ├── g1_wrapper.py
│   ├── config.py
│   └── configs/g1.yaml
├── data_utils/                           # TWIST obs utilities
│   ├── params.py
│   └── rot_utils.py
├── shm_store.py                          # file-based shared memory (replaces Redis)
├── run_robot.py                          # real robot controller
├── run_motion.py                         # boxing motion streamer
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Connect the robot

Connect an Ethernet cable from the robot computer to the G1's Ethernet port.
Find your interface name with `ip link show` (typically `eth0` on the robot computer).

### 3. Run the robot controller (Terminal 1) — **start this first**

```bash
python3 run_robot.py --net eth0
```

Wait until you see:
```
[1/3] Zero-torque mode — press START on remote to continue...
```

### 4. Run the motion streamer (Terminal 2)

```bash
python3 run_motion.py --loop
```

It will print:
```
Waiting for run_robot.py to start...
run_robot.py detected — waiting for START then A on remote...
```

### 5. Startup sequence (wireless remote)

| Step | Button | What happens |
|------|--------|--------------|
| 1 | — | Zero-torque (limp); lift robot onto safety guard |
| 2 | **START** | Moves to boxing guard pose over 2 s |
| 3 | **A** | RSI history pre-fill, then policy loop begins; motion warm-up starts |
| 4 | — | 3 s warm-up (holds frame 0), then full boxing motion |
| 5 | **SELECT** | Exits; robot returns to zero-torque |

> **Safety**: always have the robot on a safety guard during testing.

---

## How it works

### Architecture

```
punching03.pkl
  └─ run_motion.py  ──►  /tmp/g1_shm/action_mimic_g1  (33 dims @ 50 Hz)
                                         │
                         run_robot.py  ──►  G1 robot (23 DOF @ 50 Hz)
```

### Observation space (1155 dims)

```
mimic_31  (31): action_mimic target — 33-dim mimic obs, wrist dims [27,32] removed
proprio   (74): ang_vel×0.25(3) + rpy[:2](2) + dof_pos-default(23) + dof_vel×0.05(23) + last_action(23)
history  (1050): 10 × 105 stacked past obs_full frames
```

### Startup handshake

```
run_robot.py                         run_motion.py
     │                                     │
     ├─ sets policy_running = "0"          │
     ├─ seeds action_mimic_g1             │
     │                                     ├─ writes init_pose_g1 (frame-0 joints)
     │                                     ├─ waits for policy_running == "0"  ←──┘
     │                                     ├─ waits for policy_running == "1"
     ├─ START → move to boxing guard       │
     ├─ A → reads init_pose_g1             │
     ├─ RSI: pre-fills history            │
     ├─ sets policy_running = "1"  ───────►│
     │                                     ├─ 3 s warm-up (streams frame 0)
     └─ policy loop @ 50 Hz               └─ streams full boxing motion
```

### Reference State Initialization (RSI)

The pretrained TWIST policy was trained with RSI: at episode start the simulation
resets to a random frame of the motion, so the policy always begins with a coherent
history of 10 past observations.

On the real robot we replicate this:

1. **Hardware**: `arm_waist_target` in `g1.yaml` is set to boxing frame 0 values, so
   `move_to_default_pos()` places the robot in the boxing guard pose before the policy starts.

2. **Observation default**: the policy's `dof_pos - default` term uses the **training
   standing-pose default** (`TRAINING_ARM_WAIST` in `run_robot.py`), not the boxing guard,
   so the deviation seen by the policy matches what it saw in training.

3. **History pre-fill**: after pressing A, `run_robot.py` reads the actual robot state
   (at boxing guard) and pre-fills all 10 history slots with the corresponding observation,
   using the steady-state action for the boxing guard as `last_action`. This gives the policy
   a coherent starting context instead of 10 frames of zeros.

### Shared memory (no Redis)

`shm_store.py` replaces Redis with atomic file writes to `/tmp/g1_shm/`.
`os.replace()` ensures readers always see a complete value, never a partially written file.

Keys used:

| Key | Writer | Reader | Content |
|-----|--------|--------|---------|
| `policy_running` | run_robot.py | run_motion.py | `"0"` = robot alive, `"1"` = policy loop started |
| `init_pose_g1` | run_motion.py | run_robot.py | JSON `{height, dof_25}` of motion frame 0 |
| `action_mimic_g1` | run_motion.py | run_robot.py | 33-dim mimic obs array |
| `state_body_g1` | run_robot.py | (monitoring) | 74-dim proprio obs array |

---

## Robot state machine

```
run_robot.py starts
  └─ [1] zero_torque_state()        # limp, START to continue
  └─ [2] move_to_default_pos()      # smooth 2 s move to boxing guard
  └─ [3] default_pos_state()        # hold guard, wait for A
  └─ [RSI] pre-fill history         # coherent 10-step context from frame 0
  └─ [4] policy loop @ 50 Hz        # TWIST policy tracks boxing motion
  └─ SELECT → zero_torque_state()   # safe exit
```

---

## Troubleshooting

**Arms fall immediately after pressing A**
→ Stale shm files from a previous run. Clear them:
```bash
rm -rf /tmp/g1_shm/
```
Then restart both scripts.

**run_motion.py stuck on "Waiting for run_robot.py to start..."**
→ run_robot.py must be started first. If it was already started, check that
`/tmp/g1_shm/policy_running` exists and contains `0`.

**Robot falls immediately**
→ Press SELECT to stop. Increase the safety guard height. The robot needs
the full 3 s warm-up before tracking the motion.

**`unitree_sdk2py` import error**
→ Install from [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2py).

---

## Notes on the pretrained policy

The policy `twist_general_motion_tracker.pt` is the generic TWIST tracker from the
original paper. It works for `punching03.pkl` after RSI is applied correctly.

For improved arm tracking, the `docs/TWIST_BOXING_NOTES.md` describes a PPO fine-tuning
pipeline (`src/finetune_twist.py`) that can be used to fine-tune the policy specifically
on the boxing motion in MuJoCo before deploying to the real robot.

---

## Acknowledgements

- [TWIST](https://github.com/YanjieZe/TWIST) — CoRL 2025 whole-body imitation policy by Yanjie Ze et al.
- [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2py) — Unitree G1 SDK
