# G1 Boxing — Real Robot Deployment

Plug-and-play deployment of the [TWIST](https://github.com/YanjieZe/TWIST) whole-body imitation policy on the Unitree G1 robot, pre-loaded with a boxing motion (`kit/punching03.pkl`).

The pretrained policy (`twist_general_motion_tracker.pt`) tracks the boxing reference motion streamed from `run_motion.py` and sends 23-DOF joint position commands to the real robot at 50 Hz.

---

## Requirements

- Ubuntu 20.04 / 22.04
- Python 3.8+
- Unitree G1 robot connected via Ethernet (e.g. `eno1`)
- [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2py) installed
- Redis server on PATH (`sudo apt install redis-server` or build from source)

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
│       └── punching03.pkl                # boxing reference motion
├── robot_control/                        # G1 low-level interface (unitree_sdk2py)
│   ├── g1_wrapper.py
│   ├── config.py
│   └── configs/g1.yaml
├── data_utils/                           # TWIST obs utilities
│   ├── params.py
│   └── rot_utils.py
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

Make sure `redis-server` is on your PATH:
```bash
redis-server --version   # should print Redis version
```

### 2. Connect the robot

Connect an Ethernet cable from your PC to the G1's Ethernet port.
Set your PC's network interface to a static IP in the G1's subnet (usually `192.168.123.x`).

Find your interface name:
```bash
ip link show
```

### 3. Run the controller (Terminal 1)

```bash
python3 run_robot.py --net eno1
```

Replace `eno1` with your actual network interface name.

The terminal will print:
```
[1/3] Zero-torque mode — press START on remote to continue...
```

### 4. Run the motion streamer (Terminal 2)

Start this **within 15 seconds** of launching `run_robot.py`:
```bash
python3 run_motion.py --loop
```

The streamer loads the boxing motion, waits for Redis (started by `run_robot.py`), then streams mimic observations at 50 Hz.

### 5. Startup sequence (wireless remote)

| Step | Button | Robot state |
|------|--------|-------------|
| 1 | — | Zero-torque (limp); lift robot onto safety guard |
| 2 | **START** | Moves to default standing pose over 2 s |
| 3 | **A** | Holds default pose; policy loop begins |
| 4 | **SELECT** | Exits; robot returns to zero-torque |

> **Safety**: always have the robot hanging from a safety guard during initial testing.

---

## Robot state machine

```
run_robot.py starts
  └─ [1] zero_torque_state()        # limp, START to continue
  └─ [2] move_to_default_pos()      # smooth 2s move to standing
  └─ [3] default_pos_state()        # hold pose, wait for A
  └─ [4] policy loop @ 50 Hz        # TWIST policy tracks boxing motion
  └─ SELECT → zero_torque_state()   # safe exit
```

---

## Motion pipeline

```
punching03.pkl
  └─ run_motion.py  ──►  Redis: action_mimic_g1 (33 dims @ 50 Hz)
                                         │
                         run_robot.py  ──►  G1 robot (23 DOF @ 50 Hz)
```

**Observation space** (1155 dims):
```
action_mimic[31] + obs_proprio[74] + history×10[1050]

action_mimic: 33-dim from run_motion.py, minus 2 wrist indices → 31 dims
obs_proprio:  ang_vel×0.25(3) + rpy[:2](2) + dof_pos_diff(23) + dof_vel×0.05(23) + last_action(23)
```

---

## Troubleshooting

**`ERROR: Redis did not start in time.`**
→ Make sure `redis-server` is on your PATH.

**Robot falls immediately**
→ Press SELECT to stop. Increase the safety guard height. The robot needs ~3 s of warm-up before tracking the motion.

**`unitree_sdk2py` import error**
→ Install from [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2py).

**Policy runs but arms don't move**
→ Normal for the first 3 s (warm-up). Motion tracking begins after warm-up completes.

---

## Acknowledgements

- [TWIST](https://github.com/YanjieZe/TWIST) — CoRL 2025 whole-body imitation policy by Yanjie Ze et al.
- [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2py) — Unitree G1 SDK
