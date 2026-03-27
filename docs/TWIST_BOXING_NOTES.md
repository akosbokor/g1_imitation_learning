# TWIST Boxing / Punching — Simulation & Real-Robot Notes

## Overview

This document covers running boxing and punching motions on the G1 robot using the TWIST
whole-body imitation policy, including:

- Which motions work in simulation
- The PPO fine-tuning pipeline we built
- All bugs that were fixed to make it work in MuJoCo
- Why it failed on the real robot and what needs to be done

---

## Available boxing / punching motions

The TWIST dataset contains motions in several categories that map to boxing:

| Category | File | Notes |
|----------|------|-------|
| `kit/` | `punching03.pkl` | **Recommended** — upright stance, alternating jabs, 576 frames (11.5s at 50 Hz). Works well in sim. |
| `kit/` | `punch_left01..05.pkl` | Single-arm left jab series |
| `kit/` | `punch_right01..05.pkl` | Single-arm right jab series |
| `dfaust/` | `50002_punching.pkl` .. `50027_punching.pkl` | 10 captures of people punching, various body shapes |
| `ssm_synced/` | `punching.pkl` | Synchronized two-person punching capture |
| `ssm_synced/` | `punch_kick_sync.pkl` | Punch + kick combo |
| `transitions/` | `punchboxing_stand.pkl` | Boxing stance hold |
| `transitions/` | `punchboxing_walk.pkl` | Boxing walk |
| `transitions/` | `punchboxing_running.pkl` | Running boxing combo |
| `transitions/` | `punchboxing_kick.pkl` | Boxing + kick |
| `transitions/` | `punchboxing_push.pkl` | Boxing + push |
| `transitions/` | `punchboxing_jumpinplace.pkl` | Boxing + jump in place |

---

## Running boxing in simulation

### Standard (pretrained policy)

```bash
cd /home/sztaki/g1_project

# Terminal 1 — start sim (starts Redis, waits up to 15s for motion init)
python3 run_twist_sim.py

# Terminal 2 — stream boxing motion (start within 15s)
python3 run_twist_motion.py --motion kit/punching03.pkl --loop
```

`punching03.pkl` has `height=0.772 m` (slightly crouched stance) and
`max_joint=0.862 rad` (≈49° elbow bend for guard). Both are within the policy's
training distribution.

### Fine-tuned policy (improved arm tracking)

After PPO fine-tuning on `punching03` (see below), use the saved actor instead of the
default pretrained model by modifying `POLICY` in `run_twist_sim.py`:

```python
POLICY = "output/finetune/punching03/best_actor.pt"
```

---

## PPO fine-tuning pipeline

We built a full PPO fine-tuning pipeline that runs entirely in MuJoCo — no Isaac Gym required.

### Files

| File | Purpose |
|------|---------|
| `src/finetune_twist.py` | PPO training loop — loads pretrained JIT actor, adds learnable log_std + new critic, runs PPO |
| `src/twist_mujoco_env.py` | Gymnasium environment for fine-tuning — matches obs/action space of `server_low_level_g1_sim.py` exactly |

### Architecture

```
pretrained twist_general_motion_tracker.pt
        │
        └── TwistActor (wrapper)
              ├── JIT model weights (updated at lr=3e-6)
              └── log_std (new param, trained at lr=3e-6)

TwistCritic (new, 512→256→128→1, trained at lr=1e-4)

TwistG1Env
├── MuJoCo physics (1 kHz, 20-step decimation → 50 Hz policy)
├── Reference State Init: reset to random frame of motion
├── Reward: 0.6*joint_dof + 0.2*joint_vel + 0.6*root_pose + 1.0*root_vel
└── Termination: pelvis_z < 0.35 m OR |roll/pitch| > 1.2 rad
```

### Run fine-tuning

```bash
python3 src/finetune_twist.py --motion kit/punching03.pkl --iters 2000

# With GUI (slower, useful to watch):
python3 src/finetune_twist.py --motion kit/punching03.pkl --iters 2000 --gui

# Other boxing motions:
python3 src/finetune_twist.py --motion kit/punch_right01.pkl --iters 2000
python3 src/finetune_twist.py --motion transitions/punchboxing_stand.pkl --iters 1000
```

Outputs saved to `output/finetune/<motion_name>/`:
- `best_actor.pt` — best JIT model by mean episode reward (drop-in replacement for pretrained)
- `checkpoint_NNNN.pt` — periodic full checkpoint (actor + critic state dicts)
- `reward_curve_NNNN.png` — training curves
- Training metrics logged to W&B (`humanoid-g1` project)

### Observation space (1155 dims)

Matches `server_low_level_g1_sim.py` exactly:

```
mimic_31  (31): action_mimic target — 33-dim mimic obs minus 2 wrist dims
proprio   (74): ang_vel×0.25(3) + rpy[:2](2) + dof_pos-default(23) + dof_vel×0.05(23) + last_action(23)
history  (1050): 10 × 105 stacked past obs_full
─────────────────
total    (1155)
```

### Reward function

Matches TWIST training (`humanoid_mimic.py`):

| Term | Weight | Formula |
|------|--------|---------|
| Joint DOF tracking | 0.6 | `exp(-0.15 * Σ w_i * (q_i - q_ref_i)²)` |
| Joint velocity tracking | 0.2 | `exp(-0.01 * Σ w_i * (dq_i - dq_ref_i)²)` |
| Root pose | 0.6 | `exp(-5.0 * (z_err² + 0.1 * rot_err²))` |
| Root velocity | 1.0 | `exp(-1.0 * (lin_vel_err² + 0.5 * ang_vel_err²))` |

---

## Bugs fixed to make it work in MuJoCo simulation

These bugs were fixed in `run_twist_motion.py` and `src/twist_mujoco_env.py`.

### Bug 1 — Quaternion convention (XYZW vs WXYZ)

**Symptom:** Robot fell immediately at step 0. Yaw computed as −178° instead of ~0°.

**Root cause:** The motion `.pkl` files store quaternions as XYZW (SciPy/AMASS convention).
All internal functions (`quat_to_rpy`, `quat_mul`, `quat_rotate_inverse`) expect WXYZ
(MuJoCo convention). Raw load gave a near-inverted orientation to the policy.

**Fix:** After loading and resampling, convert before any processing:
```python
rr = rr[:, [3, 0, 1, 2]]   # xyzw → wxyz
```

---

### Bug 2 — Angular velocity formula (wrong conjugate)

**Symptom:** `av_z ≈ 99 rad/s` — yaw rate orders of magnitude too large, robot spun and fell.

**Root cause:** Angular velocity from two successive quaternions requires `conj(q1) * q2`.
The initial code computed `conj(q2) * q1` or just `conj(q2)`, giving a large spurious rotation.

**Fix:**
```python
q1_conj = np.array([rr[i, 0], -rr[i, 1], -rr[i, 2], -rr[i, 3]])
dq = quat_mul(q1_conj, rr[i + 1])
av[i] = 2.0 * dq[1:] / dt
```

---

### Bug 3 — Yaw not wrapped / accumulated unboundedly

**Symptom:** After a few seconds of walking, yaw reached 3.8, then 6.2 rad — the policy
received an out-of-distribution yaw reference and the robot gradually veered and fell.

**Root cause:** Yaw was computed as absolute world yaw with no normalisation. Motions that
involve turning accumulate yaw indefinitely.

**Fix:** Normalise to 0 at the clip's first frame and wrap to `[−π, π]`:
```python
yaw = (yaw - yaw_0 + np.pi) % (2 * np.pi) - np.pi
```

---

### Bug 4 — Inverse rotation sign (Rodrigues formula)

**Symptom:** Root linear velocity in body frame had wrong direction — robot was told to
move backward when the reference said forward, causing erratic behaviour.

**Root cause:** The Rodrigues formula for rotating a vector v by quaternion q is:
`v' = v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))`
For the *inverse* rotation the w term changes sign: `v - w*t + ...` (not `v + w*t`).

**Fix:**
```python
t = 2.0 * np.cross([x, y, z], v)
return v - w * t + np.cross([x, y, z], t)   # minus, not plus
```

---

### Bug 5 — Reference State Initialization (RSI) missing

**Symptom:** Robot always started at the default T-pose/standing pose, even for motions
that begin mid-punch or in a boxing guard. The pose gap could be up to 67° in arm joints
at frame 0. The policy received an impossible correction signal and the robot fell at the
start of every motion.

**Root cause:** The pretrained policy was trained with RSI — the simulation always resets
to a frame sampled randomly from the motion, so the robot is *already* in the right pose
when the episode starts. Without RSI, the policy has never seen this situation.

**Fix:** `run_twist_motion.py` writes the first-frame joint positions and root height to
Redis on startup:
```python
r.set("init_pose_g1", json.dumps({"height": init_height, "dof_25": init_dof25}))
```
`run_twist_sim.py` reads this key (waiting up to 15 s) and physically sets the MuJoCo
`qpos` before starting the simulation loop:
```python
controller.mujoco_default_dof_pos[2]  = init["height"]
controller.mujoco_default_dof_pos[7:] = np.array(init["dof_25"])
```

---

### Bug 6 — Warm-up mismatch

**Symptom:** Even after RSI was added, the robot fell in the first 3 seconds. During warm-up
the mimic target was interpolated from default→first_frame, but the robot was *already at*
first_frame — the policy tried to correct what it perceived as drift, pushing the robot into
an incorrect pose.

**Root cause:** The warm-up was designed for the case where the robot starts at default
pose. With RSI the robot is already at first_frame, so blending from default is wrong.

**Fix:** Hold `first_frame` as the constant mimic target throughout the warm-up:
```python
for i in range(warmup_steps):   # 150 steps = 3s
    r.set("action_mimic_g1", json.dumps(first_frame.tolist()))
```

---

### Bug 7 — Extreme lean motions (e.g. back kick)

**Symptom:** Motions with large body lean (back kick: roll ±85°, pitch ±50°, yaw-rate ±4 rad/s)
caused the robot to fall. The policy was trained with these values clipped; at inference time
the raw motion values exceeded the clipping range used in training.

**Fix:** Optional clamp flags in `run_twist_motion.py`:
```bash
python3 run_twist_motion.py --motion eyes_japan/karate_06_back_kick_yokoyama.pkl \
    --loop --max_roll 0.2 --max_pitch 0.35 --max_av_z 2.0
```

Boxing motions (`punching03.pkl`) do **not** need clamps — roll/pitch stays within ±0.15 rad.

---

## Why it failed on the real robot

### Missing: RSI for real robot

The RSI mechanism (Bug 5) only works in simulation. `server_low_level_g1_real.py` does not
read `init_pose_g1` from Redis. On the real robot:

1. The robot is in its default standing pose (arms at sides)
2. `run_twist_motion.py` immediately streams the boxing guard frame (elbow bent ≈49°, height 0.772 m)
3. The policy receives a mimic target 49° away from the robot's actual state
4. It applies a large corrective torque command → jolt → fall

### Missing: Fine-tuned policy on real robot

The real-robot controller (`server_low_level_g1_real.py`) loads the pretrained
`twist_general_motion_tracker.pt`. The fine-tuned `best_actor.pt` from PPO was never
deployed there.

### What needs to be done for real-robot boxing

**Option A — Gradual transition (no code change to real controller):**
1. Start the motion streamer with the *default standing pose* first (or modify
   `run_twist_motion.py` to add a pre-roll transition from standing to boxing guard
   before the actual punching frames start)
2. Let the robot gradually adopt the stance over 5–10 s
3. Then send the punching frames

**Option B — Add RSI to real robot controller:**
1. In `server_low_level_g1_real.py`, before the control loop starts, read `init_pose_g1`
   from Redis
2. Use the G1 `dds` interface to command a slow joint trajectory (over 2–3 s) from
   current pose to `init_pose_g1`
3. Wait for convergence, then start the TWIST policy loop
4. This exactly mirrors the sim behaviour

**Option C — Use fine-tuned policy:**
1. In `server_low_level_g1_real.py`, change the policy path:
   ```python
   policy_path = "/home/sztaki/g1_project/output/finetune/punching03/best_actor.pt"
   ```
2. The fine-tuned policy is more robust to the boxing guard start pose since it was
   trained with RSI on that specific motion

---

## Summary

| Step | Sim | Real robot |
|------|-----|-----------|
| Pretrained policy + walking motion | Works | Works |
| Pretrained policy + boxing motion | Works (after all 7 bug fixes) | **Falls** — RSI missing |
| Fine-tuned policy + boxing motion | Works, better arm tracking | Not tested |
| Real-robot RSI transition | N/A | Not implemented |
