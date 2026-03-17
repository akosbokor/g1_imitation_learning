# G1 Locomotion with mujoco_mpc — Design Document

**Date:** 2026-03-17
**Status:** Approved
**Goal:** Adapt Google DeepMind's mujoco_mpc to drive bipedal walking/running on the Unitree G1 (29 DOF), using an incremental phased approach.

---

## 1. Architecture Overview

```
g1_project/
├── mujoco_mpc/                          ← git submodule
│   └── mjpc/tasks/g1/walk/
│       ├── walk.h                       ← C++ task header
│       ├── walk.cc                      ← residual function
│       └── task.xml                     ← planner config + sensors
├── src/
│   ├── g1_mpc_model.xml                 ← G1 model with MPC sensors added
│   └── run_g1_mpc.py                    ← Python gRPC driver
└── docs/plans/
    └── 2026-03-17-g1-mpc-walk-design.md
```

**Data flow:** Python driver → gRPC → mujoco_mpc C++ server → G1Walk task → residual → iLQG/sampling planner → actions → MuJoCo sim

---

## 2. Robot Model

**Base model:** `unitree_mujoco/unitree_robots/g1/g1_29dof.xml`

**G1 joint layout (29 DOF):**

| Index | Joint | Group |
|-------|-------|-------|
| 0–5   | left_hip_yaw/roll/pitch, left_knee, left_ankle_pitch/roll | Left leg |
| 6–11  | right_hip_yaw/roll/pitch, right_knee, right_ankle_pitch/roll | Right leg |
| 12–14 | waist_yaw/roll/pitch | Waist |
| 15–21 | left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw | Left arm |
| 22–28 | right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw | Right arm |

**Body names used by residual:**
- `pelvis` — root body
- `torso_link` — upper torso
- `left_ankle_roll_link` — left foot proxy
- `right_ankle_roll_link` — right foot proxy

**MPC model file (`src/g1_mpc_model.xml`):** includes `g1_29dof.xml` and appends required sensors (see Section 4).

---

## 3. Phased Approach

### Phase 1 — Legs + Waist (15 active DOF)

- MPC controls joints 0–14 (12 legs + 3 waist)
- Arm joints 15–28 get `damping="5"` override → hang naturally, no MPC control
- `model->nu = 15`
- Goal: stable bipedal walking at 0.5 m/s

### Phase 2 — Full Body (29 active DOF)

- Remove arm damping override
- Extend `ctrl` to 29 signals
- Add arm-swing residual term (penalise arm-COM velocity mismatch)
- `model->nu = 29`
- Goal: natural arm-swinging gait, higher speeds

---

## 4. Sensor Setup (g1_mpc_model.xml)

Sensors required by the residual function:

```xml
<!-- Position / COM -->
<framepos  name="torso_position"      objtype="body" objname="torso_link"/>
<subtreecom name="torso_subcom"       body="torso_link"/>
<subtreelinvel name="torso_subcomvel" body="torso_link"/>
<framepos  name="foot_right"          objtype="body" objname="right_ankle_roll_link"/>
<framepos  name="foot_left"           objtype="body" objname="left_ankle_roll_link"/>
<framepos  name="pelvis_position"     objtype="body" objname="pelvis"/>

<!-- Upright axes -->
<framezaxis name="torso_up"           objtype="xbody" objname="torso_link"/>
<framezaxis name="pelvis_up"          objtype="xbody" objname="pelvis"/>
<framezaxis name="foot_right_up"      objtype="xbody" objname="right_ankle_roll_link"/>
<framezaxis name="foot_left_up"       objtype="xbody" objname="left_ankle_roll_link"/>

<!-- Forward axes -->
<framexaxis name="torso_forward"      objtype="xbody" objname="torso_link"/>
<framexaxis name="pelvis_forward"     objtype="xbody" objname="pelvis"/>
<framexaxis name="foot_right_forward" objtype="xbody" objname="right_ankle_roll_link"/>
<framexaxis name="foot_left_forward"  objtype="xbody" objname="left_ankle_roll_link"/>

<!-- Velocities -->
<subtreelinvel name="pelvis_subcomvel" body="pelvis"/>
<framelinvel name="torso_velocity"    objtype="body" objname="torso_link"/>
<framelinvel name="foot_right_velocity" objtype="body" objname="right_ankle_roll_link"/>
<framelinvel name="foot_left_velocity"  objtype="body" objname="left_ankle_roll_link"/>
```

User sensors (cost terms, declared in task.xml) must appear **before** the above.

---

## 5. Residual Function (walk.cc)

8 residual terms, mirroring `humanoid/walk/walk.cc`:

| # | Name | Dim | G1 notes |
|---|------|-----|----------|
| 0 | Torso height | 1 | goal = **0.75 m** (G1 torso height at stand) |
| 1 | Pelvis-feet alignment | 1 | feet z avg − pelvis z − 0.2 |
| 2 | Balance (capture point) | 2 | identical capture-point logic |
| 3 | Upright | 8 | torso_up, pelvis_up, foot_*_up (×2) |
| 4 | Posture | **29** | `qpos[7..35]` toward zero (vs 21 for humanoid) |
| 5 | Walk forward | 1 | COM vel · forward − speed_param |
| 6 | Move feet | 2 | foot vel vs COM vel |
| 7 | Control effort | **15** (P1) / **29** (P2) | `ctrl[0..nu-1]` |

**Total residual dim Phase 1:** 1+1+2+8+29+1+2+15 = **59**
**Total residual dim Phase 2:** 1+1+2+8+29+1+2+29 = **73**

**Key parameters:**
- `residual_Torso`: `[0.75, 0.5, 1.1]` (goal, min, max)
- `residual_Speed`: `[0.5, -2.0, 2.0]`

---

## 6. Task Configuration (task.xml)

```xml
<custom>
  <numeric name="agent_planner"        data="2" />      <!-- sampling -->
  <numeric name="agent_horizon"        data="0.35" />
  <numeric name="agent_timestep"       data="0.015" />
  <numeric name="sampling_spline_points" data="3" />
  <numeric name="sampling_exploration"   data="0.05" />
  <numeric name="gradient_spline_points" data="5" />
  <numeric name="residual_Torso"       data="0.75 0.5 1.1" />
  <numeric name="residual_Speed"       data="0.5 -2.0 2.0" />
</custom>
```

User sensors mirror humanoid walk with G1-adjusted dims.

---

## 7. Build System Changes

**`mjpc/tasks/CMakeLists.txt`** — add:
```cmake
# G1 Walk task
configure_file(tasks/g1/walk/task.xml   ... COPYONLY)
configure_file(src/g1_mpc_model.xml     ... COPYONLY)

cc_library(g1_walk  SRCS tasks/g1/walk/walk.cc  HDRS tasks/g1/walk/walk.h ...)
```

**`mjpc/tasks/tasks.cc`** — add:
```cpp
#include "mjpc/tasks/g1/walk/walk.h"
// in GetTasks():
std::make_shared<g1::Walk>(),
```

---

## 8. Python Driver (src/run_g1_mpc.py)

```python
from mujoco_mpc import Agent
import mujoco

model = mujoco.MjModel.from_xml_path("src/g1_mpc_model.xml")
agent = Agent(task_name="G1 Walk", model=model)

# Set initial parameters
agent.set_task_parameter("Torso", 0.75)
agent.set_task_parameter("Speed", 0.5)

# Planning loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        agent.planner_step(num_iterations=10)
        ctrl = agent.get_action()
        data.ctrl[:len(ctrl)] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
```

---

## 9. Implementation Steps (ordered)

1. Clone mujoco_mpc as git submodule
2. Verify G1 body names by loading `g1_29dof.xml` and inspecting
3. Create `src/g1_mpc_model.xml` (include + sensors)
4. Create `mjpc/tasks/g1/walk/walk.h` and `walk.cc`
5. Create `mjpc/tasks/g1/walk/task.xml`
6. Register task in `tasks.cc` and `CMakeLists.txt`
7. Build mujoco_mpc (`cmake + ninja`)
8. Write and test `src/run_g1_mpc.py`
9. Tune weights and height goal
10. Phase 2: extend to 29 DOF with arm swing

---

## 10. Success Criteria

- Phase 1: G1 walks forward at ~0.5 m/s without falling for ≥10 seconds
- Phase 2: Natural arm swing emerges, speed range -2 to +2 m/s controllable
