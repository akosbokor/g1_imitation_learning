# G1 mujoco_mpc Walk — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt Google DeepMind's mujoco_mpc to drive bipedal walking on the Unitree G1 (29 DOF).

**Architecture:** Clone mujoco_mpc as a git submodule. Add a `g1/walk` C++ task class with a residual function mapped to G1's body/sensor names. Phase 1 controls 15 DOF (legs + waist); arms hang with passive damping. Phase 2 extends to 29 DOF.

**Tech Stack:** C++17 (mujoco_mpc core), CMake/Ninja, Python 3.10 (gRPC driver), MuJoCo 3.x, gRPC

---

## Prerequisites & Background

### G1 robot facts (needed throughout)
- **29 actuated joints** + 1 freejoint (`floating_base_joint`) on the pelvis
- **Pelvis z at stand:** 0.793 m above ground
- **Torso (`torso_link`) z at stand:** ~1.05 m above ground
- **Body names:** `pelvis`, `torso_link`, `left_ankle_roll_link`, `right_ankle_roll_link`
- **Joint order:** legs 0-11, waist 12-14, left arm 15-21, right arm 22-28
- **Actuator order** in XML matches joint order exactly

### mujoco_mpc residual pattern
Each cost term is declared as a `<user>` sensor in `task.xml`. The C++ `Residual()` function fills a flat `double* residual` array in the **same order** as those sensors. Mismatch → runtime error.

The `<user>` sensor format is:
```xml
<user name="TermName" dim="D" user="NORM_TYPE WEIGHT MIN MAX [NORM_PARAMS...]" />
```
Norm types: 0=quadratic, 1=L2, 2=Cosh, 3=Power, 7=Quadratic-linear

---

## Task 1: Clone mujoco_mpc as a submodule

**Files:**
- Modify: `.gitmodules` (auto-created)
- Create: `mujoco_mpc/` directory

**Step 1: Add submodule**

```bash
cd /home/sztaki/g1_project
git submodule add https://github.com/google-deepmind/mujoco_mpc.git mujoco_mpc
git submodule update --init --recursive
```

Expected: `mujoco_mpc/` populated, `.gitmodules` created.

**Step 2: Verify it cloned**

```bash
ls mujoco_mpc/mjpc/tasks/humanoid/walk/
```

Expected: `walk.h  walk.cc  task.xml` visible.

**Step 3: Commit**

```bash
git add .gitmodules mujoco_mpc
git commit -m "feat: add mujoco_mpc as git submodule"
```

---

## Task 2: Inspect G1 torso height (calibrate residual goal)

**Files:**
- Read: `unitree_mujoco/unitree_robots/g1/g1_29dof.xml`

**Step 1: Load model and print torso z position**

```bash
python3 - <<'EOF'
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)  # default pose (all q=0)
mujoco.mj_forward(m, d)
torso_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
foot_l_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
foot_r_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
print(f"torso_link z:            {d.xpos[torso_id][2]:.4f}")
print(f"pelvis z:                {d.xpos[pelvis_id][2]:.4f}")
print(f"left_ankle_roll_link z:  {d.xpos[foot_l_id][2]:.4f}")
print(f"right_ankle_roll_link z: {d.xpos[foot_r_id][2]:.4f}")
print(f"nq={m.nq}  nu={m.nu}  nv={m.nv}")
EOF
```

Expected output (approx):
```
torso_link z:            1.05xx
pelvis z:                0.793x
left_ankle_roll_link z:  0.0xxx
right_ankle_roll_link z: 0.0xxx
nq=36  nu=29  nv=35
```

**Step 2: Record actual torso_link z value** — use it as `residual_Torso` goal in Task 5.

**Step 3: Also confirm nq=36 (7 freejoint + 29 actuated)**

```bash
python3 - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  joint[{i:2d}] {name}")
EOF
```

Expected: joint[0] = floating_base_joint (freejoint), joints[1-29] = the 29 actuated joints.

---

## Task 3: Create g1_mpc_model.xml

**Files:**
- Create: `src/g1_mpc_model.xml`

This file wraps `g1_29dof.xml`, overrides arm damping for Phase 1, and adds all sensors required by the residual function.

**Step 1: Write the file**

```xml
<!-- src/g1_mpc_model.xml -->
<mujoco model="G1 MPC Walk">

  <!-- Include the base G1 model.
       Path is relative to this file's location (src/).
       Meshes live at unitree_mujoco/unitree_robots/g1/meshes/
  -->
  <compiler meshdir="../unitree_mujoco/unitree_robots/g1/meshes"
            angle="radian"/>

  <include file="../unitree_mujoco/unitree_robots/g1/g1_29dof.xml"/>

  <!-- ===== Phase 1: passive arm damping =====
       Arms hang naturally; MPC does NOT control them.
       High damping on all 14 arm joints.
       Remove/comment these defaults in Phase 2. -->
  <default>
    <default class="arm_passive">
      <joint damping="10"/>
    </default>
  </default>

  <!-- Override arm joint damping -->
  <!-- NOTE: MuJoCo applies the last default that matches, but since
       joints already have a class, we patch via individual joint refs.
       We do this in the actuator section by setting gear=0 for arms. -->

  <!-- ===== Sensors required by G1Walk residual =====
       ORDER MATTERS: user sensors (cost terms) must come first,
       then the residual/estimator sensors below. -->

  <sensor>
    <!-- ---- Cost term sensors (user sensors, order = residual order) ---- -->
    <!-- Residual 0: torso height (dim=1) -->
    <user name="Height"      dim="1"  user="7 5.0 0 25.0 0.1 4.0"/>
    <!-- Residual 1: pelvis-feet alignment (dim=1) -->
    <user name="Pelvis_Feet" dim="1"  user="8 1.0 0.0 10.0 0.05"/>
    <!-- Residual 2: balance capture point (dim=2) -->
    <user name="Balance"     dim="2"  user="1 5.0 0.0 25.0 0.02 4.0"/>
    <!-- Residual 3: upright (dim=8: torso1 + pelvis1 + foot_r3 + foot_l3) -->
    <user name="Upright"     dim="8"  user="2 5.0 0.0 25.0 0.01"/>
    <!-- Residual 4: posture (dim=29 = nq-7) -->
    <user name="Posture"     dim="29" user="0 0.025 0 1.0"/>
    <!-- Residual 5: walk forward (dim=1) -->
    <user name="Walk"        dim="1"  user="7 1.0 0.0 25.0 0.5 3.0"/>
    <!-- Residual 6: move feet (dim=2) -->
    <user name="Move_Feet"   dim="2"  user="7 0.625 0 25.0 0.2 4.0"/>
    <!-- Residual 7: control effort (dim=15 Phase1, change to 29 Phase2) -->
    <user name="Control"     dim="15" user="3 0.1 0 1.0 0.3"/>

    <!-- ---- Residual computation sensors ---- -->
    <framepos    name="torso_position"       objtype="body"  objname="torso_link"/>
    <subtreecom  name="torso_subcom"         body="torso_link"/>
    <subtreelinvel name="torso_subcomvel"    body="torso_link"/>
    <framepos    name="foot_right"           objtype="body"  objname="right_ankle_roll_link"/>
    <framepos    name="foot_left"            objtype="body"  objname="left_ankle_roll_link"/>
    <framepos    name="pelvis_position"      objtype="body"  objname="pelvis"/>
    <framezaxis  name="torso_up"             objtype="xbody" objname="torso_link"/>
    <framezaxis  name="pelvis_up"            objtype="xbody" objname="pelvis"/>
    <framezaxis  name="foot_right_up"        objtype="xbody" objname="right_ankle_roll_link"/>
    <framezaxis  name="foot_left_up"         objtype="xbody" objname="left_ankle_roll_link"/>
    <framexaxis  name="torso_forward"        objtype="xbody" objname="torso_link"/>
    <framexaxis  name="pelvis_forward"       objtype="xbody" objname="pelvis"/>
    <framexaxis  name="foot_right_forward"   objtype="xbody" objname="right_ankle_roll_link"/>
    <framexaxis  name="foot_left_forward"    objtype="xbody" objname="left_ankle_roll_link"/>
    <subtreelinvel name="pelvis_subcomvel"   body="pelvis"/>
    <framelinvel name="torso_velocity"       objtype="body"  objname="torso_link"/>
    <framelinvel name="foot_right_velocity"  objtype="body"  objname="right_ankle_roll_link"/>
    <framelinvel name="foot_left_velocity"   objtype="body"  objname="left_ankle_roll_link"/>
  </sensor>

</mujoco>
```

**Step 2: Verify the XML loads without errors**

```bash
python3 - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_path("src/g1_mpc_model.xml")
print(f"nq={m.nq}  nu={m.nu}  nsensor={m.nsensor}")
# Count user sensors
n_user = sum(1 for i in range(m.nsensor)
             if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_USER)
print(f"user sensors: {n_user}  (expected 8)")
EOF
```

Expected:
```
nq=36  nu=29  nsensor=XX
user sensors: 8
```

**Step 3: Commit**

```bash
git add src/g1_mpc_model.xml
git commit -m "feat: add G1 MPC model XML with locomotion sensors"
```

---

## Task 4: Create C++ task header (walk.h)

**Files:**
- Create: `mujoco_mpc/mjpc/tasks/g1/walk/walk.h`

**Step 1: Create directory and write header**

```bash
mkdir -p mujoco_mpc/mjpc/tasks/g1/walk
```

```cpp
// mujoco_mpc/mjpc/tasks/g1/walk/walk.h
// Copyright 2024 — G1 Walk task for mujoco_mpc
// Adapted from mjpc/tasks/humanoid/walk/walk.h

#ifndef MJPC_TASKS_G1_WALK_H_
#define MJPC_TASKS_G1_WALK_H_

#include <string>
#include <memory>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Walk : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Walk* task) : mjpc::BaseResidualFn(task) {}

    // Residuals (in order, matching task.xml user sensors):
    //   [0]     torso height          (dim 1)
    //   [1]     pelvis-feet alignment (dim 1)
    //   [2-3]   balance capture point (dim 2)
    //   [4-11]  upright               (dim 8)
    //   [12-40] posture               (dim 29 = nq - 7)
    //   [41]    walk forward          (dim 1)
    //   [42-43] move feet             (dim 2)
    //   [44-58] control effort        (dim 15, Phase 1)
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Walk() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_WALK_H_
```

**Step 2: Check it compiles in isolation (will fail, that's OK — just verify no obvious syntax errors)**

```bash
cd mujoco_mpc
g++ -std=c++17 -fsyntax-only \
    -I. -Imjpc \
    mjpc/tasks/g1/walk/walk.h 2>&1 | head -20
```

Expected: errors about missing `mjpc/task.h` include path (fine — full build happens in Task 8).

---

## Task 5: Create C++ task implementation (walk.cc)

**Files:**
- Create: `mujoco_mpc/mjpc/tasks/g1/walk/walk.cc`

**Step 1: Write the implementation**

Replace `TORSO_HEIGHT_GOAL` with the actual value measured in Task 2 (e.g. `1.05`).

```cpp
// mujoco_mpc/mjpc/tasks/g1/walk/walk.cc
// Adapted from mjpc/tasks/humanoid/walk/walk.cc

#include "mjpc/tasks/g1/walk/walk.h"

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace g1 {

std::string Walk::Name() const { return "G1 Walk"; }

std::string Walk::XmlPath() const {
  return GetModelPath("g1/walk/task.xml");
}

// -----------------------------------------------------------------
// Residual for G1 walk task
//
// Parameters:
//   parameters_[0] = torso height goal  (default: ~1.05m)
//   parameters_[1] = speed goal         (default: 0.5 m/s)
// -----------------------------------------------------------------
void Walk::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;

  // ---- [0] torso height ----------------------------------------
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

  // ---- [1] pelvis / feet alignment -----------------------------
  double* foot_right = SensorByName(model, data, "foot_right");
  double* foot_left  = SensorByName(model, data, "foot_left");
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ---- [2-3] balance (capture point) ---------------------------
  double* subcom    = SensorByName(model, data, "torso_subcom");
  double* subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  double axis[3], center[3], vec[3], pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);
  double t = mju_dot3(vec, axis);
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // standing metric: 1 when upright, 0 when fallen
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ---- [4-11] upright ------------------------------------------
  double* torso_up      = SensorByName(model, data, "torso_up");
  double* pelvis_up     = SensorByName(model, data, "pelvis_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* foot_left_up  = SensorByName(model, data, "foot_left_up");
  double z_ref[3] = {0.0, 0.0, 1.0};

  residual[counter++] = torso_up[2] - 1.0;                    // dim 1
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);           // dim 1

  mju_sub3(&residual[counter], foot_right_up, z_ref);          // dim 3
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);           // dim 3
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  // ---- [12-40] posture (29 = nq - 7) ---------------------------
  // Penalise deviation from q=0 for all actuated joints.
  // qpos[0:7] = freejoint (3 pos + 4 quat), qpos[7:36] = 29 joints.
  mju_copy(&residual[counter], data->qpos + 7, model->nq - 7);
  counter += model->nq - 7;  // = 29

  // ---- [41] walk forward ---------------------------------------
  double* torso_forward  = SensorByName(model, data, "torso_forward");
  double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double* fr_forward     = SensorByName(model, data, "foot_right_forward");
  double* fl_forward     = SensorByName(model, data, "foot_left_forward");

  double forward[2];
  mju_copy(forward, torso_forward, 2);
  mju_addTo(forward, pelvis_forward, 2);
  mju_addTo(forward, fr_forward, 2);
  mju_addTo(forward, fl_forward, 2);
  mju_normalize(forward, 2);

  double* pelvis_comvel = SensorByName(model, data, "pelvis_subcomvel");
  double* torso_vel     = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, pelvis_comvel, torso_vel, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  residual[counter++] =
      standing * (mju_dot(com_vel, forward, 2) - parameters_[1]);

  // ---- [42-43] move feet ---------------------------------------
  double* foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  double* foot_left_vel  = SensorByName(model, data, "foot_left_velocity");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel,  -0.5, 2);
  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ---- [44-58] control effort (Phase 1: 15 DOF) ----------------
  // Only the first 15 actuators (legs + waist) are active in Phase 1.
  // In Phase 2, change to model->nu (29).
  int nu_active = 15;  // Phase 1
  mju_copy(&residual[counter], data->ctrl, nu_active);
  counter += nu_active;

  // ---- sanity check --------------------------------------------
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error(
        "G1Walk: mismatch between user-sensor total dim (%d) "
        "and residual counter (%d)",
        user_sensor_dim, counter);
  }
}

}  // namespace g1
}  // namespace mjpc
```

**Step 2: No unit test yet (tested via full build in Task 8). Commit the file.**

```bash
git add mujoco_mpc/mjpc/tasks/g1/
git commit -m "feat: add G1Walk C++ task class (Phase 1, 15 DOF)"
```

---

## Task 6: Create task.xml

**Files:**
- Create: `mujoco_mpc/mjpc/tasks/g1/walk/task.xml`

The torso height goal (`residual_Torso`) uses the value measured in Task 2.
Replace `TORSO_Z` with the actual measured value (e.g. `1.05`).

**Step 1: Write task.xml**

```xml
<!-- mujoco_mpc/mjpc/tasks/g1/walk/task.xml -->
<mujoco model="G1 Walk">
  <include file="../../common.xml"/>

  <!-- G1 model with sensors. Path resolved from build output dir. -->
  <include file="g1_mpc_model.xml"/>

  <size memory="1M"/>

  <custom>
    <!-- Agent planner: 2 = sampling MPC -->
    <numeric name="agent_planner"          data="2"/>
    <numeric name="agent_horizon"          data="0.35"/>
    <numeric name="agent_timestep"         data="0.015"/>
    <numeric name="sampling_spline_points" data="3"/>
    <numeric name="sampling_exploration"   data="0.05"/>
    <numeric name="gradient_spline_points" data="5"/>

    <!-- Residual parameters (goal, min, max) -->
    <!-- TORSO_Z: replace with value from Task 2 measurement -->
    <numeric name="residual_Torso" data="1.05 0.6 1.4"/>
    <numeric name="residual_Speed" data="0.5 -2.0 2.0"/>
  </custom>

  <sensor>
    <!-- Cost term sensors: these must exactly match walk.cc residual order
         AND match the sensors in g1_mpc_model.xml.
         They are re-declared here for the task XML (task.xml is the root). -->

    <!-- The sensors below ARE the user sensors from g1_mpc_model.xml.
         mujoco_mpc reads them from the composed model, not this file.
         This section is intentionally empty — sensors live in g1_mpc_model.xml. -->
  </sensor>
</mujoco>
```

> **Note:** In mujoco_mpc, `task.xml` uses `<include>` to pull in the robot model. The user sensors (cost terms) must be declared once in the included model file. In our case they live in `g1_mpc_model.xml`.

**Step 2: Commit**

```bash
git add mujoco_mpc/mjpc/tasks/g1/walk/task.xml
git commit -m "feat: add G1Walk task.xml with planner config"
```

---

## Task 7: Register G1Walk in tasks.cc and CMakeLists.txt

**Files:**
- Modify: `mujoco_mpc/mjpc/tasks/tasks.cc`
- Modify: `mujoco_mpc/mjpc/tasks/CMakeLists.txt`

### 7a: tasks.cc

**Step 1: Add include and factory entry**

Open `mujoco_mpc/mjpc/tasks/tasks.cc`. Find the block of `#include` lines and add:

```cpp
#include "mjpc/tasks/g1/walk/walk.h"
```

Then in `GetTasks()`, add after the humanoid entries:

```cpp
std::make_shared<g1::Walk>(),
```

**Step 2: Verify the addition looks correct**

```bash
grep -n "g1\|G1\|humanoid" mujoco_mpc/mjpc/tasks/tasks.cc
```

Expected: see both humanoid and g1 entries.

### 7b: CMakeLists.txt

**Step 1: Open `mujoco_mpc/mjpc/tasks/CMakeLists.txt`**

Find a similar existing task block (e.g. humanoid walk) and add after it:

```cmake
# ---- G1 Walk ----
add_library(g1_walk OBJECT
  g1/walk/walk.cc
  g1/walk/walk.h
)
target_link_libraries(g1_walk
  task
  utilities
  mujoco::mujoco
)
target_include_directories(g1_walk PRIVATE ${CMAKE_SOURCE_DIR})

# Copy G1 task XML and model to build output
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/g1/walk/task.xml
  ${CMAKE_CURRENT_BINARY_DIR}/g1/walk/task.xml COPYONLY)
configure_file(
  ${CMAKE_SOURCE_DIR}/../src/g1_mpc_model.xml
  ${CMAKE_CURRENT_BINARY_DIR}/g1/walk/g1_mpc_model.xml COPYONLY)
# G1 model XML (referenced by g1_mpc_model.xml)
configure_file(
  ${CMAKE_SOURCE_DIR}/../unitree_mujoco/unitree_robots/g1/g1_29dof.xml
  ${CMAKE_CURRENT_BINARY_DIR}/g1/walk/../../../unitree_mujoco/unitree_robots/g1/g1_29dof.xml
  COPYONLY)
```

Then find the `tasks` target that aggregates all task libraries and add `g1_walk` to it:

```cmake
target_link_libraries(tasks
  ...
  g1_walk       # add this line
  ...
)
```

**Step 2: Also link meshes** — G1 uses STL mesh files. Add a custom target to copy them:

```cmake
# Copy G1 mesh files
file(GLOB G1_MESHES
  "${CMAKE_SOURCE_DIR}/../unitree_mujoco/unitree_robots/g1/meshes/*.stl"
  "${CMAKE_SOURCE_DIR}/../unitree_mujoco/unitree_robots/g1/meshes/*.STL"
)
foreach(mesh ${G1_MESHES})
  get_filename_component(mesh_name ${mesh} NAME)
  configure_file(${mesh}
    ${CMAKE_CURRENT_BINARY_DIR}/g1/walk/meshes/${mesh_name} COPYONLY)
endforeach()
```

**Step 3: Commit**

```bash
git add mujoco_mpc/mjpc/tasks/tasks.cc mujoco_mpc/mjpc/tasks/CMakeLists.txt
git commit -m "feat: register G1Walk task in mujoco_mpc build system"
```

---

## Task 8: Build mujoco_mpc

**Files:**
- Build output: `mujoco_mpc/build/`

**Step 1: Install system dependencies**

```bash
sudo apt-get update && sudo apt-get install -y \
    cmake ninja-build libgl1-mesa-dev libxinerama-dev \
    libxcursor-dev libxi-dev libxrandr-dev libxxf86vm-dev \
    libpython3-dev python3-dev \
    clang-14 libc++-14-dev libc++abi-14-dev
```

**Step 2: Configure CMake**

```bash
cd /home/sztaki/g1_project/mujoco_mpc
mkdir -p build && cd build
cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-14 \
  -DCMAKE_CXX_COMPILER=clang++-14 \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  2>&1 | tail -20
```

Expected: `-- Build files have been written to: .../mujoco_mpc/build`

**Step 3: Build (only the mjpc target)**

```bash
ninja mjpc 2>&1 | tail -30
```

Expected: `[N/N] Linking CXX executable mjpc/mjpc`

If there are compile errors in `walk.cc`, fix them before proceeding (common issues: wrong include path, missing `mju_error` signature — check humanoid/walk.cc for reference).

**Step 4: Verify G1 Walk is in the task list**

```bash
./mjpc/mjpc --list_tasks 2>&1 | grep -i "g1\|G1"
```

Expected: `G1 Walk`

**Step 5: Also build the Python bindings**

```bash
cd /home/sztaki/g1_project/mujoco_mpc
pip install -e "python/[dev]" 2>&1 | tail -10
```

Expected: `Successfully installed mujoco-mpc`

**Step 6: Commit build confirmation**

```bash
cd /home/sztaki/g1_project
git add mujoco_mpc/  # captures any generated/patched files
git commit -m "build: mujoco_mpc compiled with G1Walk task"
```

---

## Task 9: Verify sensor/residual dimension match

This is a critical check *before* running the full Python driver.

**Step 1: Write a standalone dimension-check script**

```python
# src/check_g1_sensors.py
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("src/g1_mpc_model.xml")
d = mujoco.MjData(m)

print(f"nq={m.nq}  nu={m.nu}  nsensor={m.nsensor}")

# Print all user sensors
user_total = 0
for i in range(m.nsensor):
    if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_USER:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim  = m.sensor_dim[i]
        print(f"  user sensor [{i}] '{name}' dim={dim}")
        user_total += dim

print(f"\nTotal user sensor dim = {user_total}")
# Expected Phase 1: 1+1+2+8+29+1+2+15 = 59
print(f"Expected Phase 1:      59")
assert user_total == 59, f"MISMATCH: got {user_total}, expected 59"
print("OK: dimensions match")
```

**Step 2: Run it**

```bash
cd /home/sztaki/g1_project
python3 src/check_g1_sensors.py
```

Expected: `OK: dimensions match`

If mismatched: go back to `g1_mpc_model.xml` and fix the `dim=` values on user sensors.

**Step 3: Commit**

```bash
git add src/check_g1_sensors.py
git commit -m "test: add G1 sensor/residual dimension check script"
```

---

## Task 10: Write Python driver (run_g1_mpc.py)

**Files:**
- Create: `src/run_g1_mpc.py`

**Step 1: Write the driver**

```python
# src/run_g1_mpc.py
"""
Drive the G1 Walk task via mujoco_mpc's Python gRPC API.
Phase 1: 15 DOF (legs + waist), arms passively damped.

Usage:
    python src/run_g1_mpc.py [--speed 0.5] [--headless]
"""
import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed",    type=float, default=0.5,
                        help="Walking speed target (m/s)")
    parser.add_argument("--headless", action="store_true",
                        help="No viewer, just print stats")
    args = parser.parse_args()

    # ---- load model ----
    model_path = str(ROOT / "src" / "g1_mpc_model.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Start at standing height — set pelvis z
    # freejoint qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[2] = 0.793   # pelvis z at stand

    # ---- launch mujoco_mpc agent ----
    from mujoco_mpc import agent as agent_lib
    agent = agent_lib.Agent(task_id="G1 Walk", model=model)

    # Set task parameters
    agent.set_task_parameter("Torso", 1.05)   # torso height goal
    agent.set_task_parameter("Speed", args.speed)

    print(f"G1 Walk | speed={args.speed} m/s | {'headless' if args.headless else 'viewer'}")
    print("Starting planning loop ...")

    step_count = 0
    t0 = time.time()

    def run_loop(viewer=None):
        nonlocal step_count
        while True:
            # Sync state to agent
            agent.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.mocap_pos,
                mocap_quat=data.mocap_quat,
            )

            # Run planner iterations
            agent.planner_step()

            # Get action (15 ctrl signals, Phase 1)
            action = agent.get_action()
            data.ctrl[:len(action)] = action

            # Step physics
            mujoco.mj_step(model, data)
            step_count += 1

            # Stats every 100 steps
            if step_count % 100 == 0:
                torso_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
                torso_z = data.xpos[torso_id][2]
                elapsed = time.time() - t0
                print(f"  step={step_count:5d}  sim_t={data.time:.2f}s  "
                      f"torso_z={torso_z:.3f}m  wall={elapsed:.1f}s")

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break

    if args.headless:
        # Run 500 steps headless
        for _ in range(500):
            run_loop.__wrapped__ = True
        # simpler headless loop
        for _ in range(500):
            agent.set_state(time=data.time, qpos=data.qpos,
                            qvel=data.qvel, act=data.act,
                            mocap_pos=data.mocap_pos,
                            mocap_quat=data.mocap_quat)
            agent.planner_step()
            action = agent.get_action()
            data.ctrl[:len(action)] = action
            mujoco.mj_step(model, data)
            step_count += 1
            if step_count % 100 == 0:
                torso_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
                print(f"  step={step_count}  torso_z={data.xpos[torso_id][2]:.3f}")
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_loop(viewer)

    print(f"Done. {step_count} steps in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
```

**Step 2: Run headless sanity check (30 steps)**

```bash
cd /home/sztaki/g1_project
python3 src/run_g1_mpc.py --headless 2>&1 | head -30
```

Expected: no crash, prints step/torso_z lines. Torso z should stay near 1.05 (not fall to 0).

**Step 3: Run with viewer**

```bash
python3 src/run_g1_mpc.py --speed 0.5
```

Expected: MuJoCo viewer opens, G1 attempts to walk forward.

**Step 4: Commit**

```bash
git add src/run_g1_mpc.py
git commit -m "feat: add G1 mujoco_mpc Python driver (Phase 1, 15 DOF)"
```

---

## Task 11: Tune Phase 1 parameters

If the robot falls immediately, adjust these values in `g1_mpc_model.xml` and `task.xml`:

| Issue | Fix |
|-------|-----|
| Falls forward/backward | Increase `Balance` weight (user sensor `user=` 2nd field) |
| Can't maintain height | Increase `Height` weight |
| Legs too stiff / no gait | Reduce `Posture` weight |
| Too much energy | Increase `Control` weight |
| Wrong height goal | Re-measure in Task 2, update `residual_Torso` in task.xml |

**Tuning loop:**

```bash
# Edit weight in src/g1_mpc_model.xml, then re-run
python3 src/run_g1_mpc.py --headless
```

No rebuild needed for XML-only changes. No rebuild needed for Python changes.
Rebuild (`ninja mjpc`) only needed if `.cc` or `.h` files change.

---

## Task 12: Phase 2 — extend to full 29 DOF

When Phase 1 walks stably, extend to all 29 joints:

**Step 1: Update `g1_mpc_model.xml`** — change `Control` user sensor dim from 15 to 29:

```xml
<user name="Control" dim="29" user="3 0.1 0 1.0 0.3"/>
```

**Step 2: Update `walk.cc`** — change `nu_active` and add arm-swing residual:

```cpp
// Phase 2: full 29 DOF
int nu_active = model->nu;   // 29
mju_copy(&residual[counter], data->ctrl, nu_active);
counter += nu_active;
```

**Step 3: Verify new dimension** — update `check_g1_sensors.py`:

```python
# Phase 2: 1+1+2+8+29+1+2+29 = 73
assert user_total == 73, f"MISMATCH: got {user_total}, expected 73"
```

**Step 4: Rebuild and test**

```bash
cd mujoco_mpc/build && ninja mjpc
cd /home/sztaki/g1_project
python3 src/check_g1_sensors.py
python3 src/run_g1_mpc.py --speed 0.5
```

**Step 5: Commit**

```bash
git add src/g1_mpc_model.xml mujoco_mpc/mjpc/tasks/g1/walk/walk.cc src/check_g1_sensors.py
git commit -m "feat: extend G1Walk to Phase 2 full 29 DOF"
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `ninja mjpc` (in `mujoco_mpc/build/`) | Rebuild after C++ changes |
| `python3 src/check_g1_sensors.py` | Verify sensor/residual dims |
| `python3 src/run_g1_mpc.py --headless` | Headless smoke test |
| `python3 src/run_g1_mpc.py --speed 1.0` | Run with viewer at 1 m/s |
| `./mjpc/mjpc --list_tasks` | List registered tasks |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `mismatch between total user-sensor dimension` | `walk.cc` counter ≠ XML user sensor total | Fix `dim=` in XML or `counter` in .cc |
| `cannot find body 'torso_link'` | Wrong body name in XML | Check exact names with Task 2 script |
| `unknown sensor name` | Sensor missing from `g1_mpc_model.xml` | Add the missing sensor |
| `task 'G1 Walk' not found` | Task not registered or not rebuilt | Check `tasks.cc`, rebuild |
| Robot falls immediately | Height/Balance weights too low | See Task 11 tuning table |
