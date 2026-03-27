import numpy as np


DEFAULT_MIMIC_OBS_G1 = np.concatenate([
                    np.array([ 0.772]),          # height (matches boxing frame 0)
                    np.array([0, 0, 0]),          # rpy
                    np.array([0, 0, 0]),          # root_vel_local
                    np.array([ 0.0]),             # ang_vel_z
                    # 25 dof (leg + waist + arms, wrist roll zeros at [19] and [24])
                    np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg (6)
                        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,          # right leg (6)
                        -0.052, -0.017, 0.089,                    # waist (3) — boxing init
                        0.534,  0.806,  0.355, -0.316, 0.0,       # left arm (4) + wrist roll (1)
                        0.524, -0.862, -0.302, -0.279, 0.0,       # right arm (4) + wrist roll (1)
                        ])
                ])

DEFAULT_MIMIC_OBS_G1_WITHOUT_WRIST_ROLL = np.concatenate([
                    np.array([ 0.793]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([ 0.0]),
                    # 25 dof
                    np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                        0.0, 0.0, 0.0, # torso (1)
                        # 0.0, 0.2, 0.0, 1.2, # left arm (4)
                        # 0.0, -0.2, 0.0, 1.2, # right arm (4)
                        0.0, 0.2, 0.0, 1.2, # left arm (4)
                        0.0, -0.2, 0.0, 1.2, # right arm (4)
                        ])
                ])


DEFAULT_MIMIC_OBS_T1 = np.concatenate([
                    np.array([ 0.6]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([ 0.0]),
                    # 21 dof
                    np.array([
                        0.25, -1.4, 0.0, -0.5, # left arm
                        0.25, 1.4, 0.0, 0.5, # right arm
                        0.0, # waist
                        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0, # left leg
                        -0.1, 0.0, 0.0, 0.2, -0.1, 0.0, # right leg
                    ])
                ])

DEFAULT_MIMIC_OBS_TODDY = np.concatenate([
                    np.array([ 0.3]),
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0]),
                    np.array([ 0.0]),
                    # 21 dof
                    np.array([
                        0.0, 0.0, # waist (2)
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # left leg (6)
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # right leg (6)
                        0, -0.3, 0, 0.0, # left arm (4)
                        0, -0.3, 0, 0.0, # right arm (4)
                      
                    ])
                ])

DEFAULT_MIMIC_OBS = {
    "g1": DEFAULT_MIMIC_OBS_G1,
    "g1_without_wrist_roll": DEFAULT_MIMIC_OBS_G1_WITHOUT_WRIST_ROLL,
    "t1": DEFAULT_MIMIC_OBS_T1,
    "toddy": DEFAULT_MIMIC_OBS_TODDY,
}


DEFAULT_ACTION_HAND = {
    "g1": np.concatenate([
        np.array([0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0]), # 7 Dof hands
    ]),
    "t1": np.concatenate([
        np.array([0, 0]), # parallel gripper
    ]),
    "toddy": np.concatenate([
        np.array([0, 0]), # parallel gripper
    ]),
}

DEX31_QPOS_CLOSE = {
    "left": np.array([
        # left (thumb, index, middle)
        0, 1.0, 1.74, -1.57, -1.74, -1.57, -1.74,
    ]),
    "right": np.array([
        # right (thumb, index, middle)
        0, -1.0, -1.74, 1.57, 1.74, 1.57, 1.74,
    ]),
}


DEX31_QPOS_OPEN = {
    "left": np.array([
        # left (thumb, index, middle)
        0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]),
    "right": np.array([
        # right (thumb, index, middle)
        0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]),
}