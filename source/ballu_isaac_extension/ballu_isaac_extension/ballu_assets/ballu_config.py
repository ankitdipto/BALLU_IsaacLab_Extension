import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, SpringPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import math
import os
##
# Configuration
##

root_usd_path = os.path.dirname(os.path.abspath(__file__)) + "/robots"

def degree_to_radian(degree):
    return degree * math.pi / 180.0

def get_robot_usd_path():
    """Get robot USD path, with support for morphology override via environment variable."""
    # Check for morphology override environment variable
    morphology_usd_rel_path = os.environ.get('BALLU_USD_REL_PATH')
    print(f"ENV variable received: {morphology_usd_rel_path}")
    if morphology_usd_rel_path:
        morphology_usd_path = os.path.join(root_usd_path, morphology_usd_rel_path)
        print(f"🤖 Using morphology override USD: {morphology_usd_path}")
        return morphology_usd_path
    else:
        # Default to original robot
        # default_path = os.path.join(root_usd_path, "original", "original")
        default_path = os.path.join(
            root_usd_path, "11.04.2025", 
            "trial64_fl0.305_tl0.422_spc0.002_gcr0.859", 
            "trial64_fl0.305_tl0.422_spc0.002_gcr0.859.usd"
        )
        print(f"🤖 Using default robot USD: {default_path}")
        return default_path

BALLU_REAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=get_robot_usd_path(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), 
        # rot=(0.9238795, 0.0, 0.0, 0.3826834),
        joint_pos={"NECK": 0.0, 
                   "HIP_LEFT": degree_to_radian(1),
                   "HIP_RIGHT": degree_to_radian(1),
                   "KNEE_LEFT": degree_to_radian(27.35),
                   "KNEE_RIGHT": degree_to_radian(27.35),
                   "MOTOR_LEFT": degree_to_radian(10),
                   "MOTOR_RIGHT": degree_to_radian(10)}
    ),
    actuators={
        # Define actuators for MOTOR joints to accept position commands from action space
        "motor_actuators": ImplicitActuatorCfg(
            joint_names_expr=["MOTOR_LEFT", "MOTOR_RIGHT"],
            effort_limit_sim=1.44 * 9.81 * 1e-2, # 0.1412 Nm
            velocity_limit_sim=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            stiffness=1.0,
            damping=0.01,
        ),
        # Define effort-control actuator for KNEE joints
        "knee_effort_actuators": SpringPDActuatorCfg(
            joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
            effort_limit=1.44 * 9.81 * 1e-2, # 0.141264 Nm
            velocity_limit=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            spring_coeff=0.00507, #0.0807, #0.00807, #0.1409e-3 / degree_to_radian(1.0), # 0.00807 Nm/rad
            spring_damping=1.0e-3,
            spring_preload=degree_to_radian(180 - 135 + 27.35),
            pd_p=0.20, #1.00, #0.9, #1.0,
            pd_d=0.02, #0.08, #0.02
            stiffness=float("inf"), # Should not be used (If used, then I will understand by simulation instability)
            damping=float("inf"), # Should not be used (If used, then I will understand by simulation instability)
        ),
        # Keep other joints passive
        "other_passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT"], 
            stiffness=0.0,
            damping=0.001,
        ),
    },
)
"""Configuration for the real BALLU robot."""

BALLU_REAL_HETERO_CFG = ArticulationCfg(
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            os.path.join(
                root_usd_path, 
                "morphologies", 
                "trial30_FLr0.578_knKd0.103_spD0.035_GCR0.891", 
                "trial30_FLr0.578_knKd0.103_spD0.035_GCR0.891.usd"
            ),
            os.path.join(
                root_usd_path, 
                "morphologies", 
                "trial36_FLr0.641_knKd0.062_spD0.023_GCR0.891",
                "trial36_FLr0.641_knKd0.062_spD0.023_GCR0.891.usd"
            ),
        ],
        random_choice=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.1), 
        # rot=(0.9238795, 0.0, 0.0, 0.3826834),
        joint_pos={"NECK": 0.0, 
                   "HIP_LEFT": degree_to_radian(1),
                   "HIP_RIGHT": degree_to_radian(1),
                   "KNEE_LEFT": degree_to_radian(27.35),
                   "KNEE_RIGHT": degree_to_radian(27.35),
                   "MOTOR_LEFT": degree_to_radian(10),
                   "MOTOR_RIGHT": degree_to_radian(10)}
    ),
    actuators={
        # Define actuators for MOTOR joints to accept position commands from action space
        "motor_actuators": ImplicitActuatorCfg(
            joint_names_expr=["MOTOR_LEFT", "MOTOR_RIGHT"],
            effort_limit_sim=1.44 * 9.81 * 1e-2, # 0.1412 Nm
            velocity_limit_sim=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            stiffness=1.0,
            damping=0.01,
        ),
        # Define effort-control actuator for KNEE joints
        "knee_effort_actuators": SpringPDActuatorCfg(
            joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
            effort_limit=1.44 * 9.81 * 1e-2, # 0.141264 Nm
            velocity_limit=degree_to_radian(60) / 0.14, # 60 deg/0.14 sec = 428.57 rad/s
            spring_coeff=0.0807, #0.00807, #0.1409e-3 / degree_to_radian(1.0), # 0.00807 Nm/rad
            spring_damping=1.0e-2,
            spring_preload=degree_to_radian(180 - 135 + 27.35),
            pd_p=1.0, #1.0,
            pd_d=0.08, #0.02
            stiffness=float("inf"), # Should not be used (If used, then I will understand by simulation instability)
            damping=float("inf"), # Should not be used (If used, then I will understand by simulation instability)
        ),
        # Keep other joints passive
        "other_passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT"], 
            stiffness=0.0,
            damping=0.001,
        ),
    },
)
"""Configuration for the real BALLU robot."""
# BALLU_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#             fix_root_link=False
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 1.0), 
#         joint_pos={"NECK": 0.0, 
#                    "HIP_LEFT": 0.0,
#                    "HIP_RIGHT": 0.0,
#                    "KNEE_LEFT": 0.0,
#                    "KNEE_RIGHT": 0.0,
#                    "MOTOR_LEFT": 0.0,
#                    "MOTOR_RIGHT": 0.0}
#     ),
#     actuators={
#         "servo_motor_actuators": ImplicitActuatorCfg(
#             joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=10000.0, 
#             damping=10000,
#         ),
#         "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
#             joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT", 
#                               "MOTOR_LEFT", "MOTOR_RIGHT"],
#             stiffness=0,
#             damping=0.1,
#         ),
#     },
# )
"""Configuration for a BALLU robot."""

# BALLU_fixed_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#             fix_root_link=True
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 1.0), 
#         joint_pos={"NECK": 0.0, 
#                    "HIP_LEFT": 0.0,
#                    "HIP_RIGHT": 0.0,
#                    "KNEE_LEFT": 0.0,
#                    "KNEE_RIGHT": 0.0,
#                    "MOTOR_LEFT": 0.0,
#                    "MOTOR_RIGHT": 0.0}
#     ),
#     actuators={
#         "servo_motor_actuators": ImplicitActuatorCfg(
#             joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=10000.0,
#             damping=10000,
#         ),
#         "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
#             joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT", 
#                               "MOTOR_LEFT", "MOTOR_RIGHT"],
#             stiffness=0,
#             damping=10000,
#         ),
#     },
# )
"""Configuration for a BALLU robot with fixed base."""

# BALLU_HIP_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#             fix_root_link=False
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 1.0), 
#         joint_pos={"NECK": 0.0, 
#                    "HIP_LEFT": 0.0,
#                    "HIP_RIGHT": 0.0,
#                    "KNEE_LEFT": 0.0,
#                    "KNEE_RIGHT": 0.0,
#                    "MOTOR_LEFT": 0.0,
#                    "MOTOR_RIGHT": 0.0}
#     ),
#     actuators={
#         "servo_motor_actuators": ImplicitActuatorCfg(
#             joint_names_expr=["HIP_LEFT", "HIP_RIGHT"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=10000.0,
#             damping=10000,
#         ),
#         "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
#             joint_names_expr=["NECK", "KNEE_LEFT", "KNEE_RIGHT", 
#                               "MOTOR_LEFT", "MOTOR_RIGHT"],
#             stiffness=0,
#             damping=10000,
#         ),
#     },
# )
"""Configuration for a BALLU robot with actuated hips instead of knees."""

# BALLU_HIP_fixed_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#             fix_root_link=True
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 1.0), 
#         joint_pos={"NECK": 0.0, 
#                    "HIP_LEFT": 0.0,
#                    "HIP_RIGHT": 0.0,
#                    "KNEE_LEFT": 0.0,
#                    "KNEE_RIGHT": 0.0,
#                    "MOTOR_LEFT": 0.0,
#                    "MOTOR_RIGHT": 0.0}
#     ),
#     actuators={
#         "servo_motor_actuators": ImplicitActuatorCfg(
#             joint_names_expr=["HIP_LEFT", "HIP_RIGHT"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=10000.0,
#             damping=10000,
#         ),
#         "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
#            joint_names_expr=["NECK", "KNEE_LEFT", "KNEE_RIGHT", 
#                              "MOTOR_LEFT", "MOTOR_RIGHT"],
#            stiffness=0,
#            damping=0.1,
#         ),
#     },
# )
"""Configuration for a BALLU robot with actuated hips instead of knees and fixed base."""

# BALLU_HIP_KNEE_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=100.0,
#             enable_gyroscopic_forces=True,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#             fix_root_link=False
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 1.0), 
#         joint_pos={"NECK": 0.0, 
#                    "HIP_LEFT": 0.0,
#                    "HIP_RIGHT": 0.0,
#                    "KNEE_LEFT": 0.0,
#                    "KNEE_RIGHT": 0.0,
#                    "MOTOR_LEFT": 0.0,
#                    "MOTOR_RIGHT": 0.0}
#     ),
#     actuators={
#         "servo_motor_actuators": ImplicitActuatorCfg(
#             joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT", "HIP_LEFT", "HIP_RIGHT"],
#             effort_limit=400.0,
#             velocity_limit=100.0,
#             stiffness=10000.0,
#             damping=10000,
#         ),
#         "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
#             joint_names_expr=["NECK", "MOTOR_LEFT", "MOTOR_RIGHT"],
#             stiffness=0,
#             damping=10000,
#         ),
#     },
# )
"""Configuration for a BALLU robot with actuated hips and knees."""