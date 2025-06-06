import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, SpringPDActuatorCfg
from isaaclab.assets import ArticulationCfg
import math
##
# Configuration
##

def degree_to_radian(degree):
    return degree * math.pi / 180.0

BALLU_REAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/original/original.usd",
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
        pos=(0.0, 0.0, 0.7), 
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
            spring_coeff=0.1409e-3 / degree_to_radian(1.0), # 0.4021 Nm/rad
            spring_damping=1.0e-2,
            spring_preload=degree_to_radian(180 - 135 + 27.35),
            pd_p=0.08, #1.0,
            pd_d=0.01,
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