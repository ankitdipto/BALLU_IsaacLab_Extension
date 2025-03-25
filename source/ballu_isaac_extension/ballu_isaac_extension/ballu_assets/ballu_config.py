import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

BALLU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/asinha389/Documents/Projects/MorphologyOPT/BALLU_IsaacLab_Extension/source/ballu_isaac_extension/ballu_isaac_extension/ballu_assets/robots/ballu_v0.usd",
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
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0), 
        joint_pos={"NECK": 0.0, 
                   "HIP_LEFT": 0.0,
                   "HIP_RIGHT": 0.0,
                   "KNEE_LEFT": 0.0,
                   "KNEE_RIGHT": 0.0,
                   "MOTOR_LEFT": 0.0,
                   "MOTOR_RIGHT": 0.0}
    ),
    actuators={
        "servo_motor_actuators": ImplicitActuatorCfg(
            joint_names_expr=["KNEE_LEFT", "KNEE_RIGHT"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=10000.0,
            damping=10000,
        ),
        "dummy_actuator_all_passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["NECK", "HIP_LEFT", "HIP_RIGHT", 
                              "MOTOR_LEFT", "MOTOR_RIGHT"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0,
            damping=1000,
        ),
    },
)
"""Configuration for a BALLU robot."""