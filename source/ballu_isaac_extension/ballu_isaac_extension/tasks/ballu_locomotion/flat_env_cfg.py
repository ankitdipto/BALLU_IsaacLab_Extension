# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

import ballu_isaac_extension.tasks.ballu_locomotion.mdp as mdp

##
# Pre-defined configs
##

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_REAL_CFG

##
# Scene definition
##


@configclass
class BALLUSceneCfg(InteractiveSceneCfg):
    """Configuration for a BALLU robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            # physics_material=sim_utils.RigidBodyMaterialCfg(
            #     static_friction=0.5,  # Default: 0.5
            #     dynamic_friction=0.5,  # Default: 0.5
            #     restitution=0.0,      # Default: 0.0
            #     friction_combine_mode="multiply",  # Default: "average"
            #     restitution_combine_mode="multiply",  # Default: "average"
            # ),
        ),
    )

    # BALLU
    robot: ArticulationCfg = BALLU_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )

    # contact sensors at feet
    #contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/TIBIA_(LEFT|RIGHT)", 
    #                                  history_length=3, 
    #                                  track_air_time=True)
    
    # IMU sensors on tibias
    # imu_tibia_left = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/TIBIA_LEFT",
    #     update_period=0.05,  # Corresponds to 20Hz
    #     gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
    #     debug_vis=True,
    # )
    
    # imu_tibia_right = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/TIBIA_RIGHT",
    #     update_period=0.05,  # Corresponds to 20Hz
    #     gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
    #     debug_vis=True,
    # )

    # IMU sensors on femurs
    # imu_femur_left = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/FEMUR_LEFT",
    #     update_period=0.05,  # Corresponds to 20Hz
    #     gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
    #     debug_vis=True,
    # )

    # imu_femur_right = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/FEMUR_RIGHT",
    #     update_period=0.05,  # Corresponds to 20Hz
    #     gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
    #     debug_vis=True,
    # )

@configclass
class ConstantVelCommandCfg:
    """Command specifications for the MDP with constant velocity."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,  # No standing environments
        rel_heading_envs=0.0,   # No heading environments
        heading_command=False,  # Not using heading commands
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.23, 0.23),  # Constant 0.1927 m/s along +x
            lin_vel_y=(0.0, 0.0),  # No y-velocity
            ang_vel_z=(0.0, 0.0),  # No angular velocity
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP - Targetting Motor Joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["MOTOR_LEFT", "MOTOR_RIGHT"], # Target motor joints
        scale=3.14159265, #1.0, #0.5,
        use_default_offset=False,
        clip = {
           "MOTOR_LEFT": (0.0, 3.14159265), # Motor limits 0 to pi
           "MOTOR_RIGHT": (0.0, 3.14159265)
        } # TODO: Consider keeping or removing this
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # TODO: Add velocity commands to the observation space for velocity tracking task
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        #actions = ObsTerm(func=mdp.last_action)

        # IMU sensor readings on tibias
        #left_tibia_orientation = ObsTerm(func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_tibia_left")})
        #left_tibia_angular_velocity = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_tibia_left")})
        #left_tibia_linear_acceleration = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_tibia_left")})

        #right_tibia_orientation = ObsTerm(func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_tibia_right")})
        #right_tibia_angular_velocity = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_tibia_right")})
        #right_tibia_linear_acceleration = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_tibia_right")})

        # IMU sensor readings on femurs
        # left_femur_orientation = ObsTerm(func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_femur_left")})
        # left_femur_angular_velocity = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_femur_left")})
        # left_femur_linear_acceleration = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_femur_left")})

        # right_femur_orientation = ObsTerm(func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_femur_right")})
        # right_femur_angular_velocity = ObsTerm(func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_femur_right")})
        # right_femur_linear_acceleration = ObsTerm(func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_femur_right")})
        
        # Contact sensor readings
        #feet_air_time = ObsTerm(func=mdp.feet_air_time, params={"sensor_cfg": SceneEntityCfg("contact_forces")})
        #feet_contact_time = ObsTerm(func=mdp.feet_contact_time, params={"sensor_cfg": SceneEntityCfg("contact_forces")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_ballu_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reward to encourage tracking the command velocity
    track_lin_vel_xy_base_l2 = RewTerm(
        func=mdp.track_lin_vel_xy_base_l2,
        weight=0.0,
        params={"command_name": "base_velocity"}
    )

    # Reward to encourage tracking the command direction
    forward_vel_base = RewTerm(
        func=mdp.forward_velocity_x,
        weight=4.0,
    )

    # Rewards to encourage tracking the exact command velocity
    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_exp_ballu, 
        weight=0.0, 
        params=
            {
                "command_name": "base_velocity", 
                "std": 0.5
            }
    )
    track_lin_vel_xy_world_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp_ballu, 
        weight=0.0, #1.0, 
        params=
            {
                "command_name": "base_velocity", 
                "std": math.sqrt(0.25)
            }
    )

    # Penalize lateral velocity
    lateral_vel_base = RewTerm(
        func=mdp.lateral_velocity_y,
        weight=-2.0,
    )
    # Penalty to enforce joint actions are within bounds
    # joint_torques_out_of_bounds = RewTerm(
    #     func=mdp.joint_torques_out_of_bounds,
    #     weight=-1.0,
    #     params=
    #         {
    #             "thresh_min": 0.0,
    #             "thresh_max": 3.14
    #         }
    # )

    # Penalty for yaw and linear velocity along z axis
    # ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.05)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)

    # Penalties to enforce action smoothness
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-9)

    # Penalties to enforce joint torque limits
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)

    # Penalties to penalize feet sliding
    #feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.01, 
    #                     params={"sensor_cfg": SceneEntityCfg("contact_forces", 
    #                                                          body_names="TIBIA_(LEFT|RIGHT)")})

    # Reward to encourage long feet air time
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=2.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="TIBIA_(LEFT|RIGHT)"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.3,
    #     },
    # )

@configclass
class CurriculumsCfg:
    """Curriculums for the MDP."""

    # vel_command_dir_reward = CurrTerm(
    #     func=mdp.scale_reward_weight,
    #     params={
    #         "term_name": "vel_command_dir",
    #         "final_weight": 1.0,
    #         "global_start_step": 0,
    #         "global_stop_step": 1,
    #     }
    # )
    # feet_air_time_reward = CurrTerm(
    #     func=mdp.scale_reward_weight,
    #     params={
    #         "term_name": "feet_air_time",
    #         "final_weight": 2.5,
    #         "global_start_step": 0,
    #         "global_stop_step": 1,
    #     }
    # )
    # action_rate_l2_penalty = CurrTerm(
    #     func=mdp.scale_reward_weight,
    #     params={
    #         "term_name": "action_rate_l2",
    #         "final_weight": -0.01,
    #         "global_start_step": 4_000_000,
    #         "global_stop_step": 4_000_002,
    #     }
    # )
    # dof_acc_l2_penalty = CurrTerm(
    #     func=mdp.scale_reward_weight,
    #     params={
    #         "term_name": "dof_acc_l2",
    #         "final_weight": -2.5e-7,
    #         "global_start_step": 8_000_000,
    #         "global_stop_step": 8_000_001,
    #     }
    # )
    # vel_tracking_reward = CurrTerm(
    #     func=mdp.scale_reward_weight,
    #     params={
    #         "term_name": "track_lin_vel_xy_base_exp",
    #         "final_weight": 1.0,
    #         "global_start_step": 0,
    #         "global_stop_step": 1,
    #     }
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Invalid simulator state (terminal)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    # (3) Root height above hard limit (terminal)
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 3.0})

##
# Environment configuration
##


@configclass
class BalluFlatEnvCfg(ManagerBasedRLEnvCfg): # Renamed class
    """Configuration for the BALLU robot environment with flat terrain."""

    # Scene settings
    scene: BALLUSceneCfg = BALLUSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg() # Use updated ActionsCfg
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10 #8
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (2, 5, 3)
        self.viewer.lookat = (2, 0, 0.3)
        self.viewer.resolution = (1920, 1080) # Full HD resolution
        # simulation settings
        self.sim.dt = 1 / 200.0 #160.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        #self.sim.physx.solver_type = 0 # Projected Gauss-Seidel
        self.sim.physx.solver_type = 1 # Truncated Gauss-Seidel
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1