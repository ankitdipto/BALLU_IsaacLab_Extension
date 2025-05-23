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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.imu import ImuCfg
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
    """Configuration for a BALLU scene with IMU sensors on both tibias."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # BALLU robot
    robot: ArticulationCfg = BALLU_REAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # IMU sensors on tibias
    imu_tibia_left = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/TIBIA_LEFT",
        update_period=0.05,  # Corresponds to 20Hz
        gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
        debug_vis=True,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    imu_tibia_right = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/TIBIA_RIGHT",
        update_period=0.05,  # Corresponds to 20Hz
        gravity_bias=(0.0, 0.0, 9.81),  # Compensates 'g'. At rest, IMU reads (0.0, 0.0, 0.0)
        debug_vis=True,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0), 
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


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
            lin_vel_x=(0.4, 0.4),  # Constant 0.4 m/s along +x
            lin_vel_y=(0.0, 0.0),  # No y-velocity
            ang_vel_z=(0.0, 0.0),  # No angular velocity
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["MOTOR_LEFT", "MOTOR_RIGHT"],
        scale=1.0,
        use_default_offset=True,
        clip = {
            "MOTOR_LEFT": (0, 3.14159265),
            "MOTOR_RIGHT": (0, 3.14159265)
        }
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP with tibia IMU sensor data."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group using tibia IMU sensor data only."""

        # Command velocity (desired velocity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        # Left tibia IMU sensor readings
        left_tibia_orientation = ObsTerm(
            func=mdp.imu_orientation, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_left")}
        )
        left_tibia_angular_velocity = ObsTerm(
            func=mdp.imu_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_left")}
        )
        left_tibia_linear_acceleration = ObsTerm(
            func=mdp.imu_lin_acc, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_left")}
        )
        
        # Right tibia IMU sensor readings
        right_tibia_orientation = ObsTerm(
            func=mdp.imu_orientation, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_right")}
        )
        right_tibia_angular_velocity = ObsTerm(
            func=mdp.imu_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_right")}
        )
        right_tibia_linear_acceleration = ObsTerm(
            func=mdp.imu_lin_acc, 
            params={"asset_cfg": SceneEntityCfg("imu_tibia_right")}
        )
        
        # No joint positions or velocities as they are privileged information

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

    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_exp_ballu, 
        weight=1.0, 
        params=
            {
                "command_name": "base_velocity", 
                "std": math.sqrt(0.1)
            }
    )
    track_lin_vel_xy_world_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp_ballu, 
        weight=1.0, 
        params=
            {
                "command_name": "base_velocity", 
                "std": math.sqrt(0.1)
            }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class BALLU_TibiaIMU_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the BALLU robot environment with tibia IMU sensor observations."""

    # Scene settings
    scene: BALLUSceneCfg = BALLUSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (0, 7, 3.0)
        # simulation settings
        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.solver_type = 1 # TGS
