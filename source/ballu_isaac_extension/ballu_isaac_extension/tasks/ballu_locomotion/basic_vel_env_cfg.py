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
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_CFG, BALLU_HIP_fixed_CFG

##
# Scene definition
##


@configclass
class BALLUSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = BALLU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
            lin_vel_x=(0.4, 0.4),  # Constant 0.2 m/s along +x
            lin_vel_y=(0.0, 0.0),  # No y-velocity
            ang_vel_z=(0.0, 0.0),  # No angular velocity
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["KNEE_LEFT", "KNEE_RIGHT"],
        scale=1.0, #0.5,
        use_default_offset=True,
        clip = {
            "KNEE_LEFT": (0, 1.74),
            "KNEE_RIGHT": (0, 1.74)
        }
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

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

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Joint position out of limit
    #joint_pos_out_of_limit = DoneTerm(
    #    func=mdp.joint_pos_out_of_limit,
    #    params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4, 5,
    #                                                            6])}
    #)

    # (3) Joint velocity out of limit
    #joint_vel_out_of_limit = DoneTerm(
    #    func=mdp.joint_vel_out_of_manual_limit,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", joint_ids=[0, 1, 2, 3, 4,
    #                                                        5, 6]),
    #        "max_velocity": 100.0
    #    }
    #)

##
# Environment configuration
##


@configclass
class BALLUEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the BALLU robot environment."""

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
        self.decimation = 10 #8
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (0, 7, 3.0)
        #self.viewer.resolution = (1280, 720)
        # simulation settings
        self.sim.dt = 1 / 200.0 #160.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
