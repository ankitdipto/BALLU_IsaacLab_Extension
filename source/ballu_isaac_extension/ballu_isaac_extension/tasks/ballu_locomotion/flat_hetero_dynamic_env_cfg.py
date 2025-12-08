# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Heterogeneous BALLU Environment with Dynamic Morphology Loading on Flat Terrain

This configuration combines flat_env_cfg.py with dynamic morphology loading
from single_obstacle_hetero_dynamic_env_cfg.py to train a morphology-conditioned
universal controller for fast walking across diverse BALLU morphologies.

Usage:
    # Train a universal controller on flat terrain
    python train.py --task Isc-Vel-BALLU-flat-hetero-dynamic
    
    # Or specify a custom morphology library
    export BALLU_MORPHOLOGY_LIBRARY_PATH=/path/to/morphology/library
    python train.py --task Isc-Vel-BALLU-flat-hetero-dynamic
"""

import math

from ballu_isaac_extension.ballu_assets.ballu_config import (
    get_ballu_hetero_cfg_dynamic,
    has_dynamic_morphology_support,
    BALLU_WALKER_CFG
)
import ballu_isaac_extension.tasks.ballu_locomotion.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

##
# Scene definition
##


@configclass
class BALLUSceneFlatDynamicCfg(InteractiveSceneCfg):
    """Configuration for a BALLU robot scene with dynamic morphology loading on flat terrain."""

    # ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
    )

    # BALLU - will be set dynamically in __post_init__
    robot: ArticulationCfg = None
    # robot: ArticulationCfg = BALLU_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )

    # contact sensors at feet (optional, currently not used in observations)
    contact_forces_tibia = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_(LEFT|RIGHT)", 
        update_period=0.05,  # 20 Hz
        debug_vis=False
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
            lin_vel_x=(0.12, 0.12),  # Constant 0.12 m/s along +x
            lin_vel_y=(0.0, 0.0),    # No y-velocity
            ang_vel_z=(0.0, 0.0),    # No angular velocity
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP - Targetting Motor Joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["MOTOR_LEFT", "MOTOR_RIGHT"],
        scale=3.14159265,
        use_default_offset=False,
        clip = {
           "MOTOR_LEFT": (0.0, 3.14159265),
           "MOTOR_RIGHT": (0.0, 3.14159265)
        }
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state observations
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # Phase of periodic reference trajectory
        phase_of_periodic_reference_traj = ObsTerm(func=mdp.phase_of_periodic_reference_traj, params={"period": 40})

        # Previous action
        last_action = ObsTerm(func=mdp.last_action)

        # Morphology vector - key for universal controller
        morphology_vector = ObsTerm(func=mdp.morphology_vector_priv)

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

    # Reward to encourage progress towards goal
    navigation_reward_l2 = RewTerm(
        func=mdp.navigation_reward_l2,
        weight=1.0,
        params={
            "begin_iter": 200,
            "ramp_width": 300
        }
    )

    # Primary reward - forward velocity tracking
    forward_vel_base = RewTerm(
        func=mdp.forward_velocity_x,
        weight=4.0,
    )

    # Penalty for deviation from straight line
    deviation_from_straight_line = RewTerm(
        func=mdp.deviation_from_straight_line,
        weight=-1.0,
    )

    # Reward to encourage tracking the exact command velocity (exponential)
    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_exp_ballu, 
        weight=0.0, 
        params={
            "command_name": "base_velocity", 
            "std": 0.1
        }
    )

    # World frame velocity tracking (exponential)
    track_lin_vel_xy_world_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp_ballu, 
        weight=0.0,
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
        }
    )

    # L2 velocity tracking
    track_lin_vel_xy_base_l2 = RewTerm(
        func=mdp.track_lin_vel_xy_base_l2,
        weight=0.0,
        params={"command_name": "base_velocity"}
    )

    # Penalize lateral velocity
    lateral_vel_base = RewTerm(
        func=mdp.lateral_velocity_y,
        weight=0.0,
    )

    # Periodic reference trajectory following
    periodic_reference_traj = RewTerm(
        func=mdp.periodic_reference_traj,
        weight=0.0,
        params={"period": 40},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 3.0})
    feet_z_pos_above = DoneTerm(func=mdp.feet_z_pos_above, params={"z_limit": 0.76})


@configclass
class CurriculumsCfg:
    """Curriculums for the MDP."""
    # No curriculum needed for flat terrain
    pass


##
# Environment configuration
##


@configclass
class BalluFlatHeteroDynamicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the BALLU robot environment with dynamic heterogeneous morphology loading on flat terrain."""

    # Scene settings
    scene: BALLUSceneFlatDynamicCfg = BALLUSceneFlatDynamicCfg(num_envs=4096, env_spacing=4.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # Check if dynamic loading is available
        if not has_dynamic_morphology_support():
            raise ImportError(
                "Dynamic morphology loading not available. "
                "Ensure morphology_loader.py is installed."
            )
        
        # Morphology library configuration
        self.morphology_library_name = "hetero_library_tl_fl_hw_n100_11_23_15_48_01"
        # self.morphology_library_name = "test_library"
        self.max_morphologies = None  # None = load all morphologies
        
        # Load robot configuration dynamically
        print(f"\n{'='*80}")
        print(f"Loading dynamic heterogeneous morphology configuration")
        print(f"Environment: Flat terrain for fast walking")
        print(f"Library: {self.morphology_library_name}")
        print(f"Max morphologies: {self.max_morphologies if self.max_morphologies else 'ALL'}")
        print(f"{'='*80}\n")
        
        self.scene.robot = get_ballu_hetero_cfg_dynamic(
            library_name=self.morphology_library_name,
            max_morphologies=self.max_morphologies,
            spring_coeff=0.00507,
            spring_damping=1.0e-3,
            pd_p=0.09,
            pd_d=0.02,
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # general settings
        self.decimation = 10
        self.episode_length_s = 20
        
        # viewer settings
        self.viewer.eye = (2.0, 5.0, 3.0)
        self.viewer.lookat = (2.0, 0.0, 0.3)
        self.viewer.resolution = (1920, 1080)
        
        # simulation settings
        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.solver_type = 1  # Truncated Gauss-Seidel
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1

        # Environment spacing - keep reasonable spacing for visualization
        # (unlike obstacle env which overlaps all environments)
        self.scene.env_spacing = 4.0

