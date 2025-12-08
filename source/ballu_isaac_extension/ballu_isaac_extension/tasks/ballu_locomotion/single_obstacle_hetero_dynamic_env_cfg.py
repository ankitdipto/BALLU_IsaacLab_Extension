# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Heterogeneous BALLU Environment with Dynamic Morphology Loading

This configuration extends single_obstacle_hetero_env_cfg.py to support
dynamic loading of 100+ morphologies from a library directory.

Usage:
    # Set environment variable to specify library
    export BALLU_MORPHOLOGY_LIBRARY_PATH=/path/to/morphology/library
    
    # Or use default library location
    python train.py --task Isc-Vel-BALLU-1-obstacle-hetero-dynamic
"""

import math

from ballu_isaac_extension.ballu_assets.ballu_config import (
    get_ballu_hetero_cfg_dynamic,
    has_dynamic_morphology_support,
    BALLU_REAL_CFG
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class BALLUSceneDynamicCfg(InteractiveSceneCfg):
    """Configuration for a BALLU robot scene with dynamic morphology loading."""

    # ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
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
    # robot: ArticulationCfg = BALLU_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

    # contact sensors at feet
    contact_forces_tibia = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_(LEFT|RIGHT)", 
        update_period=0.05,  # 20 Hz
        debug_vis=True
    )

    # Generated obstacles are added as individual AssetBaseCfg entries in __post_init__
    obstacles = None


@configclass
class ConstantVelCommandCfg:
    """Command specifications for the MDP with constant velocity."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.23, 0.23),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
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

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        distance_to_obstacle = ObsTerm(func=mdp.distance_to_obstacle_priv)
        height_of_obstacle = ObsTerm(func=mdp.height_of_obstacle_in_front_priv)

        # Where is the robot wrt environment origin i.e. environment frame?
        robot_pos_w = ObsTerm(func=mdp.root_pos_w)

        # Where is the goal wrt environment origin i.e. environment frame?
        goal_pos_w = ObsTerm(func=mdp.goal_location_w_priv)

        # How far are the limbs from the obstacle?
        limb_dist_from_obstacle = ObsTerm(func=mdp.distance_of_limbs_from_obstacle_priv)

        # What action did the robot take last?
        last_action = ObsTerm(func=mdp.last_action)

        # Morphology vector
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

    # Primary reward - reach the goal
    position_tracking_l2_singleObj = RewTerm(
        func=mdp.position_tracking_l2_singleObj,
        weight=5.0,
        params={
            "begin_iter": 100,
            "ramp_width": 100
        }
    )

    # Shaping reward - jump to clear the obstacle
    high_jump = RewTerm(
        func=mdp.feet_z_pos_exp,
        weight=1.0,
        params={
            "slope": 1.73
        }
    )

    # Reward to encourage tracking the command direction
    forward_vel_base = RewTerm(
        func=mdp.forward_velocity_x,
        weight=3.0,
    )

    # Sparse reward to encourage reaching the goal
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=0.0,
    )

    # Reward to encourage tracking the command velocity
    track_lin_vel_xy_base_l2 = RewTerm(
        func=mdp.track_lin_vel_xy_base_l2,
        weight=0.0,
        params={"command_name": "base_velocity"}
    )

    # Rewards to encourage tracking the exact command velocity
    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_exp_ballu, 
        weight=0.0, 
        params={
            "command_name": "base_velocity", 
            "std": 0.5
        }
    )
    track_lin_vel_xy_world_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp_ballu, 
        weight=0.0,
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
        }
    )

    # Penalize lateral velocity
    lateral_vel_base = RewTerm(
        func=mdp.lateral_velocity_y,
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 5.0})
    feet_z_pos_above = DoneTerm(func=mdp.feet_z_pos_above, params={"z_limit": 2.5})


@configclass
class CurriculumsCfg:
    """Curriculums for the MDP."""
    obstacle_height_levels_custom = CurrTerm(func=mdp.obstacle_height_levels_same_row)


##
# Environment configuration
##


@configclass
class BalluSingleObstacleHeteroDynamicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the BALLU robot environment with dynamic heterogeneous morphology loading."""

    # Scene settings
    scene: BALLUSceneDynamicCfg = BALLUSceneDynamicCfg(num_envs=4096, env_spacing=4.0, replicate_physics=False)
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
        
        self.morphology_library_name = "hetero_library_20251115_215756"
        # self.morphology_library_name = "test_library"
        self.max_morphologies = None # None = load all morphologies
        # Load robot configuration dynamically
        print(f"\n{'='*80}")
        print(f"Loading dynamic heterogeneous morphology configuration")
        print(f"Library: {self.morphology_library_name}")
        print(f"Max morphologies: {self.max_morphologies if self.max_morphologies else 'ALL'}")
        print(f"{'='*80}\n")
        
        self.scene.robot = get_ballu_hetero_cfg_dynamic(
            library_name=self.morphology_library_name,
            max_morphologies=self.max_morphologies
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # general settings
        self.decimation = 10
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (1.0 - 5.5/1.414, 5.5/1.414 - 0 * 2.0, 1.5)
        self.viewer.lookat = (1.0, 0.0 - 0 * 2.0, 1.0)
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.solver_type = 1  # Truncated Gauss-Seidel
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1

        # Place all env origins at (0, 0, 0)
        self.scene.env_spacing = 2.5e-4
        
        # --- Obstacle array parameters (user-configurable) ---
        obstacle_num: int = 75
        obstacle_size_x: float = 1.0
        obstacle_size_y: float = 2.0
        obstacle_base_height: float = 0.001
        obstacle_growth_delta: float = 0.010
        obstacle_spacing_y: float = 2.0
        obstacle_x_line: float = 1.0
        obstacle_y_start: float = 0.0
        
        # Build global obstacles as extras (not replicated per env)
        if obstacle_num > 0:
            base_h = float(obstacle_base_height)
            sx = float(obstacle_size_x)
            sy = float(obstacle_size_y)
            x_line = float(obstacle_x_line)
            y0 = float(obstacle_y_start)
            dy = float(obstacle_spacing_y)

            for i in range(int(obstacle_num)):
                height_i = base_h + i * obstacle_growth_delta
                pos_i = (x_line, y0 - i * dy, height_i / 2.0)
                name = f"obstacle_{i}"
                setattr(
                    self.scene,
                    name,
                    AssetBaseCfg(
                        prim_path=f"/World/{name}",
                        collision_group=-1,
                        spawn=sim_utils.CuboidCfg(
                            size=(sx, sy, height_i),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            physics_material=sim_utils.RigidBodyMaterialCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.2, 0.2, 0.2),
                                roughness=0.2,
                                metallic=0.1,
                            ),
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(
                            pos=pos_i,
                            rot=(1.0, 0.0, 0.0, 0.0),
                        ),
                    ),
                )
                self.obstacle_height_list.append(height_i)

        # check if terrain levels curriculum is enabled
        tg_cfg = getattr(self.scene.terrain, "terrain_generator", None)
        if getattr(self, "curriculums", None) is not None and getattr(self.curriculums, "terrain_levels", None) is not None:
            if tg_cfg is not None:
                tg_cfg.curriculum = True
        else:
            if tg_cfg is not None:
                tg_cfg.curriculum = False

