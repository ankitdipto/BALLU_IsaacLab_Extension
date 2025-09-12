# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_REAL_CFG
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
class BALLUSceneCfg(InteractiveSceneCfg):
    """Configuration for a BALLU robot scene."""

    # ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,  # Default: 0.5
                dynamic_friction=0.5,  # Default: 0.5
                restitution=0.0,      # Default: 0.0
                friction_combine_mode="multiply",  # Default: "average"
                restitution_combine_mode="multiply",  # Default: "average"
            ),
        ),
    )
    # obstacle - cuboid
    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.055),
            # Make it collide and fix it in place (kinematic)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            # Use default reasonable friction/restitution
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            # Marble-like visual material
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),  # Light gray marble
                roughness=0.2,  # Slightly glossy
                metallic=0.1,  # Subtle metallic sheen
            ),
        ),
        # Place so it rests on ground (height/2) and rotate 45 deg about z
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.5, 0.0, 0.0275),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # BALLU
    robot: ArticulationCfg = BALLU_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # robot.init_state.pos = (0.0, 0.0, 0.9)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=500.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # contact sensors at feet
    #contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/TIBIA_(LEFT|RIGHT)", 
    #                                  history_length=3, 
    #                                  track_air_time=True)


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
            lin_vel_x=(0.23, 0.23),  # Constant x-velocity of 0.23 m/s
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

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Invalid simulator state (terminal)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    # (3) Root height above hard limit (terminal)
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 3.0})

@configclass
class CurriculumsCfg:
    """Curriculums for the MDP."""

##
# Environment configuration
##


@configclass
class BalluSingleObstacleEnvCfg(ManagerBasedRLEnvCfg): # Renamed class
    """Configuration for the BALLU robot environment with indirect actuation."""

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
        self.viewer.eye = (1.0, 6.0, 2.0)
        self.viewer.lookat = (1.0, 0.0, 1.0)
        self.viewer.resolution = (1920, 1080) # Full HD resolution
        # simulation settings
        self.sim.dt = 1 / 200.0 #160.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        #self.sim.physx.solver_type = 0 # Projected Gauss-Seidel
        self.sim.physx.solver_type = 1 # Truncated Gauss-Seidel
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        tg_cfg = getattr(self.scene.terrain, "terrain_generator", None)
        if getattr(self, "curriculums", None) is not None and getattr(self.curriculums, "terrain_levels", None) is not None:
            if tg_cfg is not None:
                tg_cfg.curriculum = True
        else:
            if tg_cfg is not None:
                tg_cfg.curriculum = False
        # else:
        #   if self.scene.terrain.terrain_generator is not None:
        #       self.scene.terrain.terrain_generator.curriculum = False

# @configclass
# class BalluSingleObstacleEnvCfg_PLAY(BalluSingleObstacleEnvCfg):
#     """Configuration for the BALLU robot environment with indirect actuation."""

#     def __post_init__(self):
#         """Post initialization."""
#         super().__post_init__()
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         self.scene.terrain.max_init_terrain_level = None
#         self.scene.terrain.terrain_generator = BALLU_TERRAINS_CFG_PLAY

#         if self.scene.terrain.terrain_generator is not None:
#             self.scene.terrain.terrain_generator.curriculum = False