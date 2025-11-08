import math

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_REAL_CFG
import ballu_isaac_extension.tasks.ballu_locomotion.mdp as mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class BALLUSceneCfg(InteractiveSceneCfg):
    """Scene configuration with a single per-environment obstacle and IMUs on the electronics boxes."""

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

    robot: ArticulationCfg = BALLU_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 2.0, 0.2),
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
            pos=(1.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    imu_electronics_left = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_LEFT",
        update_period=0.05,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    imu_electronics_right = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_RIGHT",
        update_period=0.05,
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


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
    """Action specifications for the MDP targeting motor joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["MOTOR_LEFT", "MOTOR_RIGHT"],
        scale=math.pi,
        use_default_offset=False,
        clip={
            "MOTOR_LEFT": (0.0, math.pi),
            "MOTOR_RIGHT": (0.0, math.pi),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        distance_to_obstacle = ObsTerm(func=mdp.distance_to_obstacle_priv)
        height_of_obstacle = ObsTerm(func=mdp.height_of_obstacle_in_front_priv)
        robot_pos_w = ObsTerm(func=mdp.root_pos_w)
        goal_pos_w = ObsTerm(func=mdp.goal_location_w_priv)
        base_velocity = ObsTerm(func=mdp.base_lin_vel)
        limb_dist_from_obstacle = ObsTerm(func=mdp.distance_of_limbs_from_obstacle_priv)
        left_electronics_orientation = ObsTerm(
            func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_electronics_left")}
        )
        left_electronics_angular_velocity = ObsTerm(
            func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_electronics_left")}
        )
        left_electronics_linear_acceleration = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_electronics_left")}
        )
        right_electronics_orientation = ObsTerm(
            func=mdp.imu_orientation, params={"asset_cfg": SceneEntityCfg("imu_electronics_right")}
        )
        right_electronics_angular_velocity = ObsTerm(
            func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu_electronics_right")}
        )
        right_electronics_linear_acceleration = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu_electronics_right")}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_ballu_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    position_tracking_l2_singleObj = RewTerm(
        func=mdp.position_tracking_l2_singleObj,
        weight=5.0,
        params={
            "begin_iter": 100,
            "ramp_width": 100,
        },
    )

    high_jump = RewTerm(
        func=mdp.feet_z_pos_exp,
        weight=0.0,
        params={
            "slope": 1.73,
        },
    )

    forward_vel_base = RewTerm(func=mdp.forward_velocity_x, weight=4.0)

    goal_reached_bonus = RewTerm(func=mdp.goal_reached_bonus, weight=0.0)

    track_lin_vel_xy_base_l2 = RewTerm(
        func=mdp.track_lin_vel_xy_base_l2,
        weight=0.0,
        params={"command_name": "base_velocity"},
    )

    track_lin_vel_xy_base_exp = RewTerm(
        func=mdp.track_lin_vel_xy_base_exp_ballu,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    track_lin_vel_xy_world_exp = RewTerm(
        func=mdp.track_lin_vel_xy_world_exp_ballu,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    lateral_vel_base = RewTerm(func=mdp.lateral_velocity_y, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 5.0})
    feet_z_pos_above = DoneTerm(func=mdp.feet_z_pos_above, params={"z_limit": 2.5})


@configclass
class BalluObstacleImuDbgEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment with a per-environment obstacle and electronics IMUs."""

    scene: BALLUSceneCfg = BALLUSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 20

        self.viewer.eye = (1.0 - 5.5 / 1.414, 5.5 / 1.414, 1.5)
        self.viewer.lookat = (1.0, 0.0, 1.0)
        self.viewer.resolution = (1920, 1080)

        self.sim.dt = 1 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.solver_type = 1
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1


