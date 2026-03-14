"""BALLU ramp locomotion task — single triangular ramp with curriculum.

Curriculum:
    75 triangular ramps of increasing height are placed globally along the -Y
    axis at 2 m intervals.  Each env's Y-origin selects which ramp it faces.
    On episode reset:
        - robot x > 2.0 m  →  upgrade (harder ramp, origin_y -= 2.0)
        - robot x < 0.5 m  →  downgrade (easier ramp, origin_y += 2.0)

    Ramp heights: h_i = 0.001 + i * 0.020 m  (level 0: ~flat, level 74: ~45°)
"""

import math

from ballu_isaac_extension.ballu_assets.ballu_config import BALLU_REAL_CFG
import ballu_isaac_extension.tasks.ballu_locomotion.mdp as mdp
from ballu_isaac_extension.tasks.ballu_locomotion.ramp_spawner import TriangularRampCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##


@configclass
class BALLURampSceneCfg(InteractiveSceneCfg):
    """Scene for BALLU ramp locomotion: ground plane + globally spawned ramps."""

    # Ground plane
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(300.0, 300.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
    )

    # BALLU robot
    robot: ArticulationCfg = BALLU_REAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

    # Sky light
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=500.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )

    # Contact sensors at feet (ELECTRONICS bodies act as foot contacts)
    contact_forces_tibia = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_(LEFT|RIGHT)",
        update_period=0.05,  # 20 Hz
        debug_vis=True,
    )

    # Ramps are added dynamically in BalluSingleRampEnvCfg.__post_init__
    ramps = None


##
# MDP specifications
##


@configclass
class ConstantVelCommandCfg:
    """Constant forward velocity command (0.23 m/s along x)."""

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
    """Motor joint position actions (2-D: MOTOR_LEFT, MOTOR_RIGHT ∈ [0, π])."""

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
    """Observations for the ramp locomotion task.

    Identical to the obstacle stepping task but with all obstacle-specific terms
    removed and the goal position updated to x = 2.5 m.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

        # Joint state
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        # Robot root position in env frame
        robot_pos_w = ObsTerm(func=mdp.root_pos_w)

        # Base linear velocity (body frame)
        base_velocity = ObsTerm(func=mdp.base_lin_vel)

        # Privileged IMU readings from physics API
        # imu_info_from_api = ObsTerm(
        #     func=mdp.imu_information_combined,
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        # Phase of periodic reference trajectory (period = 40 control steps)
        # phase_of_periodic_reference_traj = ObsTerm(
        #     func=mdp.phase_of_periodic_reference_traj,
        #     params={"period": 40},
        # )

        # Previous action
        last_action = ObsTerm(func=mdp.last_action)

        # Goal at x = 2.5 m in env frame (constant [2.5, 0.0])
        goal_pos_ramp = ObsTerm(func=mdp.goal_location_ramp_w_priv)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset event: return robot to default state."""

    # reset_ballu_to_default = EventTerm(
    #    func=mdp.reset_scene_to_default,
    #    mode="reset",
    # )

    # TODO: Uncomment this block before next training experiments
    reset_ballu_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi/6, math.pi/6),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_ballu_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for ramp locomotion.

    Three active signals:
        1. forward_vel  — immediate forward-progress reward (w=3)
        2. feet_lift    — feet height reward with no obstacle region logic (w=1)
        3. goal_track   — L2 distance to ramp goal at x=2.5, ramped in (w=5)
    """

    # Forward progress (caps at 0.8 m/s; negative vel → exponential penalty)
    forward_vel = RewTerm(func=mdp.forward_velocity_x, weight=3.0)

    # Encourage lifting feet — exponential of min feet height, no region check
    feet_lift = RewTerm(
        func=mdp.feet_z_pos_exp_flat,
        weight=0.0, #1.0,
        params={"slope": 1.73},
    )

    # Navigate to goal at x = 2.5 m; activates at iter 100, ramps over 100 iters
    position_tracking_ramp = RewTerm(
        func=mdp.position_tracking_l2_ramp,
        weight=5.0,
        params={"begin_iter": 100, "ramp_width": 100},
    )


@configclass
class TerminationsCfg:
    """Termination conditions — identical to single obstacle task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"max_root_speed": 10.0})
    root_height_above = DoneTerm(func=mdp.root_height_above, params={"z_limit": 5.0})
    feet_z_pos_above = DoneTerm(func=mdp.feet_z_pos_above, params={"z_limit": 2.5})


@configclass
class CurriculumsCfg:
    """Ramp height curriculum: shift env Y-origin to select harder/easier ramps."""

    ramp_height_levels = CurrTerm(
        func=mdp.ramp_height_levels_same_row, 
        params={"upgrade_threshold": 0.8, "downgrade_threshold": 0.53}
    )


##
# Environment configuration
##


@configclass
class BalluSingleRampEnvCfg(ManagerBasedRLEnvCfg):
    """BALLU environment for ramp locomotion with progressive curriculum.

    Scene layout (world frame)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    75 triangular ramp prisms are placed globally along the -Y axis.
    Ramp i occupies:
        x ∈ [0.5, 2.0]  (base_len = 1.5 m)
        y ∈ [-i*2 - 1, -i*2 + 1]  (width = 2 m)
        z ∈ [0, h_i]  where h_i = 0.001 + i*0.020 m

    All environments share the same world-frame ramp array; each env's Y-origin
    determines which ramp it is currently facing.

    Curriculum thresholds
    ~~~~~~~~~~~~~~~~~~~~~
        Upgrade:   robot x > 2.0 m after episode end
        Downgrade: robot x < 0.5 m after episode end
        Warmup:    disabled for first 100 PPO iterations
    """

    scene: BALLURampSceneCfg = BALLURampSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self) -> None:
        """Post-init: simulation settings + procedural ramp generation."""
        # --- General settings ---
        self.decimation = 10              # 20 Hz control (physics at 200 Hz)
        self.episode_length_s = 20.0

        # --- Viewer ---
        self.viewer.eye = (1.0 - 5.5 / 1.414, 5.5 / 1.414, 1.5)
        self.viewer.lookat = (1.0, 0.0, 1.0)
        self.viewer.resolution = (1920, 1080)

        # --- Simulation ---
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.solver_type = 1           # Truncated Gauss-Seidel
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 1

        # All env origins overlap (physics replication separates envs)
        self.scene.env_spacing = 2.5e-4

        # --- Ramp array parameters ---
        ramp_num: int = 75
        ramp_base_len: float = 1.5       # ramp spans x = 0.5 to 2.0 m in world
        ramp_width_y: float = 2.0
        ramp_height_base: float = 0.001
        ramp_height_delta: float = 0.020  # +2 cm per curriculum level
        ramp_start_x: float = 0.5
        ramp_spacing_y: float = 2.0

        for i in range(ramp_num):
            height_i = ramp_height_base + i * ramp_height_delta
            y_center = -i * ramp_spacing_y

            setattr(
                self.scene,
                f"ramp_{i}",
                AssetBaseCfg(
                    prim_path=f"/World/ramp_{i}",
                    collision_group=-1,
                    spawn=TriangularRampCfg(
                        base_len=ramp_base_len,
                        height=height_i,
                        width_y=ramp_width_y,
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True
                        ),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            static_friction=0.8,
                            dynamic_friction=0.8,
                        ),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.2, 0.2),  # dark gray
                            roughness=0.3,
                            metallic=0.6,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        # Place ramp so that local x=0 lands at world x=ramp_start_x
                        pos=(ramp_start_x, y_center, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            )
            self.obstacle_height_list.append(height_i)

        # Disable terrain-generator curriculum (ramps use custom origin shifting)
        tg_cfg = getattr(self.scene.terrain, "terrain_generator", None)
        if tg_cfg is not None:
            tg_cfg.curriculum = False
