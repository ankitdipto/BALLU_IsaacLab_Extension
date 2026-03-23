"""BALLU ramp locomotion task with dynamic heterogeneous morphology loading.

Extends ramp_hetero_general_env_cfg.py to load many kinematic designs from a
library directory.  Deterministic spawning (random_choice=False, the default)
assigns env i → morphology i % N via Isaac Lab's MultiUsdFileCfg round-robin,
making it suitable for PEC kinematic evaluation where we must know which design
lives in which parallel environment.

Usage (training — stochastic):
    env_cfg = BalluRampHeteroDynamicEnvCfg()
    env_cfg.random_choice = True

Usage (eval — deterministic):
    env_cfg = BalluRampHeteroDynamicEnvCfg()   # random_choice=False by default
"""

import json as _json
import os as _os

from ballu_isaac_extension.ballu_assets.ballu_config import (
    get_ballu_hetero_cfg_dynamic,
    has_dynamic_morphology_support,
)
from ballu_isaac_extension.ballu_assets.morphology_loader import create_hetero_config
from ballu_isaac_extension.tasks.ballu_locomotion.ramp_spawner import TriangularRampCfg

# Reuse all MDP configs from the single-morphology ramp hetero env
from ballu_isaac_extension.tasks.ballu_locomotion.ramp_hetero_general_env_cfg import (
    ConstantVelCommandCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumsCfg,
)

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


##
# Scene definition
##


@configclass
class BALLURampHeteroDynamicSceneCfg(InteractiveSceneCfg):
    """Scene for dynamic heterogeneous ramp locomotion.

    Robot is left as None and populated in BalluRampHeteroDynamicEnvCfg.__post_init__
    so that the MultiUsdFileCfg is built from the library at config time.
    Ramps are also added dynamically in __post_init__.
    """

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

    # Robot — set dynamically in __post_init__
    robot: ArticulationCfg = None

    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

    # Contact sensors at feet
    contact_forces_tibia = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ELECTRONICS_(LEFT|RIGHT)",
        update_period=0.05,  # 20 Hz
        debug_vis=True,
    )

    # Ramps — populated dynamically in __post_init__
    ramps = None


##
# Environment configuration
##


@configclass
class BalluRampHeteroDynamicEnvCfg(ManagerBasedRLEnvCfg):
    """BALLU ramp locomotion with dynamic multi-morphology loading.

    Identical MDP to BalluSingleRampHeteroEnvCfg but the robot articulation is
    built from a full morphology library (MultiUsdFileCfg).  The key difference
    is random_choice=False (default) which gives deterministic round-robin
    assignment: env i spawns morphology i % N_morphologies.

    Set random_choice=True before construction for stochastic pretraining.
    """

    # Scene — replicate_physics=False required for per-env different USD assets
    scene: BALLURampHeteroDynamicSceneCfg = BALLURampHeteroDynamicSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: ConstantVelCommandCfg = ConstantVelCommandCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    # Morphology library — matches the ramp PEC config
    morphology_library_name: str = "hetero_library_hvyBloon_ramp_lab03.17.2026"
    # Set to an integer to cap the number of loaded morphologies (None = all)
    max_morphologies: int = None
    # False → deterministic round-robin (env i → morphology i % N)
    # True  → random per-env selection (for stochastic pretraining)
    random_choice: bool = False

    def __post_init__(self) -> None:
        """Post-init: load morphology library or ordered USD list, build ramp array.

        Two modes:
        - BALLU_USD_ORDER_FILE set (3D PEC mode): reads an ordered JSON list of
          USD paths; env i → usd_paths[i].  Used during PEC training and eval so
          each parallel environment gets its designated design.
        - Otherwise (legacy library mode): loads a full morphology library via
          get_ballu_hetero_cfg_dynamic with optional random_choice.
        """
        order_file = _os.environ.get("BALLU_USD_ORDER_FILE")

        if order_file:
            # 3D PEC mode: deterministic per-env USD assignment.
            with open(order_file) as f:
                usd_paths = _json.load(f)

            morphologies = [{"usd_path": p} for p in usd_paths]
            self.scene.robot = create_hetero_config(
                morphologies=morphologies,
                spring_coeff=0.0807,
                spring_damping=0.001,
                pd_p=1.00,
                pd_d=0.08,
                init_pos=(0.0, 0.0, 1.2),
                random_choice=False,
            ).replace(prim_path="{ENV_REGEX_NS}/Robot")

            print(f"\n{'='*80}")
            print(f"BALLU_USD_ORDER_FILE mode: {len(usd_paths)} ordered USD(s) loaded")
            print(f"  File: {order_file}")
            print(f"{'='*80}\n")

        else:
            # Legacy library-based mode.
            if not has_dynamic_morphology_support():
                raise ImportError(
                    "Dynamic morphology loading not available. "
                    "Ensure morphology_loader.py is installed."
                )

            print(f"\n{'='*80}")
            print("Loading dynamic heterogeneous ramp morphology configuration")
            print(f"Library:        {self.morphology_library_name}")
            print(f"Max morphs:     {self.max_morphologies if self.max_morphologies else 'ALL'}")
            print(f"Random choice:  {self.random_choice}  "
                  f"({'stochastic' if self.random_choice else 'deterministic round-robin'})")
            print(f"{'='*80}\n")

            self.scene.robot = get_ballu_hetero_cfg_dynamic(
                library_name=self.morphology_library_name,
                max_morphologies=self.max_morphologies,
                spring_coeff=0.0807,
                spring_damping=0.001,
                pd_p=1.00,
                pd_d=0.08,
                init_pos=(0.0, 0.0, 1.2),
                random_choice=self.random_choice,
            ).replace(prim_path="{ENV_REGEX_NS}/Robot")

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

        # Pack all env origins together (global ramps serve all envs)
        self.scene.env_spacing = 2.5e-4

        # --- Ramp array parameters (identical to BalluSingleRampHeteroEnvCfg) ---
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
                            diffuse_color=(0.2, 0.2, 0.2),
                            roughness=0.3,
                            metallic=0.6,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
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
