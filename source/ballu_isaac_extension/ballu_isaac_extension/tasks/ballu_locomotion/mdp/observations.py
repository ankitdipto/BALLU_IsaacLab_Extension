from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.managers import SceneEntityCfg
from .geometry_utils import *
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Root state.
"""
MORPH_VECTOR: torch.Tensor | None = None
SPRING_COEFF: torch.Tensor | None = None
BUOY_MASSES: torch.Tensor | None = None

def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")) -> torch.Tensor:
    """Feet air time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return air_time

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")) -> torch.Tensor:
    """Feet contact time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return contact_time

def distance_to_obstacle_priv(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Distance to obstacle"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins
    # The obstacle edge starts at x = 0.5m from the env origin
    obstacle_edge_x = env_origins[:, 0] + 0.5
    distance_to_obstacle = obstacle_edge_x - asset.data.root_pos_w[:, 0]
    # Ensure shape is (num_envs, 1) for concatenation compatibility
    return distance_to_obstacle.unsqueeze(-1)

def height_of_obstacle_in_front_priv(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Height of obstacle in front"""
    # extract the used quantities (to enable type-hinting)
    env_origins = env.scene.env_origins
    # Inter-obstacle spacing is 2.0 m
    obstacle_spacing_y = -2.0
    # We need to extract the difficulty index from the location of the env origin
    difficulty_indices = env_origins[:, 1] / obstacle_spacing_y
    # We need to extract the obstacle height from the difficulty index
    all_obstacle_heights_t = torch.tensor(env.obstacle_height_list, device=env.device, dtype=torch.float32)
    difficulty_indices_int = difficulty_indices.long()
    obstacle_heights_t = all_obstacle_heights_t[difficulty_indices_int]
    # Ensure shape is (num_envs, 1) for concatenation compatibility
    return obstacle_heights_t.unsqueeze(-1)

def distance_of_limbs_from_obstacle_priv(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Distance of limbs from obstacle"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins
    # The obstacle edge starts at x = 0.5m from the env origin
    obstacle_edge_x = env_origins[:, 0] + 0.5  # (num_envs,)
    tibia_ids, _ = asset.find_bodies("TIBIA_(LEFT|RIGHT)")  # (2,)
    tibia_pos_w = asset.data.body_link_pos_w[:, tibia_ids, :]  # (num_envs, 2, 3)
    tibia_pos_w_x = tibia_pos_w[:, :, 0]  # (num_envs, 2)
    obstacle_edge_x_repeated = obstacle_edge_x.unsqueeze(-1).repeat(1, 2)  # (num_envs, 2)
    tibia_dist_from_obstacle = obstacle_edge_x_repeated - tibia_pos_w_x  # (num_envs, 2)
    return tibia_dist_from_obstacle

def goal_location_w_priv(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Goal location"""
    # extract the used quantities (to enable type-hinting)
    env_origins = env.scene.env_origins
    # The goal is at x = 2.0m from the env origin
    goal_pos_w = env_origins[:, :2] + torch.tensor([2.0, 0.0], device=env.device, dtype=env_origins.dtype) # shape: (num_envs, 2)
    goal_pos_w = goal_pos_w - env_origins[:, :2] # shape: (num_envs, 2) - subtract the env origin from the goal position
    # Ensure shape is (num_envs, 2) for concatenation compatibility
    assert goal_pos_w.shape == (env.num_envs, 2), f"Goal position shape mismatch, expected (num_envs, 2), got {goal_pos_w.shape}"
    return goal_pos_w

def generated_spring_coeff(env: ManagerBasedRLEnv, low: float = 1e-3, high: float = 1e-2) -> torch.Tensor:
    """Generated spring coefficient"""
    # extract the used quantities (to enable type-hinting)
    spring_coeff = (high - low) * torch.rand(env.num_envs, 1, dtype=torch.float32) + low
    assert spring_coeff.shape == (env.num_envs, 1), f"Spring coefficient shape mismatch, expected (num_envs, 1), got {spring_coeff.shape}"
    return spring_coeff

def morphology_vector_priv(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Morphology vector"""
    # extract the used quantities (to enable type-hinting)
    global MORPH_VECTOR
    global SPRING_COEFF
    global BUOY_MASSES
    if BUOY_MASSES is None:
        # Try to use buoyancy masses already computed on the environment. If they
        # do not exist yet (during manager/observation initialization), compute
        # them here and store back on the environment.
        balloon_masses = getattr(env, "balloon_buoyancy_mass_t", None)
        if balloon_masses is None:
            robot: Articulation = env.scene[asset_cfg.name]
            robot_total_mass = robot.data.default_mass.sum(dim=1)
            # Prefer a GCR range if provided, otherwise fall back to a fixed GCR.
            gcr_range = getattr(env, "GCR_range", None)
            if gcr_range is not None:
                gcr_tensor = (
                    torch.rand(env.scene.num_envs, 1, device=env.device)
                    * (gcr_range[1] - gcr_range[0])
                    + gcr_range[0]
                )
                balloon_masses = gcr_tensor * robot_total_mass.mean().item()
            else:
                gcr = getattr(env, "GCR", 0.84)
                balloon_mass = gcr * robot_total_mass.mean().item()
                balloon_masses = torch.full(
                    (env.scene.num_envs, 1),
                    balloon_mass,
                    device=env.device,
                    dtype=robot_total_mass.dtype,
                )
            # Cache on env so subsequent physics code can use the same values.
            env.balloon_buoyancy_mass_t = balloon_masses
        BUOY_MASSES = env.balloon_buoyancy_mass_t.clone().to('cpu')
        print(f"[DEBUG] Balloon buoyancy masses: {BUOY_MASSES}")

    if SPRING_COEFF is None:
        if env.spcf_range is not None:
            SPRING_COEFF = generated_spring_coeff(env, low=env.spcf_range[0], high=env.spcf_range[1])
        else:
            SPRING_COEFF = torch.full((env.num_envs, 1), env.spcf, dtype=torch.float32)
        assert SPRING_COEFF.shape == (env.num_envs, 1), f"Spring coefficient shape mismatch, expected (num_envs, 1), got {SPRING_COEFF.shape}"
        print(f"[DEBUG] Spring coefficient: {SPRING_COEFF}")
        robot: Articulation = env.scene[asset_cfg.name]
        # `robot.actuators["knee_effort_actuators"]` is a SpringPDActuator instance,
        # so we assign to its attribute directly instead of treating it like a dict.
        knee_actuators = robot.actuators["knee_effort_actuators"]
        knee_actuators.spring_coeff = SPRING_COEFF.clone().to(env.device)

    if MORPH_VECTOR is None:
        all_dims = get_robot_dimensions(env_indices=slice(0, env.num_envs))
        morphology_vector = torch.cat([
            all_dims.pelvis.height.unsqueeze(-1),
            all_dims.femur_left.height.unsqueeze(-1),
            all_dims.femur_right.height.unsqueeze(-1),
            all_dims.tibia_left.height.unsqueeze(-1),
            all_dims.tibia_right.height.unsqueeze(-1),
            all_dims.electronics_left.cylinder.height.unsqueeze(-1),
            all_dims.electronics_right.cylinder.height.unsqueeze(-1),
            all_dims.electronics_left.sphere.radius.unsqueeze(-1),
            all_dims.electronics_right.sphere.radius.unsqueeze(-1),
            SPRING_COEFF,
            BUOY_MASSES
        ], dim=-1).to(env.device)
        MORPH_VECTOR = morphology_vector
    # print(f"Morphology vector shape: {MORPH_VECTOR.shape}")
    return MORPH_VECTOR

def imu_information_combined(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """IMU information combined"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_ids, _ = asset.find_bodies("ELECTRONICS_(LEFT|RIGHT)")  # (2,)
    imu_orient_w = asset.data.body_link_quat_w[:, feet_ids, :]  # (num_envs, 2, 4)
    imu_ang_vel_w = asset.data.body_link_ang_vel_w[:, feet_ids, :]  # (num_envs, 2, 3)
    imu_lin_acc_w = asset.data.body_lin_acc_w[:, feet_ids, :]  # (num_envs, 2, 3)
    # Convert world-frame signals to the respective body (IMU) frame
    imu_ang_vel_b = math_utils.quat_rotate_inverse(imu_orient_w, imu_ang_vel_w)
    imu_lin_acc_b = math_utils.quat_rotate_inverse(imu_orient_w, imu_lin_acc_w)
    assert imu_orient_w.shape == (env.num_envs, 2, 4), f"IMU orientation shape mismatch, expected (num_envs, 2, 4), got {imu_orient_w.shape}"
    assert imu_ang_vel_b.shape == (env.num_envs, 2, 3), f"IMU angular velocity shape mismatch, expected (num_envs, 2, 3), got {imu_ang_vel_b.shape}"
    assert imu_lin_acc_b.shape == (env.num_envs, 2, 3), f"IMU linear acceleration shape mismatch, expected (num_envs, 2, 3), got {imu_lin_acc_b.shape}"
    
    imu_information_combined = torch.cat([imu_orient_w, imu_ang_vel_b, imu_lin_acc_b], dim=-1)
    # I need to flatten this tensor from (N, 2, 10) to (N, 20)
    imu_information_combined_flattened = imu_information_combined.view(env.num_envs, -1)
    assert imu_information_combined_flattened.shape == (env.num_envs, 20), f"IMU information combined shape mismatch, expected (num_envs, 20), got {imu_information_combined_flattened.shape}"
    # assert imu_lin_acc_b.allclose(imu_lin_acc_w), f"IMU linear acceleration mismatch, expected {imu_lin_acc_w}, got {imu_lin_acc_b}"
    return imu_information_combined_flattened

def phase_of_periodic_reference_traj(env: ManagerBasedRLEnv, period: int) -> torch.Tensor:
    """Phase of periodic reference trajectory"""
    # extract the used quantities (to enable type-hinting)
    try:
        curr_env_step = env.episode_length_buf
    except AttributeError:
        curr_env_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = curr_env_step % period
    # print(f"Phase of periodic reference trajectory: {phase}")
    return phase.unsqueeze(-1)