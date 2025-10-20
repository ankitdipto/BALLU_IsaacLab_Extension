from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Root state.
"""


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