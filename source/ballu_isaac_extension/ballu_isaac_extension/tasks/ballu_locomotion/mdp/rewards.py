"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast, Any

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def forward_velocity_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward velocity along the x-axis"""
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    Vx = torch.nan_to_num(asset.data.root_lin_vel_b[:, 0], nan=0.0)
    Vx = torch.where(Vx > 0.8, torch.zeros_like(Vx), Vx)
    # Transform negative velocities using exponential function
    Vx = torch.where(Vx < 0.0, torch.exp(Vx) - 1, Vx)
    # Vx = torch.clamp(Vx, min=-0.5) # This component introduced severe learning slowdown in the rough terrain environment. Better to keep it disabled unless absolutely necessary.
    return Vx

# 2\left(e^{2x}-1\right)
def feet_z_pos_exp(env: ManagerBasedRLEnv, slope: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet z position."""
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    tibia_ids, _ = asset.find_bodies("TIBIA_(LEFT|RIGHT)") # (2,)
    tibia_pos_w = asset.data.body_link_pos_w[:,tibia_ids, :] # (num_envs, 2, 3)
    tibia_quat_w = asset.data.body_link_quat_w[:,tibia_ids, :] # (num_envs, 2, 4)
    feet_offset_b = torch.tensor([0.0, 0.38485 + 0.004, 0.0], 
                                device=env.device, dtype=tibia_pos_w.dtype)
    feet_offset_b = feet_offset_b.unsqueeze(0).unsqueeze(0).expand(tibia_pos_w.shape) # (num_envs, 2, 3)
    pose_offset_w = math_utils.quat_apply(tibia_quat_w.reshape(-1, 4), feet_offset_b.reshape(-1, 3)).reshape_as(tibia_pos_w)
    feet_pos_w = tibia_pos_w + pose_offset_w # (num_envs, 2, 3)
    feet_z_pos_w = feet_pos_w[:, :, 2] # (num_envs, 2)
    feet_x_pos_w = feet_pos_w[:, :, 0] # (num_envs, 2)

    min_feet_z_pos_w = feet_z_pos_w.min(dim = 1)[0]
    
    obstacle_spacing_y = -2.0
    difficulty_indices = env.scene.env_origins[:, 1] / obstacle_spacing_y
    # Vectorized indexing: convert obstacle_height_list to tensor and use advanced indexing
    obstacle_height_tensor = torch.tensor(env.obstacle_height_list, device=env.device, dtype=torch.float32)
    difficulty_indices_int = difficulty_indices.long()  # Convert to integer indices
    obstacle_heights_t = obstacle_height_tensor[difficulty_indices_int]

    # print(f"[DEBUG] obstacle_heights_t shape: {obstacle_heights_t.shape}")
    # print(f"[DEBUG] min_feet_z_pos_w shape: {min_feet_z_pos_w.shape}")

    # Check if both feet are in the obstacle region (x between 0.5 and 1.5)
    feet_in_obstacle_region = ((feet_x_pos_w >= 0.5) & (feet_x_pos_w <= 1.5)).all(dim=1)  # (num_envs,)
    min_feet_z_pos_w = torch.where(feet_in_obstacle_region, min_feet_z_pos_w - obstacle_heights_t, min_feet_z_pos_w)
    rew = torch.where(min_feet_z_pos_w > 1.8, 
                                        0.0, 
                                        torch.exp(slope * min_feet_z_pos_w) - 1)
    # rew = torch.exp(slope * min_feet_z_pos_w) - 1
    rew = torch.nan_to_num(rew, nan=0.0)
    # assert rew has no nan
    assert not torch.isnan(rew).any()
    return rew

def position_tracking_l2_singleObj(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
        begin_iter: int|None = None,
        ramp_width: int|None = None,
    ) -> torch.Tensor:
    """Reward position tracking using L2 norm."""
    if begin_iter is not None and env.rsl_rl_iteration < begin_iter:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins
    # Each env has an obstacle of width 2.0 m, so the center of the obstacle is at distance of 1.0 m in the corresponding y-direction.
    # The y-coordinate of goal center is same as obstacle which happens to be same as env_origin_y.
    # The x-coordinate of goal center = env_origin_X + 2.0 m
    goal_pos_w = env_origins[:, :2] + torch.tensor([2.0, 0.0], device=env.device, dtype=env_origins.dtype)
    motor_arm_ids, _ = asset.find_bodies("MOTORARM_(LEFT|RIGHT)") # (2,)
    motorarm_pos_w_XY = asset.data.body_link_pos_w[:, motor_arm_ids, :2] # (num_envs, 2, 2)
    mean_motorarm_pos_w_XY = motorarm_pos_w_XY.mean(dim=1) # (num_envs, 2)
    error = torch.norm(goal_pos_w - mean_motorarm_pos_w_XY, p=2, dim=1)
    rew = 1.0 - 0.33 * error
    rew = torch.nan_to_num(rew, nan=0.0)
    rew = torch.clip(rew, min=0.0)
    if ramp_width is not None and env.rsl_rl_iteration < begin_iter + ramp_width:
        rew = rew * (env.rsl_rl_iteration - begin_iter) / ramp_width
    return rew

def goal_reached_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward goal reached."""
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins
    goal_pos_w = env_origins[:, :2] + torch.tensor([2.0, 0.0], device=env.device, dtype=env_origins.dtype)
    error = torch.norm(goal_pos_w - asset.data.root_pos_w[:, :2], p=2, dim=1)
    rew = torch.where(error < 0.1, 1.0, 0.0)
    return rew

def lateral_velocity_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lateral velocity along the y-axis"""
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    Vy = torch.nan_to_num(asset.data.root_lin_vel_b[:, 1], nan=0.0)
    Vy = torch.clamp(Vy, min=-0.5, max=0.5)
    Vy = torch.abs(Vy)
    return Vy

def track_lin_vel_xy_base_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - \
                    asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # replace NaNs with large finite numbers
    lin_vel_error = torch.nan_to_num(lin_vel_error, nan=1e6)
    
    return 1 - lin_vel_error

def track_lin_vel_xy_base_exp_ballu(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - \
                    asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    # replace NaNs with large finite numbers
    lin_vel_error = torch.nan_to_num(lin_vel_error, nan=1e6)

    return torch.exp(-lin_vel_error / std**2)

def track_lin_vel_xy_world_exp_ballu(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - \
                    asset.data.root_lin_vel_w[:, :2]),
        dim=1,
    )
    # replace NaNs with large finite numbers
    lin_vel_error = torch.nan_to_num(lin_vel_error, nan=1e6)

    return torch.exp(-lin_vel_error / std**2)

def feet_z_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet z position."""
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    tibia_ids, _ = asset.find_bodies("TIBIA_(LEFT|RIGHT)") # (2,)
    tibia_pos_w = asset.data.body_link_pos_w[:,tibia_ids, :] # (num_envs, 2, 3)
    tibia_quat_w = asset.data.body_link_quat_w[:,tibia_ids, :] # (num_envs, 2, 4)
    feet_offset_b = torch.tensor([0.0, 0.38485 + 0.004, 0.0], 
                                device=env.device, dtype=tibia_pos_w.dtype)
    feet_offset_b = feet_offset_b.unsqueeze(0).unsqueeze(0).expand(tibia_pos_w.shape) # (num_envs, 2, 3)
    pose_offset_w = math_utils.quat_apply(tibia_quat_w.reshape(-1, 4), feet_offset_b.reshape(-1, 3)).reshape_as(tibia_pos_w)
    feet_pos_w = tibia_pos_w + pose_offset_w # (num_envs, 2, 3)
    feet_z_pos_w = feet_pos_w[:, :, 2] # (num_envs, 2)
    # print("feet_z_pos_w: ", feet_z_pos_w)
    # print("feet_z_pos_w.min(dim = 1): ", feet_z_pos_w.min(dim = 1)[0])
    return feet_z_pos_w.min(dim = 1)[0]

def feet_air_time_positive_biped(env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # scene.sensors[...] is typed as SensorBase; cast to ContactSensor
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    # Guard: body_ids may be Optional in SceneEntityCfg
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("sensor_cfg.body_ids is None; please configure body_ids for the contact sensor.")
    # Help static type narrowing: from Optional[...] to concrete type
    assert body_ids is not None
    idxs = cast(Any, body_ids)
    # compute the reward
    air_time_arr = contact_sensor.data.current_air_time
    contact_time_arr = contact_sensor.data.current_contact_time
    if air_time_arr is None or contact_time_arr is None:
        raise RuntimeError("ContactSensor data arrays (air/contact time) are not available.")
    air_time = air_time_arr[:, idxs]
    contact_time = contact_time_arr[:, idxs]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("sensor_cfg.body_ids is None; please configure body_ids for the contact sensor.")
    assert body_ids is not None
    idxs = body_ids
    forces_hist = contact_sensor.data.net_forces_w_history
    if forces_hist is None:
        raise RuntimeError("ContactSensor net_forces_w_history is not available.")
    contacts = forces_hist[:, :, idxs, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    asset_body_ids = asset_cfg.body_ids
    if asset_body_ids is None:
        raise RuntimeError("asset_cfg.body_ids is None; please configure body_ids for the robot.")
    assert asset_body_ids is not None
    body_vel = asset.data.body_lin_vel_w[:, asset_body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def joint_torques_out_of_bounds(env: ManagerBasedRLEnv, thresh_min: float, thresh_max: float, 
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation that are out of bounds.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    The joint torques are computed as the product of the joint positions and the joint torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    joint_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # Penalty if joint torques are out of range [thresh_min, thresh_max]
    penalty = torch.where(joint_torques > thresh_max, joint_torques - thresh_max, 0.0) + \
              torch.where(thresh_min > joint_torques, thresh_min - joint_torques, 0.0)
    
    return torch.sum(penalty, dim=1)

def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObjectSpawnerCfg = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])
