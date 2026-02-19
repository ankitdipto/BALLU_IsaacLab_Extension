import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils


def invalid_state(env: ManagerBasedRLEnv, max_root_speed: float = 10.0) -> torch.Tensor:
    """Returns a boolean mask per environment for invalid simulator state.

    An environment is marked invalid (True) when any of the following hold:
    - Joint positions contain NaN/Inf
    - Joint velocities contain NaN/Inf
    - Root linear speed in world frame exceeds ``max_root_speed`` (m/s)
    - Root position or quaternion contain NaN/Inf

    Args:
        env: The manager-based RL environment.
        max_root_speed: Maximum allowed root linear speed (m/s) before termination.

    Returns:
        torch.BoolTensor of shape (num_envs,) where True indicates the env should terminate.
    """
    robot = env.scene["robot"]

    # Finite checks per environment
    joint_positions_are_finite = torch.isfinite(robot.data.joint_pos).all(dim=1)
    joint_velocities_are_finite = torch.isfinite(robot.data.joint_vel).all(dim=1)
    root_position_is_finite = torch.isfinite(robot.data.root_pos_w).all(dim=1)
    root_quaternion_is_finite = torch.isfinite(robot.data.root_quat_w).all(dim=1)
    body_link_positions_are_finite = torch.isfinite(robot.data.body_link_pos_w).all(dim = (1, 2))
    body_link_quaternions_are_finite = torch.isfinite(robot.data.body_link_quat_w).all(dim = (1, 2))
    body_link_velocities_are_finite = torch.isfinite(robot.data.body_link_vel_w).all(dim = (1, 2))

    # Root linear speed threshold in world frame
    root_linear_speed_w = torch.linalg.norm(robot.data.root_lin_vel_w, dim=1)
    root_speed_exceeds_limit = root_linear_speed_w > max_root_speed

    # INSERT_YOUR_CODE
    # Diagnostic: print which individual checks are causing invalid_mask=True
    # checks = {
    #     "joint_positions_are_finite": joint_positions_are_finite,
    #     "joint_velocities_are_finite": joint_velocities_are_finite,
    #     "root_position_is_finite": root_position_is_finite,
    #     "root_quaternion_is_finite": root_quaternion_is_finite,
    #     "body_link_positions_are_finite": body_link_positions_are_finite,
    #     "body_link_quaternions_are_finite": body_link_quaternions_are_finite,
    #     "body_link_velocities_are_finite": body_link_velocities_are_finite,
    #     "root_speed_exceeds_limit": root_speed_exceeds_limit,
    # }
    # # If any invalid environment detected, print which check failed for which env
    # if torch.any(
    #     ~joint_positions_are_finite
    #     | ~joint_velocities_are_finite
    #     | ~root_position_is_finite
    #     | ~root_quaternion_is_finite
    #     | ~body_link_positions_are_finite
    #     | ~body_link_quaternions_are_finite
    #     | ~body_link_velocities_are_finite
    #     | root_speed_exceeds_limit
    # ):
    #     num_envs = joint_positions_are_finite.shape[0]
    #     for env_idx in range(num_envs):
    #         for k, v in checks.items():
    #             flag = v[env_idx] if v.shape == (num_envs,) else v[env_idx].item()
    #             if (k == "root_speed_exceeds_limit" and flag) or (k != "root_speed_exceeds_limit" and not flag):
    #                 print(f"[invalid_state] Env {env_idx}: {k} -> {flag}")
    # Invalid if any of the above checks fail
    invalid_mask = (
        (~joint_positions_are_finite)
        | (~joint_velocities_are_finite)
        | (~root_position_is_finite)
        | (~root_quaternion_is_finite)
        | (~body_link_positions_are_finite)
        | (~body_link_quaternions_are_finite)
        | (~body_link_velocities_are_finite)
        | (root_speed_exceeds_limit)
    )

    return invalid_mask


def root_height_above(env: ManagerBasedRLEnv, z_limit: float = 3.0) -> torch.Tensor:
    """Terminate when the articulation root's z-position exceeds the given limit.

    Args:
        env: The manager-based RL environment.
        z_limit: Maximum allowed height (meters) for the root position in world frame.

    Returns:
        torch.BoolTensor of shape (num_envs,) where True indicates termination.
    """
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] > z_limit

def feet_z_pos_above(env: ManagerBasedRLEnv, z_limit: float = 1.5) -> torch.Tensor:
    """Terminate when the feet z position exceeds the given limit."""
    asset = env.scene["robot"]
    tibia_ids, _ = asset.find_bodies("TIBIA_(LEFT|RIGHT)") # (2,)
    tibia_pos_w = asset.data.body_link_pos_w[:,tibia_ids, :] # (num_envs, 2, 3)
    tibia_quat_w = asset.data.body_link_quat_w[:,tibia_ids, :] # (num_envs, 2, 4)
    feet_offset_b = torch.tensor([0.0, 0.38485 + 0.004, 0.0], 
                                device=env.device, dtype=tibia_pos_w.dtype)
    feet_offset_b = feet_offset_b.unsqueeze(0).unsqueeze(0).expand(tibia_pos_w.shape) # (num_envs, 2, 3)
    pose_offset_w = math_utils.quat_apply(tibia_quat_w.reshape(-1, 4), feet_offset_b.reshape(-1, 3)).reshape_as(tibia_pos_w)
    feet_pos_w = tibia_pos_w + pose_offset_w # (num_envs, 2, 3)
    feet_z_pos_w = feet_pos_w[:, :, 2] # (num_envs, 2)
    min_feet_z_pos_w = feet_z_pos_w.min(dim = 1)[0]
    # print("min_feet_z_pos_w: ", min_feet_z_pos_w)
    return min_feet_z_pos_w > z_limit

def robot_crosses_env_boundary(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the robot crosses the env boundary."""
    env_origins = env.scene.env_origins
    robot = env.scene["robot"]
    inter_env_spacing_y = 2.0
    # Check if robot is outside [env_origin_y - inter_env_spacing_y / 2, env_origin_y + inter_env_spacing_y / 2]
    robot_pos_w_y = robot.data.root_pos_w[:, 1]
    return torch.logical_or(robot_pos_w_y < env_origins[:, 1] - inter_env_spacing_y / 2, robot_pos_w_y > env_origins[:, 1] + inter_env_spacing_y / 2)