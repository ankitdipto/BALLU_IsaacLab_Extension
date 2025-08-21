import torch
from isaaclab.envs import ManagerBasedRLEnv


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

    # Root linear speed threshold in world frame
    root_linear_speed_w = torch.linalg.norm(robot.data.root_lin_vel_w, dim=1)
    root_speed_exceeds_limit = root_linear_speed_w > max_root_speed

    # Invalid if any of the above checks fail
    invalid_mask = (
        (~joint_positions_are_finite)
        | (~joint_velocities_are_finite)
        | (~root_position_is_finite)
        | (~root_quaternion_is_finite)
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
