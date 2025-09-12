from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
    from isaaclab.assets import Articulation


def threshold_based_verification(
    env: ManagerBasedRLEnv | DirectRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_x: float = 1.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate success for the single-obstacle BALLU task.

    Success is True for env k if the robot's local x-position (world position minus env origin)
    is at least ``threshold_x`` meters. Returns a boolean tensor of shape (num_envs,) indicating
    success per environment, and the final world positions tensor of shape (num_envs, 3).

    Args:
        env: The running manager-based RL environment.
        asset_cfg: Scene entity config pointing to the robot articulation in the scene.
        threshold_x: Local x threshold for success (in meters). Default is 1.7 m.

    Returns:
        A tuple ``(success, final_world_positions)`` where:
        - success: torch.BoolTensor of shape (num_envs,)
        - final_world_positions: torch.FloatTensor of shape (num_envs, 3)
    """
    # Access robot articulation and positions
    robot: "Articulation" = env.scene[asset_cfg.name]
    final_world_positions: torch.Tensor = robot.data.root_pos_w  # (num_envs, 3)

    # Convert to local/env-relative coordinates
    env_origins: torch.Tensor = env.scene.env_origins  # (num_envs, 3)
    local_positions: torch.Tensor = final_world_positions - env_origins

    # Success if local x >= threshold_x
    success: torch.Tensor = local_positions[:, 0] >= threshold_x

    return success, local_positions

