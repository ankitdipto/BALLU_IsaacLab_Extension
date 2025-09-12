from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def scale_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    final_weight: float,
    global_start_step: int,
    global_stop_step: int,
):
    """Curriculum that scales a reward weight linearly given over a number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        final_weight: The weight of the reward term at the end of the scaling.
        global_start_step: The global step at which the scaling starts.
        global_stop_step: The global step at which the scaling stops.
    """
    #print(f"[DEBUG] Updating curriculum for term: {term_name} at step: {env.common_step_counter}")
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    total_env_steps = env.num_envs * env.common_step_counter
    
    if total_env_steps < global_start_step:
        term_cfg.weight = 0.0

    elif total_env_steps >= global_start_step and total_env_steps <= global_stop_step:
        # update term settings
        term_cfg.weight = final_weight * (env.common_step_counter - global_start_step) / (global_stop_step - global_start_step)

    elif total_env_steps > global_stop_step:
        # update term settings
        term_cfg.weight = final_weight

    env.reward_manager.set_term_cfg(term_name, term_cfg)
    #if term_name == "action_rate_l2" and env.common_step_counter > global_start_step:
    #    print(f"[DEBUG] Action rate l2: {env.reward_manager.get_term_cfg(term_name)}")

def terrain_levels_ballu(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> float:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    assert (
        env.scene.terrain is not None
    ), "There must be a terrain in the scene for the terrain levels curriculum to work."
    terrain: TerrainImporter = env.scene.terrain

    # ensure env_ids is a tensor (type-checker + API expects Tensor)
    env_ids_t = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    # compute the distance the robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    # NOTE: terrain.cfg.terrain_generator is Optional; assert at runtime to satisfy type-checker and avoid None access
    assert (
        terrain.cfg.terrain_generator is not None
    ), "Terrain type must be 'generator' with a valid terrain_generator config for curriculum to work."
    
    half_row_length = terrain.cfg.terrain_generator.size[0] / 2
    one_third_row_length = terrain.cfg.terrain_generator.size[0] / 3

    move_up = distance > half_row_length

    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < one_third_row_length
    # ensure mutual exclusivity: if moving up, don't move down
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids_t, move_up, move_down)
    # return the mean terrain level for logging/monitoring as a Python float
    return torch.mean(terrain.terrain_levels.float()).item()