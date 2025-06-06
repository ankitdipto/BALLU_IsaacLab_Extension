from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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