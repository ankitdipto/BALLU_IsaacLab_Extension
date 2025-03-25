"""
Python module serving as a project/extension template.
"""

import gymnasium as gym

from . import agents

# Register the BALLU environment
gym.register(
    id="Isaac-Velocity-BALLU-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.basic_vel_env_cfg:BALLUEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)