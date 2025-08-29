"""
Python module serving as a project/extension template.
"""

import gymnasium as gym

from . import agents

# Register the BALLU environments
gym.register(
    id="Isc-Vel-BALLU-encoder",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.indirect_act_vel_env_cfg:BalluIndirectActEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Vel-BALLU-priv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.basic_vel_env_cfg:BALLUEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Vel-BALLU-mocap",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mocap_vel_env_cfg:BALLUMocapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Vel-BALLU-imu-base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.imu_base_env_cfg:BALLU_IMU_base_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)

gym.register(
    id="Isaac-Vel-BALLU-imu-tibia",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tibia_imu_env_cfg:BALLU_TibiaIMU_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)

gym.register(
    id="Isc-Vel-BALLU-rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BalluRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BALLUPPORunnerCfg"
    }
)