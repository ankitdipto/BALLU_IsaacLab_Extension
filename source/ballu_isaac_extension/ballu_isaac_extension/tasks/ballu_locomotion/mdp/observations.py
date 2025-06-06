from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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