# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .geometry_utils import get_robot_dimensions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initialize_morphology_clusters(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    """
    Compute and cache morphology cluster assignments for all environments.

    Current clustering is intentionally 2-bit (4-way) for specialist routing:
    - GCR bit:   gcr >= 0.82   -> bit1
    - SPCF bit:  spcf >= 0.006 -> bit0

    Cluster index:
        cluster_id = (gcr_bit << 1) | spcf_bit

    Note:
    - FL/TL-based logic is intentionally kept commented for future extension.
    """
    # 1) Optional FL/TL extraction kept for future 4-parameter clustering.
    # dims = get_robot_dimensions(slice(0, env.num_envs))
    # femur_len = (dims.femur_left.height + dims.femur_right.height) / 2.0
    # tibia_len = (dims.tibia_left.height + dims.tibia_right.height) / 2.0
    # femur_len = femur_len.to(env.device)
    # tibia_len = tibia_len.to(env.device)
    # b3 = (femur_len >= 0.39).long() << 3
    # b2 = (tibia_len >= 0.39).long() << 2

    # 2) Resolve per-environment GCR values.
    # Prefer an already-cached tensor, otherwise sample from range, otherwise scalar fallback.
    gcr = getattr(env, "gcr_t", None)
    if gcr is None:
        gcr_range = getattr(env, "GCR_range", None)
        if gcr_range is not None:
            gcr = (
                torch.rand(env.num_envs, device=env.device, dtype=torch.float32)
                * (gcr_range[1] - gcr_range[0])
                + gcr_range[0]
            )
            env.gcr_t = gcr
        else:
            gcr_val = getattr(env, "GCR", 0.84)
            gcr = torch.full((env.num_envs,), gcr_val, device=env.device, dtype=torch.float32)
    else:
        gcr = gcr.to(env.device).view(-1)

    # 3) Resolve per-environment spring coefficient values.
    # Prefer cached env.spcf_t, otherwise sample from range, otherwise scalar fallback.
    spcf_t = getattr(env, "spcf_t", None)
    if spcf_t is None:
        spcf_range = getattr(env, "spcf_range", None)
        if spcf_range is not None:
            spcf_t = (
                torch.rand(env.num_envs, device=env.device, dtype=torch.float32)
                * (spcf_range[1] - spcf_range[0])
                + spcf_range[0]
            )
        else:
            spcf = getattr(env, "spcf", 0.005)
            spcf_t = torch.full((env.num_envs,), spcf, device=env.device, dtype=torch.float32)
        env.spcf_t = spcf_t
    else:
        spcf_t = spcf_t.to(env.device).view(-1)

    # Push spring coefficients to actuators so physics and observations use the same values.
    robot = env.scene["robot"]
    knee_actuators = robot.actuators["knee_effort_actuators"]
    knee_actuators.spring_coeff = spcf_t.view(-1, 1)
    spring_coeff = spcf_t

    # 4) Apply thresholds and compute 2-bit index.
    # Optional FL/TL term kept commented for future extension:
    # b2 = (femur_len + tibia_len >= 0.75).long() << 2
    b1 = (gcr >= 0.82).long() << 1
    b0 = (spring_coeff >= 0.006).long() << 0
    cluster_ids = b1 | b0

    # 5) Cache on the environment object
    env.cluster_assignments = cluster_ids.to(env.device)
    print(f"[INFO] Initialized {env.num_envs} morphology clusters (4-way grid over GCR/SPCF).")
