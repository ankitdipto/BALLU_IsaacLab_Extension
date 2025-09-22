from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt, Usd

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

def obstacle_height_levels_same_row(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    obstacle_center_x: float = 1.0,
    obstacle_half_size_x: float = 0.5,
    inter_obstacle_spacing_y: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> float:
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    # ensure env_ids is a tensor (type-checker + API expects Tensor)
    env_ids_t = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)
    # check if the robot has passed the obstacle
    # Obstacle center line is at x = 1.0
    final_robot_positions_x = robot.data.root_pos_w[env_ids_t, 0]
    upgrade = final_robot_positions_x > obstacle_center_x
    downgrade = final_robot_positions_x < obstacle_center_x - obstacle_half_size_x
    # print("----------------------------------------------------------------------------------------")
    # print(f"[DEBUG] final_robot_positions_x: {final_robot_positions_x} obstacle_center_x: {obstacle_center_x} obstacle_half_size_x: {obstacle_half_size_x}")
    #print(f"[DEBUG] upgrade: {upgrade} downgrade: {downgrade}")
    # ensure mutual exclusivity: if upgrading, don't downgrade
    downgrade *= ~upgrade
    new_env_origins_y = env.scene.env_origins[env_ids_t, 1] - \
                         inter_obstacle_spacing_y * (1 * upgrade) + \
                         inter_obstacle_spacing_y * (1 * downgrade)

    new_env_origins_y = new_env_origins_y.clip(min=-100.0, max=0.0)
    # update env origins
    env.scene._default_env_origins[env_ids_t, 1] = new_env_origins_y
    # return the mean height of obstacle across all environments
    curr_mean_level = - new_env_origins_y.mean(dim=0).detach().cpu().item() / inter_obstacle_spacing_y
    curr_mean_obstacle_height = env.obstacle_height_list[int(curr_mean_level)]
    return curr_mean_obstacle_height

def obstacle_height_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    scale_fraction: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> float:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # ensure env_ids is a tensor (type-checker + API expects Tensor)
    env_ids_t = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    # Check if the robot has passed the obstacle => Final position > sizeX / 2 from the obstacle center
    obstacle_asset : RigidObject = env.scene["obstacle"]

    print(f"[DEBUG] Obstacle asset: {obstacle_asset}")
    print(f"[DEBUG] robot asset: {asset}")
    final_pos_w_x = asset.data.root_pos_w[env_ids_t, 0]
    obstacle_half_size_x = obstacle_asset.cfg.spawn.size[0] / 2
    obstacle_center_w_x = obstacle_asset.data.root_pos_w[env_ids_t, 0]
    upgrade = final_pos_w_x > obstacle_center_w_x
    downgrade = final_pos_w_x < obstacle_center_w_x - obstacle_half_size_x
    # ensure mutual exclusivity: if upgrading, don't downgrade
    downgrade *= ~upgrade
    # Prepare scale factor tensor
    scale_factor_tensor = torch.ones_like(env_ids_t, device=asset.device)
    scale_factor_tensor[upgrade] = 1.0 + scale_fraction
    scale_factor_tensor[downgrade] = 1.0 - scale_fraction
    # Update obstacle heights
    new_obstacle_heights = scale_obstacle_height(env, 
    env_ids_t, scale_factor_tensor)
    # Return the mean height of obstacle across all environments
    return torch.mean(new_obstacle_heights).item()

def scale_obstacle_height(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    scale_factors: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    relative_child_path: str | None = None,
) -> torch.Tensor:
    """Scale an asset along the z-axis only using USD xform scale, per environment.

    This sets the prim's ``xformOp:scale`` to (1.0, 1.0, z_scale) for each targeted environment.

    .. attention::
        Modifies USD properties parsed before simulation starts. Call this only before the
        simulation is playing (i.e., during pre-startup/"usd" event mode).

    Args:
        env: The learning environment.
        env_ids: The environments to update. If None, applies to all environments.
        scale_factors: Tensor of shape (num_envs_targeted,) with z-scale per environment.
        asset_cfg: The scene entity to scale.
        relative_child_path: Optional relative child prim path under the asset to scale.
            Example: "mesh" or "/mesh".
    """
    # ensure simulation is not running
    # if env.sim.is_playing():
    #     raise RuntimeError(
    #         "Scaling via USD while simulation is running leads to unpredictable behaviors."
    #         " Please call this function before the simulation starts (use 'usd' mode)."
    #     )

    # extract the asset
    asset = env.scene[asset_cfg.name]
    print(f"[DEBUG] Asset: {asset}")
    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation via USD is not supported. Generate separate USDs per scale"
            " and use multi-asset spawning instead."
        )
    # if not isinstance(asset, RigidObject):
    #     raise ValueError(
    #         f"Z-scaling is only supported for RigidObject assets. Got: {type(asset)}"
    #     )

    print(f"[DEBUG] Scaling obstacle height")
    # resolve env ids
    if env_ids is None:
        env_ids_t = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids_t = torch.as_tensor(env_ids, device="cpu")

    # validate scale_factors shape
    scale_factors_cpu = torch.as_tensor(scale_factors, device="cpu").flatten()
    if len(scale_factors_cpu) != len(env_ids_t):
        raise ValueError(
            "scale_factors length must match number of targeted environments."
            f" Expected {len(env_ids_t)}, got {len(scale_factors_cpu)}."
        )

    # resolve stage and prim paths
    stage = omni.usd.get_context().get_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # normalize relative child path
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    new_obstacle_heights = []
    # apply per-env scaling on z-axis only
    with Sdf.ChangeBlock():
        for idx, env_id in enumerate(env_ids_t.tolist()):
            prim_path = prim_paths[env_id] + relative_child_path
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # fetch/create scale attribute
            scale_attr_path = prim_path + ".xformOp:scale"
            scale_spec = prim_spec.GetAttributeAtPath(scale_attr_path)
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, scale_attr_path, Sdf.ValueTypeNames.Double3)

            # set (1, 1, z)
            z_scale = float(scale_factors_cpu[idx].item())
            scale_spec.default = Gf.Vec3f(1.0, 1.0, z_scale)

            # ensure xformOpOrder includes scale in canonical order if we created it
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            # fetch the prim
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None:
                raise ValueError(f"Could not find prim at path: {prim_path}")
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], False)
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            aligned_box = world_bbox.ComputeAlignedBox()
            min_pt = aligned_box.GetMin()
            max_pt = aligned_box.GetMax()
            size_z = max_pt[2] - min_pt[2]

            new_obstacle_heights.append(size_z)

    return torch.tensor(new_obstacle_heights)
