"""Custom triangular ramp (wedge) spawner for BALLU ramp locomotion task.

Follows the Isaac Lab MeshCfg pattern from
``isaaclab.sim.spawners.meshes``.

Key design decision: ``spawn_triangular_ramp`` is defined BEFORE
``TriangularRampCfg`` (with a forward-reference string type hint for ``cfg``)
so that ``TriangularRampCfg.func`` can reference the live function object
at class-definition time.  This is required because ``@configclass`` (backed
by dataclasses) captures field defaults at class-definition time; a
post-hoc class-attribute assignment is never seen by instances.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import isaacsim.core.utils.prims as prim_utils
from pxr import Usd

from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass  # TriangularRampCfg is referenced as a string below


@clone
def spawn_triangular_ramp(
    prim_path: str,
    cfg: "TriangularRampCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a triangular prism (ramp/wedge) as a USD mesh prim.

    The ramp cross-section in the XZ-plane is a right triangle that slopes
    from z=0 at x=0 up to z=``cfg.height`` at x=``cfg.base_len``.
    The prism extends ``cfg.width_y`` metres symmetrically along ±Y.

    Args:
        prim_path: USD prim path (or regex pattern) where the ramp is spawned.
        cfg: Ramp configuration.
        translation: Translation (x, y, z) relative to parent prim.
        orientation: Orientation quaternion (w, x, y, z) relative to parent prim.

    Returns:
        The created USD prim.
    """
    L = cfg.base_len
    h = cfg.height
    w = cfg.width_y / 2.0

    # 6 vertices of the triangular prism.
    # The right-triangle cross-section lies in the XZ-plane:
    #   vertex 0/3: (0, ∓w, 0)  — ramp start, ground level
    #   vertex 1/4: (L, ∓w, 0)  — ramp end,   ground level
    #   vertex 2/5: (L, ∓w, h)  — ramp end,   peak
    verts = np.array(
        [
            [0.0, -w, 0.0],  # 0: start, -y side, ground
            [L,   -w, 0.0],  # 1: end,   -y side, ground
            [L,   -w, h  ],  # 2: end,   -y side, peak
            [0.0, +w, 0.0],  # 3: start, +y side, ground
            [L,   +w, 0.0],  # 4: end,   +y side, ground
            [L,   +w, h  ],  # 5: end,   +y side, peak
        ],
        dtype=float,
    )

    # 8 triangular faces (all-triangles, consistent outward normals).
    # Front/back triangular end-caps + bottom, right-end, and slope quads
    # each split into two triangles.
    faces = np.array(
        [
            [0, 2, 1],          # front triangle  (normal: -y)
            [3, 4, 5],          # back triangle   (normal: +y)
            [0, 1, 4], [0, 4, 3],  # bottom face    (normal: -z)
            [1, 2, 5], [1, 5, 4],  # right-end face (normal: +x)
            [0, 5, 2], [0, 3, 5],  # slope face     (normal: upper-left diagonal)
        ],
        dtype=int,
    )

    ramp_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, ramp_mesh, translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TriangularRampCfg(MeshCfg):
    """Configuration for a triangular prism (ramp/wedge) mesh prim.

    The ramp cross-section in the XZ-plane is a right triangle:

    .. code-block:: text

         z
         |   /|
         |  / |  <- height
         | /  |
         |/___|__ x
        x=0  x=base_len

    The prism extends ``width_y`` metres symmetrically along ±Y.

    When placed via ``AssetBaseCfg.InitialStateCfg.pos = (x_start, y_center, 0)``,
    the ramp occupies world x ∈ [x_start, x_start + base_len].

    Collision approximation: ``convexHull`` (default for non-standard
    MeshCfg subclasses in ``_spawn_mesh_geom_from_mesh``).
    """

    func: Callable = spawn_triangular_ramp  # defined above — captured at class-def time

    base_len: float = 1.5
    """Length of the ramp base along the local x-axis (m)."""

    height: float = 0.1
    """Peak height of the ramp at x=``base_len`` (m)."""

    width_y: float = 2.0
    """Full width of the ramp along the y-axis (m)."""
