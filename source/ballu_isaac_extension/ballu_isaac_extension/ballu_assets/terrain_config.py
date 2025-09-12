import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


BALLU_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(7.0, 7.0),
    border_width=1.4,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.4,
            grid_height_range=(0.00, 0.04), #(0.027124, 0.027124),
            platform_width=0.5,
            holes=False,
        ),
    },
)
"""Easy terrain featuring only box grid cells with modest height variation suitable for bipeds."""

BALLU_TERRAINS_CFG_PLAY = TerrainGeneratorCfg(
    size=(7.0, 7.0),
    border_width=1.4,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.4,
            grid_height_range=(0.03, 0.03), #(0.027124, 0.027124),
            platform_width=0.5,
            holes=False,
        ),
    },
)