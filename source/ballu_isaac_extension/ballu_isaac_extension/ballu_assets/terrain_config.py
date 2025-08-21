import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


BALLU_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=6.0,
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
            grid_height_range=(0.03, 0.03),
            platform_width=0.4,
            holes=False,
        ),
    },
)
"""Easy terrain featuring only box grid cells with modest height variation suitable for bipeds."""
