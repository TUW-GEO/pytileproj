import math
from pathlib import Path

import numpy as np
import pytest
from morecantile.models import Tile

from pytileproj._errors import GeomOutOfZoneError
from pytileproj.tile import RasterTile
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    ProjSystem,
    ProjTilingSystem,
    RegularProjTilingSystem,
    TilingSystem,
)


@pytest.fixture(scope="module")
def e7eu_grid_t1():
    return RegularTiling(
        name="e7eut1",
        extent=[0, 0, 8_660_000, 6_020_000],
        sampling=10,
        tile_shape_px=(10_000, 10_000),
        tiling_level=1,
        axis_orientation=["E", "S"],
    )


@pytest.fixture(scope="module")
def e7eu_grid_t3():
    return RegularTiling(
        name="e7eut3",
        extent=[0, 0, 8_660_000, 6_020_000],
        sampling=20,
        tile_shape_px=(15_000, 15_000),
        tiling_level=0,
        axis_orientation=["E", "S"],
    )


@pytest.fixture(scope="module")
def e7eu_grid_invalid():
    return RegularTiling(
        name="e7_invalid",
        extent=[0, 0, 300_000, 300_000],
        sampling=20,
        tile_shape_px=(15_000, 15_000),
        tiling_level=2,
        axis_orientation=["E", "S"],
    )


@pytest.fixture(scope="module")
def e7eu_psb(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    return ProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
        crs=27704,
    )


@pytest.fixture(scope="module")
def e7eu_rpsb(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    return RegularProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
        crs=27704,
    )


def test_projsystembase():
    e7eu = ProjSystem(crs=27704)

    lon, lat = 16.37, 48.19
    e7_coord = e7eu.lonlat_to_xy(lon, lat)
    geog_coord_2 = e7eu.xy_to_lonlat(e7_coord.x, e7_coord.y)
    assert math.isclose(geog_coord_2.x, lon, rel_tol=0.01)
    assert math.isclose(geog_coord_2.y, lat, rel_tol=0.01)

    lon, lat = 0, 0
    try:
        e7_coord = e7eu.lonlat_to_xy(lon, lat)
        raise AssertionError
    except GeomOutOfZoneError:
        assert True


def test_gridsystembase(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    grids = {grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]}
    gsb = TilingSystem(name="e7eu", tilings=grids)
    ref_len = 2
    assert len(gsb) == ref_len

    json_path = Path("test_gridsystembase.json")
    gsb.to_file(json_path)
    gsb2 = TilingSystem.from_file(json_path)

    assert gsb[0].to_ogc_standard() == gsb2[0].to_ogc_standard()
    assert gsb[1].to_ogc_standard() == gsb2[1].to_ogc_standard()

    json_path.unlink()


def test_projgridsystembase_tile(e7eu_psb: ProjTilingSystem):
    e7_tile = RasterTile.from_extent(
        [3700000, 2300000, 3800000, 2400000], 27704, 10, 10
    )
    assert e7_tile in e7eu_psb


def test_projgridsystembase_mask(e7eu_psb: ProjTilingSystem):
    e7_tile = RasterTile.from_extent(
        [3700000, 2300000, 3800000, 2400000], 27704, 10, 10
    )
    tile_mask = e7eu_psb.get_tile_mask(e7_tile)
    assert np.array_equal(tile_mask, np.ones((10000, 10000)))

    e7_tile = RasterTile.from_extent([0, 0, 100000, 100000], 27704, 10, 10)
    tile_mask = e7eu_psb.get_tile_mask(e7_tile)
    assert np.array_equal(tile_mask, np.zeros((10000, 10000)))


def test_reg_pgs_invalid(
    e7eu_grid_t1: RegularTiling,
    e7eu_grid_t3: RegularTiling,
    e7eu_grid_invalid: RegularTiling,
):
    grids = {
        grid.tiling_level: grid
        for grid in [e7eu_grid_t1, e7eu_grid_t3, e7eu_grid_invalid]
    }
    try:
        _ = RegularProjTilingSystem(name="e7eu", tilings=grids, crs=27704)
        raise AssertionError
    except ValueError:
        assert True


def test_reg_pgs_tile_conv(e7eu_rpsb: RegularProjTilingSystem):
    tile_1 = Tile(x=15, y=10, z=0)
    tilename_1 = e7eu_rpsb._tile_to_name(tile_1)  # noqa: SLF001
    raster_tile = e7eu_rpsb.get_tile_from_index(*tile_1)
    assert tilename_1 == raster_tile.name


def test_reg_pgs_raster_tile_conv(e7eu_rpsb: RegularProjTilingSystem):
    tile = Tile(x=37, y=23, z=1)
    raster_tile = e7eu_rpsb.get_tile_from_index(*tile)
    assert raster_tile.name == "X37Y23T01"
    assert raster_tile.shape == (10000, 10000)
    assert raster_tile.geotrans == (
        e7eu_rpsb[0].origin_xy[0] + 10000 * 10 * tile.x,
        10.0,
        0.0,
        e7eu_rpsb[0].origin_xy[1] - 10000 * 10 * tile.y,
        0.0,
        -10.0,
    )


def test_congruency(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    rpts = RegularProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
        crs=27704,
    )

    assert not rpts.is_congruent

    new_grid = RegularTiling(
        name="e7eut3",
        extent=[0, 0, 8_660_000, 6_020_000],
        sampling=20,
        tile_shape_px=(10_000, 10_000),
        tiling_level=0,
        axis_orientation=["E", "S"],
    )
    rpts = RegularProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, new_grid]},
        crs=27704,
    )

    assert rpts.is_congruent


def test_allowed_samplings(e7eu_grid_t1: RegularTiling):
    allowed_samplings = [1, 5]
    try:
        _ = RegularProjTilingSystem(
            name="e7eu",
            tilings={e7eu_grid_t1.tiling_level: e7eu_grid_t1},
            crs=27704,
            allowed_samplings={e7eu_grid_t1.tiling_level: allowed_samplings},
        )
        raise AssertionError
    except ValueError:
        assert True

    new_grid = RegularTiling(
        name="e7eut3",
        extent=[0, 0, 8660000, 6020000],
        sampling=5,
        tile_shape_px=(10000, 10000),
        tiling_level=0,
    )
    _ = RegularProjTilingSystem(
        name="e7eu",
        tilings={new_grid.tiling_level: new_grid},
        crs=27704,
        allowed_samplings={new_grid.tiling_level: allowed_samplings},
    )


@pytest.mark.vis_installed
def test_plot(e7eu_rpsb: RegularProjTilingSystem):
    e7eu_rpsb.plot()


def test_proj_zone_geog_io():
    e7eu_ref = ProjSystem(crs=27704)
    json_path = Path("test_proj_zone_geog.json")
    e7eu_ref.export_proj_zone_geog(json_path)

    e7eu = ProjSystem(crs=27704, proj_zone_geog=json_path)
    json_path.unlink()

    assert e7eu.proj_zone_geog.geom.wkt == e7eu_ref.proj_zone_geog.geom.wkt


if __name__ == "__main__":
    pass
