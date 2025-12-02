import math
from pathlib import Path

import numpy as np
import pytest
from morecantile.models import Tile
from osgeo import osr

from pytileproj.tile import RasterTile
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    ProjSystemBase,
    ProjTilingSystemBase,
    RegularProjTilingSystem,
    TilingSystemBase,
)

osr.UseExceptions()


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
    return ProjTilingSystemBase(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
        epsg=27704,
    )


@pytest.fixture(scope="module")
def e7eu_rpsb(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    return RegularProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
        epsg=27704,
    )


def test_projsystembase():
    e7eu = ProjSystemBase(epsg=27704)

    lon, lat = 16.37, 48.19
    e7_coord = e7eu.lonlat_to_xy(lon, lat)
    geog_coord_2 = e7eu.xy_to_lonlat(e7_coord.x, e7_coord.y)
    assert math.isclose(geog_coord_2.x, lon, rel_tol=0.01)
    assert math.isclose(geog_coord_2.y, lat, rel_tol=0.01)

    lon, lat = 0, 0
    e7_coord = e7eu.lonlat_to_xy(lon, lat)
    assert e7_coord is None


def test_gridsystembase(e7eu_grid_t1: RegularTiling, e7eu_grid_t3: RegularTiling):
    grids = {grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]}
    gsb = TilingSystemBase(name="e7eu", tilings=grids)
    ref_len = 2
    assert len(gsb) == ref_len

    json_path = Path("test_gridsystembase.json")
    gsb.to_file(json_path)
    gsb2 = TilingSystemBase.from_file(json_path)

    assert gsb[0].to_ogc_repr() == gsb2[0].to_ogc_repr()
    assert gsb[1].to_ogc_repr() == gsb2[1].to_ogc_repr()

    json_path.unlink()


def test_projgridsystembase_tile(e7eu_psb: ProjTilingSystemBase):
    e7_tile = RasterTile.from_extent(
        [3700000, 2300000, 3800000, 2400000], 27704, 10, 10
    )
    assert e7_tile in e7eu_psb


def test_projgridsystembase_mask(e7eu_psb: ProjTilingSystemBase):
    e7_tile = RasterTile.from_extent(
        [3700000, 2300000, 3800000, 2400000], 27704, 10, 10
    )
    tile_mask = e7eu_psb.tile_mask(e7_tile)
    assert np.array_equal(tile_mask, np.ones((10000, 10000)))

    e7_tile = RasterTile.from_extent([0, 0, 100000, 100000], 27704, 10, 10)
    tile_mask = e7eu_psb.tile_mask(e7_tile)
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
        _ = RegularProjTilingSystem(name="e7eu", tilings=grids, epsg=27704)
        raise AssertionError
    except ValueError:
        assert True


def test_reg_pgs_tile_conv(e7eu_rpsb: RegularProjTilingSystem):
    tilename_1 = e7eu_rpsb._create_tilename(Tile(x=37, y=23, z=0))  # noqa: SLF001
    tile = e7eu_rpsb._create_tile(tilename_1)  # noqa: SLF001
    tilename_2 = e7eu_rpsb._create_tilename(tile)  # noqa: SLF001
    assert tilename_1 == tilename_2


def test_reg_pgs_raster_tile_conv(e7eu_rpsb: RegularProjTilingSystem):
    tile = Tile(x=37, y=23, z=1)
    tilename_1 = e7eu_rpsb._create_tilename(tile)  # noqa: SLF001
    raster_tile = e7eu_rpsb.create_tile(tilename_1)
    assert raster_tile.name == "E37S23T1"
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
    try:
        _ = RegularProjTilingSystem(
            name="e7eu",
            tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, e7eu_grid_t3]},
            epsg=27704,
            congruent=True,
        )
        raise AssertionError
    except ValueError:
        assert True

    new_grid = RegularTiling(
        name="e7eut3",
        extent=[0, 0, 8660000, 6020000],
        sampling=20,
        tile_shape_px=(10000, 10000),
        tiling_level=0,
        axis_orientation=["E", "S"],
    )
    _ = RegularProjTilingSystem(
        name="e7eu",
        tilings={grid.tiling_level: grid for grid in [e7eu_grid_t1, new_grid]},
        epsg=27704,
        congruent=True,
    )


def test_allowed_samplings(e7eu_grid_t1: RegularTiling):
    allowed_samplings = [1, 5]
    try:
        _ = RegularProjTilingSystem(
            name="e7eu",
            tilings={e7eu_grid_t1.tiling_level: e7eu_grid_t1},
            epsg=27704,
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
        epsg=27704,
        allowed_samplings={new_grid.tiling_level: allowed_samplings},
    )


@pytest.mark.vis_installed
def test_plot(e7eu_rpsb: RegularProjTilingSystem):
    e7eu_rpsb.plot()


def test_proj_zone_geog_io():
    e7eu_ref = ProjSystemBase(epsg=27704)
    json_path = Path("test_proj_zone_geog.json")
    e7eu_ref.export_proj_zone_geog(json_path)

    e7eu = ProjSystemBase(epsg=27704, proj_zone_geog=json_path)
    json_path.unlink()

    assert e7eu._proj_zone_geog.ExportToWkt() == e7eu_ref._proj_zone_geog.ExportToWkt()  # noqa: SLF001


if __name__ == "__main__":
    pass
