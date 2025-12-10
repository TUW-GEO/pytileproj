import copy
import random

import numpy as np
import pyproj
import pytest
from shapely import Polygon

from pytileproj._const import DECIMALS, VIS_INSTALLED
from pytileproj.projgeom import ProjGeom, transform_geometry
from pytileproj.tile import RasterTile

if VIS_INSTALLED:
    import cartopy.crs as ccrs


@pytest.fixture(scope="session")
def ref_extent() -> tuple:
    ll_x = random.randrange(-50, 50, 10)  # noqa: S311
    ll_y = random.randrange(-50, 50, 10)  # noqa: S311
    ur_x = ll_x + random.randrange(10, 50, 10)  # noqa: S311
    ur_y = ll_y + random.randrange(10, 50, 10)  # noqa: S311

    return tuple(map(float, (ll_x, ll_y, ur_x, ur_y)))


@pytest.fixture(scope="session")
def ref_boundary(ref_extent: tuple) -> ProjGeom:
    ll_x, ll_y, ur_x, ur_y = ref_extent
    ref_points = [(ll_x, ll_y), (ll_x, ur_y), (ur_x, ur_y), (ur_x, ll_y)]
    sh_geom = Polygon(ref_points)

    return ProjGeom(sh_geom, pyproj.CRS.from_epsg(4326))


@pytest.fixture(scope="session")
def pixel_size() -> float:
    return 0.5


@pytest.fixture(scope="session")
def epsg() -> int:
    return 4326


@pytest.fixture(scope="session")
def ref_proj_tile(ref_extent: tuple, epsg: int, pixel_size: float) -> RasterTile:
    return RasterTile.from_extent(ref_extent, epsg, pixel_size, pixel_size)


def assert_extent(this_extent: tuple, other_extent: tuple):
    this_extent = np.around(np.array(this_extent), decimals=DECIMALS)
    other_extent = np.around(np.array(other_extent), decimals=DECIMALS)
    assert np.all(this_extent == other_extent)


def test_from_extent(ref_proj_tile: RasterTile, ref_extent: tuple):
    assert_extent(ref_proj_tile.outer_boundary_extent, ref_extent)


def test_from_geom(ref_boundary: ProjGeom, pixel_size: float):
    proj_tile = RasterTile.from_geometry(ref_boundary, pixel_size, pixel_size)

    assert_extent(
        proj_tile.outer_boundary_corners,
        tuple(ref_boundary.geom.exterior.coords)[:-1],
    )


def test_is_axis_parallel(ref_proj_tile: RasterTile):
    assert ref_proj_tile.is_axis_parallel


def test_pixel_size(ref_proj_tile: RasterTile):
    assert ref_proj_tile.x_pixel_size == ref_proj_tile.h_pixel_size
    assert ref_proj_tile.y_pixel_size == ref_proj_tile.v_pixel_size


def test_size(ref_proj_tile: RasterTile, ref_extent: tuple, pixel_size: float):
    extent_px_width = (ref_extent[2] - ref_extent[0]) * int(1.0 / pixel_size)
    extent_px_height = (ref_extent[3] - ref_extent[1]) * int(1.0 / pixel_size)

    extent_px_size = extent_px_height * extent_px_width

    assert extent_px_size == ref_proj_tile.size


def test_vertices(
    ref_proj_tile: RasterTile,
    ref_extent: tuple,
):
    vertices = (
        (ref_extent[0], ref_extent[1]),
        (ref_extent[0], ref_extent[3]),
        (ref_extent[2], ref_extent[3]),
        (ref_extent[2], ref_extent[1]),
    )

    assert_extent(ref_proj_tile.outer_boundary_corners, vertices)


def test_x_coords(ref_proj_tile: RasterTile):
    assert len(ref_proj_tile.x_coords) == ref_proj_tile.n_cols
    assert (
        ref_proj_tile.x_coords[-1]
        == ref_proj_tile.rc2xy(0, ref_proj_tile.n_cols - 1)[0]
    )
    assert ref_proj_tile.x_coords[0] == ref_proj_tile.rc2xy(0, 0)[0]
    rand_idx = random.randrange(1, ref_proj_tile.n_cols - 2, 1)  # noqa: S311
    assert ref_proj_tile.x_coords[rand_idx] == ref_proj_tile.rc2xy(0, rand_idx)[0]


def test_y_coords(ref_proj_tile: RasterTile):
    assert len(ref_proj_tile.y_coords) == ref_proj_tile.n_rows
    assert (
        ref_proj_tile.y_coords[-1]
        == ref_proj_tile.rc2xy(ref_proj_tile.n_rows - 1, 0)[1]
    )
    assert ref_proj_tile.y_coords[0] == ref_proj_tile.rc2xy(0, 0)[1]
    rand_idx = random.randrange(1, ref_proj_tile.n_rows - 2, 1)  # noqa: S311
    assert ref_proj_tile.y_coords[rand_idx] == ref_proj_tile.rc2xy(rand_idx, 0)[1]


def test_xy_coords(ref_proj_tile: RasterTile, pixel_size: float):
    x_coords_ref, y_coords_ref = np.meshgrid(
        np.arange(
            ref_proj_tile.ul_x, ref_proj_tile.ul_x + ref_proj_tile.x_size, pixel_size
        ),
        np.arange(
            ref_proj_tile.ul_y, ref_proj_tile.ul_y - ref_proj_tile.y_size, -pixel_size
        ),
        indexing="ij",
    )
    x_coords, y_coords = ref_proj_tile.xy_coords
    assert np.array_equal(x_coords_ref, x_coords)
    assert np.array_equal(y_coords_ref, y_coords)


def test_touches(
    ref_proj_tile: RasterTile, ref_extent: tuple, epsg: int, pixel_size: float
):
    # create raster geometry which touches the previous one
    extent_tchs = (
        ref_extent[2],
        ref_extent[3],
        ref_extent[2] + 5.0,
        ref_extent[3] + 5.0,
    )
    raster_geom_tchs = RasterTile.from_extent(extent_tchs, epsg, pixel_size, pixel_size)
    assert ref_proj_tile.touches(raster_geom_tchs)

    # create raster geometry which does not touch the previous one
    extent_no_tchs = (
        ref_extent[2] + 1.0,
        ref_extent[3] + 1.0,
        ref_extent[2] + 5.0,
        ref_extent[3] + 5.0,
    )
    raster_geom_no_tchs = RasterTile.from_extent(
        extent_no_tchs, epsg, pixel_size, pixel_size
    )
    assert not ref_proj_tile.touches(raster_geom_no_tchs)


def test_coord_conversion(ref_proj_tile: RasterTile):
    r_0 = random.randint(0, ref_proj_tile.n_rows)  # noqa: S311
    c_0 = random.randint(0, ref_proj_tile.n_cols)  # noqa: S311

    x, y = ref_proj_tile.rc2xy(r_0, c_0)
    r, c = ref_proj_tile.xy2rc(x, y)

    assert (r_0, c_0) == (r, c)


def test_different_sref(
    ref_proj_tile: RasterTile, ref_extent: tuple, epsg: int, pixel_size: float
):
    # create raster geometry which touches the previous one
    extent_tchs = (
        ref_extent[2],
        ref_extent[3],
        ref_extent[2] + 5.0,
        ref_extent[3] + 5.0,
    )
    raster_geom_tchs = RasterTile.from_extent(extent_tchs, epsg, pixel_size, pixel_size)

    # reproject to different system
    geom = transform_geometry(raster_geom_tchs.boundary, 3857)

    assert ref_proj_tile.touches(geom)


@pytest.mark.vis_installed
def test_plot(ref_proj_tile: RasterTile):
    ref_proj_tile.plot(add_country_borders=True)

    # test plotting with labelling and different output projection
    ref_proj_tile_cp = copy.deepcopy(ref_proj_tile)
    ref_proj_tile_cp.name = "E048N018T1"
    ref_proj_tile_cp.plot(
        proj=ccrs.EckertI(), label_tile=True, add_country_borders=True
    )

    # test plotting with different input projection
    extent = [527798, 94878, 956835, 535687]
    proj_tile = RasterTile.from_extent(extent, 3857, x_pixel_size=500, y_pixel_size=500)
    proj_tile.plot(add_country_borders=True)


def test_equal(ref_proj_tile: RasterTile, ref_boundary: ProjGeom, pixel_size: float):
    other_proj_tile = RasterTile.from_geometry(ref_boundary, pixel_size, pixel_size)

    assert ref_proj_tile == other_proj_tile


def test_not_equal(
    ref_proj_tile: RasterTile, ref_extent: tuple, epsg: int, pixel_size: float
):
    other_extent = (
        ref_extent[2],
        ref_extent[3],
        ref_extent[2] + 5.0,
        ref_extent[3] + 5.0,
    )
    other_proj_tile = RasterTile.from_extent(other_extent, epsg, pixel_size, pixel_size)

    assert ref_proj_tile != other_proj_tile


def test_in(ref_proj_tile: RasterTile, ref_extent: tuple, epsg: int, pixel_size: float):
    inside_extent = (
        ref_extent[0] + pixel_size,
        ref_extent[1] + pixel_size,
        ref_extent[2] - pixel_size,
        ref_extent[3] - pixel_size,
    )
    inside_proj_tile = RasterTile.from_extent(
        inside_extent, epsg, pixel_size, pixel_size
    )
    assert inside_proj_tile in ref_proj_tile

    outside_extent = (
        ref_extent[2] + pixel_size,
        ref_extent[3] + pixel_size,
        ref_extent[2] + pixel_size * 2,
        ref_extent[3] + pixel_size * 2,
    )
    outside_proj_tile = RasterTile.from_extent(
        outside_extent, epsg, pixel_size, pixel_size
    )
    assert outside_proj_tile not in ref_proj_tile


def test_tile_dump(
    ref_proj_tile: RasterTile, ref_extent: tuple, epsg: int, pixel_size: float
):
    ref_tile_dict = ref_proj_tile.model_dump()
    extent_px_width = int((ref_extent[2] - ref_extent[0]) / pixel_size)
    extent_px_height = int((ref_extent[3] - ref_extent[1]) / pixel_size)
    ul_x, ul_y = ref_extent[0], ref_extent[3]
    geotrans = (ul_x, pixel_size, 0, ul_y, 0, -pixel_size)

    assert ref_tile_dict["name"] is None
    assert ref_tile_dict["crs"] == epsg
    assert ref_tile_dict["px_origin"] == "ul"
    assert ref_tile_dict["n_rows"] == extent_px_height
    assert ref_tile_dict["n_cols"] == extent_px_width
    assert ref_tile_dict["geotrans"] == geotrans


if __name__ == "__main__":
    pass
