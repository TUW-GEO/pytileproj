import numpy as np
import pytest
from osgeo import osr

from pytileproj.tile import IrregularTile
from pytileproj.tiling import IrregularTiling, RegularTiling

osr.UseExceptions()


@pytest.fixture(scope="module")
def reg_grid():
    return RegularTiling(
        name="grid", extent=[0, 0, 180, 90], sampling=1, tile_shape_px=(10, 10)
    )


@pytest.fixture
def irreg_grid():
    tile_0 = IrregularTile(id="0", z=0, extent=[-180, -90, 180, -45])
    tile_1 = IrregularTile(id="3", z=0, extent=[-180, 0, 0, 90])
    tile_2 = IrregularTile(id="2", z=0, extent=[0, 0, 180, 90])
    tile_4 = IrregularTile(id="1", z=0, extent=[-180, -45, 180, 0])
    tiles = {tile.id: tile for tile in [tile_0, tile_1, tile_2, tile_4]}

    return IrregularTiling(name="grid", tiles=tiles)


def test_reg_ori(reg_grid: RegularTiling):
    assert reg_grid.origin_xy == (0, 90)


def test_reg_grid(reg_grid: RegularTiling):
    assert reg_grid.n_tiles == 18 * 9


def test_iter_reg_tiles(reg_grid: RegularTiling):
    test_tile = None
    for i, tile in enumerate(reg_grid):
        if i == 38:
            test_tile = tile
            break

    assert (test_tile.x, test_tile.y, test_tile.z) == (4, 2, 0)


def test_to_ogc(reg_grid: RegularTiling):
    ogc_grid_dict = reg_grid.to_ogc_repr()
    assert ogc_grid_dict == {
        "title": "grid",
        "description": None,
        "keywords": None,
        "id": "0",
        "scaleDenominator": 3571.4285714285716,
        "cellSize": 1.0,
        "cornerOfOrigin": "bottomLeft",
        "pointOfOrigin": (0.0, 90.0),
        "tileWidth": 10,
        "tileHeight": 10,
        "matrixWidth": 18,
        "matrixHeight": 9,
        "variableMatrixWidths": None,
    }


def test_irreg_grid(irreg_grid: IrregularTiling):
    assert irreg_grid.tile_ids == ["0", "3", "2", "1"]


def test_adj_matrix(irreg_grid: IrregularTiling):
    adj_matrix = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=np.bool_
    )
    assert np.array_equal(irreg_grid.adjacency_matrix, adj_matrix)


def test_neighbours(irreg_grid: IrregularTiling):
    tiles = irreg_grid.neighbours("1")
    assert sorted([tile.id for tile in tiles]) == ["0", "2", "3"]

    tiles = irreg_grid.neighbours("0")
    assert sorted([tile.id for tile in tiles]) == ["1"]


def test_iter_irreg_tiles(irreg_grid: IrregularTiling):
    test_tile = None
    for i, tile in enumerate(irreg_grid):
        if i == 1:
            test_tile = tile
            break

    assert (test_tile.id, test_tile.z) == ("3", 0)


def test_irreg_tiles_bbox(irreg_grid: IrregularTiling):
    tiles = irreg_grid.tiles_intersecting_bbox([-10, -10, 10, 10])
    assert sorted([tile.id for tile in tiles]) == ["1", "2", "3"]


if __name__ == "__main__":
    pass
