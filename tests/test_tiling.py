import numpy as np
import pytest
from pydantic_core import ValidationError

from pytileproj.tile import IrregularTile
from pytileproj.tiling import IrregularTiling, RegularTiling


@pytest.fixture(scope="module")
def reg_grid():
    return RegularTiling(
        name="grid", extent=[0, 0, 180, 90], sampling=1, tile_shape=(10, 10)
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
    assert reg_grid.origin_xy == (0, 0)


def test_reg_grid(reg_grid: RegularTiling):
    assert reg_grid.n_tiles == 18 * 9


def test_iter_reg_tiles(reg_grid: RegularTiling):
    test_tile = None
    x_should, y_should = 4, 2
    i_stop = x_should * reg_grid.n_tiles_y + y_should
    for i, tile in enumerate(reg_grid):
        if i == i_stop:
            test_tile = tile
            break

    assert (test_tile.x, test_tile.y, test_tile.z) == (x_should, y_should, 0)


def test_to_ogc(reg_grid: RegularTiling):
    ogc_grid_dict = reg_grid.to_ogc_standard()
    assert ogc_grid_dict == {
        "title": "grid",
        "description": None,
        "keywords": None,
        "id": "0",
        "scaleDenominator": 3571.4285714285716,
        "cellSize": 1.0,
        "cornerOfOrigin": "bottomLeft",
        "pointOfOrigin": (0.0, 0.0),
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
    stop_idx = 1
    for i, tile in enumerate(irreg_grid):
        if i == stop_idx:
            test_tile = tile
            break

    assert (test_tile.id, test_tile.z) == ("3", 0)


def test_irreg_tiles_bbox(irreg_grid: IrregularTiling):
    tiles = irreg_grid.tiles_intersecting_bbox([-10, -10, 10, 10])
    assert sorted([tile.id for tile in tiles]) == ["1", "2", "3"]


def test_allowed_samplings():
    tile_size = 3000
    assert RegularTiling.allowed_samplings(tile_size) == [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
        20,
        24,
        25,
        30,
        40,
        50,
        60,
        75,
        100,
        120,
        125,
        150,
        200,
        250,
        300,
        375,
        500,
        600,
        750,
        1000,
        1500,
        3000,
    ]


def test_validate_sampling():
    _ = RegularTiling(
        name="grid", extent=[0, 0, 180, 90], sampling=5, tile_shape=(10, 10)
    )

    try:
        _ = RegularTiling(
            name="grid", extent=[0, 0, 180, 90], sampling=3, tile_shape=(10, 10)
        )
        raise AssertionError
    except ValidationError:
        assert True


if __name__ == "__main__":
    pass
