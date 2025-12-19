from typing import cast

import numpy as np
import pytest
from pydantic_core import ValidationError

from pytileproj.tile import IrregularTile
from pytileproj.tiling import IrregularTiling, RegularTiling


@pytest.fixture(scope="module")
def reg_tiling():
    return RegularTiling(
        name="grid", extent=(0, 0, 180, 90), sampling=1, tile_shape=(10, 10)
    )


@pytest.fixture
def irreg_tiling():
    tile_0 = IrregularTile(name="0", z=0, extent=(-180, -90, 180, -45))
    tile_1 = IrregularTile(name="3", z=0, extent=(-180, 0, 0, 90))
    tile_2 = IrregularTile(name="2", z=0, extent=(0, 0, 180, 90))
    tile_4 = IrregularTile(name="1", z=0, extent=(-180, -45, 180, 0))
    tiles = {tile.name: tile for tile in [tile_0, tile_1, tile_2, tile_4]}

    return IrregularTiling(name="grid", tiles_map=tiles)


def test_reg_ori(reg_tiling: RegularTiling):
    assert reg_tiling.origin_xy == (0, 0)


def test_reg_grid(reg_tiling: RegularTiling):
    assert reg_tiling.n_tiles == 18 * 9


def test_iter_reg_tiles(reg_tiling: RegularTiling):
    test_tile = None
    x_should, y_should = 4, 2
    i_stop = x_should * reg_tiling.n_tiles_y + y_should
    for i, tile in enumerate(reg_tiling.tiles()):
        if i == i_stop:
            test_tile = tile
            break

    assert (test_tile.x, test_tile.y, test_tile.z) == (x_should, y_should, 0)


def test_to_ogc(reg_tiling: RegularTiling):
    ogc_grid_dict = reg_tiling.to_ogc_standard()
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


def test_irreg_grid(irreg_tiling: IrregularTiling):
    assert irreg_tiling.tile_ids == ["0", "3", "2", "1"]


def test_adj_matrix(irreg_tiling: IrregularTiling):
    adj_matrix = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=np.bool_
    )
    assert np.array_equal(cast("np.ndarray", irreg_tiling.adjacency_matrix), adj_matrix)


def test_neighbours(irreg_tiling: IrregularTiling):
    tiles = irreg_tiling.neighbours("1")
    assert sorted([tile.name for tile in tiles]) == ["0", "2", "3"]

    tiles = irreg_tiling.neighbours("0")
    assert sorted([tile.name for tile in tiles]) == ["1"]


def test_iter_irreg_tiles(irreg_tiling: IrregularTiling):
    test_tile = None
    stop_idx = 1
    for i, tile in enumerate(irreg_tiling.tiles()):
        if i == stop_idx:
            test_tile = tile
            break

    assert (test_tile.name, test_tile.z) == ("3", 0)


def test_irreg_tiles_bbox(irreg_tiling: IrregularTiling):
    tiles = irreg_tiling.tiles_intersecting_bbox((-10, -10, 10, 10))
    assert sorted([tile.name for tile in tiles]) == ["1", "2", "3"]


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
        name="grid", extent=(0, 0, 180, 90), sampling=5, tile_shape=(10, 10)
    )

    try:
        _ = RegularTiling(
            name="grid", extent=(0, 0, 180, 90), sampling=3, tile_shape=(10, 10)
        )
        raise AssertionError
    except ValidationError:
        assert True


if __name__ == "__main__":
    pass
