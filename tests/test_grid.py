from pathlib import Path

import pytest

from pytileproj.grid import RegularGrid
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    ProjSystemDefinition,
    RegularProjTilingSystem,
    RegularTilingDefinition,
)


@pytest.fixture(scope="module")
def tiling_defs():
    return {1: RegularTilingDefinition(name="t1", tile_shape=(100_000, 100_000))}


@pytest.fixture(scope="module")
def tiling_defs_multi():
    return {
        1: RegularTilingDefinition(name="t1", tile_shape=(200_000, 200_000)),
        2: RegularTilingDefinition(name="t2", tile_shape=(100_000, 100_000)),
    }


@pytest.fixture(scope="module")
def rpts_defs(*, epsg_api_accessible: bool):
    return {
        "e7eu": ProjSystemDefinition(
            name="e7eu",
            crs=27704,
            min_xy=(0, 0),
            max_xy=(8_660_000, 6_020_000),
            axis_orientation=("E", "S"),
            proj_zone_geog=Path(__file__).parent / "data" / "eu_zone.parquet"
            if not epsg_api_accessible
            else None,
        ),
        "e7af": ProjSystemDefinition(
            name="e7af",
            crs=27701,
            min_xy=(0, 0),
            max_xy=(12_000_000, 9_600_000),
            axis_orientation=("E", "S"),
            proj_zone_geog=Path(__file__).parent / "data" / "af_zone.parquet"
            if not epsg_api_accessible
            else None,
        ),
    }


@pytest.fixture(scope="module")
def euas_defs(*, epsg_api_accessible: bool):
    return {
        "e7eu": ProjSystemDefinition(
            name="e7eu",
            crs=27704,
            min_xy=(0, 0),
            max_xy=(8_660_000, 6_020_000),
            axis_orientation=("E", "S"),
            proj_zone_geog=Path(__file__).parent / "data" / "eu_zone.parquet"
            if not epsg_api_accessible
            else None,
        ),
        "e7as": ProjSystemDefinition(
            name="e7as",
            crs=27703,
            min_xy=(0, -1_800_000),
            axis_orientation=("E", "S"),
            proj_zone_geog=Path(__file__).parent / "data" / "as_zone.parquet"
            if not epsg_api_accessible
            else None,
        ),
    }


@pytest.fixture(scope="module")
def e7eu_grid_t1():
    return RegularTiling(
        name="t1",
        extent=(0, 0, 8_660_000, 6_020_000),
        sampling=10,
        tile_shape=(100_000, 100_000),
        tiling_level=1,
        axis_orientation=("E", "S"),
    )


@pytest.fixture(scope="module")
def e7af_grid_t1():
    return RegularTiling(
        name="t1",
        extent=(0, 0, 12_000_000, 9_600_000),
        sampling=10,
        tile_shape=(100_000, 100_000),
        tiling_level=1,
        axis_orientation=("E", "S"),
    )


@pytest.fixture(scope="module")
def e7grid(
    e7eu_grid_t1: RegularTiling,
    e7af_grid_t1: RegularTiling,
    *,
    epsg_api_accessible: bool,
):
    rpts_eu = RegularProjTilingSystem(
        name="e7eu",
        tilings={e7eu_grid_t1.tiling_level: e7eu_grid_t1},
        crs=27704,
        proj_zone_geog=Path(__file__).parent / "data" / "eu_zone.parquet"
        if not epsg_api_accessible
        else None,
    )
    rpts_af = RegularProjTilingSystem(
        name="e7af",
        tilings={e7af_grid_t1.tiling_level: e7af_grid_t1},
        crs=27701,
        proj_zone_geog=Path(__file__).parent / "data" / "af_zone.parquet"
        if not epsg_api_accessible
        else None,
    )

    return RegularGrid(
        system_order=None, **{rpts_eu.name: rpts_eu, rpts_af.name: rpts_af}
    )


def test_io_grid_def(e7grid: RegularGrid):
    json_path = Path("test_io_grid_def.json")
    e7grid.to_grid_def(json_path)

    e7_grid_io = RegularGrid.from_grid_def(json_path, 10)
    json_path.unlink()

    assert e7grid.model_dump() == e7_grid_io.model_dump()


def test_io_file(e7grid: RegularGrid):
    json_path = Path("test_io_file.json")
    e7grid.to_file(json_path)

    e7_grid_io = RegularGrid.from_file(json_path)
    json_path.unlink()

    assert e7grid.model_dump() == e7_grid_io.model_dump()


def test_from_sampling(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs: dict[int, RegularTilingDefinition],
    e7grid: RegularGrid,
):
    e7grid_from_def = RegularGrid.from_sampling(10, rpts_defs, tiling_defs)

    assert e7grid.model_dump() == e7grid_from_def.model_dump()


def test_tiling_defs_multi(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid_to_def = RegularGrid.from_sampling(10, rpts_defs, tiling_defs_multi)
    json_path = Path("test_tiling_defs_multi.json")
    e7grid_to_def.to_grid_def(json_path)

    e7grid_from_def = RegularGrid.from_grid_def(json_path, 10)
    json_path.unlink()

    assert e7grid_from_def._proj_defs == e7grid_to_def._proj_defs  # noqa: SLF001
    assert e7grid_from_def._tiling_defs == e7grid_to_def._tiling_defs  # noqa: SLF001


def test_tiling_defs_multi_sampling_diff(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid = RegularGrid.from_sampling(10, rpts_defs, tiling_defs_multi)
    json_path = Path("test_tiling_defs_multi_sampling_diff.json")
    e7grid.to_grid_def(json_path)

    e7grid_from_def = RegularGrid.from_grid_def(json_path, 1000)
    json_path.unlink()

    e7grid = RegularGrid.from_sampling(1000, rpts_defs, tiling_defs_multi)

    assert e7grid_from_def.model_dump() == e7grid.model_dump()


def test_tiling_defs_multi_io(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid = RegularGrid.from_sampling(10, rpts_defs, tiling_defs_multi)
    json_path = Path("test_tiling_defs_multi_mismatch.json")
    e7grid.to_file(json_path)

    e7grid_from_file = RegularGrid.from_file(json_path)
    json_path.unlink()

    e7grid_from_file.to_grid_def(json_path)
    e7grid_from_def = RegularGrid.from_grid_def(json_path, 10)
    json_path.unlink()

    len_tiling_defs = 2
    assert e7grid_from_def._tiling_defs is not None  # noqa: SLF001
    assert len(e7grid_from_def._tiling_defs) == len_tiling_defs  # noqa: SLF001


def test_system_order_fail(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    try:
        _ = RegularGrid.from_sampling(
            10, rpts_defs, tiling_defs_multi, system_order=["a"]
        )
        raise AssertionError
    except ValueError:
        assert True


def test_system_order_one(
    rpts_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid = RegularGrid.from_sampling(
        10, rpts_defs, tiling_defs_multi, system_order=["e7eu"]
    )
    assert len(e7grid) == 1


def test_system_tiles_different_order(
    euas_defs: dict[str, ProjSystemDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    euas_bbox = (49.5, 58, 50.5, 58.5)
    e7grid = RegularGrid.from_sampling(
        10, euas_defs, tiling_defs_multi, system_order=["e7as", "e7eu"]
    )
    tilenames = [
        tile.name for tile in e7grid.get_tiles_in_geog_bbox(euas_bbox, tiling_id="t2")
    ]
    assert tilenames == [
        "e7as_X018Y028T02",
        "e7as_X019Y028T02",
        "e7as_X018Y029T02",
        "e7as_X019Y029T02",
        "e7eu_X72Y30T02",
        "e7eu_X73Y30T02",
    ]

    e7grid = RegularGrid.from_sampling(
        10, euas_defs, tiling_defs_multi, system_order=["e7eu", "e7as"]
    )
    tilenames = [
        tile.name for tile in e7grid.get_tiles_in_geog_bbox(euas_bbox, tiling_id="t2")
    ]
    assert tilenames == [
        "e7eu_X72Y30T02",
        "e7eu_X73Y30T02",
        "e7as_X018Y028T02",
        "e7as_X019Y028T02",
        "e7as_X018Y029T02",
        "e7as_X019Y029T02",
    ]

    e7grid = RegularGrid.from_sampling(
        10, euas_defs, tiling_defs_multi, system_order=["e7eu"]
    )
    tilenames = [
        tile.name for tile in e7grid.get_tiles_in_geog_bbox(euas_bbox, tiling_id="t2")
    ]
    assert tilenames == ["e7eu_X72Y30T02", "e7eu_X73Y30T02"]


if __name__ == "__main__":
    pass
