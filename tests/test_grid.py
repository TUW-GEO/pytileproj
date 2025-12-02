from pathlib import Path

import pytest
from pydantic_core import ValidationError

from pytileproj.grid import RegularGrid
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    RegularProjTilingSystem,
    RegularTilingDefinition,
    RPTSDefinition,
)


@pytest.fixture(scope="module")
def tiling_defs():
    return {1: RegularTilingDefinition(name="t1", tile_size=100_000)}


@pytest.fixture(scope="module")
def tiling_defs_multi():
    return {
        1: RegularTilingDefinition(name="t1", tile_size=100_000),
        2: RegularTilingDefinition(name="t2", tile_size=200_000),
    }


@pytest.fixture(scope="module")
def rpts_defs():
    return {
        "e7eu": RPTSDefinition(
            name="e7eu",
            epsg=27704,
            extent=[0, 0, 8_660_000, 6_020_000],
            axis_orientation=("E", "S"),
        ),
        "e7af": RPTSDefinition(
            name="e7af",
            epsg=27701,
            extent=[0, 0, 12_000_000, 9_600_000],
            axis_orientation=("E", "S"),
        ),
    }


@pytest.fixture(scope="module")
def e7eu_grid_t1():
    return RegularTiling(
        name="t1",
        extent=[0, 0, 8_660_000, 6_020_000],
        sampling=10,
        tile_shape_px=(10_000, 10_000),
        tiling_level=1,
        axis_orientation=["E", "S"],
    )


@pytest.fixture(scope="module")
def e7af_grid_t1():
    return RegularTiling(
        name="t1",
        extent=[0, 0, 12_000_000, 9_600_000],
        sampling=10,
        tile_shape_px=(10_000, 10_000),
        tiling_level=1,
        axis_orientation=["E", "S"],
    )


@pytest.fixture(scope="module")
def e7grid(e7eu_grid_t1: RegularTiling, e7af_grid_t1: RegularTiling):
    rpts_eu = RegularProjTilingSystem(
        name="e7eu",
        tilings={e7eu_grid_t1.tiling_level: e7eu_grid_t1},
        epsg=27704,
    )
    rpts_af = RegularProjTilingSystem(
        name="e7af",
        tilings={e7af_grid_t1.tiling_level: e7af_grid_t1},
        epsg=27701,
    )

    return RegularGrid(**{rpts_eu.name: rpts_eu, rpts_af.name: rpts_af})


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
    rpts_defs: dict[str, RegularTilingDefinition],
    tiling_defs: dict[int, RegularTilingDefinition],
    e7grid: RegularGrid,
):
    e7grid_from_def = RegularGrid.from_sampling(10, rpts_defs, tiling_defs)

    assert e7grid.model_dump() == e7grid_from_def.model_dump()


def test_tiling_defs_multi(
    rpts_defs: dict[str, RegularTilingDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid_to_def = RegularGrid.from_sampling(10, rpts_defs, tiling_defs_multi)
    json_path = Path("test_tiling_defs_multi.json")
    e7grid_to_def.to_grid_def(json_path)

    e7grid_from_def = RegularGrid.from_grid_def(json_path, 10)
    json_path.unlink()

    assert e7grid_from_def._rpts_defs == e7grid_to_def._rpts_defs  # noqa: SLF001
    assert e7grid_from_def._tiling_defs == e7grid_to_def._tiling_defs  # noqa: SLF001
    assert e7grid_from_def._allowed_samplings == e7grid_to_def._allowed_samplings  # noqa: SLF001
    assert e7grid_from_def._congruent == e7grid_to_def._congruent  # noqa: SLF001


def test_tiling_defs_multi_sampling_diff(
    rpts_defs: dict[str, RegularTilingDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid = RegularGrid.from_sampling(10, rpts_defs, tiling_defs_multi)
    json_path = Path("test_tiling_defs_multi_sampling_diff.json")
    e7grid.to_grid_def(json_path)

    e7grid_from_def = RegularGrid.from_grid_def(json_path, 1000)
    json_path.unlink()

    e7grid = RegularGrid.from_sampling(1000, rpts_defs, tiling_defs_multi)

    assert e7grid_from_def.model_dump() == e7grid.model_dump()


def test_tiling_defs_multi_allwd_wrng_smpl(
    rpts_defs: dict[str, RegularTilingDefinition],
    tiling_defs_multi: dict[int, RegularTilingDefinition],
):
    e7grid_to_def = RegularGrid.from_sampling(
        10, rpts_defs, tiling_defs_multi, allowed_samplings={1: [10]}
    )
    json_path = Path("test_tiling_defs_multi_allwd_wrng_smpl.json")
    e7grid_to_def.to_grid_def(json_path)

    try:
        _ = RegularGrid.from_grid_def(json_path, 100)
        raise AssertionError
    except ValidationError:
        assert True

    json_path.unlink()


def test_tiling_defs_multi_mismatch(
    rpts_defs: dict[str, RegularTilingDefinition],
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

    assert len(e7grid_from_def._tiling_defs) == 1  # noqa: SLF001


if __name__ == "__main__":
    pass
