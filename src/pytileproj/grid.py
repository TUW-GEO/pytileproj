# Copyright (c) 2026, TU Wien
# Licensed under the MIT License. See LICENSE file.

"""Grid module defining regular and irregular grids."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import orjson
import shapely
from pydantic import BaseModel, PrivateAttr, TypeAdapter, model_validator

from pytileproj._const import JSON_INDENT
from pytileproj._errors import GeomOutOfZoneError
from pytileproj._types import RasterTileGenerator, SamplingFloatOrMap, T_co
from pytileproj.projgeom import GeogCoord, ProjCoord
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    PSD,
    RPTS,
    ProjSystemDefinition,
    RegularProjTilingSystem,
    RegularTilingDefinition,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RG = TypeVar("RG", bound="RegularGrid[Any]")
SYSTEM_ORDER_NAME = "system_order"

__all__ = ["RegularGrid"]


class RegularGrid(BaseModel, Generic[T_co], extra="allow"):
    """Define a regular grid.

    A collection of regular, projected, multi-level tiling systems
    sharing the same tiling scheme.

    """

    system_order: list[str] | None = None

    _proj_defs: Mapping[str, ProjSystemDefinition] = PrivateAttr()
    _tiling_defs: Mapping[int, RegularTilingDefinition] = PrivateAttr()

    _rpts_cls = RegularProjTilingSystem

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the projection system object."""
        super().model_post_init(context)
        if self.system_order is None:
            self.system_order = self._get_system_order()

    def _get_system_order(self) -> list[str]:
        """Return internal tiling system order."""
        return sorted(set(dict(self).keys()) - {SYSTEM_ORDER_NAME})

    @model_validator(mode="after")
    def check_system_order(self) -> Self:
        """Check if system orders correspond."""
        internal_system_order = self._get_system_order()
        system_order = self.system_order or []
        for ts_name in system_order:
            if ts_name not in internal_system_order:
                err_msg = (
                    f"The tiling system '{ts_name}' specified "
                    "in the tiling system order is not available."
                )
                raise ValueError(err_msg)

        return self

    @staticmethod
    def _create_rpts_from_def(
        proj_def: ProjSystemDefinition,
        sampling: SamplingFloatOrMap,
        tiling_defs: Mapping[int, RegularTilingDefinition],
    ) -> RPTS:
        """Create regular projected tiling system from grid definitions.

        Create a regular, projected tiling system instance from given tiling system
        definitions and a grid sampling.

        Parameters
        ----------
        proj_def: ProjSystemDefinition
            Projection system definition (stores name, CRS, extent,
            and axis orientation).
        sampling: float | int | Dict[int | str, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling IDs as keys and samplings as values.
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        return RegularProjTilingSystem.from_sampling(  # type: ignore[reportReturnType]
            sampling,
            proj_def,
            tiling_defs,
        )

    @staticmethod
    def allowed_samplings(tile_size: float) -> list[float]:
        """Compute samplings which fit into the given tile size.

        Parameters
        ----------
        tile_size: float
            Tile size.

        Returns
        -------
        list[float]
            Divisors/samplings of the given tile size.

        """
        return RegularTiling.allowed_samplings(tile_size)

    @classmethod
    def from_sampling(
        cls,
        sampling: SamplingFloatOrMap,
        proj_defs: Mapping[str, ProjSystemDefinition],
        tiling_defs: Mapping[int, RegularTilingDefinition],
        system_order: list[str] | None = None,
    ) -> Self:
        """Create a regular grid from grid definitions.

        Create a regular grid instance from given tiling system definitions
        and a grid sampling.

        Parameters
        ----------
        sampling: float | int | Dict[int | str, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling IDs as keys and samplings as values.
        proj_defs: Mapping[str, ProjSystemDefinition]
            Projection system definitions (stores name, CRS, extent,
            and axis orientation).
        tiling_defs: Mapping[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        system_order: list[str] | None, optional
            Defines the usage and order of the tiling systems.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        tiling_systems = {}
        for name, rpts_def in proj_defs.items():
            if (system_order is not None) and (name not in system_order):
                continue

            tiling_systems[name] = cls._create_rpts_from_def(
                rpts_def,
                sampling,
                tiling_defs,
            )

        rgrid = cls(system_order=system_order, **tiling_systems)
        rgrid._proj_defs = proj_defs
        rgrid._tiling_defs = tiling_defs

        return rgrid

    @classmethod
    def from_grid_def(
        cls,
        json_path: Path,
        sampling: SamplingFloatOrMap,
        system_order: list[str] | None = None,
    ) -> Self:
        """Create a regular grid from a grid definition file.

        Create a regular grid instance from given tiling system definitions stored
        in a JSON file and a grid sampling.

        Parameters
        ----------
        json_path: Path
            Path to JSON file storing grid definition.
        sampling: float | int | Dict[int | str, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling IDs as keys and samplings as values.
        system_order: list[str] | None, optional
            Defines the usage and order of the tiling systems.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        with json_path.open() as f:
            grid_def = orjson.loads(f.read())

        rpts_defs = {
            name: ProjSystemDefinition(**rpts_def)
            for name, rpts_def in grid_def["proj_defs"].items()
        }
        tiling_defs = {
            int(name): RegularTilingDefinition(**rpts_def)
            for name, rpts_def in grid_def["tiling_defs"].items()
        }

        return cls.from_sampling(
            sampling=sampling,
            proj_defs=rpts_defs,
            tiling_defs=tiling_defs,
            system_order=system_order,
        )

    def get_systems_from_lonlat(self, lon: float, lat: float) -> list[RPTS]:
        """Get regular, projected tiling system from geographic coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        list[RegularProjTilingSystem]
            The regular, projected tiling system which intersects with
            the given coordinate.

        """
        coord = GeogCoord(x=lon, y=lat)
        return self.get_systems_from_coord(coord)

    def get_systems_from_coord(self, coord: ProjCoord) -> list[RPTS]:
        """Get regular, projected tiling system from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates.

        Returns
        -------
        list[RegularProjTilingSystem]
            The regular, projected tiling systems which intersect with
            the given coordinate.

        """
        rptss = []
        system_order = self.system_order or []
        for ts_name in system_order:
            if coord in self[ts_name]:
                rpts = self[ts_name]
                rptss.append(rpts)

        if len(rptss) == 0:
            raise GeomOutOfZoneError(shapely.Point((coord.x, coord.y)))

        return rptss

    def lonlat_to_xy(self, lon: float, lat: float) -> Mapping[str, ProjCoord]:
        """Convert geographic to projected coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        Mapping[str, ProjCoord]
            X and Y coordinate for each tiling system corresponding
            to the given geographic point. Tiling system names
            serve as key.

        Raises
        ------
        GeomOutOfZoneError
            If the given point is outside the projection boundaries.

        """
        proj_coords = {}
        for rpts in self.get_systems_from_lonlat(lon, lat):
            proj_coords[rpts.name] = rpts.lonlat_to_xy(lon, lat)

        return proj_coords

    def _fetch_mod_grid_def(
        self,
    ) -> tuple[
        Mapping[str, ProjSystemDefinition[T_co]], Mapping[int, RegularTilingDefinition]
    ]:
        """Create regular grid system definitions.

        Create required regular tiling system definitions from the tiling systems
        of the regular grid.

        Returns
        -------
        Dict[str, ProjSystemDefinition]
            Projection system definitions (stores name, CRS, extent,
            and axis orientation).
        Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).

        Notes
        -----
        The returned tiling definitions will only represent the tiling levels, which fit
        to the sampling of the current regular grid instance.

        """
        proj_defs = {}
        tiling_defs = {}
        ref_tiling_level = None
        system_order = self.system_order or []
        for name in system_order:
            rpts = self[name]
            if ref_tiling_level is None:
                ref_tiling_level = rpts.tiling_levels[0]
                for tiling_level in rpts.tiling_levels:
                    tile_shape = rpts[tiling_level].tile_shape
                    tiling_def = RegularTilingDefinition(
                        name=rpts[tiling_level].name, tile_shape=tile_shape
                    )
                    tiling_defs[tiling_level] = tiling_def
            proj_defs[name] = ProjSystemDefinition(
                name=name,
                crs=rpts.crs,
                min_xy=rpts[ref_tiling_level].extent[:2],
                max_xy=rpts[ref_tiling_level].extent[2:],
                axis_orientation=rpts[ref_tiling_level].axis_orientation,
            )

        return proj_defs, tiling_defs

    def to_grid_def(self, json_path: Path) -> None:
        """Write the regular grid definition to a JSON file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file, where the grid definition should be stored.

        """
        if not hasattr(self, "_proj_defs") or not hasattr(self, "_tiling_defs"):
            proj_defs, tiling_defs = self._fetch_mod_grid_def()
        else:
            proj_defs, tiling_defs = self._proj_defs, self._tiling_defs

        write_grid_def(json_path, proj_defs, tiling_defs)

    @staticmethod
    def _validate_json(
        grid_def: str,
        rgrid_cls: RegularGrid,
        rpts_cls: RegularProjTilingSystem,
    ) -> RG:
        """Create a regular grid object from the JSON class representation.

        Parameters
        ----------
        grid_def: str
            JSON string representing a regular grid instance.
        rgrid_cls: class
            Regular grid class.
        rpts_cls: class
            Regular projected tiling system class.

        Returns
        -------
        RegularGrid
            Regular grid.

        """
        rgrid = TypeAdapter(rgrid_cls).validate_json(grid_def)
        rpts_names = list(rgrid.model_dump().keys())
        for rpts_name in rpts_names:
            if rpts_name == SYSTEM_ORDER_NAME:
                continue
            setattr(
                rgrid,
                rpts_name,
                TypeAdapter(rpts_cls).validate_python(rgrid[rpts_name]),
            )

        return rgrid

    @classmethod
    def from_file(cls, json_path: Path) -> Self:
        """Create a regular grid instance from a file.

        Create a regular grid instance from its JSON representation stored
        within the given file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        with json_path.open() as f:
            pp_def = f.read()

        return cls._validate_json(pp_def, cls, cls._rpts_cls.default)  # type: ignore[reportArgumentType]

    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str = 0,
    ) -> RasterTileGenerator:
        """Get all tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_id: int | str
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTileGenerator
            Yields raster tile after tile, which intersects with the given
            bounding box.

        """
        system_order = self.system_order or []
        for ts_name in system_order:
            yield from self[ts_name].get_tiles_in_geog_bbox(bbox, tiling_id=tiling_id)

    def to_file(self, json_path: Path) -> None:
        """Write the JSON representation of the regular grid instance to a file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        """
        pp_def = self.model_dump_json(indent=JSON_INDENT)
        with json_path.open("w") as f:
            f.writelines(pp_def)

    def __getitem__(self, arg: str | ProjCoord) -> RPTS | list[RPTS]:
        """Return a regular, projected tiling system instance.

        Parameters
        ----------
        arg: str | ProjCoord
            Name/identifier of the projected tiling system instance or
            a projected coordinate.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        if isinstance(arg, str):
            rpts = getattr(self, arg)
        elif isinstance(arg, ProjCoord):
            rpts = self.get_systems_from_coord(arg)
        else:
            err_msg = f"Item type '{type(arg)}' is not supported."
            raise TypeError(err_msg)

        return rpts

    def __len__(self) -> int:
        """Get number of tiling systems."""
        return len(self._get_system_order())


def write_grid_def(
    json_path: Path,
    proj_defs: Mapping[str, PSD],
    tiling_defs: Mapping[int, RegularTilingDefinition],
) -> None:
    """Write grid definitions to a JSON file.

    Parameters
    ----------
    json_path: Path
            Path to JSON file.
    proj_defs: Mapping[str, ProjSystemDefinition]
            Projection system definitions (stores name, CRS, extent,
            and axis orientation).
    tiling_defs: Mapping[int, RegularTilingDefinition]
        Tiling definition (stores name/tiling level and tile size).

    """
    grid_def = {}
    grid_def["proj_defs"] = {k: v.model_dump() for k, v in proj_defs.items()}
    grid_def["tiling_defs"] = {k: v.model_dump() for k, v in tiling_defs.items()}
    grid_def = json.dumps(grid_def, indent=JSON_INDENT)
    with json_path.open("w") as f:
        f.writelines(grid_def)
