# Copyright (c) 2025, TU Wien
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of the FreeBSD Project.

"""Grid module defining regular and irregular grids."""

import json
from pathlib import Path

import orjson
import pyproj
from pydantic import BaseModel, PrivateAttr, TypeAdapter

from pytileproj._const import GEOG_EPSG, JSON_INDENT
from pytileproj._errors import GeomOutOfZoneError
from pytileproj._types import RasterTileGenerator
from pytileproj.projgeom import ProjCoord
from pytileproj.tiling import RegularTiling
from pytileproj.tiling_system import (
    ProjSystemDefinition,
    RegularProjTilingSystem,
    RegularTilingDefinition,
)

__all__ = ["RegularGrid"]


class RegularGrid(BaseModel, extra="allow"):
    """Define a regular grid.

    A collection of regular, projected, multi-level tiling systems
    sharing the same tiling scheme.

    """

    _proj_defs: dict[str, ProjSystemDefinition] = PrivateAttr()
    _tiling_defs: dict[int, RegularTilingDefinition] = PrivateAttr()

    _rpts_cls = RegularProjTilingSystem

    def __init__(self, **rpts: RegularProjTilingSystem) -> None:
        """Initialise a regular grid object."""
        super().__init__(**rpts)

    @staticmethod
    def _create_rpts_from_def(
        proj_def: ProjSystemDefinition,
        sampling: float | dict[int, float | int],
        tiling_defs: dict[int, RegularTilingDefinition],
    ) -> RegularProjTilingSystem:
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
        return RegularProjTilingSystem.from_sampling(
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
        sampling: float | dict[int | str, float | int],
        proj_defs: dict[str, ProjSystemDefinition],
        tiling_defs: dict[int, RegularTilingDefinition],
    ) -> "RegularGrid":
        """Create a regular grid from grid definitions.

        Create a regular grid instance from given tiling system definitions
        and a grid sampling.

        Parameters
        ----------
        sampling: float | int | Dict[int | str, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling IDs as keys and samplings as values.
        proj_defs: dict[str, ProjSystemDefinition]
            Projection system definitions (stores name, CRS, extent,
            and axis orientation).
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        tiling_systems = {}
        for name, rpts_def in proj_defs.items():
            tiling_systems[name] = cls._create_rpts_from_def(
                rpts_def,
                sampling,
                tiling_defs,
            )

        rgrid = cls(**tiling_systems)
        rgrid._proj_defs = proj_defs
        rgrid._tiling_defs = tiling_defs

        return rgrid

    @classmethod
    def from_grid_def(
        cls, json_path: Path, sampling: float | dict[int | str, float | int]
    ) -> "RegularGrid":
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
        )

    def get_system_from_lonlat(self, lon: float, lat: float) -> RegularProjTilingSystem:
        """Get regular, projected tiling system from geographic coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        RegularProjTilingSystem
            The regular, projected tiling system which intersects with
            the given coordinate.

        """
        coord = ProjCoord(x=lon, y=lat, crs=pyproj.CRS.from_epsg(GEOG_EPSG))
        return self.get_system_from_coord(coord)

    def get_system_from_coord(self, coord: ProjCoord) -> RegularProjTilingSystem:
        """Get regular, projected tiling system from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates.

        Returns
        -------
        RegularProjTilingSystem
            The regular, projected tiling system which intersects with
            the given coordinate.

        """
        rpts_sel = None
        for rpts in dict(self).values():
            if coord in rpts:
                rpts_sel = rpts
                break

        if rpts_sel is None:
            err_msg = (
                f"The given coordinate ({coord}) is "
                "outside any tiling system boundaries."
            )
            raise GeomOutOfZoneError(err_msg)

        return rpts_sel

    def _fetch_mod_grid_def(
        self,
    ) -> tuple[dict[str, ProjSystemDefinition], dict[int, RegularTilingDefinition]]:
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
        rtps_names = list(self.model_dump().keys())
        for name in rtps_names:
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
        rgrid_cls: "RegularGrid",
        rpts_cls: RegularProjTilingSystem,
    ) -> "RegularGrid":
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
            setattr(
                rgrid,
                rpts_name,
                TypeAdapter(rpts_cls).validate_python(rgrid[rpts_name]),
            )

        return rgrid

    @classmethod
    def from_file(cls, json_path: Path) -> "RegularGrid":
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

        return cls._validate_json(pp_def, cls, cls._rpts_cls.default)

    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int = 0,
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
        for rpts in dict(self).values():
            yield from rpts.get_tiles_in_geog_bbox(bbox, tiling_id)

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

    def __getitem__(self, arg: str | ProjCoord) -> RegularProjTilingSystem:
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
            rpts = self.get_system_from_coord(arg)
        else:
            err_msg = f"Item type '{type(arg)}' is not supported."
            raise TypeError(err_msg)

        return rpts


def write_grid_def(
    json_path: Path,
    proj_defs: dict[str, ProjSystemDefinition],
    tiling_defs: dict[int, RegularTilingDefinition],
) -> None:
    """Write grid definitions to a JSON file.

    Parameters
    ----------
    json_path: Path
            Path to JSON file.
    proj_defs: dict[str, ProjSystemDefinition]
            Projection system definitions (stores name, CRS, extent,
            and axis orientation).
    tiling_defs: Dict[int, RegularTilingDefinition]
        Tiling definition (stores name/tiling level and tile size).

    """
    grid_def = {}
    grid_def["proj_defs"] = {k: v.model_dump() for k, v in proj_defs.items()}
    grid_def["tiling_defs"] = {k: v.model_dump() for k, v in tiling_defs.items()}
    grid_def = json.dumps(grid_def, indent=JSON_INDENT)
    with json_path.open("w") as f:
        f.writelines(grid_def)
