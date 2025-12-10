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
from typing import Any

from pydantic import BaseModel, TypeAdapter

from pytileproj._const import JSON_INDENT
from pytileproj.tiling_system import (
    RegularProjTilingSystem,
    RegularTilingDefinition,
    RPTSDefinition,
)

__all__ = ["RegularGrid"]


class RegularGrid(BaseModel, extra="allow"):
    """Define a regular grid.

    A collection of regular, projected, multi-level tiling systems
    sharing the same tiling scheme.

    """

    _rpts_defs: dict[int, RPTSDefinition] | None = None
    _tiling_defs: dict[int, RegularTilingDefinition] | None = None
    _allowed_samplings: dict[int, list[float | int]] | None = None
    _congruent: bool = True

    _rpts_cls = RegularProjTilingSystem

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise a regular grid object."""
        super().__init__(**kwargs)

    @staticmethod
    def _create_rpts_from_def(
        rpts_def: RPTSDefinition,
        sampling: float | dict[int, float | int],
        tiling_defs: dict[int, RegularTilingDefinition],
        *,
        allowed_samplings: dict[int, list[float | int]] | None = None,
        congruent: bool = False,
    ) -> RegularProjTilingSystem:
        """Create regular projected tiling system from grid definitions.

        Create a regular, projected tiling system instance from given tiling system
        definitions and a grid sampling.

        Parameters
        ----------
        rpts_def: RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent,
            and axis orientation).
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling levels as keys and samplings as values.
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        allowed_samplings: Dict[int, List[float | int]] | None, optional
            Dictionary with tiling levels as keys and allowed samplings as values.
            Defaults to None, which means there are no restrictions for the specified
            sampling.
        congruent: bool, optional
            If true, then tilings from adjacent tiling levels need to be congruent,
            which means that tiles from the higher tiling level need to be exactly
            in one tile of the lower level. Defaults to false.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        return RegularProjTilingSystem.from_sampling(
            sampling,
            rpts_def,
            tiling_defs,
            allowed_samplings=allowed_samplings,
            congruent=congruent,
        )

    @classmethod
    def from_sampling(
        cls,
        sampling: float | dict[int, float | int],
        rpts_defs: dict[str, RPTSDefinition],
        tiling_defs: dict[int, RegularTilingDefinition],
        *,
        allowed_samplings: dict[int, list[float | int]] | None = None,
        congruent: bool = False,
    ) -> "RegularGrid":
        """Create a regular grid from grid definitions.

        Create a regular grid instance from given tiling system definitions
        and a grid sampling.

        Parameters
        ----------
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary
            with tiling levels as keys and samplings as values.
        rpts_defs: RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code,
            extent, and axis orientation).
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        allowed_samplings: Dict[int, List[float | int]] | None, optional
            Dictionary with tiling levels as keys and allowed samplings as values.
            Defaults to None, which means there are no restrictions for the
            specified sampling.
        congruent: bool, optional
            If true, then tilings from adjacent tiling levels need to be congruent,
            which means that tiles from the higher tiling level need to be exactly
            in one tile of the lower level. Defaults to false.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        tiling_systems = {}
        for name, rpts_def in rpts_defs.items():
            tiling_systems[name] = cls._create_rpts_from_def(
                rpts_def,
                sampling,
                tiling_defs,
                allowed_samplings=allowed_samplings,
                congruent=congruent,
            )

        rgrid = cls(**tiling_systems)
        rgrid._rpts_defs = rpts_defs
        rgrid._tiling_defs = tiling_defs
        rgrid._allowed_samplings = allowed_samplings
        rgrid._congruent = congruent

        return rgrid

    @classmethod
    def from_grid_def(
        cls, json_path: Path, sampling: float | dict[int, float | int]
    ) -> "RegularGrid":
        """Create a regular grid from a grid definition file.

        Create a regular grid instance from given tiling system definitions stored
        in a JSON file and a grid sampling.

        Parameters
        ----------
        json_path: Path
            Path to JSON file storing grid definition.
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with
            tiling levels as keys and samplings as values.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        with json_path.open() as f:
            grid_def = json.load(f)

        rpts_defs = {
            name: RPTSDefinition(**rpts_def)
            for name, rpts_def in grid_def["rpts_defs"].items()
        }
        tiling_defs = {
            int(name): RegularTilingDefinition(**rpts_def)
            for name, rpts_def in grid_def["tiling_defs"].items()
        }
        allowed_samplings = grid_def["allowed_samplings"]
        if allowed_samplings is not None:
            allowed_samplings = {int(k): v for k, v in allowed_samplings.items()}
        congruent = grid_def["congruent"]

        return cls.from_sampling(
            sampling=sampling,
            rpts_defs=rpts_defs,
            tiling_defs=tiling_defs,
            allowed_samplings=allowed_samplings,
            congruent=congruent,
        )

    def _fetch_mod_grid_def(
        self,
    ) -> tuple[
        dict[str, RPTSDefinition],
        dict[int, RegularTilingDefinition],
        dict[int, list[float | int]],
        bool,
    ]:
        """Create regular grid system definitions.

        Create required regular tiling system definitions from the tiling systems
        of the regular grid.

        Returns
        -------
        RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent,
            and axis orientation).
        Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        Dict[int, List[float | int]]
            Dictionary with tiling levels as keys and allowed samplings as values.
        bool
            Are the tiles in the regular tiling systems of the grid congruent?

        Notes
        -----
        The returned tiling definitions will only represent the tiling levels, which fit
        to the allowed samplings and the sampling of the current regular grid instance.

        """
        rpts_defs = {}
        tiling_defs = {}
        allowed_samplings = {}
        congruent = False
        ref_tiling_level = None
        rtps_names = list(self.model_dump().keys())
        for name in rtps_names:
            rpts = self[name]
            if ref_tiling_level is None:
                ref_tiling_level = rpts.tiling_levels[0]
                for tiling_level in rpts.tiling_levels:
                    tile_size = (
                        rpts[tiling_level].sampling
                        * rpts[tiling_level].tile_shape_px[0]
                    )
                    tiling_def = RegularTilingDefinition(
                        name=rpts[tiling_level].name, tile_size=tile_size
                    ).model_dump()
                    tiling_defs[tiling_level] = tiling_def
                allowed_samplings = rpts.allowed_samplings
                congruent = rpts.congruent
            rpts_defs[name] = RPTSDefinition(
                name=name,
                crs=rpts.crs,
                extent=rpts[ref_tiling_level].extent,
                axis_orientation=rpts[ref_tiling_level].axis_orientation,
            ).model_dump()

        return rpts_defs, tiling_defs, allowed_samplings, congruent

    def _fetch_ori_grid_def(
        self,
    ) -> tuple[
        dict[str, RPTSDefinition],
        dict[int, RegularTilingDefinition],
        dict[int, list[float | int]],
        bool,
    ]:
        """Create regular grid system definitions.

        Create regular grid definitions from the attributes
        stored in the regular grid.

        Returns
        -------
        RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code,
            extent, and axis orientation).
        Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        Dict[int, List[float | int]]
            Dictionary with tiling levels as keys and allowed samplings as values.
        bool
            Are the tiles in the regular tiling systems of the grid congruent?

        """
        rpts_def = {k: v.model_dump() for k, v in self._rpts_defs.items()}
        tilings_def = {k: v.model_dump() for k, v in self._tiling_defs.items()}

        return rpts_def, tilings_def, self._allowed_samplings, self._congruent

    def to_grid_def(self, json_path: Path) -> None:
        """Write the regular grid definition to a JSON file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file, where the grid definition should be stored.

        """
        if not self._rpts_defs:
            rpts_defs, tiling_defs, allowed_samplings, congruent = (
                self._fetch_mod_grid_def()
            )
        else:
            rpts_defs, tiling_defs, allowed_samplings, congruent = (
                self._fetch_ori_grid_def()
            )

        grid_def = {}
        grid_def["rpts_defs"] = rpts_defs
        grid_def["tiling_defs"] = tiling_defs
        grid_def["allowed_samplings"] = allowed_samplings
        grid_def["congruent"] = congruent
        grid_def = json.dumps(grid_def, indent=JSON_INDENT)
        with json_path.open("w") as f:
            f.writelines(grid_def)

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

    def __getitem__(self, name: str) -> RegularProjTilingSystem:
        """Return a regular, projected tiling system instance.

        Parameters
        ----------
        name: str
            Name/identifier of the projected tiling system instance.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        return getattr(self, name)
