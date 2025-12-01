import json
from pathlib import Path

from pydantic import BaseModel

from pytileproj.tiling_system import (
    RegularProjTilingSystem,
    RegularTilingDefinition,
    RPTSDefinition,
)


class RegularGrid(BaseModel, extra="allow"):
    """Defines a collection of regular, projected, multi-level tiling systems sharing the same tiling scheme."""

    _rpts_defs: dict[int, RPTSDefinition]
    _tiling_defs: dict[int, RegularTilingDefinition]
    _allowed_samplings: dict[int, list[float | int]]
    _congruent: bool

    def __init__(
        self, tiling_systems: dict[str, RegularProjTilingSystem] | None = None
    ):
        """
        Constructs a regular grid object from a collection of sub-grids represented by a dictionary of regular, projected tiling systems.

        Parameters
        ----------
        tiling_systems: Dict[str, RegularProjTilingSystem] | None
            Dictionary with the name of the tiling system as a key and a `RegularProjTilingSystem` instance as a value.

        """
        super().__init__(**tiling_systems)

    @staticmethod
    def _create_rpts_from_def(
        rpts_def: RPTSDefinition,
        sampling: float | int | dict[int, float | int],
        tiling_defs: dict[int, RegularTilingDefinition],
        allowed_samplings: dict[int, list[float | int]] | None = None,
        congruent: bool = False,
    ) -> RegularProjTilingSystem:
        """
        Creates a regular, projected tiling system instance from given tiling system definitions and a grid sampling.

        Parameters
        ----------
        rpts_def: RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent, and axis orientation).
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with tiling levels as keys and samplings as values.
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        allowed_samplings: Dict[int, List[float | int]] | None, optional
            Dictionary with tiling levels as keys and allowed samplings as values. Defaults to None, which means there are no restrictions for the specified sampling.
        congruent: bool, optional
            If true, then tilings from adjacent tiling levels need to be congruent, which means that tiles from the higher tiling level need to be exactly in one tile of the lower level.
            Defaults to false.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        return RegularProjTilingSystem.from_sampling(
            sampling, rpts_def, tiling_defs, allowed_samplings, congruent
        )

    @classmethod
    def from_sampling(
        cls,
        sampling: float | int | dict[int, float | int],
        rpts_defs: dict[str, RPTSDefinition],
        tiling_defs: dict[int, RegularTilingDefinition],
        allowed_samplings: dict[int, list[float | int]] | None = None,
        congruent: bool = False,
    ) -> "RegularGrid":
        """
        Creates a regular grid instance from given tiling system definitions and a grid sampling.

        Parameters
        ----------
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with tiling levels as keys and samplings as values.
        rpts_def: RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent, and axis orientation).
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        allowed_samplings: Dict[int, List[float | int]] | None, optional
            Dictionary with tiling levels as keys and allowed samplings as values. Defaults to None, which means there are no restrictions for the specified sampling.
        congruent: bool, optional
            If true, then tilings from adjacent tiling levels need to be congruent, which means that tiles from the higher tiling level need to be exactly in one tile of the lower level.
            Defaults to false.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        tiling_systems = {}
        for name, rpts_def in rpts_defs.items():
            tiling_systems[name] = cls._create_rpts_from_def(
                rpts_def, sampling, tiling_defs, allowed_samplings, congruent
            )

        rgrid = cls(**tiling_systems)
        rgrid._rpts_defs = rpts_defs
        rgrid._tiling_defs = tiling_defs
        rgrid._allowed_samplings = allowed_samplings
        rgrid._congruent = congruent

        return rgrid

    @classmethod
    def from_grid_def(
        cls, json_path: Path, sampling: float | int | dict[int, float | int]
    ) -> "RegularGrid":
        """
        Creates a regular grid instance from given tiling system definitions stored in a JSON file and a grid sampling.

        Parameters
        ----------
        json_path: Path
            Path to JSON file storing grid definition.
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary with tiling levels as keys and samplings as values.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        with open(json_path) as f:
            grid_def = json.load(f)

        rpts_defs = {
            name: RPTSDefinition(**rpts_def)
            for name, rpts_def in grid_def["rpts_defs"].items()
        }
        tiling_defs = {
            int(name): RegularTilingDefinition(**rpts_def)
            for name, rpts_def in grid_def["tiling_defs"].items()
        }
        allowed_samplings = {
            int(k): v for k, v in grid_def["allowed_samplings"].items()
        }
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
        """
        Creates required regular tiling system definitions from the tiling systems of the regular grid.

        Returns
        -------
        RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent, and axis orientation).
        Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        Dict[int, List[float | int]]
            Dictionary with tiling levels as keys and allowed samplings as values.
        bool
            Are the tiles in the regular tiling systems of the grid congruent?

        Notes
        -----
        Attention:
        The returned tiling definitions will only represent the tiling levels, which fit to the allowed samplings and the sampling of the current regular grid instance.

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
                name=name, epsg=rpts.epsg, extent=rpts[ref_tiling_level].extent
            ).model_dump()

        return rpts_defs, tiling_defs, allowed_samplings, congruent

    def _fetch_ori_grid_def(self):
        """
        Creates regular tiling system definitions from the attributes stored in the regular grid.

        Returns
        -------
        RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent, and axis orientation).
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

    def to_grid_def(self, json_path: Path):
        """
        Writes the regular grid definition to a JSON file.

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
        grid_def = json.dumps(grid_def, indent=2)
        with open(json_path, "w") as f:
            f.writelines(grid_def)

    @classmethod
    def from_file(cls, json_path: Path) -> "RegularGrid":
        """
        Creates a regular grid instance from its JSON representation stored within the given file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        Returns
        -------
        RegularGrid
            Regular grid instance.

        """
        with open(json_path) as f:
            pp_def = json.load(f)

        return cls(**pp_def)

    def to_file(self, json_path: Path):
        """
        Writes the JSON representation of the regular grid instance to a file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        """
        pp_def = self.model_dump_json(indent=2)
        with open(json_path, "w") as f:
            f.writelines(pp_def)

    def __getitem__(self, name: str) -> RegularProjTilingSystem:
        """
        Returns a regular, projected tiling system instance.

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
