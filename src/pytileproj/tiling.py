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

"""Tiling module defining classes for irregular and regular tilings."""

from collections.abc import Generator
from enum import Enum
from typing import Annotated, Any, Literal

import numpy as np
import shapely
from morecantile.models import Tile as RegularTile
from morecantile.models import TileMatrix
from pydantic import AfterValidator, BaseModel, NonNegativeFloat, model_validator
from shapely.geometry import Polygon

from pytileproj.tile import IrregularTile

__all__ = []


class CornerOfOrigin(Enum):
    """Defines a corner of origin in an OGC compliant manner."""

    bottom_left = "bottomLeft"
    top_left = "topLeft"


class RegularTiling(BaseModel, arbitrary_types_allowed=True):
    """Defines regular tiling scheme following the OGC standard."""

    name: str
    extent: tuple[float, float, float, float]
    sampling: NonNegativeFloat
    tile_shape: tuple[NonNegativeFloat, NonNegativeFloat]
    tiling_level: int | None = 0
    axis_orientation: tuple[Literal["W", "E"], Literal["N", "S"]] | None = ("E", "N")

    _tm: TileMatrix

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise regular tiling object."""
        super().__init__(**kwargs)

        tile_width, tile_height = (
            int(self.tile_shape[0] / self.sampling),
            int(self.tile_shape[1] / self.sampling),
        )
        matrix_width = int(
            np.ceil((self.extent[2] - self.extent[0]) / self.tile_shape[0])
        )
        matrix_height = int(
            np.ceil((self.extent[3] - self.extent[1]) / self.tile_shape[1])
        )

        self._tm = TileMatrix(
            scaleDenominator=self.sampling / 0.28e-3,  # per OGC definition
            cellSize=self.sampling,
            cornerOfOrigin=self.corner_of_origin,
            pointOfOrigin=self.origin_xy,
            tileWidth=tile_width,
            tileHeight=tile_height,
            matrixWidth=matrix_width,
            matrixHeight=matrix_height,
            id=str(self.tiling_level),
            title=self.name,
        )

    @model_validator(mode="after")
    def validate_sampling(self) -> "RegularTiling":
        """Check if sampling is a divisor of the tile shape."""
        err_msg = (
            "Tiling {}'s sampling {} is not allowed."
            " The following samplings are allowed: {} and {}"
        )
        width_okay = self.tile_shape[0] % self.sampling == 0
        height_okay = self.tile_shape[1] % self.sampling == 0
        if not (width_okay and height_okay):
            raise ValueError(
                err_msg.format(
                    self.name,
                    self.sampling,
                    RegularTiling.allowed_samplings(self.tile_shape[0]),
                    RegularTiling.allowed_samplings(self.tile_shape[1]),
                )
            )

        return self

    @property
    def corner_of_origin(self) -> CornerOfOrigin:
        """Corner of origin of the tiling."""
        return (
            CornerOfOrigin.bottom_left.value
            if self.axis_orientation[1] == "N"
            else CornerOfOrigin.top_left.value
        )

    @property
    def origin_xy(self) -> tuple:
        """Origin of the tiling."""
        return (
            (self.extent[0], self.extent[3])
            if self.corner_of_origin == CornerOfOrigin.top_left.value
            else (self.extent[0], self.extent[1])
        )

    @property
    def n_tiles(self) -> int:
        """Number of tiles in the tiling."""
        return self.n_tiles_x * self.n_tiles_y

    @property
    def n_tiles_x(self) -> int:
        """Number of tiles in X direction."""
        return self._tm.matrixWidth

    @property
    def n_tiles_y(self) -> int:
        """Number of tiles in Y direction."""
        return self._tm.matrixHeight

    @property
    def tm(self) -> TileMatrix:
        """Morecantile's TileMatrix instance."""
        return self._tm

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
        samplings = []
        for i in range(1, int(np.sqrt(tile_size)) + 1):
            if tile_size % i == 0:
                samplings.append(i)
                if i != tile_size // i:
                    samplings.append(tile_size // i)

        return sorted(samplings)

    def __iter__(self) -> Generator[RegularTile, RegularTile, RegularTile]:
        """Iterate over tiles in the tiling."""
        for x in range(self._tm.matrixWidth):
            for y in range(self._tm.matrixHeight):
                yield RegularTile(x, y, self.tiling_level)

    def to_ogc_standard(self) -> dict:
        """OGC representation of the tiling."""
        return self._tm.model_dump()


def validate_adj_matrix(ar: np.ndarray | None) -> np.ndarray | None:
    """Test if input array representing an adjacency matrix is 2D.

    Parameters
    ----------
    ar: np.ndarray | None
        Array representing

    Returns
    -------
    np.ndarray | None
        Forwarded input.

    Raises
    ------
    ValueError
        If input array is not 2D.

    """
    n_dims_adj = 2
    if ar is not None and (ar.ndim != n_dims_adj):
        err_msg = "Adjacency matrix is expected to be passed as a 2D numpy array."
        raise ValueError(err_msg)

    return ar


class IrregularTiling(BaseModel, arbitrary_types_allowed=True):
    """Define irregular tiling scheme."""

    name: str
    tiles: dict[str, IrregularTile]
    adjacency_matrix: Annotated[
        np.ndarray | None, AfterValidator(validate_adj_matrix)
    ] = None
    tiling_level: int | None = 0

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise irregular tiling object."""
        super().__init__(**kwargs)
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self._build_adjacency_matrix()

    @property
    def tile_ids(self) -> list[str]:
        """All tile ID's of the tiling."""
        return list(self.tiles.keys())

    def neighbours(self, tile_id: str) -> list[IrregularTile]:
        """Return the neighbouring tiles for a given tile ID.

        Parameters
        ----------
        tile_id: str
            Tile ID.

        Returns
        -------
        list[IrregularTile]
            List of neighbouring tiles.

        """
        tile_idx = self.tile_ids.index(tile_id)
        nbr_idxs = self.adjacency_matrix[tile_idx, :]

        return [self[tile_id] for tile_id in np.array(self.tile_ids)[nbr_idxs]]

    def tiles_intersecting_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> Generator[IrregularTile, IrregularTile, IrregularTile]:
        """Return tiles intersecting with the given bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.

        Returns
        -------
        Generator[IrregularTile, IrregularTile, IrregularTile]
            Yields tile after tile, which intersects with the given bounding box.

        """
        min_x, min_y, max_x, max_y = bbox
        bbox = Polygon(
            [
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, max_y),
                (min_x, min_y),
            ]
        )
        for tile in self.tiles.values():
            if shapely.intersects(tile.boundary, bbox):
                yield tile

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Create adjacency matrix based on tiles touching each other."""
        n_tiles = len(self.tiles)
        adjacency_matrix = np.zeros((n_tiles, n_tiles), dtype=bool)
        for i in range(n_tiles):
            for j in range(i, n_tiles):
                if i != j:
                    tile_i = self.tiles[self.tile_ids[i]]
                    tile_j = self.tiles[self.tile_ids[j]]
                    if shapely.touches(tile_i.boundary, tile_j.boundary):
                        adjacency_matrix[i, j] = True
                        adjacency_matrix[j, i] = True

        return adjacency_matrix

    def __iter__(self) -> Generator[IrregularTile, IrregularTile, IrregularTile]:
        """Yield one tile after the other."""
        yield from self.tiles.values()

    def __getitem__(self, tile_id: str) -> IrregularTile:
        """Return tile instance corresponding to the given tile ID.

        Parameters
        ----------
        tile_id: str
            Tile ID.

        Returns
        -------
        IrregularTile
            Tile instance corresponding to the given tile ID.

        """
        return self.tiles[tile_id]


if __name__ == "__main__":
    pass
