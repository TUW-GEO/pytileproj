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
"""
Code for Tiled Projection Systems.
"""

from enum import Enum
from typing import Annotated, Literal

import numpy as np
import shapely
from morecantile.models import Tile as RegularTile, TileMatrix
from pydantic import AfterValidator, BaseModel, NonNegativeFloat, NonNegativeInt
from shapely.geometry import Polygon

from pytileproj.tile import IrregularTile


class CornerOfOrigin(Enum):
    bottom_left = "bottomLeft"
    top_left = "topLeft"


class RegularGrid(BaseModel, arbitrary_types_allowed=True):
    name: str
    extent: tuple[float, float, float, float]
    sampling: NonNegativeFloat
    tile_shape_px: tuple[NonNegativeInt, NonNegativeInt]
    tiling_level: int | None = 0
    axis_orientation: tuple[Literal["W", "E"], Literal["N", "S"]] | None = ("E", "N")

    _tm: TileMatrix

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.axis_orientation = (
            "E",
            "S",
        )  # hardcoded because of the cornerOfOrigin issue below

        matrix_width = int(
            (self.extent[2] - self.extent[0]) / (self.tile_shape_px[0] * self.sampling)
        )
        matrix_height = int(
            (self.extent[3] - self.extent[1]) / (self.tile_shape_px[1] * self.sampling)
        )

        self._tm = TileMatrix(
            scaleDenominator=self.sampling / 0.28e-3,  # per OGC definition
            cellSize=self.sampling,
            cornerOfOrigin=CornerOfOrigin.top_left.value,  # unfortunately, this value is hardcoded within morecantile (see )
            pointOfOrigin=self.origin_xy,
            tileWidth=self.tile_shape_px[0],
            tileHeight=self.tile_shape_px[1],
            matrixWidth=matrix_width,
            matrixHeight=matrix_height,
            id=str(self.tiling_level),
            title=self.name,
        )

    @property
    def corner_of_origin(self) -> CornerOfOrigin:
        return (
            CornerOfOrigin.bottom_left
            if self.axis_orientation[1] == "N"
            else CornerOfOrigin.top_left
        )

    @property
    def origin_xy(self) -> tuple:
        return self.extent[0], self.extent[3]

    @property
    def n_tiles(self) -> int:
        return self._tm.matrixHeight * self._tm.matrixWidth

    @property
    def tm(self) -> TileMatrix:
        return self._tm

    def __iter__(self):
        for x in range(self._tm.matrixWidth):
            for y in range(self._tm.matrixHeight):
                yield RegularTile(x, y, self.tiling_level)

    def to_ogc_repr(self) -> dict:
        return self._tm.model_dump()


def validate_adj_matrix(input: np.ndarray | None) -> np.ndarray | None:
    if input is not None:
        if input.ndim != 2:
            err_msg = "Adjacency matrix is expected to be passed as a 2D numpy array."
            raise ValueError(err_msg)

    return input


class IrregularGrid(BaseModel, arbitrary_types_allowed=True):
    name: str
    tiles: dict[str, IrregularTile]
    adjacency_matrix: Annotated[
        np.ndarray | None, AfterValidator(validate_adj_matrix)
    ] = None
    tiling_level: int | None = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self._build_adjacency_matrix()

    @property
    def tile_ids(self) -> list[str]:
        return list(self.tiles.keys())

    def neighbours(self, tile_id: str) -> list[IrregularTile]:
        tile_idx = self.tile_ids.index(tile_id)
        nbr_idxs = self.adjacency_matrix[tile_idx, :]
        nbr_tiles = [self[tile_id] for tile_id in np.array(self.tile_ids)[nbr_idxs]]

        return nbr_tiles

    def tiles_intersecting_bbox(self, bbox: tuple[float, float, float, float]):
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

    def __iter__(self):
        yield from self.tiles.values()

    def __getitem__(self, tile_id: str) -> IrregularTile:
        return self.tiles[tile_id]


if __name__ == "__main__":
    pass
