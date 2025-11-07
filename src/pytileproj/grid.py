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

from morecantile.models import TileMatrix, TileMatrixSet, Tile, CRS
from pydantic import BaseModel, AfterValidator, NonNegativeFloat, NonNegativeInt
from typing import Annotated, Literal, Optional, Tuple, List, Dict
import pyproj
from osgeo import ogr, osr
import requests
import warnings
import json
import shapely
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from pytileproj.geometry import transform_geometry

from pytileproj.tile import RegularTile, IrregularTile
from pytileproj.utils import fetch_proj_zone


class Grid(BaseModel, arbitrary_types_allowed=True):
    name: str
    extent:  Tuple[float, float, float, float]
    sampling: NonNegativeFloat
    origin_xy: Tuple[float, float]
    tile_shape_px: Tuple[NonNegativeInt, NonNegativeInt]
    tiling_level: Optional[NonNegativeInt] = 0
    axis_orientation: Optional[Tuple[Literal["W", "E"], Literal["N", "S"]]] = ("E", "N") 
    
    _tm: TileMatrix 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        corner_of_ori = 'bottomLeft' if self.axis_orientation[1] == 'N' else 'topLeft'
        matrix_width = int((self.extent[2] - self.extent[0]) / (self.tile_shape_px[0] * self.sampling))
        matrix_height = int((self.extent[3] - self.extent[1]) / (self.tile_shape_px[1] * self.sampling))

        self._tm = TileMatrix(scaleDenominator= self.sampling/ 0.28e-3,  # per OGC definition
                              cellSize=self.sampling,
                              cornerOfOrigin=corner_of_ori,
                              pointOfOrigin=self.origin_xy,
                              tileWidth=self.tile_shape_px[0],
                              tileHeight=self.tile_shape_px[1],
                              matrixWidth=matrix_width,
                              matrixHeight=matrix_height,
                              id=str(self.tiling_level), title=f"{self.name} at tiling level {self.tiling_level}")
    
    @property
    def n_tiles(self) -> int:
        return self._tm.matrixHeight * self._tm.matrixWidth
    

    def __iter__(self):
        for x in range(self._tm.matrixHeight):
            for y in range(self._tm.matrixWidth):
                yield RegularTile(x, y, self.tiling_level)


    def to_ogc_repr(self) -> dict:
        return self._tm.model_dump()


def validate_adj_matrix(input: np.ndarray | None) -> np.ndarray | None:
    if input is not None:
        if input.ndim != 2:
            err_msg = "Adjacency matrix is expected to be passed as a 2D numpy array."
            raise ValueError(err_msg)
    
    return input


class IrregularTileMatrix(BaseModel):

    tiles: Dict[IrregularTile]
    adjacency_matrix: Annotated[np.ndarray | None, AfterValidator(validate_adj_matrix)] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self._build_adjacency_matrix()

    @property
    def tile_ids(self) -> List[str]:
        return list(self.tiles.keys())
    
    def neighbours(self, tile_id: str) -> List[IrregularTile]:
        tile_idx = self.tile_ids.index(tile_id)
        nbr_idxs = self.adjacency_matrix[tile_idx, :]
        nbr_tiles = [self[self.tile_ids[nbr_idx]] for nbr_idx in nbr_idxs]

        return nbr_tiles
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        n_tiles = len(self.tiles)
        adjacency_matrix = np.zeros((n_tiles, n_tiles), dtype=bool)
        for i in range(n_tiles):
            for j in range(i, n_tiles):
                if i != j:
                    tile_i = self.tiles[self.tiles_ids[i]]
                    tile_j = self.tiles[self.tiles_ids[j]]
                    if shapely.touches(tile_i.boundary(), tile_j.boundary()):
                        adjacency_matrix[i, j] = True
                        adjacency_matrix[j, i] = True

        return adjacency_matrix
    
    def __iter__(self):
        for tile in self.tiles:
            yield tile
    
    def __getitem__(self, tile_id: str) -> IrregularTile:
        return self.tiles[tile_id]


if __name__ == "__main__":
    epsg = 27704
    extent = [0, 0, 300e3 * 20, 300e3 * 15]
    sampling = 10
    origin_xy = (0, 0)
    tile_shape_px = 10000, 10000
    tiling_level = 1
    grid_t1 = Grid(name="E7EUT1", extent=extent, sampling=sampling, origin_xy=origin_xy, tile_shape_px=tile_shape_px, tiling_level=tiling_level)
    sampling = 20
    tile_shape_px = 15000, 15000
    tiling_level = 0
    grid_t3 = Grid(name="E7EUT3", extent=extent, sampling=sampling, origin_xy=origin_xy, tile_shape_px=tile_shape_px, tiling_level=tiling_level)
    grids = [grid_t3, grid_t1]
    pp = ProjPyramid(name="Equi7Europe", epsg=epsg, grids=grids)
    pp.to_file("e7_def.json")
    pp2 = ProjPyramid.from_file("e7_def.json")
    pass