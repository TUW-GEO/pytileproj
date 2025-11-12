from morecantile.models import TileMatrixSet
from pydantic import BaseModel, AfterValidator, NonNegativeInt
from typing import Annotated, Optional, Tuple, List
import pyproj
from osgeo import ogr
import json
from pathlib import Path
import numpy as np

from pytileproj.tile import ProjTile, RegularTile, IrregularTile
from pytileproj.geom import get_lonlat_sref, transform_geom_to_geog, rasterise_polygon
from pytileproj.grid import RegularGrid, IrregularGrid


def validate_grids(grids: List[RegularGrid] | None) -> List[RegularGrid] | None:
    if grids is not None:
        ref_grid = grids[0]
        for grid in grids[1:]:
            same_extent = ref_grid.extent == grid.extent
            if not same_extent:
                raise ValueError(f"The given grids do not have the same extent: {ref_grid.tiling_level}:{ref_grid.extent} vs. {grid.tiling_level}:{grid.extent}")
            
            same_origin = ref_grid.origin_xy == grid.origin_xy
            if not same_origin:
                raise ValueError(f"The given grids do not have the same origin: {ref_grid.tiling_level}:{ref_grid.origin_xy} vs. {grid.tiling_level}:{grid.origin_xy}")
            
            same_orientation = ref_grid.axis_orientation == grid.axis_orientation
            if not same_orientation:
                raise ValueError(f"The given grids do not have the same axis orientation: {ref_grid.tiling_level}:{ref_grid.axis_orientation} vs. {grid.tiling_level}:{grid.axis_orientation}")
            
            correct_order = ref_grid.tiling_level < grid.tiling_level
            if not correct_order:
                raise ValueError(f"The given grids need to be in order, from low to high tiling levels.")
            
            ref_grid = grid

    return grids


class ProjCoord:
    x: float
    y: float
    epsg: int


class ProjSystemBase(BaseModel):
    epsg: int
    
    _proj_zone: ogr.Geometry
    _to_geographic: pyproj.Transformer
    _from_geographic: pyproj.Transformer
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        this_crs = pyproj.CRS(self.epsg)
        geog_crs = pyproj.CRS(4326)
        self._to_geographic = pyproj.Transformer.from_crs(
                this_crs, geog_crs, always_xy=True
            )
        self._from_geographic = pyproj.Transformer.from_crs(
                geog_crs, this_crs, always_xy=True
            )
    
    def _lonlat_inside_proj(self, lon: float, lat: float) -> bool:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        point.AssignSpatialReference(get_lonlat_sref())

        return point.Within(self._proj_zone)

    def lonlat_to_xy(self, lon: float, lat: float) -> ProjCoord | None:
        coord = None
        if self._lonlat_inside_proj(lon, lat):
            x, y = self._from_geographic.transform(lon, lat)
            coord = ProjCoord(x, y, self.epsg)

        return coord
    
    def xy_to_lonlat(self, x: float, y: float) -> ProjCoord | None:
        lon, lat = self._to_geographic.transform(x, y)
        coord = ProjCoord(lon, lat, 4326)
        if self._lonlat_inside_proj(lon, lat):
            coord = None

        return coord


class GridSystemBase(BaseModel):
    name: str
    grids: List[RegularGrid | IrregularGrid]

    @classmethod
    def from_file(cls, json_path: Path):
        with open(json_path) as f:
            pp_def = json.load(f)

        return cls(**pp_def)
    
    def to_file(self, json_path: Path):
        pp_def = self.model_dump_json(indent=2)
        with open(json_path, "w") as f:
            f.writelines(pp_def)    

    def _create_tilename(self, tile: RegularTile | IrregularTile) -> str:
        raise NotImplementedError

    def _create_tile(self, tilename: str) -> RegularTile | IrregularTile:
        raise NotImplementedError
    
    def _tilenames_at_level(self, tiling_level: int):
        grid = self[tiling_level]
        for tile in grid:
            yield self._create_tilename(tile)

    def _tiles_at_level(self, tiling_level: int):
        grid = self[tiling_level]
        for tile in grid:
            yield tile

    def __len__(self) -> int:
        return len(self.grids)
    
    def __getitem__(self, tiling_level: int) -> RegularGrid:
        return self.grids[tiling_level]


class ProjGridSystemBase(GridSystemBase, ProjSystemBase):
    tiles_in_zone_only: bool = True

    def create_tile(self, tilename: str) -> ProjTile:
        raise NotImplementedError

    def get_tile_bbox_geog(self, tilename: str) -> ogr.Geometry:
        proj_tile = self.create_tile(tilename)
        return transform_geom_to_geog(proj_tile.boundary_ogr)
    
    def _search_tiles_in_bbox_geog(self, bbox: tuple[float, float, float, float], tiling_level: int) -> list[RegularTile | IrregularTile]:
        raise NotImplementedError
    
    def tile_mask(self, proj_tile: ProjTile) -> np.ndarray:
        intrsct_geom = self._proj_zone.Intersection(proj_tile.boundary_ogr)
        if intrsct_geom.Area() == 0.0:
            mask = np.zeros(proj_tile.shape, dtype=np.uint8)
        else:
            # first, using 'outer_boundary_extent' as a pixel buffer for generating the rasterised
            # pixel skeleton
            # second, reduce this pixel buffer again to the coordinate extent by skipping the last
            # row and column
            mask = rasterise_polygon(intrsct_geom, proj_tile.x_pixel_size, proj_tile.y_pixel_size,
                                    proj_tile.outer_boundary_extent)[:-1, :-1]
        return mask


class RegularProjGridSystem(ProjGridSystemBase):
    grids: Annotated[List[RegularGrid], AfterValidator(validate_grids)]
    
    _tms: TileMatrixSet

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        wkt = pyproj.CRS(self.epsg).to_json_dict()
        self._tms = TileMatrixSet(crs={"wkt": wkt}, 
                                  tileMatrices=[grid._tm for grid in self.grids])
    
    @property
    def axis_orientation(self) -> tuple[str, str]:
        return self[0].axis_orientation
    
    @property
    def max_n_tiles_x(self) -> int:
        return self[-1].tm.matrixWidth
    
    @property
    def max_n_tiles_y(self) -> int:
        return self[-1].tm.matrixHeight
    
    def n_tiles(self, tiling_level: int) -> int:
        return self[tiling_level].tm.matrixHeight * self[tiling_level].tm.matrixWidth
    
    @classmethod
    def default(cls,
                name: str, 
                epsg: int,
                extent:  Tuple[float, float, float, float],
                tile_shape_px: Tuple[NonNegativeInt, NonNegativeInt],
                tiling_level_limits: Optional[Tuple[NonNegativeInt, NonNegativeInt]] = (0, 24)) -> 'RegularProjGridSystem':
            
            min_zoom, max_zoom = tiling_level_limits 
            tms = TileMatrixSet.custom(extent, pyproj.CRS(epsg), 
                                                tile_width=tile_shape_px[0], 
                                                tile_height=tile_shape_px[1],
                                                minzoom=min_zoom,
                                                maxzoom=max_zoom)
            grids = []
            for i, tm in enumerate(tms.tileMatrices):
                grid = RegularGrid(name=str(i), 
                            extent=tms.bbox, 
                            sampling=tm.cellSize,
                            )
                grids.append(grid)

            return cls(name, epsg, grids)

    def _create_tilename(self, tile: RegularTile) -> str:
        x_ori, y_ori = self.axis_orientation
        n_digits_xy = len(str(max(self.max_n_tiles_x, self.max_n_tiles_y)))
        n_digits_z = len(str(len(self) - 1))
        return  f"{x_ori}{tile.x:0{n_digits_xy}}{y_ori}{tile.y:0{n_digits_xy}}T{tile.z:0{n_digits_z}}"
    
    def _create_tile(self, tilename: str) -> RegularTile:
        _, y_ori = self.axis_orientation
        tiling_level = int(tilename.split("T")[-1])
        x = int(tilename.split(y_ori)[0][1:])
        y = int(tilename.split(y_ori)[1].split("T")[0])
        tile = RegularTile(x, y, tiling_level)
        return tile

    def create_tile(self, tilename: str) -> ProjTile:
        tile = self._create_tile(tilename)
        proj_tile = self._to_proj_tile(tile, name=tilename) 
        if self.tiles_in_zone_only:
            bbox_geog = transform_geom_to_geog(proj_tile.boundary_ogr)
            if not bbox_geog.Intersect(self._proj_zone):
                proj_tile = None
        
        return proj_tile
    
    def _to_proj_tile(self, tile: RegularTile, name: str = None) -> ProjTile:
        extent = self._tms.xy_bounds(tile)
        sampling = self[tile.z].sampling
        return ProjTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def _search_tiles_in_bbox_geog(self, bbox, tiling_level):
        min_x, min_y, max_x, max_y = bbox
        return self._tms.tiles(min_x, min_y, max_x, max_y, [tiling_level])

    def search_tiles_in_lonlat_bbox(self, bbox: tuple[float, float, float, float], tiling_level: int):
        for tile in self._search_tiles_in_bbox_geog(bbox, tiling_level):
            tilename = self._create_tilename(tile)
            proj_tile = self._to_proj_tile(tile, name=tilename)
            if self.tiles_in_zone_only:
                if not transform_geom_to_geog(proj_tile.boundary_ogr).Intersect(self._proj_zone):
                    continue

            yield proj_tile
    

class IrregularProjGridSystem(ProjGridSystemBase):

    def _create_tilename(self, tile: IrregularTile) -> str:
        return tile.id
    
    def _tilename_to_level(self, tilename: str) -> str:
        tiling_level = int(tilename.split("T")[-1])
        return tiling_level
    
    def _create_tile(self, tilename: str) -> RegularTile:
        tiling_level = self._tilename_to_level(tilename)
        return self[tiling_level].tiles[tilename]

    def create_tile(self, tilename: str) -> ProjTile:
        tile = self._create_tile(tilename)
        proj_tile = self._to_proj_tile(tile, name=tilename) 
        if self.tiles_in_zone_only:
            bbox_geog = transform_geom_to_geog(proj_tile.boundary_ogr)
            if not bbox_geog.Intersect(self._proj_zone):
                proj_tile = None
        
        return proj_tile

    def get_tile_bbox_proj(self, tilename: str) -> ogr.Geometry:
        proj_tile = self.create_tile(tilename)
        return proj_tile.boundary_ogr
    
    def _to_proj_tile(self, tile: IrregularTile, name: str = None) -> ProjTile:
        extent = tile.boundary.bounds
        sampling = self[tile.z].sampling
        return ProjTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def search_tiles_in_lonlat_bbox(self, bbox: tuple[float, float, float, float], tiling_level: int):
        for tile in self[tiling_level].tiles_in_bbox(bbox):
            tilename = self._create_tilename(tile)
            proj_tile = self._to_proj_tile(tile, name=tilename)
            if self.tiles_in_zone_only:
                if not transform_geom_to_geog(proj_tile.boundary_ogr).Intersect(self._proj_zone):
                    continue

            yield proj_tile
    
