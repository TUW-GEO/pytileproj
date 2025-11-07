from morecantile.models import TileMatrixSet, Tile
from pydantic import BaseModel, AfterValidator, NonNegativeInt
from typing import Annotated, Optional, Tuple, List
import pyproj
from osgeo import ogr, osr
import json
from pathlib import Path
from shapely.geometry import Polygon
from pytileproj.geometry import transform_geometry

from pytileproj.tile import ProjTile
from pytileproj.utils import fetch_proj_zone
from pytileproj.grid import Grid 



def validate_grids(grids: List[Grid] | None) -> List[Grid] | None:
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


class ProjPyramid(BaseModel):
    name: str
    epsg: int
    grids: Annotated[List[Grid] | None, AfterValidator(validate_grids)]
    tiles_in_zone_only: bool = True

    _tms: TileMatrixSet | AbstractTMS
    _proj_zone: ogr.Geometry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        wkt = pyproj.CRS(self.epsg).to_json_dict()
        self._tms = TileMatrixSet(crs={"wkt": wkt}, 
                                  tileMatrices=[grid._tm for grid in self.grids])
        self._proj_zone = fetch_proj_zone(self.epsg) if self.epsg else None
    
    @property
    def axis_orientation(self) -> tuple[str, str]:
        return self.grids[0].axis_orientation
    
    @property
    def max_n_tiles_x(self) -> int:
        return self.grids[-1]._tm.matrixWidth
    
    @property
    def max_n_tiles_y(self) -> int:
        return self.grids[-1]._tm.matrixHeight
    

    @classmethod
    def default(cls,
                name: str, 
                epsg: int,
                extent:  Tuple[float, float, float, float],
                tile_shape_px: Tuple[NonNegativeInt, NonNegativeInt],
                tiling_level_limits: Optional[Tuple[NonNegativeInt, NonNegativeInt]] = (0, 24)) -> 'ProjPyramid':
            
            min_zoom, max_zoom = tiling_level_limits 
            tms = TileMatrixSet.custom(extent, pyproj.CRS(epsg), 
                                                tile_width=tile_shape_px[0], 
                                                tile_height=tile_shape_px[1],
                                                minzoom=min_zoom,
                                                maxzoom=max_zoom)
            grids = []
            for i, tm in enumerate(tms.tileMatrices):
                grid = Grid(name=str(i), 
                            extent=tms.bbox, 
                            sampling=tm.cellSize,
                            )
                grids.append(grid)

            return cls(name, epsg, grids)
    
    @classmethod
    def from_file(cls, json_path: Path) -> 'ProjPyramid':
        with open(json_path) as f:
            pp_def = json.load(f)

        return cls(**pp_def)
    
    def to_file(self, json_path: Path):
        pp_def = self.model_dump_json(indent=2)
        with open(json_path, "w") as f:
            f.writelines(pp_def)

    def _lonlat_inside_proj(self, lon: float, lat: float) -> bool:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        sref = osr.SpatialReference()
        sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        sref.ImportFromEPSG(4326)
        point.AssignSpatialReference(sref)

        return point.Within(self._proj_zone)

    def lonlat2xy(self, lon: float, lat: float) -> tuple[float, float]:
        x, y = None, None
        if self._lonlat_inside_proj(lon, lat):
            xy_coords = self._tms.xy(lon, lat)
            x, y = xy_coords.x, xy_coords.y

        return x, y
    
    def xy2lonlat(self, x: float, y: float) -> tuple[float, float]:
        lonlat_coords = self._tms.lnglat(x, y)
        lon, lat = lonlat_coords.x, lonlat_coords.y
        if not self._lonlat_inside_proj(lon, lat):
            lon, lat = None, None
        
        return lon, lat
    
    def create_tilename(self, tile: Tile) -> str:
        x_ori, y_ori = self.axis_orientation
        n_digits_xy = len(str(max(self.max_n_tiles_x, self.max_n_tiles_y)))
        n_digits_z = len(str(len(self) - 1))
        return  f"{x_ori}{tile.x:0{n_digits_xy}}{y_ori}{tile.y:0{n_digits_xy}}T{tile.z:0{n_digits_z}}"

    def _create_tile(self, tilename: str) -> ProjTile:
        _, y_ori = self.axis_orientation
        tiling_level = int(tilename.split("T")[-1])
        x = int(tilename.split(y_ori)[0][1:])
        y = int(tilename.split(y_ori)[1].split("T")[0])
        tile = Tile(x, y, tiling_level)
        extent = self._tms.xy_bounds(tile)
        sampling = self[tiling_level].sampling

        return ProjTile.from_extent(extent, self.epsg, sampling, sampling, name=tilename)
    

    def create_tile(self, tilename: str) -> ProjTile | None:
        ptile = self._create_tile(tilename)
        if self.tiles_in_zone_only:
            if not self.get_tile_bbox_geog(tilename).Intersect(self._proj_zone):
                ptile = None

        return ptile
    
    def get_tile_bbox_proj(self, tilename: str) -> ogr.Geometry:
        return self._create_tile(tilename).boundary
    
    def get_tile_bbox_geog(self, tilename: str) -> ogr.Geometry:
        ptile = self._create_tile(tilename)
        sref = osr.SpatialReference()
        sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        sref.ImportFromEPSG(4326)
        
        return transform_geometry(ptile.boundary, sref, segment=25000)
    

    def search_tiles_in_lonlat_bbox(self, bbox: tuple[float, float, float, float], tiling_level: int) -> list[ProjTile]:
        min_x, min_y, max_x, max_y = bbox
        tiles = list(self._tms.tiles(min_x, min_y, max_x, max_y, [tiling_level]))
        tiles = [self.create_tile(self.create_tilename(tile)) for tile in tiles]

        return tiles

    def __len__(self) -> int:
        return len(self.grids)
    
    def __getitem__(self, tiling_level: int):
        return self.grids[tiling_level]