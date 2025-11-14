import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyproj
from morecantile.models import Tile as RegularTile, TileMatrixSet
from osgeo import ogr, osr
from pydantic import BaseModel, NonNegativeInt, model_validator

from pytileproj.geom import (
    get_geog_sref,
    rasterise_polygon,
    transform_geom_to_geog,
    transform_geometry,
)
from pytileproj.grid import IrregularGrid, RegularGrid
from pytileproj.tile import IrregularTile, ProjTile
from pytileproj.utils import fetch_proj_zone


class ProjCoord(NamedTuple):
    x: float
    y: float
    epsg: int


class ProjSystemBase(BaseModel):
    epsg: int

    _proj_zone_geog: ogr.Geometry
    _proj_zone_native: ogr.Geometry
    _to_geog: pyproj.Transformer
    _from_geog: pyproj.Transformer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        this_crs = pyproj.CRS(self.epsg)
        geog_crs = pyproj.CRS(4326)
        self._to_geog = pyproj.Transformer.from_crs(this_crs, geog_crs, always_xy=True)
        self._from_geog = pyproj.Transformer.from_crs(
            geog_crs, this_crs, always_xy=True
        )
        self._proj_zone_geog = fetch_proj_zone(self.epsg)
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(self.epsg)
        self._proj_zone_native = transform_geometry(self._proj_zone_geog, sref)

    def _lonlat_inside_proj(self, lon: float, lat: float) -> bool:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        point.AssignSpatialReference(get_geog_sref())

        return point in self

    def lonlat_to_xy(self, lon: float, lat: float) -> ProjCoord | None:
        coord = None
        if self._lonlat_inside_proj(lon, lat):
            x, y = self._from_geog.transform(lon, lat)
            coord = ProjCoord(x=x, y=y, epsg=self.epsg)

        return coord

    def xy_to_lonlat(self, x: float, y: float) -> ProjCoord | None:
        lon, lat = self._to_geog.transform(x, y)
        coord = ProjCoord(x=lon, y=lat, epsg=4326)
        if not self._lonlat_inside_proj(lon, lat):
            coord = None

        return coord

    def __contains__(self, geom: ogr.Geometry) -> bool:
        other_sref = geom.GetSpatialReference()
        if other_sref is None:
            err_msg = "Spatial reference of the given geometry is not set."
            raise AttributeError(err_msg)

        geog_sref = get_geog_sref()
        if not geog_sref.IsSame(other_sref):
            wrpd_geom = transform_geometry(geom, geog_sref)
        else:
            wrpd_geom = geom

        return wrpd_geom.Within(self._proj_zone_geog)


def validate_samplings(
    grids: dict[int, RegularGrid | IrregularGrid],
    allowed_samplings: dict[int, list[int | float]] | None,
):
    if allowed_samplings is not None:
        tiling_levels = sorted(grids.keys())
        for tiling_level in tiling_levels:
            grid = grids[tiling_level]
            samplings_tl = allowed_samplings.get(tiling_level, [])
            if samplings_tl and grid.sampling not in samplings_tl:
                raise ValueError(
                    f"Grid {grid.name}'s sampling {grid.sampling} at {tiling_level} is not allowed. The following samplings are allowed: {', '.join(map(str, samplings_tl))}"
                )


class GridSystemBase(BaseModel):
    name: str
    grids: dict[int, RegularGrid | IrregularGrid]
    allowed_samplings: dict[int, list[int | float]] | None = None

    @model_validator(mode="after")
    def check_samplings(self) -> "GridSystemBase":
        validate_samplings(self.grids, self.allowed_samplings)
        return self

    @classmethod
    def from_file(cls, json_path: Path):
        with open(json_path) as f:
            pp_def = json.load(f)

        return cls(**pp_def)

    def to_file(self, json_path: Path):
        pp_def = self.model_dump_json(indent=2)
        with open(json_path, "w") as f:
            f.writelines(pp_def)

    @property
    def tiling_levels(self) -> list[int]:
        return list(self.grids.keys())

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
        yield from grid

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

    def _search_tiles_in_bbox_geog(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> list[RegularTile | IrregularTile]:
        raise NotImplementedError

    def tile_mask(self, proj_tile: ProjTile) -> np.ndarray:
        if proj_tile.epsg != self.epsg:
            raise ValueError("Projection of tile and grid system must match.")

        intrsct_geom = self._proj_zone_native.Intersection(proj_tile.boundary_ogr)
        if intrsct_geom.Area() == 0.0:
            mask = np.zeros(proj_tile.shape, dtype=np.uint8)
        elif proj_tile in self:
            mask = np.ones(proj_tile.shape, dtype=np.uint8)
        else:
            # first, using 'outer_boundary_extent' as a pixel buffer for generating the rasterised
            # pixel skeleton
            # second, reduce this pixel buffer again to the coordinate extent by skipping the last
            # row and column
            mask = rasterise_polygon(
                intrsct_geom,
                proj_tile.x_pixel_size,
                proj_tile.y_pixel_size,
                proj_tile.outer_boundary_extent,
            )[:-1, :-1]

        return mask

    def __contains__(self, geom: ProjTile | ogr.Geometry) -> bool:
        if isinstance(geom, ProjTile):
            arg = geom.boundary_ogr
        else:
            arg = geom

        return super().__contains__(arg)


def validate_grids(grids: dict[int, RegularGrid], congruent: bool):
    tiling_levels = sorted(grids.keys())
    ref_grid = grids[tiling_levels[0]]
    for tiling_level in tiling_levels[1:]:
        grid = grids[tiling_level]

        same_origin = ref_grid.origin_xy == grid.origin_xy
        if not same_origin and congruent:
            raise ValueError(
                f"The given grids do not have the same origin: {ref_grid.tiling_level}:{ref_grid.origin_xy} vs. {grid.tiling_level}:{grid.origin_xy}"
            )

        same_extent = ref_grid.extent == grid.extent
        if not same_extent and congruent:
            raise ValueError(
                f"The given grids do not have the same extent: {ref_grid.tiling_level}:{ref_grid.extent} vs. {grid.tiling_level}:{grid.extent}"
            )

        same_orientation = ref_grid.axis_orientation == grid.axis_orientation
        if not same_orientation:
            raise ValueError(
                f"The given grids do not have the same axis orientation: {ref_grid.tiling_level}:{ref_grid.axis_orientation} vs. {grid.tiling_level}:{grid.axis_orientation}"
            )

        ref_n_rows, ref_n_cols = ref_grid.tm.matrixHeight, ref_grid.tm.matrixWidth
        n_rows, n_cols = grid.tm.matrixHeight, grid.tm.matrixWidth

        if (ref_n_rows >= n_rows) or (ref_n_cols >= n_cols):
            raise ValueError(
                f"The given grids do not grow with increasing tiling level: {ref_grid.tiling_level} ({ref_n_rows},{ref_n_cols}) vs.{tiling_level} ({n_rows},{n_cols})."
            )

        if congruent:
            if (n_rows % ref_n_rows != 0) or (n_cols % ref_n_cols != 0):
                raise ValueError(
                    f"The given tiles in the grids are not congruent: {ref_grid.tiling_level} ({ref_n_rows},{ref_n_cols}) vs.{tiling_level} ({n_rows},{n_cols})."
                )

        ref_grid = grid


class RegularProjGridSystem(ProjGridSystemBase):
    congruent: bool | None = False

    _tms: TileMatrixSet

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        wkt = pyproj.CRS(self.epsg).to_json_dict()
        self._tms = TileMatrixSet(
            crs={"wkt": wkt},
            tileMatrices=[
                self.grids[tiling_level].tm
                for tiling_level in sorted(self.tiling_levels)
            ],
        )

    @model_validator(mode="after")
    def check_grids(self) -> "RegularProjGridSystem":
        validate_grids(self.grids, self.congruent)
        return self

    @property
    def axis_orientation(self) -> tuple[str, str]:
        return self[0].axis_orientation

    @property
    def max_n_tiles_x(self) -> int:
        max_tiling_level = max(self.tiling_levels)
        return self[max_tiling_level].tm.matrixWidth

    @property
    def max_n_tiles_y(self) -> int:
        max_tiling_level = max(self.tiling_levels)
        return self[max_tiling_level].tm.matrixHeight

    def n_tiles(self, tiling_level: int) -> int:
        return self[tiling_level].tm.matrixHeight * self[tiling_level].tm.matrixWidth

    @classmethod
    def default(
        cls,
        name: str,
        epsg: int,
        extent: tuple[float, float, float, float],
        tile_shape_px: tuple[NonNegativeInt, NonNegativeInt],
        tiling_level_limits: tuple[NonNegativeInt, NonNegativeInt] | None = (0, 24),
    ) -> "RegularProjGridSystem":
        min_zoom, max_zoom = tiling_level_limits
        tms = TileMatrixSet.custom(
            extent,
            pyproj.CRS(epsg),
            tile_width=tile_shape_px[0],
            tile_height=tile_shape_px[1],
            minzoom=min_zoom,
            maxzoom=max_zoom,
        )
        grids = []
        for i, tm in enumerate(tms.tileMatrices):
            grid = RegularGrid(
                name=str(i),
                extent=tms.bbox,
                sampling=tm.cellSize,
            )
            grids.append(grid)

        return cls(name, epsg, grids)

    def _create_tilename(self, tile: RegularTile) -> str:
        x_ori, y_ori = self.axis_orientation
        n_digits_xy = len(str(max(self.max_n_tiles_x, self.max_n_tiles_y)))
        n_digits_z = len(str(len(self) - 1))
        return f"{x_ori}{tile.x:0{n_digits_xy}}{y_ori}{tile.y:0{n_digits_xy}}T{tile.z:0{n_digits_z}}"

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
            if not proj_tile.boundary_ogr.Intersect(self._proj_zone_native):
                proj_tile = None

        return proj_tile

    def _to_proj_tile(self, tile: RegularTile, name: str = None) -> ProjTile:
        extent = self._tms.xy_bounds(tile)
        sampling = self[tile.z].sampling
        return ProjTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def _search_tiles_in_bbox_geog(self, bbox, tiling_level):
        min_x, min_y, max_x, max_y = bbox
        return self._tms.tiles(min_x, min_y, max_x, max_y, [tiling_level])

    def search_tiles_in_lonlat_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ):
        for tile in self._search_tiles_in_bbox_geog(bbox, tiling_level):
            tilename = self._create_tilename(tile)
            proj_tile = self._to_proj_tile(tile, name=tilename)
            if self.tiles_in_zone_only:
                if not proj_tile.boundary_ogr.Intersect(self._proj_zone_native):
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
            if not proj_tile.boundary_ogr.Intersect(self._proj_zone_native):
                proj_tile = None

        return proj_tile

    def get_tile_bbox_proj(self, tilename: str) -> ogr.Geometry:
        proj_tile = self.create_tile(tilename)
        return proj_tile.boundary_ogr

    def _to_proj_tile(self, tile: IrregularTile, name: str = None) -> ProjTile:
        extent = tile.boundary.bounds
        sampling = self[tile.z].sampling
        return ProjTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def search_tiles_in_lonlat_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ):
        for tile in self[tiling_level].tiles_in_bbox(bbox):
            tilename = self._create_tilename(tile)
            proj_tile = self._to_proj_tile(tile, name=tilename)
            if self.tiles_in_zone_only:
                if not proj_tile.boundary_ogr.Intersect(self._proj_zone_native):
                    continue

            yield proj_tile
