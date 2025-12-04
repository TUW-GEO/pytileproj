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

"""Tiling system module defining a irregular and regular tiling systems."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, Union

import numpy as np
import pyproj
import shapely.wkt
from morecantile.models import Tile as RegularTile
from morecantile.models import TileMatrixSet
from osgeo import ogr, osr
from pydantic import AfterValidator, BaseModel, NonNegativeInt, model_validator

from pytileproj._const import JSON_INDENT, VIS_INSTALLED
from pytileproj.geom import (
    convert_any_to_geog_ogr_geom,
    get_geog_sref,
    rasterise_polygon,
    transform_geom_to_geog,
    transform_geometry,
)
from pytileproj.proj import fetch_proj_zone, pyproj_to_cartopy_crs
from pytileproj.tile import IrregularTile, RasterTile
from pytileproj.tiling import IrregularTiling, RegularTiling

if VIS_INSTALLED:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature
    import matplotlib.axes as mplax
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

AnyTile = RegularTile | IrregularTile
TileGenerator = Generator[AnyTile, AnyTile, AnyTile]
RasterTileGenerator = Generator[RasterTile, RasterTile, RasterTile]


__all__ = [
    "IrregularProjTilingSystem",
    "ProjCoord",
    "ProjSystemBase",
    "ProjTilingSystemBase",
    "RPTSDefinition",
    "RegularProjTilingSystem",
    "RegularTilingDefinition",
    "TilingSystemBase",
]


class ProjCoord(NamedTuple):
    """Define a coordinate in a certain projection."""

    x: float
    y: float
    epsg: int


class ProjSystemBase(BaseModel, arbitrary_types_allowed=True):
    """Base class defining a projection represented by an EPSG code and a zone."""

    epsg: int
    proi_zone_geog: Annotated[
        Path | shapely.Polygon | ogr.Geometry | None,
        AfterValidator(convert_any_to_geog_ogr_geom),
    ] = None

    _proj_zone_geog: ogr.Geometry
    _proj_zone: ogr.Geometry
    _to_geog: pyproj.Transformer
    _from_geog: pyproj.Transformer

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise projected system base object."""
        super().__init__(**kwargs)
        this_crs = pyproj.CRS(self.epsg)
        geog_crs = pyproj.CRS(4326)
        self._to_geog = pyproj.Transformer.from_crs(this_crs, geog_crs, always_xy=True)
        self._from_geog = pyproj.Transformer.from_crs(
            geog_crs, this_crs, always_xy=True
        )
        self._proj_zone_geog = self.proi_zone_geog or fetch_proj_zone(self.epsg)
        sref = osr.SpatialReference()
        sref.ImportFromEPSG(self.epsg)
        proj_zone_faulty = transform_geometry(self._proj_zone_geog, sref)
        # buffer of 0 removes wrap-arounds along the anti-meridian
        proj_zone = shapely.buffer(shapely.wkt.loads(proj_zone_faulty.ExportToWkt()), 0)
        proj_zone = ogr.CreateGeometryFromWkt(proj_zone.wkt)
        proj_zone.AssignSpatialReference(sref)
        self._proj_zone = proj_zone

    def _lonlat_inside_proj(self, lon: float, lat: float) -> bool:
        """Check if a longitude and latitude coordinate is within the projection zone.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        bool
            True if the given coordinate is within the projection zone, false if not.

        """
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        point.AssignSpatialReference(get_geog_sref())

        return point in self

    def lonlat_to_xy(self, lon: float, lat: float) -> ProjCoord | None:
        """Convert geographic to a projected coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        ProjCoord | None
            X and Y coordinates if the given input coordinates are
            within the projection zone, None if not.

        """
        coord = None
        if self._lonlat_inside_proj(lon, lat):
            x, y = self._from_geog.transform(lon, lat)
            coord = ProjCoord(x=x, y=y, epsg=self.epsg)

        return coord

    def xy_to_lonlat(self, x: float, y: float) -> ProjCoord | None:
        """Convert projected to a geographic coordinates.

        Parameters
        ----------
        x: float
            World system coordinate in X direction.
        y: float
            World system coordinate in Y direction.

        Returns
        -------
        ProjCoord | None
            Longitude and latitude coordinates if the given input
            coordinates are within the projection zone, None if not.

        """
        lon, lat = self._to_geog.transform(x, y)
        coord = ProjCoord(x=lon, y=lat, epsg=4326)
        if not self._lonlat_inside_proj(lon, lat):
            coord = None

        return coord

    def export_proj_zone_geog(self, path: Path) -> None:
        """Write the projection zone in geographic coordinates to a GeoJSON file.

        Parameters
        ----------
        path: Path
            Output path (.geojson)

        """
        geojson = self._proj_zone_geog.ExportToJson()
        with path.open("w") as f:
            f.writelines(geojson)

    def __contains__(self, geom: ogr.Geometry | ProjCoord) -> bool:
        """Evaluate if the given geometry is fully within the projection zone.

        Parameters
        ----------
        geom : ogr.Geometry | ProjCoord
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry is within the projection zone, false if not.

        """
        if isinstance(geom, ogr.Geometry):
            other_sref = geom.GetSpatialReference()
            if other_sref is None:
                err_msg = "Spatial reference of the given geometry is not set."
                raise AttributeError(err_msg)
        elif isinstance(geom, ProjCoord):
            point = shapely.Point((geom.x, geom.y))
            other_sref = osr.SpatialReference()
            other_sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            other_sref.ImportFromEPSG(geom.epsg)
            geom = ogr.CreateGeometryFromWkt(point.wkt)
            geom.AssignSpatialReference(other_sref)
        else:
            err_msg = f"Geometry type {type(geom)} is not supported."
            raise TypeError(err_msg)

        geog_sref = get_geog_sref()
        if not geog_sref.IsSame(other_sref):
            wrpd_geom = transform_geometry(geom, geog_sref)
        else:
            wrpd_geom = geom

        return wrpd_geom.Within(self._proj_zone_geog)


def validate_samplings(
    tilings: dict[int, RegularTiling | IrregularTiling],
    allowed_samplings: dict[int, list[int | float]] | None,
) -> None:
    """Cross-check the sampling of a tiling with given allowed samplings.

    Parameters
    ----------
    tilings: dict[int, RegularTiling | IrregularTiling]
        Dictionary with tiling levels as keys and tilings as values.
    allowed_samplings: dict[int, list[int | float]] | None
        Dictionary with tiling levels as keys and allowed samplings as values.

    Raises
    ------
    ValueError
        If the sampling is not allowed.

    """
    err_msg = (
        "Grid {}'s sampling {} at {} is not allowed."
        " The following samplings are allowed: {}"
    )
    if allowed_samplings is not None:
        tiling_levels = sorted(tilings.keys())
        for tiling_level in tiling_levels:
            tiling = tilings[tiling_level]
            samplings_tl = allowed_samplings.get(tiling_level, [])
            if samplings_tl and tiling.sampling not in samplings_tl:
                allwd_smpls_str = ", ".join(map(str, samplings_tl))
                err_msg = err_msg.format(
                    tiling.name, tiling.sampling, tiling_level, allwd_smpls_str
                )
                raise ValueError(err_msg)


class TilingSystemBase(BaseModel):
    """Base class defining a multi-level tiling system."""

    name: str
    tilings: dict[int, RegularTiling | IrregularTiling]
    allowed_samplings: dict[int, list[int | float]] | None = None

    @model_validator(mode="after")
    def check_samplings(self) -> "TilingSystemBase":
        """Check the given user samplings.

        Check the sampling defined in a tiling corresponds to the
        given allowed samplings.
        """
        validate_samplings(self.tilings, self.allowed_samplings)
        return self

    @classmethod
    def from_file(cls, json_path: Path) -> "TilingSystemBase":
        """Initialise tiling system class from the settings stored within a JSON file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        Returns
        -------
        TilingSystemBase
            Tiling system object initialised from the given JSON definition.

        """
        with json_path.open() as f:
            pp_def = json.load(f)

        return cls(**pp_def)

    def to_file(self, json_path: Path) -> None:
        """Write class attributes to a JSON file.

        Parameters
        ----------
        json_path: Path
            Path to JSON file.

        """
        pp_def = self.model_dump_json(indent=JSON_INDENT)
        with json_path.open("w") as f:
            f.writelines(pp_def)

    @property
    def tiling_levels(self) -> list[int]:
        """Return tiling levels."""
        return list(self.tilings.keys())

    def _create_tilename(self, tile: AnyTile) -> str:
        """Create a tilename from a given tile object.

        Parameters
        ----------
        tile: AnyTile
            Tile object.

        Returns
        -------
        str
            Tilename.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def _create_tile(self, tilename: str) -> AnyTile:
        """Create a tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        AnyTile
            Tile object.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def _tilenames_at_level(self, tiling_level: int) -> Generator[str, str, str]:
        """Return all tilenames at a specific tiling level or zoom.

        Parameters
        ----------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        Generator[str, str, str]
            Yields one tilename after the other.

        """
        tiling = self[tiling_level]
        for tile in tiling:
            yield self._create_tilename(tile)

    def _tiles_at_level(self, tiling_level: int) -> TileGenerator:
        """Return all tiles at a specific tiling level or zoom.

        Parameters
        ----------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        TileGenerator
            Yields one tile after the other.

        """
        tiling = self[tiling_level]
        yield from tiling

    def __len__(self) -> int:
        """Return number of tiling levels."""
        return len(self.tilings)

    def __getitem__(self, tiling_level: int) -> RegularTiling:
        """Return the tiling at a specific tiling level.

        Parameter
        ---------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RegularTiling
            Tiling object at a specific tiling level.

        """
        return self.tilings[tiling_level]


class ProjTilingSystemBase(TilingSystemBase, ProjSystemBase):
    """Base class defining a projected, multi-level tiling system."""

    tiles_in_zone_only: bool = True

    def create_tile_from_lonlat(
        self, lon: float, lat: float, tiling_level: int
    ) -> RasterTile:
        """Create a raster tile object from geographic coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        geog_coord = ProjCoord(lon, lat, 4326)
        return self.create_tile_from_coord(geog_coord, tiling_level)

    def create_tile_from_xy(self, x: float, y: float, tiling_level: int) -> RasterTile:
        """Create a raster tile object from projected coordinates.

        Parameters
        ----------
        x: float
            World system coordinate in X direction.
        y: float
            World system coordinate in Y direction.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        proj_coord = ProjCoord(x, y, self.epsg)
        return self.create_tile_from_coord(proj_coord, tiling_level)

    def create_tile_from_coord(self, coord: ProjCoord, tiling_level: int) -> RasterTile:
        """Create a raster tile object from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates object.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTile
            Raster tile object.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def create_tile(self, tilename: str) -> RasterTile:
        """Create a raster tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def _to_raster_tile(self, tile: AnyTile, name: str | None = None) -> RasterTile:
        """Create a raster tile object from a given tile.

        Parameters
        ----------
        tile: AnyTile
            Simple tile object.
        name: str | None, optional
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def _tile_in_zone(self, tile: RasterTile) -> bool:
        """Check if the given tile is within the projection zone.

        Parameters
        ----------
        tile: RasterTile
            Raster tile object.

        Returns
        -------
        bool
            True if the given tile is within the projection zone.

        """
        tile_in_zone = True
        if self.tiles_in_zone_only:
            tile_in_zone = tile.boundary_ogr.Intersect(self._proj_zone)

        return tile_in_zone

    def get_tile_bbox_geog(self, tilename: str) -> ogr.Geometry:
        """Return the boundary of the tile corresponding to the given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        ogr.Geometry
            Tile boundary.

        """
        raster_tile = self.create_tile(tilename)
        return transform_geom_to_geog(raster_tile.boundary_ogr)

    def _search_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> TileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        TileGenerator
            Yields tile after tile, which intersects with the given bounding box.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def tile_mask(self, raster_tile: RasterTile) -> np.ndarray:
        """Compute tile mask w.r.t. projection zone.

        Compute a binary array representation of the given raster tile, where each
        pixel is either inside (1) or outside the projection zone (0).

        Parameters
        ----------
        raster_tile: RasterTile
            Raster tile object.

        Returns
        -------
        np.ndarray
            Tile mask.

        Raises
        ------
        ValueError
            If the projection of the tiling system and the given raster tile differs.

        """
        if raster_tile.epsg != self.epsg:
            err_msg = "Projection of tile and tiling system must match."
            raise ValueError(err_msg)

        intrsct_geom = self._proj_zone.Intersection(raster_tile.boundary_ogr)
        if intrsct_geom.Area() == 0.0:
            mask = np.zeros(raster_tile.shape, dtype=np.uint8)
        elif raster_tile in self:
            mask = np.ones(raster_tile.shape, dtype=np.uint8)
        else:
            # first, using 'outer_boundary_extent' as a pixel buffer for
            # generating the rasterised pixel skeleton
            # second, reduce this pixel buffer again to the coordinate extent by
            # skipping the last row and column
            mask = rasterise_polygon(
                intrsct_geom,
                raster_tile.x_pixel_size,
                raster_tile.y_pixel_size,
                raster_tile.outer_boundary_extent,
            )[:-1, :-1]

        return mask

    def plot(  # noqa: C901, PLR0913
        self,
        *,
        ax: Union["mplax.Axes", None] = None,
        tiling_level: int = 0,
        facecolor: str = "tab:red",
        edgecolor: str = "black",
        edgewidth: int = 1,
        alpha: float = 1.0,
        proj: Union["ccrs.CRS", None] = None,
        show: bool = False,
        label_tile: bool = False,
        label_size: int = 12,
        add_country_borders: bool = True,
        extent: tuple | None = None,
        plot_zone: bool = False,
    ) -> "mplax.Axes":
        """Plot all tiles at a specific tiling level.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            Pre-defined Matplotlib axis.
        tiling_level: int, optional
            Tiling level or zoom. Defaults to 0.
        facecolor : str, optional
            Color code as described at:
              https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
            Defaults to 'tab:red'.
        edgecolor : str, optional
            Color code as described at:
              https://matplotlib.org/3.1.0/tutorials/colors/colors.html.
            Defaults to 'black'.
        edgewidth : float, optional
            Width the of edge line (defaults to 1).
        alpha : float, optional
            Opacity (default is 1.).
        proj : cartopy.crs, optional
            Cartopy projection instance defining the projection of the axes.
            Defaults to None, which means the projection of the spatial
            reference system of the projected tiling system is taken.
        show : bool, optional
            If True, the plot result is shown. Defaults to false.
        label_tile : bool, optional
            If True, the geometry name is plotted at the center of the raster geometry.
            Defaults to false.
        label_size: int, optional
            Fontsize of the tile labels. Defaults to 12.
        add_country_borders : bool, optional
            If True, country borders are added to the plot (`cartopy.feature.BORDERS`).
            Defaults to false.
        extent : tuple or list, optional
            Coordinate/map extent of the plot, given as
            [min_x, min_y, max_x, max_y] (default is None, meaning global extent).
        plot_zone: bool, optional
            True if the projection zone should be added to the plot.
            False if not.


        Returns
        -------
        matplotlib.pyplot.axes
            Matplotlib axis containing a Cartopy map with the plotted
            raster tile boundary.

        """
        if not VIS_INSTALLED:
            err_msg = (
                "Modules 'matplotlib' and 'cartopy' are mandatory "
                "for plotting a projected tiling system object."
            )
            raise ImportError(err_msg)

        this_proj = pyproj_to_cartopy_crs(pyproj.CRS.from_epsg(self.epsg))
        other_proj = this_proj if proj is None else proj

        if ax is None:
            ax = plt.axes(projection=other_proj)
            ax.set_global()
            ax.gridlines()

        if add_country_borders:
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)

        for tile in self[tiling_level]:
            tilename = self._create_tilename(tile)
            raster_tile = self._to_raster_tile(tile, tilename)
            if self._tile_in_zone(raster_tile):
                patch = PolygonPatch(
                    list(raster_tile.boundary_shapely.exterior.coords),
                    facecolor=facecolor,
                    alpha=alpha,
                    zorder=0,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    transform=this_proj,
                )
                ax.add_patch(patch)

                if label_tile:
                    transform = this_proj._as_mpl_transform(ax)  # noqa: SLF001
                    ax.annotate(
                        raster_tile.name,
                        xy=raster_tile.centre,
                        xycoords=transform,
                        va="center",
                        ha="center",
                        fontsize=label_size,
                    )

        if extent is not None:
            ax.set_xlim([extent[0], extent[2]])
            ax.set_ylim([extent[1], extent[3]])

        if plot_zone:
            transform = this_proj._as_mpl_transform(ax)  # noqa: SLF001
            zone_boundary = shapely.wkt.loads(self._proj_zone.ExportToWkt())
            x_coords_bound, y_coords_bound = [], []
            if isinstance(zone_boundary, shapely.MultiPolygon):
                for poly in zone_boundary.geoms:
                    x_coords_bound, y_coords_bound = list(
                        zip(*poly.exterior.coords, strict=False)
                    )
                    ax.plot(
                        x_coords_bound,
                        y_coords_bound,
                        color="k",
                        linewidth=2,
                        transform=transform,
                    )
            else:
                x_coords_bound, y_coords_bound = list(
                    zip(*zone_boundary.exterior.coords, strict=False)
                )
                ax.plot(
                    x_coords_bound,
                    y_coords_bound,
                    color="k",
                    linewidth=2,
                    transform=transform,
                )

        if show:
            plt.show()

        return ax

    def search_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTileGenerator
            Yields raster tile after tile, which intersects with the given
            bounding box.

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def __contains__(self, geom: RasterTile | ogr.Geometry) -> bool:
        """Contain wrapper.

        Evaluate if the given geometry or raster tile is fully within the
        projection zone.

        Parameters
        ----------
        geom : ogr.Geometry | RasterTile
            Other geometry or raster tile to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry or raster tile is within the projection
            zone, false if not.

        """
        arg = geom.boundary_ogr if isinstance(geom, RasterTile) else geom

        return super().__contains__(arg)


def validate_tilings(
    tilings: dict[int, RegularTiling], *, congruent: bool = False
) -> None:
    """Validate if different regular tilings are compliant with each other.

    This means they need to:
        - have the same origin
        - have the same extent
        - have the same orientation
        - increase the number of tiles with increasing tiling level
        - be congruent if required.

    Parameters
    ----------
    tilings: dict[int, RegularTiling]
        Dictionary with tiling/zoom levels as keys and regular tilings as values.
    congruent: bool, optional
        If true, then tilings from adjacent tiling levels need to be congruent,
        which means that tiles from the higher tiling level need to be exactly
        in one tile of the lower level. Defaults to false.

    Raises
    ------
    ValueError
        If one of the conditions above is not met.

    """
    tiling_levels = sorted(tilings.keys())
    ref_tiling = tilings[tiling_levels[0]]
    for tiling_level in tiling_levels[1:]:
        tiling = tilings[tiling_level]

        same_origin = ref_tiling.origin_xy == tiling.origin_xy
        if not same_origin and congruent:
            ref_tiling_str = f"{ref_tiling.tiling_level}:{ref_tiling.origin_xy}"
            tiling_str = f"{tiling.tiling_level}:{tiling.origin_xy}"
            err_hdr = "The given tilings do not have the same origin:"
            err_msg = f"{err_hdr} {ref_tiling_str} vs. {tiling_str}"
            raise ValueError(err_msg)

        same_extent = ref_tiling.extent == tiling.extent
        if not same_extent and congruent:
            ref_tiling_str = f"{ref_tiling.tiling_level}:{ref_tiling.extent}"
            tiling_str = f"{tiling.tiling_level}:{tiling.extent}"
            err_hdr = "The given tilings do not have the same extent:"
            err_msg = f"{err_hdr} {ref_tiling_str} vs. {tiling_str}"
            raise ValueError(err_msg)

        same_orientation = ref_tiling.axis_orientation == tiling.axis_orientation
        if not same_orientation:
            ref_tiling_str = f"{ref_tiling.tiling_level}:{ref_tiling.axis_orientation}"
            tiling_str = f"{tiling.tiling_level}:{tiling.axis_orientation}"
            err_hdr = "The given tilings do not have the same axis orientation:"
            err_msg = f"{err_hdr} {ref_tiling_str} vs. {tiling_str}"
            raise ValueError(err_msg)

        ref_n_rows, ref_n_cols = ref_tiling.tm.matrixHeight, ref_tiling.tm.matrixWidth
        n_rows, n_cols = tiling.tm.matrixHeight, tiling.tm.matrixWidth

        if (ref_n_rows >= n_rows) or (ref_n_cols >= n_cols):
            ref_tiling_str = f"{ref_tiling.tiling_level} ({ref_n_rows},{ref_n_cols})"
            tiling_str = f"{tiling_level} ({n_rows},{n_cols})"
            err_hdr = "The given tilings do not grow with increasing tiling level:"
            err_msg = f"{err_hdr} {ref_tiling_str} vs. {tiling_str}."
            raise ValueError(err_msg)

        if congruent and ((n_rows % ref_n_rows != 0) or (n_cols % ref_n_cols != 0)):
            ref_tiling_str = f"{ref_tiling.tiling_level} ({ref_n_rows},{ref_n_cols})"
            tiling_str = f"{tiling_level} ({n_rows},{n_cols})"
            err_hdr = "The given tiles in the tilings are not congruent:"
            err_msg = f"{err_hdr} {ref_tiling_str} vs. {tiling_str}."
            raise ValueError(err_msg)

        ref_tiling = tiling


class RPTSDefinition(BaseModel):
    """Definition for a specific Regular Projected Tiling System."""

    name: str
    epsg: int
    extent: tuple[float, float, float, float]
    axis_orientation: tuple[Literal["W", "E"], Literal["N", "S"]] | None = ("E", "N")


class RegularTilingDefinition(BaseModel):
    """Definition for a specific Regular Tiling System."""

    name: str
    tile_size: float | int


class RegularProjTilingSystem(ProjTilingSystemBase):
    """Regular projected, multi-level tiling system."""

    congruent: bool | None = False

    _tms: TileMatrixSet

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise regular projected tiling system."""
        super().__init__(**kwargs)
        wkt = pyproj.CRS(self.epsg).to_json_dict()
        self._tms = TileMatrixSet(
            crs={"wkt": wkt},
            tileMatrices=[
                self.tilings[tiling_level].tm
                for tiling_level in sorted(self.tiling_levels)
            ],
        )

    @classmethod
    def from_sampling(
        cls,
        sampling: float | dict[int, float | int],
        rpts_def: RPTSDefinition,
        tiling_defs: dict[int, RegularTilingDefinition],
        *,
        allowed_samplings: dict[int, list[float | int]] | None = None,
        congruent: bool = False,
    ) -> "RegularProjTilingSystem":
        """Classmethod for creating a regular, projected tiling system.

        Create a regular, projected tiling system instance from given tiling system
        definitions and a grid sampling.

        Parameters
        ----------
        rpts_def: RPTSDefinition
            Regular, projected tiling system definition (stores name, EPSG code, extent,
            and axis orientation).
        sampling: float | int | Dict[int, float | int]
            Grid sampling/pixel size specified as a single value or a dictionary
            with tiling levels as keys and
            samplings as values.
        tiling_defs: Dict[int, RegularTilingDefinition]
            Tiling definition (stores name/tiling level and tile size).
        allowed_samplings: Dict[int, List[float | int]] | None, optional
            Dictionary with tiling levels as keys and allowed samplings as values.
            Defaults to None, which means there are no restrictions for the specified
            sampling.
        congruent: bool, optional
            If true, then tilings from adjacent tiling levels need to be congruent,
            which means that tiles from the higher tiling level need to be exactly in
            one tile of the lower level. Defaults to false.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        if isinstance(sampling, dict):
            samplings = sampling
        else:
            samplings = {}
            samplings[1] = sampling

        tilings = {}
        for k, s in samplings.items():
            tiling_def = tiling_defs.get(k)
            if tiling_def is None:
                err_msg = f"There is no tile definition for the tiling level {k}"
                raise ValueError(err_msg)
            tile_size_px = int(tiling_def.tile_size / s)
            tiling = RegularTiling(
                name=tiling_def.name,
                extent=rpts_def.extent,
                sampling=s,
                tile_shape_px=(tile_size_px, tile_size_px),
                tiling_level=k,
                axis_orientation=rpts_def.axis_orientation,
            )
            tilings[k] = tiling

        return cls(
            name=rpts_def.name,
            epsg=rpts_def.epsg,
            tilings=tilings,
            congruent=congruent,
            allowed_samplings=allowed_samplings,
        )

    @model_validator(mode="after")
    def check_tilings(self) -> "RegularProjTilingSystem":
        """Validate if different regular tilings are compliant with each other."""
        validate_tilings(self.tilings, congruent=self.congruent)
        return self

    @property
    def axis_orientation(self) -> tuple[str, str]:
        """Axis orientation taken from the first tiling level."""
        def_tiling_level = self.tiling_levels[0]
        return self[def_tiling_level].axis_orientation

    @property
    def max_n_tiles_x(self) -> int:
        """Number of tiles at the highest tiling level in X direction."""
        max_tiling_level = max(self.tiling_levels)
        return self[max_tiling_level].tm.matrixWidth

    @property
    def max_n_tiles_y(self) -> int:
        """Number of tiles at the highest tiling level in Y direction."""
        max_tiling_level = max(self.tiling_levels)
        return self[max_tiling_level].tm.matrixHeight

    def n_tiles_x(self, tiling_level: int) -> int:
        """Return the number of tiles in X direction at the given tiling.

        Parameters
        ----------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        int
            Number of tiles in X direction.

        """
        return self[tiling_level].tm.matrixWidth

    def n_tiles_y(self, tiling_level: int) -> int:
        """Return the number of tiles in Y direction at the given tiling.

        Parameters
        ----------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        int
            Number of tiles in Y direction.

        """
        return self[tiling_level].tm.matrixHeight

    def n_tiles(self, tiling_level: int) -> int:
        """Return the number of tiles at the given tiling.

        Parameters
        ----------
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        int
            Number of tiles.

        """
        return self.n_tiles_x(tiling_level) * self.n_tiles_y(tiling_level)

    @classmethod
    def default(
        cls,
        name: str,
        epsg: int,
        extent: tuple[float, float, float, float],
        tile_shape_px: tuple[NonNegativeInt, NonNegativeInt],
        tiling_level_limits: tuple[NonNegativeInt, NonNegativeInt] | None = (0, 24),
    ) -> "RegularProjTilingSystem":
        """Classmethod for creating a regular projected tiling system.

        Create a regular projected tiling system from a given extent, projection,
        and tile shape. morecantile's `TileMatrixSet` class is used in the
        background.

        Parameters
        ----------
        name: str
            Name of the tiling system.
        epsg: int
            Projection of the tiling system given as an EPSG code.
        extent: tuple[float, float, float, float]
            Extent of the tiling system (x_min, y_min, x_max, y_max).
        tile_shape_px: tuple[NonNegativeInt, NonNegativeInt]
            Shape of a tile in pixels (number of rows, number of columns).
        tiling_level_limits: tuple[NonNegativeInt, NonNegativeInt] | None, optional
            Lower and upper tiling/zoom level limits. Defaults to (0, 24).

        Returns
        -------
        RegularProjTilingSystem
            Regular projected tiling system.

        """
        min_zoom, max_zoom = tiling_level_limits
        tms = TileMatrixSet.custom(
            extent,
            pyproj.CRS(epsg),
            tile_width=tile_shape_px[0],
            tile_height=tile_shape_px[1],
            minzoom=min_zoom,
            maxzoom=max_zoom,
        )
        tilings = []
        for i, tm in enumerate(tms.tileMatrices):
            tiling = RegularTiling(
                name=str(i),
                extent=tms.bbox,
                sampling=tm.cellSize,
            )
            tilings.append(tiling)

        return cls(name, epsg, tilings)

    def _create_tilename(self, tile: RegularTile) -> str:
        """Create a tilename from a given regular tile object.

        Parameters
        ----------
        tile: RegularTile
            Regular tile object.

        Returns
        -------
        str
            Tilename.

        """
        x_ori, y_ori = self.axis_orientation
        n_digits_xy = len(str(max(self.max_n_tiles_x, self.max_n_tiles_y)))
        n_digits_z = len(str(len(self) - 1))
        x_label = f"{x_ori}{tile.x:0{n_digits_xy}}"
        y_label = f"{y_ori}{tile.y:0{n_digits_xy}}"
        z_label = f"T{tile.z:0{n_digits_z}}"
        return x_label + y_label + z_label

    def _create_tile(self, tilename: str) -> RegularTile:
        """Create a regular tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RegularTile
            Regular tile object.

        """
        _, y_ori = self.axis_orientation
        tiling_level = int(tilename.split("T")[-1])
        x = int(tilename.split(y_ori)[0][1:])
        y = int(tilename.split(y_ori)[1].split("T")[0])
        return RegularTile(x, y, tiling_level)

    def create_tile(self, tilename: str) -> RasterTile:
        """Create a raster tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = self._create_tile(tilename)
        raster_tile = self._to_raster_tile(tile, name=tilename)
        if self.tiles_in_zone_only and not self._tile_in_zone(raster_tile):
            raster_tile = None

        return raster_tile

    def create_tile_from_coord(self, coord: ProjCoord, tiling_level: int) -> RasterTile:
        """Create a raster tile object from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates object.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = self._tms.tile(
            coord.x,
            coord.y,
            tiling_level,
            geographic_crs=pyproj.CRS.from_epsg(coord.epsg),
        )
        tilename = self._create_tilename(tile)
        return self._to_raster_tile(tile, name=tilename)

    def _to_raster_tile(self, tile: RegularTile, name: str | None = None) -> RasterTile:
        """Create a raster tile object from a given regular tile.

        Parameters
        ----------
        tile: RegularTile
            Regular tile object.
        name: str | None, optional
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        extent = self._tms.xy_bounds(tile)
        sampling = self[tile.z].sampling
        return RasterTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def _search_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> TileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        TileGenerator
            Yields tile after tile, which intersects with the given bounding box.

        """
        min_x, min_y, max_x, max_y = bbox
        return self._tms.tiles(min_x, min_y, max_x, max_y, [tiling_level])

    def search_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTileGenerator
            Yields raster tile after tile, which intersects with the given bounding box.

        """
        for tile in self._search_tiles_in_geog_bbox(bbox, tiling_level):
            tilename = self._create_tilename(tile)
            raster_tile = self._to_raster_tile(tile, name=tilename)
            if self.tiles_in_zone_only and not self._tile_in_zone(raster_tile):
                continue

            yield raster_tile


class IrregularProjTilingSystem(ProjTilingSystemBase):
    """Irregular projected, multi-level tiling system."""

    def _create_tilename(self, tile: IrregularTile) -> str:
        """Create a tilename from a given irregular tile object.

        Parameters
        ----------
        tile: IrregularTile
            Irregular tile object.

        Returns
        -------
        str
            Tilename.

        """
        return tile.id

    def _tilename_to_level(self, tilename: str) -> str:
        return int(tilename.split("T")[-1])

    def _create_tile(self, tilename: str) -> IrregularTile:
        """Create an irregular tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        IrregularTile
            Irregular tile object.

        """
        tiling_level = self._tilename_to_level(tilename)
        return self[tiling_level].tiles[tilename]

    def create_tile(self, tilename: str) -> RasterTile:
        """Create a raster tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = self._create_tile(tilename)
        raster_tile = self._to_raster_tile(tile, name=tilename)
        if self.tiles_in_zone_only and not raster_tile.boundary_ogr.Intersect(
            self._proj_zone
        ):
            raster_tile = None

        return raster_tile

    def _to_raster_tile(
        self, tile: IrregularTile, name: str | None = None
    ) -> RasterTile:
        """Create a raster tile object from a given irregular tile.

        Parameters
        ----------
        tile: IrregularTile
            Irregular tile object.
        name: str | None, optional
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        extent = tile.boundary.bounds
        sampling = self[tile.z].sampling
        return RasterTile.from_extent(extent, self.epsg, sampling, sampling, name=name)

    def search_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTileGenerator
            Yields raster tile after tile, which intersects with the given bounding box.

        """
        for tile in self[tiling_level].tiles_in_bbox(bbox):
            tilename = self._create_tilename(tile)
            raster_tile = self._to_raster_tile(tile, name=tilename)
            if self.tiles_in_zone_only and not raster_tile.boundary_ogr.Intersect(
                self._proj_zone
            ):
                continue

            yield raster_tile
