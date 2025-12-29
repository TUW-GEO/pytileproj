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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Union, cast

import numpy as np
import orjson
import pyproj
import shapely
from antimeridian import fix_polygon
from morecantile.models import Tile as RegularTile
from morecantile.models import TileMatrixSet
from pydantic import (
    AfterValidator,
    BaseModel,
    NonNegativeInt,
    PrivateAttr,
    model_validator,
)

from pytileproj._const import (
    DEF_SEG_LEN_DEG,
    GEO_INSTALLED,
    GEOG_EPSG,
    JSON_INDENT,
    VIS_INSTALLED,
)
from pytileproj._errors import GeomOutOfZoneError, TileOutOfZoneError
from pytileproj._types import (
    AnyTile,
    Extent,
    RasterTileGenerator,
    T_co,
    TileGenerator,
)
from pytileproj.projgeom import (
    GeogCoord,
    GeogGeom,
    ProjCoord,
    ProjGeom,
    convert_any_to_geog_geom,
    fetch_proj_zone,
    pyproj_to_cartopy_crs,
    rasterise_polygon,
    transform_coords,
    transform_geom_to_geog,
    transform_geometry,
)
from pytileproj.tile import IrregularTile, RasterTile
from pytileproj.tiling import IrregularTiling, RegularTiling

if VIS_INSTALLED:
    import cartopy
    import cartopy.feature
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    if TYPE_CHECKING:
        from cartopy.mpl.geoaxes import GeoAxes

if GEO_INSTALLED:
    from geopandas import GeoDataFrame


__all__ = [
    "IrregularProjTilingSystem",
    "ProjCoord",
    "ProjSystem",
    "ProjSystemDefinition",
    "ProjTilingSystem",
    "RegularProjTilingSystem",
    "RegularTilingDefinition",
    "TilingSystem",
    "tiling_access",
]


def tiling_access(f):  # noqa: ANN001, ANN201
    """Map tiling ID to tiling level."""

    def wrapper(self: "TilingSystem", *args: int | str, **kwargs: int | str) -> Any:  # noqa: ANN401
        use_args = f.__name__ == "__getitem__"
        tiling_id = args[0] if use_args else kwargs.get("tiling_id")
        tiling_level = self.tiling_id_to_level(tiling_id)

        if use_args:
            return f(self, tiling_level, *args[1:], **kwargs)
        kwargs["tiling_id"] = tiling_level
        return f(self, *args, **kwargs)

    return wrapper


class ProjSystem(BaseModel, arbitrary_types_allowed=True):
    """Class defining a projection represented by an EPSG code and a zone."""

    crs: Any
    proj_zone_geog: Annotated[Any, AfterValidator(convert_any_to_geog_geom)] = None

    _proj_zone_geog: ProjGeom = PrivateAttr()
    _proj_zone: ProjGeom = PrivateAttr()
    _to_geog: pyproj.Transformer = PrivateAttr()
    _from_geog: pyproj.Transformer = PrivateAttr()
    _crs: pyproj.CRS = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the projection system object."""
        super().model_post_init(context)
        self._crs = pyproj.CRS.from_user_input(self.crs)
        geog_crs = pyproj.CRS(GEOG_EPSG)
        self._to_geog = pyproj.Transformer.from_crs(self._crs, geog_crs, always_xy=True)
        self._from_geog = pyproj.Transformer.from_crs(
            geog_crs, self._crs, always_xy=True
        )

        if self.proj_zone_geog is None:
            epsg = self._crs.to_epsg()
            if epsg is not None:
                self._proj_zone_geog = fetch_proj_zone(epsg)
            else:
                err_msg = "Could not extract projection zone boundaries."
                raise ValueError(err_msg)
            self.proj_zone_geog = self._proj_zone_geog
        else:
            self._proj_zone_geog = cast("ProjGeom", self.proj_zone_geog)

        proj_zone_faulty = transform_geometry(self._proj_zone_geog, self.crs)
        # buffer of 0 removes wrap-arounds along the anti-meridian
        proj_zone = shapely.buffer(proj_zone_faulty.geom, 0)
        self._proj_zone = ProjGeom(geom=proj_zone, crs=self._crs)

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
        proj_coord = GeogCoord(lon, lat)

        return proj_coord in self

    @property
    def pyproj_crs(self) -> pyproj.CRS:
        """Return PyProj representation of CRS."""
        return self._crs

    @property
    def unit(self) -> str:
        """Return projection unit."""
        return self._crs.prime_meridian.unit_name

    def lonlat_to_xy(self, lon: float, lat: float) -> ProjCoord:
        """Convert geographic to a projected coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.

        Returns
        -------
        ProjCoord
            X and Y coordinates if the given input coordinates are
            within the projection zone, None if not.

        Raises
        ------
        GeomOutOfZoneError
            If the given point is outside the projection boundaries.

        """
        coord = None
        if self._lonlat_inside_proj(lon, lat):
            x, y = self._from_geog.transform(lon, lat)
            coord = ProjCoord(x, y, self._crs)
        else:
            raise GeomOutOfZoneError(shapely.Point((lon, lat)))

        return coord

    def xy_to_lonlat(self, x: float, y: float) -> ProjCoord:
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

        Raises
        ------
        GeomOutOfZoneError
            If the given point is outside the projection boundaries.

        """
        lon, lat = self._to_geog.transform(x, y)
        coord = GeogCoord(x=lon, y=lat)
        if not self._lonlat_inside_proj(lon, lat):
            raise GeomOutOfZoneError(shapely.Point((x, y)))

        return coord

    def export_proj_zone_geog(self, path: Path) -> None:
        """Write the projection zone in geographic coordinates to a GeoJSON file.

        Parameters
        ----------
        path: Path
            Output path (.geojson)

        """
        geojson = shapely.to_geojson(self.proj_zone_geog.geom)
        with path.open("w") as f:
            f.writelines(geojson)

    def __contains__(self, other: ProjGeom | ProjCoord) -> bool:
        """Evaluate if the given geometry is fully within the projection zone.

        Parameters
        ----------
        other : ProjGeom | ProjCoord
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry is within the projection zone, false if not.

        """
        if isinstance(other, ProjCoord):
            other = ProjGeom(geom=shapely.Point((other.x, other.y)), crs=other.crs)

        geog_sref = pyproj.CRS.from_epsg(GEOG_EPSG)
        if not geog_sref.is_exact_same(other.crs):
            wrpd_geom = transform_geometry(other, geog_sref)
        else:
            wrpd_geom = other

        return shapely.within(wrpd_geom.geom, self.proj_zone_geog.geom)


class TilingSystem(BaseModel):
    """Class defining a multi-level tiling system."""

    name: str
    tilings: dict[int, RegularTiling | IrregularTiling]

    _tilings_map: dict = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the tiling system object."""
        super().model_post_init(context)
        self._tilings_map = {
            tiling.name: level for level, tiling in self.tilings.items()
        }

    @classmethod
    def from_file(cls, json_path: Path) -> "TilingSystem":
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
            pp_def = orjson.loads(f.read())

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

    def tiling_id_to_level(self, tiling_id: str | int | None) -> int:
        """Convert tiling name to level.

        Parameters
        ----------
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        int:
            Tiling level.

        """
        if tiling_id is None:
            tiling_level = self.tiling_levels[0]
        elif isinstance(tiling_id, int):
            tiling_level = tiling_id
        elif isinstance(tiling_id, str):
            tiling_level = self._tilings_map[tiling_id]
        else:
            err_msg = (
                "The given tiling identifier has "
                f"the wrong data type ({type(tiling_id)})"
                ". Only an integer or string is supported."
            )
            raise TypeError(err_msg)

        return tiling_level

    @property
    def tiling_levels(self) -> list[int]:
        """Return tiling levels."""
        return list(self.tilings.keys())

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

    @tiling_access
    def __getitem__(self, tiling_level: int) -> RegularTiling | IrregularTiling:
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


class ProjTilingSystem(TilingSystem, ProjSystem):
    """Base class defining a projected, multi-level tiling system."""

    def get_tile_from_lonlat(
        self, lon: float, lat: float, tiling_id: int | str | None = None
    ) -> RasterTile:
        """Get a raster tile object from geographic coordinates.

        Parameters
        ----------
        lon: float
            Longitude.
        lat: float
            Latitude.
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        geog_coord = GeogCoord(x=lon, y=lat)
        return self.get_tile_from_coord(geog_coord, tiling_id=tiling_id)

    def get_tile_from_xy(
        self, x: float, y: float, tiling_id: int | str | None = None
    ) -> RasterTile:
        """Get a raster tile object from projected coordinates.

        Parameters
        ----------
        x: float
            World system coordinate in X direction.
        y: float
            World system coordinate in Y direction.
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        proj_coord = ProjCoord(x, y, self.pyproj_crs)
        return self.get_tile_from_coord(proj_coord, tiling_id=tiling_id)

    def get_tile_from_coord(
        self, coord: ProjCoord, tiling_id: int | str | None = None
    ) -> RasterTile:
        """Get a raster tile object from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates object.
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

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

    def _tile_to_name(self, tile: AnyTile) -> str:
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
        return shapely.intersects(tile.boundary.geom, self._proj_zone.geom)

    def _tile_to_raster_tile(
        self, tile: AnyTile, name: str | None = None
    ) -> RasterTile:
        """Create a raster tile object from a given regular tile.

        Parameters
        ----------
        tile: AnyTile
            A tile object.
        name: str | None, optional
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        raise NotImplementedError

    def _get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_level: int = 0,
    ) -> TileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_level: int, optional
            Tiling or zoom level.
            Defaults to the first tiling level.

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

    def get_tile_mask(self, raster_tile: RasterTile) -> np.ndarray:
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
        if not raster_tile.pyproj_crs.is_exact_same(self.pyproj_crs):
            err_msg = "Projection of tile and tiling system must match."
            raise ValueError(err_msg)

        intrsct_geom = shapely.intersection(
            self._proj_zone.geom, raster_tile.boundary.geom
        )
        if intrsct_geom.is_empty:
            mask = np.zeros(raster_tile.shape, dtype=np.uint8)
        elif raster_tile.boundary in self:
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

    @tiling_access
    def plot(  # noqa: C901, PLR0913
        self,
        *,
        ax: Union["GeoAxes", None] = None,
        tiling_id: int | str | None = None,
        facecolor: str = "tab:red",
        edgecolor: str = "black",
        edgewidth: int = 1,
        alpha: float = 1.0,
        proj: Any = None,  # noqa: ANN401
        show: bool = False,
        label_tile: bool = False,
        label_size: int = 12,
        add_country_borders: bool = True,
        extent: Extent | None = None,
        extent_proj: Any = None,  # noqa: ANN401
        plot_zone: bool = False,
    ) -> "GeoAxes":
        """Plot all tiles at a specific tiling level.

        Parameters
        ----------
        ax : GeoAxes
            Pre-defined Matplotlib axis.
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.
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
        proj : Any, optional
            CRS defining the projection of the axes and pyproj.CRS can handle.
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
        extent : Extent, optional
            Coordinate/map extent of the plot, given as
            [min_x, min_y, max_x, max_y] (default is None, meaning global extent).
        extent_proj : Any, optional
            CRS of the given extent. A projection definition pyproj.CRS can handle.
            If it is None, then it is assumed that 'extent' is referring to the
            native projection of the raster tile.
        plot_zone: bool, optional
            True if the projection zone should be added to the plot.
            False if not.


        Returns
        -------
        GeoAxes
            Matplotlib axis containing a Cartopy map with the plotted
            raster tile boundary.

        """
        if not VIS_INSTALLED:
            err_msg = (
                "Modules 'matplotlib' and 'cartopy' are mandatory "
                "for plotting a projected tiling system object."
            )
            raise ImportError(err_msg)

        this_proj = pyproj_to_cartopy_crs(self.pyproj_crs)
        other_proj = (
            this_proj
            if proj is None
            else pyproj_to_cartopy_crs(pyproj.CRS.from_user_input(proj))
        )

        if ax is None:
            ax = cast("GeoAxes", plt.axes(projection=other_proj))
            ax.set_global()
            ax.gridlines()

        if add_country_borders:
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)

        for tile in self[tiling_id].tiles():
            tilename = self._tile_to_name(tile)
            raster_tile = self._tile_to_raster_tile(tile, tilename)
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
            dst_crs = (
                pyproj.CRS.from_user_input(proj)
                if proj is not None
                else self.pyproj_crs
            )
            src_crs = (
                pyproj.CRS.from_user_input(extent_proj)
                if extent_proj is not None
                else self.pyproj_crs
            )
            min_x, min_y = transform_coords(extent[0], extent[1], src_crs, dst_crs)
            max_x, max_y = transform_coords(extent[2], extent[3], src_crs, dst_crs)
            ax.set_xlim((float(min_x), float(max_x)))
            ax.set_ylim((float(min_y), float(max_y)))

        if plot_zone:
            transform = this_proj._as_mpl_transform(ax)  # noqa: SLF001
            x_coords_bound, y_coords_bound = [], []
            if isinstance(self._proj_zone.geom, shapely.MultiPolygon):
                for poly in self._proj_zone.geom.geoms:
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
                    zip(*self._proj_zone.geom.exterior.coords, strict=False)
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

    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str = 0,
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

        Raises
        ------
        NotImplementedError
            This function needs to be overwritten by a child class.

        """
        raise NotImplementedError

    def to_geodataframe(
        self,
        tiling_ids: list[str | int] | None = None,
        exclude: set[str] | None = None,
    ) -> GeoDataFrame:
        """Create a geodataframe from all tiles within the system.

        Parameters
        ----------
        tiling_ids: list[str | int] | None, optional
            List of tiling levels or names.
            Defaults to all tiling levels.
        exclude: set[str] | None, optional
            Exclude raster tile object attributes.
            Excludes the 'crs' attribute by default.

        Returns
        -------
        GeoDataFrame
            Dataframe where each row contains a representation of a raster tile.

        """
        tiling_levels = (
            self.tiling_levels
            if tiling_ids is None
            else [self.tiling_id_to_level(tiling_id) for tiling_id in tiling_ids]
        )
        exclude_attrs = {"crs"}
        if exclude is not None:
            exclude_attrs = exclude_attrs.union(exclude)

        tiling_attrs = ["name", "tiling_level"]
        tiling_attrs_rnmd = ["tiling_name", "tiling_level"]

        rtiles = []
        for tiling_level in tiling_levels:
            tiling_dict = {}
            for i in range(len(tiling_attrs)):
                tiling_dict[tiling_attrs_rnmd[i]] = getattr(
                    self[tiling_level], tiling_attrs[i]
                )

            for tile in self[tiling_level].tiles():
                tilename = self._tile_to_name(tile)
                rtile = self._tile_to_raster_tile(tile, tilename)
                if not self._tile_in_zone(rtile):
                    continue
                rtile_dict = rtile.model_dump(exclude=exclude_attrs)
                rtile_dict.update(tiling_dict)
                rtile_dict["geometry"] = rtile.boundary_shapely
                rtiles.append(rtile_dict)

        return GeoDataFrame(rtiles, crs=self.pyproj_crs)

    def to_shapefile(
        self,
        shp_path: Path,
        tiling_ids: list[str | int] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        """Write class attributes to a JSON file.

        Parameters
        ----------
        shp_path: Path
            Path to JSON file.
        tiling_ids: list[str | int] | None, optional
            List of tiling levels or names.
            Defaults to all tiling levels.
        exclude: set[str] | None, optional
            Exclude raster tile object attributes.
            Excludes the 'crs' attribute by default.

        """
        tiling_levels = (
            self.tiling_levels
            if tiling_ids is None
            else [self.tiling_id_to_level(tiling_id) for tiling_id in tiling_ids]
        )
        for tiling_level in tiling_levels:
            layer_name = self[tiling_level].name
            shp_layer_path = shp_path.parent / f"{shp_path.stem}_{layer_name}.shp"
            self.to_geodataframe([tiling_level], exclude).to_file(shp_layer_path)


def validate_regular_tilings(tilings: dict[int, RegularTiling]) -> None:
    """Validate if different regular tilings are compliant with each other.

    This means they need to:
        - have the same origin
        - have the same extent
        - have the same orientation
        - increase the number of tiles with increasing tiling level

    Parameters
    ----------
    tilings: dict[int, RegularTiling]
        Dictionary with tiling/zoom levels as keys and regular tilings as values.

    Raises
    ------
    ValueError
        If one of the conditions above is not met.

    """
    tiling_levels = sorted(tilings.keys())
    ref_tiling = tilings[tiling_levels[0]]
    for tiling_level in tiling_levels[1:]:
        tiling = tilings[tiling_level]

        same_ll_origin = ref_tiling.extent[:2] == tiling.extent[:2]
        same_ul_origin = (ref_tiling.extent[0] == tiling.extent[0]) & (
            ref_tiling.extent[3] == tiling.extent[3]
        )
        if not any([same_ll_origin, same_ul_origin]):
            ref_tiling_str = f"{ref_tiling.tiling_level}:{ref_tiling.origin_xy}"
            tiling_str = f"{tiling.tiling_level}:{tiling.origin_xy}"
            err_hdr = "The given tilings do not have the same origin:"
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

        ref_tiling = tiling


class ProjSystemDefinition(BaseModel, Generic[T_co], arbitrary_types_allowed=True):
    """Definition for a projection system."""

    name: str
    crs: Any
    min_xy: tuple[int | float, int | float] | None = None
    max_xy: tuple[int | float, int | float] | None = None
    proj_zone_geog: Annotated[Any, AfterValidator(convert_any_to_geog_geom)] = None
    axis_orientation: tuple[Literal["W", "E"], Literal["N", "S"]] = ("E", "N")


def convert_to_tile_shape(arg: float | tuple[float, float]) -> tuple[float, float]:
    """Convert tile size/shape to tuple."""
    return (arg, arg) if isinstance(arg, float | int) else arg


class RegularTilingDefinition(BaseModel):
    """Definition for a specific Regular Tiling System."""

    name: str
    tile_shape: Annotated[
        tuple[float, float] | float | int, AfterValidator(convert_to_tile_shape)
    ]


class RegularProjTilingSystem(ProjTilingSystem, Generic[T_co]):
    """Regular projected, multi-level tiling system."""

    tilings: dict[int, RegularTiling]

    _tms: TileMatrixSet = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the regular projected tiling system."""
        super().model_post_init(context)
        wkt = self.pyproj_crs.to_json_dict()
        self._tms = TileMatrixSet(
            crs={"wkt": wkt},
            tileMatrices=[
                self.tilings[tiling_level].tm
                for tiling_level in sorted(self.tiling_levels)
            ],
        )

    @staticmethod
    def _get_tiling_from_id(
        tiling_defs: dict[int, RegularTilingDefinition], tiling_id: int | str
    ) -> tuple[int | None, RegularTilingDefinition | None]:
        if isinstance(tiling_id, int):
            tiling_def = tiling_defs.get(tiling_id)
            tiling_level = tiling_id
        else:
            tiling_def = None
            tiling_level = None
            for i, v in enumerate(tiling_defs.values()):
                if tiling_id == v.name:
                    tiling_def = v
                    tiling_level = i
                    break

        if tiling_def is None and tiling_level is None:
            err_msg = f"There is no tile definition for the tiling ID {tiling_id}"
            raise ValueError(err_msg)

        return tiling_level, tiling_def

    @staticmethod
    def _get_extent_from_proj(
        proj_def: ProjSystemDefinition, tiling_def: RegularTilingDefinition
    ) -> Extent:
        if proj_def.max_xy is None or proj_def.min_xy is None:
            pyproj_crs = pyproj.CRS.from_user_input(proj_def.crs)
            epsg = pyproj_crs.to_epsg()
            if proj_def.proj_zone_geog is not None:
                proj_zone_geog = proj_def.proj_zone_geog
            elif epsg is not None:
                proj_zone_geog = fetch_proj_zone(epsg)
            else:
                err_msg = "Could not extract projection zone boundaries."
                raise ValueError(err_msg)
            proj_zone = transform_geometry(proj_zone_geog, pyproj_crs)
            proj_extent = proj_zone.geom.bounds
            min_x, min_y = (
                proj_extent[:2] if proj_def.min_xy is None else proj_def.min_xy
            )
            max_x, max_y = (
                proj_extent[2:] if proj_def.max_xy is None else proj_def.max_xy
            )
            x_size, y_size = max_x - min_x, max_y - min_y
            tile_shape = cast("tuple", tiling_def.tile_shape)
            x_size_mod = np.ceil(x_size / tile_shape[0]) * tile_shape[0]
            y_size_mod = np.ceil(y_size / tile_shape[1]) * tile_shape[1]
            dst_extent = (min_x, min_y, min_x + x_size_mod, min_y + y_size_mod)
        else:
            min_x, min_y = proj_def.min_xy
            max_x, max_y = proj_def.max_xy
            dst_extent = (min_x, min_y, max_x, max_y)

        return dst_extent

    @classmethod
    def from_sampling(
        cls,
        sampling: float | dict[int, float | int],
        proj_def: ProjSystemDefinition,
        tiling_defs: dict[int, RegularTilingDefinition],
        **kwargs: Any,  # noqa: ANN401
    ) -> "RegularProjTilingSystem":
        """Classmethod for creating a regular, projected tiling system.

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
        **kwargs: Any
            Additional class attributes.

        Returns
        -------
        RegularProjTilingSystem
            Regular, projected tiling system instance.

        """
        if isinstance(sampling, dict):
            samplings = sampling
        else:
            samplings = dict.fromkeys(tiling_defs.keys(), sampling)

        tilings = {}
        for k, s in samplings.items():
            tiling_level, tiling_def = cls._get_tiling_from_id(tiling_defs, k)
            extent = cls._get_extent_from_proj(proj_def, tiling_def)

            tiling = RegularTiling(
                name=tiling_def.name,
                extent=extent,
                sampling=s,
                tile_shape=cast("tuple", tiling_def.tile_shape),
                tiling_level=tiling_level,
                axis_orientation=proj_def.axis_orientation,
            )
            tilings[tiling_level] = tiling

        return cls(
            name=proj_def.name,
            crs=proj_def.crs,
            tilings=tilings,
            proj_zone_geog=proj_def.proj_zone_geog,
            **kwargs,
        )

    @model_validator(mode="after")
    def check_tilings(self) -> "RegularProjTilingSystem":
        """Validate if different regular tilings are compliant with each other."""
        validate_regular_tilings(self.tilings)
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

    @property
    def is_congruent(self) -> bool:
        """Check if tilings from different levels are congruent.

        Tilings from adjacent tiling levels are congruent, if tiles from the
        higher tiling level (fine tiling) are exactly a multiple of the tiles
        at the lower level (coarse tiling).
        """
        is_congruent = True
        tiling_levels = sorted(self.tilings.keys())
        ref_tiling = self.tilings[tiling_levels[0]]
        for tiling_level in tiling_levels[1:]:
            tiling = self.tilings[tiling_level]

            if (ref_tiling.tile_shape[0] % tiling.tile_shape[0] != 0) or (
                ref_tiling.tile_shape[1] % tiling.tile_shape[1] != 0
            ):
                is_congruent = False
                break

        return is_congruent

    @tiling_access
    def n_tiles_x(self, tiling_id: int | str | None = None) -> int:
        """Return the number of tiles in X direction at the given tiling.

        Parameters
        ----------
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        int
            Number of tiles in X direction.

        """
        return self[tiling_id].tm.matrixWidth

    @tiling_access
    def n_tiles_y(self, tiling_id: int | str | None = None) -> int:
        """Return the number of tiles in Y direction at the given tiling.

        Parameters
        ----------
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        int
            Number of tiles in Y direction.

        """
        return self[tiling_id].tm.matrixHeight

    @tiling_access
    def n_tiles(self, tiling_id: int | str | None = None) -> int:
        """Return the number of tiles at the given tiling.

        Parameters
        ----------
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        int
            Number of tiles.

        """
        return self.n_tiles_x(tiling_id) * self.n_tiles_y(tiling_id)

    @classmethod
    def default(
        cls,
        name: str,
        epsg: int,
        extent: tuple[float, float, float, float],
        tile_shape_px: tuple[NonNegativeInt, NonNegativeInt],
        tiling_level_limits: tuple[NonNegativeInt, NonNegativeInt] = (0, 24),
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
            list(extent),
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
                tile_shape=(tm.cellSize * tm.tileWidth, tm.cellSize * tm.tileHeight),
            )
            tilings.append(tiling)

        return cls(name, epsg, tilings)

    def _tile_to_name(self, tile: AnyTile) -> str:
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
        n_digits_xy = len(str(max(self.max_n_tiles_x, self.max_n_tiles_y)))
        x_label = f"X{tile.x:0{n_digits_xy}}"
        y_label = f"Y{tile.y:0{n_digits_xy}}"
        z_label = f"T{tile.z:02}"
        return x_label + y_label + z_label

    def _name_to_tile(self, tilename: str) -> RegularTile:
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
        tiling_level = int(tilename.split("T")[-1])
        x = int(tilename.split("Y")[0][1:])
        y = int(tilename.split("Y")[1].split("T")[0])
        return RegularTile(x, y, tiling_level)

    def get_tile_from_index(self, x: int, y: int, tiling_level: int) -> RasterTile:
        """Get a raster tile object from the given tile index.

        Parameters
        ----------
        x: int
            Horizontal index.
        y: int
            Verctical index.
        tiling_level: int
            Tiling level or zoom.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = RegularTile(x, y, tiling_level)
        tilename = self._tile_to_name(tile)
        raster_tile = self._tile_to_raster_tile(tile, name=tilename)
        if not self._tile_in_zone(raster_tile):
            raise TileOutOfZoneError(raster_tile)

        return raster_tile

    @tiling_access
    def get_tile_from_coord(self, coord: ProjCoord, tiling_id: int = 0) -> RasterTile:
        """Get a raster tile object from projected coordinates.

        Parameters
        ----------
        coord: ProjCoord
            Projected coordinates object.
        tiling_id: int | str, optional
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = self._tms.tile(
            coord.x,
            coord.y,
            tiling_id,
            geographic_crs=coord.crs.to_proj4(),  # ty: ignore[invalid-argument-type]
        )
        tilename = self._tile_to_name(tile)
        return self._tile_to_raster_tile(tile, name=tilename)

    def _tile_to_raster_tile(
        self, tile: AnyTile, name: str | None = None
    ) -> RasterTile:
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
        extent = self._tms.xy_bounds(cast("RegularTile", tile))
        sampling = self[tile.z].sampling
        return RasterTile.from_extent(extent, self.crs, sampling, sampling, name=name)

    def _tiles(
        self, bbox: tuple[float, float, float, float], tiling_level: int
    ) -> RasterTileGenerator:
        """Get the tiles overlapped by a projected bounding box.

        Original code from https://github.com/mapbox/mercantile/blob/master/mercantile/__init__.py#L424

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box in projected coordinates (west, south, east, north).
        tiling_level: int
            Tiling level or zoom.

        Yields
        ------
        RasterTile

        Notes
        -----
        A small epsilon is used on the south and east parameters so that this
        function yields exactly one tile when given the bounds of that same tile.

        """
        west, south, east, north = bbox
        nw_tile = self._tms._tile(west, north, tiling_level, ignore_coalescence=True)  # noqa: SLF001
        se_tile = self._tms._tile(east, south, tiling_level, ignore_coalescence=True)  # noqa: SLF001
        minx = min(nw_tile.x, se_tile.x)
        maxx = max(nw_tile.x, se_tile.x)
        miny = min(nw_tile.y, se_tile.y)
        maxy = max(nw_tile.y, se_tile.y)

        matrix = self._tms.matrix(tiling_level)
        for j in range(miny, maxy + 1):
            cf = (
                matrix.get_coalesce_factor(j)
                if matrix.variableMatrixWidths is not None
                else 1
            )
            for i in range(minx, maxx + 1):
                if cf != 1 and i % cf:
                    continue

                tile = RegularTile(i, j, tiling_level)
                tilename = self._tile_to_name(tile)
                raster_tile = self._tile_to_raster_tile(tile, name=tilename)

                yield raster_tile

    def _get_tiles_in_geog_geom(
        self, geog_geom: GeogGeom, tiling_level: int = 0
    ) -> RasterTileGenerator:
        """Get all raster tiles covered by a geographic geometry.

        Parameters
        ----------
        geog_geom: GeogGeom
            Geometry given in LonLat geographic projection system.
        tiling_level: int
            Tiling level or zoom.

        Yields
        ------
        RasterTile

        """
        geom_intersection = shapely.intersection(
            geog_geom.geom, self._proj_zone_geog.geom
        )
        geom_intersects = not geom_intersection.is_empty
        if geom_intersects:
            geog_geoms = []
            if geom_intersection.geom_type == "MultiPolygon":
                geog_geoms.extend(
                    GeogGeom(geom=geom) for geom in geom_intersection.geoms
                )
            else:
                geog_geoms.append(GeogGeom(geom=geom_intersection))
            tilenames = []
            for geom in geog_geoms:
                proj_geom = transform_geometry(
                    geom, self.pyproj_crs, segment=DEF_SEG_LEN_DEG
                )
                for raster_tile in self._tiles(
                    proj_geom.geom.bounds,
                    tiling_level,
                ):
                    if raster_tile.name not in tilenames:
                        tile_intersects = shapely.intersects(
                            raster_tile.boundary_shapely, proj_geom.geom
                        )
                        if not tile_intersects:
                            continue

                        tilenames.append(raster_tile.name)
                        yield raster_tile

    @tiling_access
    def get_tiles_in_geog_bbox(
        self, bbox: tuple[float, float, float, float], tiling_id: int | str = 0
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (west, south, east, north) for selecting tiles.
        tiling_id: int | str
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTileGenerator
            Yields tile after tile, which intersects with the given bounding box.

        """
        west, south, east, north = bbox
        bbox_poly = shapely.Polygon(
            [(west, south), (east, south), (east, north), (west, north)]
        )
        if east < west:  # bbox crosses antimeridian
            bbox_poly = fix_polygon(bbox_poly)

        yield from self._get_tiles_in_geog_geom(
            GeogGeom(geom=bbox_poly), cast("int", tiling_id)
        )

    @tiling_access
    def get_tiles_in_bbox(
        self, bbox: tuple[float, float, float, float], tiling_id: int | str = 0
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the projected bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Projected bounding box (west, south, east, north) for selecting tiles.
        tiling_id: int | str
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTileGenerator
            Yields tile after tile, which intersects with the given bounding box.

        """
        yield from self._tiles(bbox, cast("int", tiling_id))

    @tiling_access
    def get_tiles_in_geom(
        self, proj_geom: ProjGeom, tiling_id: int | str = 0
    ) -> RasterTileGenerator:
        """Search for tiles intersecting with the projected geometry.

        Parameters
        ----------
        proj_geom : ProjGeom
            Projected geometry representing the region of interest.
        tiling_id: int | str
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTileGenerator
            Yields tile after tile, which intersects with the given geometry.

        """
        if proj_geom.crs.to_epsg() == GEOG_EPSG:
            geog_geom = fix_polygon(proj_geom.geom)
            geog_geom = GeogGeom(geom=geog_geom)
        else:
            geog_geom = transform_geom_to_geog(proj_geom)

        yield from self._get_tiles_in_geog_geom(geog_geom, cast("int", tiling_id))

    def get_children_from_name(self, tilename: str) -> RasterTileGenerator:
        """Get all child tiles (next higher zoom level).

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTileGenerator
            Yields all tile children as raster tiles.

        """
        tile = self._name_to_tile(tilename)
        for child_tile in self._tms.children(tile):
            child_tilename = self._tile_to_name(child_tile)
            yield self._tile_to_raster_tile(child_tile, name=child_tilename)

    def get_parent_from_name(self, tilename: str) -> RasterTile:
        """Get parent tile (next lower zoom level).

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTile
            Parent raster tile.

        """
        tile = self._name_to_tile(tilename)
        parent_tile = self._tms.parent(tile)[0]
        parent_tilename = self._tile_to_name(parent_tile)
        return self._tile_to_raster_tile(parent_tile, name=parent_tilename)

    def to_ogc_standard(self) -> dict:
        """OGC representation of the tiling system."""
        return self._tms.model_dump()

    def to_ogc_json(self, json_path: Path) -> None:
        """Write OGC representation of the tiling system to a JSON file."""
        grid_def = json.dumps(self.to_ogc_standard(), indent=JSON_INDENT)
        with json_path.open("w") as f:
            f.writelines(grid_def)


class IrregularProjTilingSystem(ProjTilingSystem):
    """Irregular projected, multi-level tiling system."""

    def _tilename_to_level(self, tilename: str) -> int:
        return int(tilename.split("T")[-1])

    def _get_tile(self, tilename: str) -> IrregularTile:
        """Get an irregular tile object from a given tilename.

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

    def _tile_to_raster_tile(
        self, tile: AnyTile, name: str | None = None
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
        return RasterTile.from_extent(extent, self.crs, sampling, sampling, name=name)

    @tiling_access
    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str = 0,
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
            Yields raster tile after tile, which intersects with the given bounding box.

        """
        for tile in self[tiling_id].tiles_in_bbox(bbox):
            raster_tile = self._tile_to_raster_tile(tile, name=tile.id)
            if not shapely.intersects(raster_tile.boundary.geom, self._proj_zone.geom):
                continue

            yield raster_tile
