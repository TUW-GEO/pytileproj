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
from typing import Annotated, Any, Literal, Union

import numpy as np
import pyproj
import shapely
from morecantile.models import Tile as RegularTile
from morecantile.models import TileMatrixSet
from pydantic import AfterValidator, BaseModel, NonNegativeInt, model_validator

from pytileproj._const import GEO_INSTALLED, JSON_INDENT, VIS_INSTALLED
from pytileproj._errors import GeomOutOfZoneError, TileOutOfZoneError
from pytileproj.projgeom import (
    ProjCoord,
    ProjGeom,
    convert_any_to_geog_geom,
    fetch_proj_zone,
    pyproj_to_cartopy_crs,
    rasterise_polygon,
    transform_coords,
    transform_geometry,
)
from pytileproj.tile import IrregularTile, RasterTile
from pytileproj.tiling import IrregularTiling, RegularTiling

if VIS_INSTALLED:
    import cartopy
    import cartopy.feature
    import matplotlib.axes as mplax
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

if GEO_INSTALLED:
    from geopandas import GeoDataFrame

AnyTile = RegularTile | IrregularTile
TileGenerator = Generator[AnyTile, AnyTile, AnyTile]
RasterTileGenerator = Generator[RasterTile, RasterTile, RasterTile]


__all__ = [
    "IrregularProjTilingSystem",
    "ProjCoord",
    "ProjSystem",
    "ProjSystemDefinition",
    "ProjTilingSystem",
    "RegularProjTilingSystem",
    "RegularTilingDefinition",
    "TilingSystem",
]


def _tiling_access(f: callable) -> callable:
    def wrapper(self: "TilingSystem", *args: tuple, **kwargs: dict) -> Any:  # noqa: ANN401
        use_args = f.__name__ == "__getitem__"
        tiling_id = args[0] if use_args else kwargs.get("tiling_id")
        tiling_level = self.tiling_id_to_level(tiling_id)

        if use_args:
            args = list(args)
            args[0] = tiling_level
        else:
            kwargs["tiling_id"] = tiling_level

        return f(self, *args, **kwargs)

    return wrapper


class ProjSystem(BaseModel, arbitrary_types_allowed=True):
    """Class defining a projection represented by an EPSG code and a zone."""

    crs: Any
    proj_zone_geog: Annotated[
        Path | shapely.Geometry | ProjGeom | None,
        AfterValidator(convert_any_to_geog_geom),
    ] = None

    _proj_zone: ProjGeom
    _to_geog: pyproj.Transformer
    _from_geog: pyproj.Transformer
    _crs: pyproj.CRS

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise projection system object."""
        super().__init__(**kwargs)
        self._crs = pyproj.CRS.from_user_input(self.crs)
        geog_crs = pyproj.CRS(4326)
        self._to_geog = pyproj.Transformer.from_crs(self._crs, geog_crs, always_xy=True)
        self._from_geog = pyproj.Transformer.from_crs(
            geog_crs, self._crs, always_xy=True
        )
        self.proj_zone_geog = self.proj_zone_geog or fetch_proj_zone(
            self._crs.to_epsg()
        )
        proj_zone_faulty = transform_geometry(self.proj_zone_geog, self.crs)
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
        proj_coord = ProjCoord(lon, lat, pyproj.CRS.from_epsg(4326))

        return proj_coord in self

    @property
    def pyproj_crs(self) -> pyproj.CRS:
        """Return PyProj representation of CRS."""
        return self._crs

    @property
    def unit(self) -> str:
        """Return projection unit."""
        return self._crs.prime_meridian.unit_name

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
        OutOfBoundsError
            If the given point is not within the projection zone.

        """
        lon, lat = self._to_geog.transform(x, y)
        coord = ProjCoord(lon, lat, pyproj.CRS.from_epsg(4326))
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

        geog_sref = pyproj.CRS.from_epsg(4326)
        if not geog_sref.is_exact_same(other.crs):
            wrpd_geom = transform_geometry(other, geog_sref)
        else:
            wrpd_geom = other

        return shapely.within(wrpd_geom.geom, self.proj_zone_geog.geom)


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


class TilingSystem(BaseModel):
    """Class defining a multi-level tiling system."""

    name: str
    tilings: dict[int, RegularTiling | IrregularTiling]
    allowed_samplings: dict[int, list[int | float]] | None = None

    _tilings_map: dict

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise tiling system object."""
        super().__init__(**kwargs)
        self._tilings_map = {
            tiling.name: level for level, tiling in self.tilings.items()
        }

    @model_validator(mode="after")
    def check_samplings(self) -> "TilingSystem":
        """Check the given user samplings.

        Check the sampling defined in a tiling corresponds to the
        given allowed samplings.
        """
        validate_samplings(self.tilings, self.allowed_samplings)
        return self

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

    @_tiling_access
    def __getitem__(self, tiling_id: int | str) -> RegularTiling:
        """Return the tiling at a specific tiling level.

        Parameter
        ---------
        tiling_id: int | str
            Tiling ID representing a tiling level or tiling name.

        Returns
        -------
        RegularTiling
            Tiling object at a specific tiling level.

        """
        return self.tilings[tiling_id]


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
        geog_coord = ProjCoord(lon, lat, pyproj.CRS.from_epsg(4326))
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

    def _tile_to_raster_tile(
        self, tile: AnyTile, name: str | None = None
    ) -> RasterTile:
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

    def _get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str | None = None,
    ) -> TileGenerator:
        """Search for tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_id: int | str | None
            Tiling level or name.
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

    @_tiling_access
    def plot(  # noqa: C901, PLR0913
        self,
        *,
        ax: Union["mplax.Axes", None] = None,
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
        extent: tuple | None = None,
        extent_proj: Any = None,  # noqa: ANN401
        plot_zone: bool = False,
    ) -> "mplax.Axes":
        """Plot all tiles at a specific tiling level.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
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
        extent : tuple or list, optional
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

        this_proj = pyproj_to_cartopy_crs(self.pyproj_crs)
        other_proj = (
            this_proj
            if proj is None
            else pyproj_to_cartopy_crs(pyproj.CRS.from_user_input(proj))
        )

        if ax is None:
            ax = plt.axes(projection=other_proj)
            ax.set_global()
            ax.gridlines()

        if add_country_borders:
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS)

        for tile in self[tiling_id]:
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
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

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
        tiling_id: int | str | None = None,
    ) -> RasterTileGenerator:
        """Get all tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_id: int | str | None
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

    def __contains__(self, geom: RasterTile | ProjGeom) -> bool:
        """Contain wrapper.

        Evaluate if the given geometry or raster tile is fully within the
        projection zone.

        Parameters
        ----------
        geom : ProjGeom | RasterTile
            Other geometry or raster tile to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry or raster tile is within the projection
            zone, false if not.

        """
        arg = geom.boundary if isinstance(geom, RasterTile) else geom

        return super().__contains__(arg)

    def to_geodataframe(
        self,
        tiling_ids: list[str | int] | None = None,
        exclude: list[str] | None = None,
    ) -> GeoDataFrame:
        """Create a geodataframe from all tiles within the system.

        Parameters
        ----------
        tiling_ids: list[str | int] | None, optional
            List of tiling levels or names.
            Defaults to all tiling levels.
        exclude: list[str] | None, optional
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
        exclude_attrs = ["crs"]
        if exclude is not None:
            exclude_attrs = list(set(exclude_attrs + exclude_attrs))

        tiling_attrs = ["name", "tiling_level"]
        tiling_attrs_rnmd = ["tiling_name", "tiling_level"]

        rtiles = []
        for tiling_level in tiling_levels:
            tiling_dict = {}
            for i in range(len(tiling_attrs)):
                tiling_dict[tiling_attrs_rnmd[i]] = getattr(
                    self[tiling_level], tiling_attrs[i]
                )

            for tile in self[tiling_level]:
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
        exclude: list[str] | None = None,
    ) -> None:
        """Write class attributes to a JSON file.

        Parameters
        ----------
        shp_path: Path
            Path to JSON file.
        tiling_ids: list[str | int] | None, optional
            List of tiling levels or names.
            Defaults to all tiling levels.
        exclude: list[str] | None, optional
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

        same_origin = ref_tiling.origin_xy == tiling.origin_xy
        if not same_origin:
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


class ProjSystemDefinition(BaseModel, arbitrary_types_allowed=True):
    """Definition for a projection system."""

    name: str
    crs: Any
    extent: tuple[float, float, float | None, float | None] | None = None
    proj_zone_geog: Annotated[
        Path | shapely.Geometry | ProjGeom | None,
        AfterValidator(convert_any_to_geog_geom),
    ] = None
    axis_orientation: tuple[Literal["W", "E"], Literal["N", "S"]] | None = ("E", "N")


class RegularTilingDefinition(BaseModel):
    """Definition for a specific Regular Tiling System."""

    name: str
    tile_size: float | int


class RegularProjTilingSystem(ProjTilingSystem):
    """Regular projected, multi-level tiling system."""

    _tms: TileMatrixSet

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise regular projected tiling system."""
        super().__init__(**kwargs)
        wkt = self._crs.to_json_dict()
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
        proj_def: ProjSystemDefinition,
        tiling_defs: dict[int, RegularTilingDefinition],
        *,
        allowed_samplings: dict[int, list[float | int]] | None = None,
    ) -> "RegularProjTilingSystem":
        """Classmethod for creating a regular, projected tiling system.

        Create a regular, projected tiling system instance from given tiling system
        definitions and a grid sampling.

        Parameters
        ----------
        proj_def: ProjSystemDefinition
            Projection system definition (stores name, CRS, extent,
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

            extent = list(proj_def.extent) or [None] * 4
            if None in extent:
                pyproj_crs = pyproj.CRS.from_user_input(proj_def.crs)
                proj_zone_geog = proj_def.proj_zone_geog or fetch_proj_zone(
                    pyproj_crs.to_epsg()
                )
                proj_zone = transform_geometry(proj_zone_geog, pyproj_crs)
                proj_extent = proj_zone.geom.bounds
                extent = [
                    proj_extent[i] if extent[i] is None else extent[i]
                    for i in range(len(extent))
                ]
                min_x, min_y, max_x, max_y = extent
                x_size, y_size = max_x - min_x, max_y - min_y
                x_size_mod = (
                    np.ceil(x_size / tiling_def.tile_size) * tiling_def.tile_size
                )
                y_size_mod = (
                    np.ceil(y_size / tiling_def.tile_size) * tiling_def.tile_size
                )
                extent = (min_x, min_y, min_x + x_size_mod, min_y + y_size_mod)

            tiling = RegularTiling(
                name=tiling_def.name,
                extent=extent,
                sampling=s,
                tile_shape_px=(tile_size_px, tile_size_px),
                tiling_level=k,
                axis_orientation=proj_def.axis_orientation,
            )
            tilings[k] = tiling

        return cls(
            name=proj_def.name,
            crs=proj_def.crs,
            tilings=tilings,
            allowed_samplings=allowed_samplings,
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
            ref_n_rows, ref_n_cols = (
                ref_tiling.tm.matrixHeight,
                ref_tiling.tm.matrixWidth,
            )
            n_rows, n_cols = tiling.tm.matrixHeight, tiling.tm.matrixWidth

            if (n_rows % ref_n_rows != 0) or (n_cols % ref_n_cols != 0):
                is_congruent = False
                break

        return is_congruent

    @_tiling_access
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

    @_tiling_access
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

    @_tiling_access
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

    def _tile_to_name(self, tile: RegularTile) -> str:
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

    @_tiling_access
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

        """
        tile = self._tms.tile(
            coord.x,
            coord.y,
            tiling_id,
            geographic_crs=coord.crs,
        )
        tilename = self._tile_to_name(tile)
        return self._tile_to_raster_tile(tile, name=tilename)

    def _tile_to_raster_tile(
        self, tile: RegularTile, name: str | None = None
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
        extent = self._tms.xy_bounds(tile)
        sampling = self[tile.z].sampling
        return RasterTile.from_extent(extent, self.crs, sampling, sampling, name=name)

    def _get_tiles_in_geog_bbox(
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

    @_tiling_access
    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str | None = None,
    ) -> RasterTileGenerator:
        """Get all tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_id: int | str | None
            Tiling level or name.
            Defaults to the first tiling level.

        Returns
        -------
        RasterTileGenerator
            Yields raster tile after tile, which intersects with the given
            bounding box.

        """
        for tile in self._get_tiles_in_geog_bbox(bbox, tiling_id):
            tilename = self._tile_to_name(tile)
            raster_tile = self._tile_to_raster_tile(tile, name=tilename)
            if not self._tile_in_zone(raster_tile):
                continue

            yield raster_tile

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

    def _tilename_to_level(self, tilename: str) -> str:
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

    def get_tile_from_name(self, tilename: str) -> RasterTile:
        """Get a raster tile object from a given tilename.

        Parameters
        ----------
        tilename: str
            Tilename.

        Returns
        -------
        RasterTile
            Raster tile object.

        """
        tile = self._get_tile(tilename)
        raster_tile = self._tile_to_raster_tile(tile, name=tilename)
        if not shapely.intersects(raster_tile.boundary.geom, self._proj_zone.geom):
            raster_tile = None

        return raster_tile

    def _tile_to_raster_tile(
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
        return RasterTile.from_extent(extent, self.crs, sampling, sampling, name=name)

    @_tiling_access
    def get_tiles_in_geog_bbox(
        self,
        bbox: tuple[float, float, float, float],
        tiling_id: int | str | None = None,
    ) -> RasterTileGenerator:
        """Get all tiles intersecting with the geographic bounding box.

        Parameters
        ----------
        bbox: tuple[float, float, float, float]
            Bounding box (x_min, y_min, x_max, y_max) for selecting tiles.
        tiling_id: int | str | None
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
