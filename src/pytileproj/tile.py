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

"""Tile module defining regular and irregular tiles."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt
import orjson
import pyproj
import shapely
import shapely.wkt
from pydantic import BaseModel, NonNegativeInt, PrivateAttr
from shapely.geometry import Polygon

from pytileproj._const import DECIMALS, JSON_INDENT, VIS_INSTALLED
from pytileproj.projgeom import (
    ProjGeom,
    ij2xy,
    pyproj_to_cartopy_crs,
    round_polygon_vertices,
    transform_coords,
    transform_geometry,
    xy2ij,
)

if VIS_INSTALLED:
    import cartopy
    import cartopy.feature
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    if TYPE_CHECKING:
        from collections.abc import Callable, Sequence

        from cartopy.mpl.geoaxes import GeoAxes

__all__ = ["RasterTile"]

Extent = tuple[float, float, float, float]
GeoTransformTuple: TypeAlias = tuple[float, float, float, float, float, float]
OriginStr: TypeAlias = Literal["ul", "ur", "ll", "lr", "c"]
T_co = TypeVar("T_co", covariant=True)
RT = TypeVar("RT", bound="RasterTile[Any]")


class IrregularTile(BaseModel, Generic[T_co]):
    """Defines an irregular tile (arbitrary extent) at a specific zoom/tiling level."""

    name: str
    z: int
    extent: Extent

    _boundary: shapely.Polygon = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the irregular tile object."""
        super().model_post_init(context)

        min_x, min_y, max_x, max_y = self.extent
        self._boundary = shapely.Polygon(
            (
                (min_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (max_x, min_y),
                (min_x, min_y),
            )
        )

    @property
    def boundary(self) -> shapely.Polygon:
        """Tile boundary represented by a shapely.Polygon."""
        return self._boundary


def _align_geom():  # noqa: ANN202
    """Align external geometries.

    A decorator which checks if a spatial reference is available for an
    ProjGeom object and optionally reprojects the given geometry to the
    spatial reference of the projected tile.

    Returns
    -------
    decorator
        Wrapper around `f`.

    Notes
    -----
    The projected geometry is assumed to be given in first place.

    """

    def decorator(f: Callable, *args, **kwargs) -> Callable:  # noqa: ARG001, ANN002, ANN003, D417
        """Call wrapper function.

        Parameters
        ----------
        f : Callable
            Function to wrap around/execute.

        Returns
        -------
        wrapper
            Wrapper function.

        """

        def wrapper(  # noqa: ANN202
            self: RasterTile,
            arg: ProjGeom | RasterTile,
            *args: Sequence[Any],
            **kwargs: dict[str, Any],
        ):
            proj_geom = arg.boundary if isinstance(arg, RasterTile) else arg

            # warp the geometry to the spatial reference of the tile
            # if they are not the same
            this_crs = self.pyproj_crs
            other_crs = proj_geom.crs
            if not this_crs.is_exact_same(other_crs):
                wrpd_geom = transform_geometry(proj_geom, this_crs)
            else:
                wrpd_geom = proj_geom

            return f(self, wrpd_geom, *args, **kwargs)

        return wrapper

    return decorator


class RasterTile(BaseModel, Generic[T_co]):
    """Defines a raster tile geometry located in a certain projection."""

    crs: Any
    n_rows: NonNegativeInt
    n_cols: NonNegativeInt
    geotrans: GeoTransformTuple = (0, 1, 0, 0, 0, -1)
    px_origin: OriginStr = "ul"
    name: str | None = None

    _boundary: ProjGeom = PrivateAttr()
    _crs: pyproj.CRS = PrivateAttr()

    def model_post_init(self, context: Any) -> None:  # noqa: ANN401
        """Initialise remaining parts of the raster tile object."""
        super().model_post_init(context)
        self._crs = pyproj.CRS.from_user_input(self.crs)
        self._boundary = self.__proj_geom_boundary()

    @classmethod
    def from_extent(
        cls,
        extent: Extent,
        crs: Any,  # noqa: ANN401
        x_pixel_size: float,
        y_pixel_size: float,
        name: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Initialise raster tile from a given extent and projection information.

        Parameters
        ----------
        extent: Extent
            Tile extent (x_min, y_min, x_max, y_max).
        crs: Any
            A projection definition pyproj.CRS can handle.
        x_pixel_size: int
            Pixel size in units of the projection in X.
        y_pixel_size: int
            Pixel size in units of the projection in Y.
        name: str | None, optional
            Name of the raster tile.
        **kwargs: Any
            Optional keyword arguments for child class.

        Returns
        -------
        RasterTile
            Raster tile instance.

        """
        ll_x, ll_y, ur_x, ur_y = extent
        width, height = ur_x - ll_x, ur_y - ll_y
        n_rows = int(round(height / y_pixel_size, DECIMALS))
        n_cols = int(round(width / x_pixel_size, DECIMALS))
        ul_x, ul_y = ll_x, ll_y + n_rows * y_pixel_size
        geotrans = (ul_x, x_pixel_size, 0, ul_y, 0, -y_pixel_size)

        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            crs=crs,
            geotrans=geotrans,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_geometry(
        cls,
        proj_geom: ProjGeom,
        x_pixel_size: float,
        y_pixel_size: float,
        name: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create a raster tile object from an existing geometry object.

        Since a raster tile can represent rectangles only, non-rectangular
        shapely objects get converted into their bounding boxes.

        Parameters
        ----------
        proj_geom : ProjGeom
            Projected geometry object from which the raster tile object
            should be created.
        x_pixel_size : float
            Pixel size in units of the projection in X.
        y_pixel_size : float
            Pixel size in units of the projection in Y.
        name: str | None, optional
            Name of the raster tile.
        **kwargs: Any
            Optional keyword arguments for child class.

        Returns
        -------
        RasterTile
            Raster tile instance.

        Notes
        -----
        The upper-left corner of the geometry/extent is assumed to be the
        (pixel) origin.

        """
        geom_ch = shapely.convex_hull(proj_geom.geom)
        return cls.from_extent(
            geom_ch.bounds,
            proj_geom.crs,
            x_pixel_size,
            y_pixel_size,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Create raster tile from JSON str.

        Parameters
        ----------
        json_str: str
            Raster tile represented by a JSON string.

        Returns
        -------
        RasterTile
            Raster tile.

        """
        return cls(**orjson.loads(json_str))

    def to_json(self) -> str:
        """Create JSON class representation."""
        return self.model_dump_json(indent=JSON_INDENT)

    @property
    def pyproj_crs(self) -> pyproj.CRS:
        """Return PyProj representation of CRS."""
        return self._crs

    @property
    def unit(self) -> str:
        """Return projection unit."""
        return self._crs.prime_meridian.unit_name

    @property
    def ori(self) -> float:
        """Counter-clockwise orientation of the raster tile.

        Counter-clockwise orientation of the raster tile in radians
        with respect to the W-E direction/horizontal.

        """
        return float(-np.arctan2(self.geotrans[2], self.geotrans[1]))

    @property
    def is_axis_parallel(self) -> bool:
        """True if the raster tile is not rotated , i.e. it is axis-parallel."""
        return bool(self.ori == 0.0)

    @property
    def ll_x(self) -> float:
        """X coordinate of the lower left corner."""
        x, _ = tuple(
            map(float, self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin))
        )
        return x

    @property
    def ll_y(self) -> float:
        """Y coordinate of the lower left corner."""
        _, y = tuple(
            map(float, self.rc2xy(self.n_rows - 1, 0, px_origin=self.px_origin))
        )
        return y

    @property
    def ul_x(self) -> float:
        """X coordinate of the upper left corner."""
        x, _ = tuple(map(float, self.rc2xy(0, 0, px_origin=self.px_origin)))
        return x

    @property
    def ul_y(self) -> float:
        """Y coordinate of the upper left corner."""
        _, y = tuple(map(float, self.rc2xy(0, 0, px_origin=self.px_origin)))
        return y

    @property
    def ur_x(self) -> float:
        """X coordinate of the upper right corner."""
        x, _ = tuple(
            map(float, self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin))
        )
        return x

    @property
    def ur_y(self) -> float:
        """Y coordinate of the upper right corner."""
        _, y = tuple(
            map(float, self.rc2xy(0, self.n_cols - 1, px_origin=self.px_origin))
        )
        return y

    @property
    def lr_x(self) -> float:
        """X coordinate of the upper right corner."""
        x, _ = tuple(
            map(
                float,
                self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin),
            )
        )
        return x

    @property
    def lr_y(self) -> float:
        """Y coordinate of the upper right corner."""
        _, y = tuple(
            map(
                float,
                self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin=self.px_origin),
            )
        )
        return y

    @property
    def x_pixel_size(self) -> float:
        """Pixel size in X direction."""
        return np.hypot(self.geotrans[1], self.geotrans[2])

    @property
    def y_pixel_size(self) -> float:
        """Pixel size in Y direction."""
        return np.hypot(self.geotrans[4], self.geotrans[5])

    @property
    def h_pixel_size(self) -> float:
        """Pixel size in W-E direction.

        Pixel size in W-E direction (equal to `x_pixel_size`
        if the raster tile is axis-parallel).

        """
        return self.x_pixel_size / np.cos(self.ori)

    @property
    def v_pixel_size(self) -> float:
        """Pixel size in N-S direction.

        Pixel size in N-S direction (equal to `y_pixel_size`
        if the raster tile is axis-parallel).

        """
        return self.y_pixel_size / np.cos(self.ori)

    @property
    def x_size(self) -> float:
        """Width of the raster tile in world system coordinates."""
        return self.n_cols * self.x_pixel_size

    @property
    def y_size(self) -> float:
        """Height of the raster tile in world system coordinates."""
        return self.n_rows * self.y_pixel_size

    @property
    def width(self) -> int:
        """Width of the raster tile in pixels."""
        return self.n_cols

    @property
    def height(self) -> int:
        """Height of the raster tile in pixels."""
        return self.n_rows

    @property
    def shape(self) -> tuple[int, int]:
        """Return shape of the raster tile.

        Return the shape of the raster tile, which is defined by
        the height and width in pixels.

        """
        return self.height, self.width

    @property
    def coord_extent(self) -> tuple[float, float, float, float]:
        """Return coordinate extent.

        Extent of the raster tile with the pixel origins defined during initialisation
        (min_x, min_y, max_x, max_y).

        """
        return (
            min([self.ll_x, self.ul_x]),
            min([self.ll_y, self.lr_y]),
            max([self.ur_x, self.lr_x]),
            max([self.ur_y, self.ul_y]),
        )

    @property
    def outer_boundary_extent(self) -> tuple[float, float, float, float]:
        """Return extent.

        Outer extent of the raster tile containing every pixel
        (min_x, min_y, max_x, max_y).

        """
        ll_x, ll_y = tuple(map(float, self.rc2xy(self.n_rows - 1, 0, px_origin="ll")))
        ur_x, ur_y = tuple(map(float, self.rc2xy(0, self.n_cols - 1, px_origin="ur")))
        lr_x, lr_y = tuple(
            map(float, self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr"))
        )
        ul_x, ul_y = tuple(map(float, self.rc2xy(0, 0, px_origin="ul")))
        return (
            min([ll_x, ul_x]),
            min([ll_y, lr_y]),
            max([ur_x, lr_x]),
            max([ur_y, ul_y]),
        )

    @property
    def size(self) -> int:
        """Return number of pixels covered by the raster tile."""
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        """Centre defined by the mass centre of the vertices."""
        cenroid = shapely.centroid(self._boundary.geom)
        return cenroid.x, cenroid.y

    @property
    def outer_boundary_corners(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Return corner coordinate tuples of the extent.

        A tuple containing all corners (convex hull, pixel extent) in a clock-wise order
        (lower left, lower right, upper right, upper left).

        """
        ll_x, ll_y = tuple(map(float, self.rc2xy(self.n_rows - 1, 0, px_origin="ll")))
        ur_x, ur_y = tuple(map(float, self.rc2xy(0, self.n_cols - 1, px_origin="ur")))
        lr_x, lr_y = tuple(
            map(float, self.rc2xy(self.n_rows - 1, self.n_cols - 1, px_origin="lr"))
        )
        ul_x, ul_y = tuple(map(float, self.rc2xy(0, 0, px_origin="ul")))
        return ((ll_x, ll_y), (ul_x, ul_y), (ur_x, ur_y), (lr_x, lr_y))

    @property
    def coord_corners(
        self,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Return corner coordinate tuples.

        A tuple containing all corners (convex hull, coordinate extent)
        in a clock-wise order (lower left, lower right, upper right, upper left).

        """
        return (
            (self.ll_x, self.ll_y),
            (self.ul_x, self.ul_y),
            (self.ur_x, self.ur_y),
            (self.lr_x, self.lr_y),
        )

    @property
    def x_coords(self) -> npt.NDArray[Any]:
        """Return all coordinates in X direction."""
        if self.is_axis_parallel:
            min_x, _ = self.rc2xy(0, 0)
            max_x, _ = self.rc2xy(0, self.n_cols)
            return np.arange(min_x, max_x, self.x_pixel_size)
        cols = np.array(range(self.n_cols))
        return np.array(self.rc2xy(0, cols)[0])

    @property
    def y_coords(self) -> npt.NDArray[Any]:
        """Return all coordinates in Y direction."""
        if self.is_axis_parallel:
            _, min_y = self.rc2xy(self.n_rows, 0)
            _, max_y = self.rc2xy(0, 0)
            return np.arange(max_y, min_y, -self.y_pixel_size)
        rows = np.array(range(self.n_rows))
        return np.array(self.rc2xy(rows, 0)[1])

    @property
    def xy_coords(
        self,
    ) -> tuple[npt.NDArray[Any] | int | float, npt.NDArray[Any] | int | float]:
        """Return meshgrid of both X and Y coordinates."""
        if self.is_axis_parallel:
            x_coords, y_coords = np.meshgrid(
                self.x_coords, self.y_coords, indexing="ij"
            )
        else:
            rows, cols = np.meshgrid(
                np.arange(self.n_rows), np.arange(self.n_cols), indexing="ij"
            )
            x_coords, y_coords = self.rc2xy(rows, cols)

        return x_coords, y_coords

    @property
    def boundary(self) -> ProjGeom:
        """Return a projected geometry representation of the raster tile."""
        return self._boundary

    @property
    def boundary_wkt(self) -> str:
        """Return Well Known Text (WKT) of the boundary of the raster tile."""
        return self._boundary.geom.wkt

    @property
    def boundary_shapely(self) -> shapely.geometry.Polygon:
        """Boundary of the raster tile represented as a Shapely polygon."""
        return self._boundary.geom

    @_align_geom()
    def intersects(self, other: ProjGeom | RasterTile) -> bool:
        """Evaluate if the raster tile instance and another geometry intersect.

        Parameters
        ----------
        other : ProjGeom | RasterTile
            Other geometry to evaluate an intersection with.

        Returns
        -------
        bool
            True if both geometries intersect, false if not.

        """
        return bool(shapely.intersects(self._boundary.geom, other.geom))

    @_align_geom()
    def touches(self, other: ProjGeom | RasterTile) -> bool:
        """Evaluate if the raster tile instance and another geometry touch each other.

        Parameters
        ----------
        other : ProjGeom | RasterTile
            Other geometry to evaluate a touch operation with.

        Returns
        -------
        bool
            True if both geometries touch each other, false if not.

        """
        return bool(
            shapely.touches(
                round_polygon_vertices(self._boundary.geom, DECIMALS),
                round_polygon_vertices(other.geom, DECIMALS),
            )
        )

    @_align_geom()
    def within(self, other: ProjGeom | RasterTile) -> bool:
        """Evaluate if the raster tile is fully within another geometry.

        Parameters
        ----------
        other : ProjGeom | RasterTile
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the raster tile is within the given geometry, false if not.

        """
        return bool(shapely.within(self._boundary.geom, other.geom))

    @_align_geom()
    def overlaps(self, other: ProjGeom | RasterTile) -> bool:
        """Evaluate if a geometry overlaps with the raster tile.

        Parameters
        ----------
        other : ProjGeom | RasterTile
            Other geometry to evaluate an overlaps operation with.

        Returns
        -------
        bool
            True if the given geometry overlaps the raster tile, false if not.

        """
        return bool(shapely.overlaps(self._boundary.geom, other.geom))

    @overload
    def xy2rc(
        self,
        x: float,
        y: float,
        crs: Any | None,  # noqa: ANN401
        px_origin: OriginStr | None,
    ) -> tuple[int, int]: ...

    @overload
    def xy2rc(
        self,
        x: npt.NDArray[Any],
        y: float,
        crs: Any | None,  # noqa: ANN401
        px_origin: OriginStr | None,
    ) -> tuple[npt.NDArray[Any], int]: ...

    @overload
    def xy2rc(
        self,
        x: float,
        y: npt.NDArray[Any],
        crs: Any | None,  # noqa: ANN401
        px_origin: OriginStr | None,
    ) -> tuple[int, npt.NDArray[Any]]: ...

    @overload
    def xy2rc(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        crs: Any | None,  # noqa: ANN401
        px_origin: OriginStr | None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

    def xy2rc(
        self,
        x: float | npt.NDArray[Any],
        y: float | npt.NDArray[Any],
        crs: Any = None,
        px_origin: OriginStr | None = None,
    ) -> tuple[int | npt.NDArray[Any], int | npt.NDArray[Any]]:
        """Convert world system to pixels coordinates.

        Calculate an index of a pixel in which a given point of a world system lies.

        Parameters
        ----------
        x : float | np.ndarray
            World system coordinate in X direction.
        y : float | np.ndarray
            World system coordinate in Y direction.
        crs: Any
            CRS of `x` and `y`. A projection definition pyproj.CRS can handle.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")
            Defaults to None, using the class internal pixel origin.

        Returns
        -------
        int | np.ndarray
            Pixel row number.
        int | np.ndarray
            Pixel column number.

        Notes
        -----
        Rounds to the closest, lower integer.

        """
        if crs is not None:
            x, y = transform_coords(x, y, crs, self.crs)
        px_origin = self.px_origin if px_origin is None else px_origin
        c, r = xy2ij(x, y, self.geotrans, origin=px_origin)
        return r, c

    @overload
    def rc2xy(
        self,
        r: int,
        c: int,
        px_origin: OriginStr | None = None,
    ) -> tuple[float, float]: ...

    @overload
    def rc2xy(
        self,
        r: npt.NDArray[Any],
        c: int,
        px_origin: OriginStr | None = None,
    ) -> tuple[npt.NDArray[Any], float]: ...

    @overload
    def rc2xy(
        self,
        r: int,
        c: npt.NDArray[Any],
        px_origin: OriginStr | None = None,
    ) -> tuple[float, npt.NDArray[Any]]: ...

    @overload
    def rc2xy(
        self,
        r: npt.NDArray[Any],
        c: npt.NDArray[Any],
        px_origin: OriginStr | None = None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

    def rc2xy(
        self,
        r: int | npt.NDArray[Any],
        c: int | npt.NDArray[Any],
        px_origin: OriginStr | None = None,
    ) -> tuple[float | npt.NDArray[Any], float | npt.NDArray[Any]]:
        """Convert pixels to world system coordinates.

        Returns the coordinates of the center or a corner (depending on
        'px_origin') of a pixel specified by a row and column number.

        Parameters
        ----------
        r : int | np.ndarray
            Pixel row number.
        c : int | np.ndarray
            Pixel column number.
        px_origin : str, optional
            Defines the world system origin of the pixel. It can be:
            - upper left ("ul")
            - upper right ("ur")
            - lower right ("lr")
            - lower left ("ll")
            - center ("c")
            Defaults to None, using the class internal pixel origin.

        Returns
        -------
        float | np.ndarray
            World system coordinate in X direction.
        float | np.ndarray
            World system coordinate in Y direction.

        """
        px_origin = self.px_origin if px_origin is None else px_origin
        return ij2xy(c, r, self.geotrans, origin=px_origin)

    def plot(  # noqa: PLR0913
        self,
        *,
        ax: GeoAxes | None = None,
        facecolor: str = "tab:red",
        edgecolor: str = "black",
        edgewidth: float = 1,
        alpha: float = 1.0,
        proj: Any = None,  # noqa: ANN401
        show: bool = False,
        label_tile: bool = False,
        add_country_borders: bool = True,
        extent: Extent | None = None,
        extent_proj: Any = None,  # noqa: ANN401
    ) -> GeoAxes | None:
        """Plot the boundary of the raster tile on a map.

        Parameters
        ----------
        ax : GeoAxes
            Pre-defined Matplotlib axis.
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
            reference system of the raster tile is taken.
        show : bool, optional
            If True, the plot result is shown (default is False).
        label_tile : bool, optional
            If True, the geometry name is plotted at the center of the raster geometry.
            Defaults to fale.
        add_country_borders : bool, optional
            If True, country borders are added to the plot (`cartopy.feature.BORDERS`).
            Defaults to false.
        extent : tuple or list, optional
            Coordinate/map extent of the plot, given as [min_x, min_y, max_x, max_y]
            (default is None, meaning global extent).
        extent_proj : Any, optional
            CRS of the given extent. A projection definition pyproj.CRS can handle.
            If it is None, then it is assumed that 'extent' is referring to the
            native projection of the raster tile.

        Returns
        -------
        GeoAxes
            Matplotlib axis containing a Cartopy map with the plotted raster tile
            boundary.

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

        patch = PolygonPatch(
            list(self.boundary_shapely.exterior.coords),
            facecolor=facecolor,
            alpha=alpha,
            zorder=0,
            edgecolor=edgecolor,
            linewidth=edgewidth,
            transform=this_proj,
        )
        ax.add_patch(patch)

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

        if self.name is not None and label_tile:
            transform = this_proj._as_mpl_transform(ax)  # noqa: SLF001
            ax.annotate(
                str(self.name),
                xy=self.centre,
                xycoords=transform,
                va="center",
                ha="center",
            )

        if show:
            plt.show()

        return ax

    def __proj_geom_boundary(self) -> ProjGeom:
        """Outer boundary of the raster tile as a projected geometry."""
        boundary = Polygon(self.outer_boundary_corners)
        return ProjGeom(geom=boundary, crs=self.crs)

    @_align_geom()
    def __contains__(self, other: ProjGeom | RasterTile) -> bool:
        """Evaluate if the given geometry is fully within the raster tile.

        Parameters
        ----------
        other : ProjGeom | RasterTile
            Other geometry to evaluate a within operation with.

        Returns
        -------
        bool
            True if the given geometry is within the raster tile, false if not.

        """
        return bool(shapely.within(other.geom, self._boundary.geom))

    def __hash__(self) -> int:
        """Return class hash."""
        this_corners = np.around(
            np.array(self.outer_boundary_corners), decimals=DECIMALS
        )
        return hash((this_corners, self.n_rows, self.n_cols))

    def __eq__(self, other: RasterTile) -> bool:  # ty: ignore[invalid-method-override]
        """Check if this and another raster tile are equal.

        Equality holds true if the vertices, rows and columns are the same.

        Parameters
        ----------
        other : RasterTile
            Raster tile to compare with.

        Returns
        -------
        bool
            True if both raster tiles are the same, otherwise false.

        """
        this_corners = np.around(
            np.array(self.outer_boundary_corners), decimals=DECIMALS
        )
        other_corners = np.around(
            np.array(other.outer_boundary_corners), decimals=DECIMALS
        )
        return bool(
            np.all(this_corners == other_corners)
            and self.n_rows == other.n_rows
            and self.n_cols == other.n_cols
        )

    def __ne__(self, other: object) -> bool:
        """Check if this and another raster tile are not equal.

        Non-equality holds true if the vertices, rows or columns differ.

        Parameters
        ----------
        other : RasterTile
            Raster tile object to compare with.

        Returns
        -------
        bool
            True if both raster tiles are the not the same, otherwise false.

        """
        return not self == other

    def __str__(self) -> str:
        """Representation of a raster tile as a Well Known Text (WKT) string."""
        return self.boundary_wkt


if __name__ == "__main__":
    pass
