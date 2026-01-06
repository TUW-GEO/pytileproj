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

"""Utility module for projected geometries."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, cast, overload

import numpy as np
import numpy.typing as npt
import orjson
import pyproj
import requests
import shapely
import shapely.wkt as swkt
from antimeridian import fix_multi_polygon, fix_polygon
from PIL import Image, ImageDraw
from pydantic import AfterValidator, BaseModel, model_serializer
from shapely.geometry import MultiPolygon, Polygon

from pytileproj._const import (
    DECIMALS,
    DEF_SEG_LEN_M,
    GEO_INSTALLED,
    GEOG_CRS,
    GEOG_EPSG,
    TIMEOUT,
    VIS_INSTALLED,
)

if VIS_INSTALLED:
    import cartopy.crs as ccrs

if GEO_INSTALLED:
    import geopandas as gpd

__all__ = [
    "ProjCoord",
    "ProjGeom",
    "ij2xy",
    "pyproj_to_cartopy_crs",
    "rasterise_polygon",
    "round_polygon_vertices",
    "split_polygon_by_antimeridian",
    "transform_coords",
    "transform_geom_to_geog",
    "transform_geometry",
    "xy2ij",
]

GeoTransformTuple: TypeAlias = tuple[float, float, float, float, float, float]
OriginStr: TypeAlias = Literal["ul", "ur", "ll", "lr", "c"]


@dataclass(frozen=True)
class ProjCoord:
    """Define a coordinate in a certain projection."""

    x: float
    y: float
    crs: pyproj.CRS


@dataclass(frozen=True)
class GeogCoord(ProjCoord):
    crs: pyproj.CRS = GEOG_CRS


def convert_geom(arg: str | shapely.Geometry) -> shapely.Geometry:
    return swkt.loads(arg) if isinstance(arg, str) else arg


def convert_crs(arg: Any) -> pyproj.CRS:  # noqa: ANN401
    return pyproj.CRS.from_user_input(arg)


class ProjGeom(BaseModel, arbitrary_types_allowed=True):
    """Define a geometry in a certain projection."""

    geom: Annotated[Any, AfterValidator(convert_geom)]
    crs: Annotated[Any, AfterValidator(convert_crs)]

    @model_serializer
    def serialize(self) -> dict:
        """Serialise/encode class variables."""
        return {"geom": self.geom.wkt, "crs": self.crs.to_proj4()}


class GeogGeom(ProjGeom):
    crs: pyproj.CRS = pyproj.CRS.from_epsg(GEOG_EPSG)


def fetch_proj_zone(epsg: int) -> ProjGeom:
    """Fetch the zone polygon of the given projection from the EPSG database.

    Parameters
    ----------
    epsg: int
        EPSG code representing the projection.

    Returns
    -------
    ProjGeom
        Projected polygon or multi-polygon object representing the projection zone.

    Notes
    -----
    This function requires a internet connection.

    """
    epsg_code_url = "https://apps.epsg.org/api/v1/ProjectedCoordRefSystem/"
    epsg_extent_url = "https://apps.epsg.org/api/v1/Extent/"

    zone_geom = None
    code_resp = requests.get(f"{epsg_code_url}/{epsg}/", timeout=TIMEOUT)
    if code_resp.ok:
        code_data = orjson.loads(code_resp.content)
        code_usages = code_data["Usage"]
        if len(code_usages):
            if len(code_usages) != 1:
                warnings.warn("Multiple EPSG code usages found!", stacklevel=1)
            code_usage = code_usages[-1]
            extent_resp = requests.get(
                f"{epsg_extent_url}/{code_usage['Extent']['Code']}/polygon",
                timeout=TIMEOUT,
            )
            if extent_resp.ok:
                extent_data = orjson.loads(extent_resp.content)
                geom_type = extent_data["type"]
                coords = extent_data["coordinates"]
                if geom_type == "Polygon":
                    zone_geom = Polygon(coords[0])
                elif geom_type == "MultiPolygon":
                    zone_geom = MultiPolygon(coords)
                else:
                    err_msg = f"Geometry type '{geom_type}' not supported."
                    raise ValueError(err_msg)
                zone_geom = GeogGeom(geom=zone_geom)

    if zone_geom is None:
        err_msg = f"No zone boundary found for EPSG {epsg}"
        raise ValueError(err_msg)

    return zone_geom


def pyproj_to_cartopy_crs(crs: pyproj.CRS) -> "ccrs.CRS":
    """Convert a pyproj to a cartopy CRS object.

    Parameters
    ----------
    crs: pyproj.CRS
        Pyproj CRS object.

    Returns
    -------
    ccrs.CRS
        Cartopy CRS object.

    """
    proj4_params = crs.to_dict()
    proj4_name = proj4_params.get("proj")
    central_longitude = proj4_params.get("lon_0", 0.0)
    central_latitude = proj4_params.get("lat_0", 0.0)
    false_easting = proj4_params.get("x_0", 0.0)
    false_northing = proj4_params.get("y_0", 0.0)
    scale_factor = proj4_params.get("k", 1.0)
    standard_parallels = (
        proj4_params.get("lat_1", 20.0),
        proj4_params.get("lat_2", 50.0),
    )

    ccrs_lut = {
        "longlat": ccrs.PlateCarree(central_longitude),
        "aeqd": ccrs.AzimuthalEquidistant(
            central_longitude, central_latitude, false_easting, false_northing
        ),
        "merc": ccrs.Mercator(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor,
        ),
        "eck1": ccrs.EckertI(central_longitude, false_easting, false_northing),
        "eck2": ccrs.EckertII(central_longitude, false_easting, false_northing),
        "eck3": ccrs.EckertIII(central_longitude, false_easting, false_northing),
        "eck4": ccrs.EckertIV(central_longitude, false_easting, false_northing),
        "eck5": ccrs.EckertV(central_longitude, false_easting, false_northing),
        "eck6": ccrs.EckertVI(central_longitude, false_easting, false_northing),
        "aea": ccrs.AlbersEqualArea(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels,
        ),
        "eqdc": ccrs.EquidistantConic(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels,
        ),
        "gnom": ccrs.Gnomonic(central_longitude, central_latitude),
        "laea": ccrs.LambertAzimuthalEqualArea(
            central_longitude, central_latitude, false_easting, false_northing
        ),
        "lcc": ccrs.LambertConformal(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels=standard_parallels,
        ),
        "mill": ccrs.Miller(central_longitude),
        "moll": ccrs.Mollweide(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
        ),
        "stere": ccrs.Stereographic(
            central_latitude,
            central_longitude,
            false_easting,
            false_northing,
            scale_factor=scale_factor,
        ),
        "ortho": ccrs.Orthographic(central_longitude, central_latitude),
        "robin": ccrs.Robinson(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
        ),
        "sinus": ccrs.Sinusoidal(central_longitude, false_easting, false_northing),
        "tmerc": ccrs.TransverseMercator(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            scale_factor,
        ),
    }

    ccrs_proj = ccrs_lut.get(proj4_name)

    if ccrs_proj is None:
        err_msg = f"Projection '{proj4_name}' is not supported."
        raise ValueError(err_msg)

    return ccrs_proj


def transform_coords(
    x: float | npt.NDArray[Any],
    y: float | npt.NDArray[Any],
    this_crs: Any,  # noqa: ANN401
    other_crs: Any,  # noqa: ANN401
) -> tuple[float | npt.NDArray[Any], float | npt.NDArray[Any]]:
    """Transform coordinate tuple from a given to another projection.

    Parameters
    ----------
    x: float | np.ndarray
        X coordinate.
    y: float | np.ndarray
        Y coordinate.
    this_crs: Any
        CRS of the input coordinates. A projection definition
        pyproj.CRS can handle.
    other_crs: Any
        CRS of the target projection. A projection definition
        pyproj.CRS can handle.

    Returns
    -------
    float | np.ndarray
        X coordinate in the target projection.
    float | np.ndarray
        Y coordinate in the target projection.

    """
    traffo = pyproj.Transformer.from_crs(this_crs, other_crs, always_xy=True)
    return traffo.transform(x, y)


@overload
def xy2ij(
    x: float,
    y: float,
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[int, int]: ...


@overload
def xy2ij(
    x: npt.NDArray[Any],
    y: float,
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[npt.NDArray[Any], int]: ...


@overload
def xy2ij(
    x: float,
    y: npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[int, npt.NDArray[Any]]: ...


@overload
def xy2ij(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...


def xy2ij(
    x: float | npt.NDArray[Any],
    y: float | npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr = "ul",
) -> tuple[int | npt.NDArray[Any], int | npt.NDArray[Any]]:
    """Transform global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    x : float | np.ndarray
        World system coordinate(s) in X direction.
    y : float | np.ndarray
        World system coordinate(s) in Y direction.
    geotrans : 6-tuple
        GDAL geo-transformation parameters/dictionary.
    origin : str, optional
        Defines the world system origin of the pixel. It can be:
        - upper left ("ul", default)
        - upper right ("ur")
        - lower right ("lr")
        - lower left ("ll")
        - center ("c")

    Returns
    -------
    i : int | np.ndarray
        Column number(s) in pixels.
    j : int | np.ndarray
        Row number(s) in pixels.

    """
    px_shift_map = {
        "ul": (0, 0),
        "ur": (1, 0),
        "lr": (1, 1),
        "ll": (0, 1),
        "c": (0.5, 0.5),
    }

    px_shift = px_shift_map.get(origin)
    if px_shift is None:
        wrng_msg = (
            "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead"
        )
        wrng_msg = wrng_msg.format(origin)
        warnings.warn(wrng_msg, stacklevel=1)
        px_shift = (0, 0)

    # shift world system coordinates to the desired pixel origin
    x -= px_shift[0] * geotrans[1]
    y -= px_shift[1] * geotrans[5]

    # solved equation system describing an affine model: https://gdal.org/user/raster_data_model.html
    i = np.around(
        (
            -1.0
            * (
                geotrans[2] * geotrans[3]
                - geotrans[0] * geotrans[5]
                + geotrans[5] * x
                - geotrans[2] * y
            )
            / (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])
        ),
        decimals=DECIMALS,
    ).astype(int)
    j = np.around(
        (
            -1.0
            * (
                -1 * geotrans[1] * geotrans[3]
                + geotrans[0] * geotrans[4]
                - geotrans[4] * x
                + geotrans[1] * y
            )
            / (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])
        ),
        decimals=DECIMALS,
    ).astype(int)

    return i, j


@overload
def ij2xy(
    i: int,
    j: int,
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[int, int]: ...


@overload
def ij2xy(
    i: npt.NDArray[Any],
    j: int,
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[npt.NDArray[Any], int]: ...


@overload
def ij2xy(
    i: int,
    j: npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[int, npt.NDArray[Any]]: ...


@overload
def ij2xy(
    i: npt.NDArray[Any],
    j: npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...


def ij2xy(
    i: int | npt.NDArray[Any],
    j: int | npt.NDArray[Any],
    geotrans: GeoTransformTuple,
    origin: OriginStr = "ul",
) -> tuple[float | npt.NDArray[Any], float | npt.NDArray[Any]]:
    """Transform global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    i : int or np.array
        Column number(s) in pixels.
    j : int or np.array
        Row number(s) in pixels.
    geotrans : 6-tuple
        GDAL geo-transformation parameters/dictionary.
    origin : str, optional
        Defines the world system origin of the pixel. It can be:
        - upper left ("ul", default)
        - upper right ("ur")
        - lower right ("lr")
        - lower left ("ll")
        - center ("c")

    Returns
    -------
    float or np.ndarray
        World system coordinate(s) in X direction.
    float or np.ndarray
        World system coordinate(s) in Y direction.

    """
    px_shift_map = {
        "ul": (0, 0),
        "ur": (1, 0),
        "lr": (1, 1),
        "ll": (0, 1),
        "c": (0.5, 0.5),
    }

    px_shift = px_shift_map.get(origin)
    if px_shift is None:
        wrng_msg = (
            "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead"
        )
        wrng_msg = wrng_msg.format(origin)
        warnings.warn(wrng_msg, stacklevel=1)
        px_shift = (0, 0)

    # shift pixel coordinates to the desired pixel origin
    i_ori = i + px_shift[0]
    j_ori = j + px_shift[1]

    # applying affine model: https://gdal.org/user/raster_data_model.html
    x = geotrans[0] + i_ori * geotrans[1] + j_ori * geotrans[2]
    y = geotrans[3] + i_ori * geotrans[4] + j_ori * geotrans[5]

    return x, y


def round_polygon_vertices(poly: shapely.Polygon, decimals: int) -> shapely.Polygon:
    """Clean coordinates of a polygon, so that it has rounded coordinates.

    Parameters
    ----------
    poly : shapely.Polygon
        A shapely polygon object.
    decimals : int
        Number of significant digits to round to.

    Returns
    -------
    shapely.Polygon
        A polygon with rounded coordinates.

    """
    xs, ys = zip(*poly.exterior.coords, strict=True)
    xs, ys = np.round(xs, decimals=decimals), np.round(ys, decimals=decimals)
    return shapely.Polygon(list(zip(xs, ys, strict=True)))


def rasterise_polygon(
    poly: shapely.Polygon,
    x_pixel_size: float,
    y_pixel_size: float,
    extent: tuple | None = None,
) -> npt.NDArray[Any]:
    """Rasterises a Shapely polygon defined by a clockwise list of points.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Clockwise list of x and y coordinates defining a polygon.
    x_pixel_size : float
        Absolute pixel size in X direction.
    y_pixel_size : float
        Absolute pixel size in Y direction.
    extent : 4-tuple, optional
        Output extent of the raster (x_min, y_min, x_max, y_max).
        If it is not set the output extent is taken from the
        given geometry.

    Returns
    -------
    raster : np.ndarray
        Binary array where zeros are background pixels and ones are
        foreground (polygon) pixels. Its shape is defined by the
        coordinate extent of the input polygon or by the specified
        'extent' parameter.

    Notes
    -----
    The coordinates are always expected to refer to the upper-left corner
    of a pixel, in a right-hand coordinate system. If the coordinates do
    not match the sampling, they are automatically aligned to upper-left.

    For rasterising the actual polygon, PIL's `ImageDraw` class is used.

    """
    # retrieve polygon points
    poly_pts = list(poly.exterior.coords)

    # split tuple points into x and y coordinates
    xs, ys = list(zip(*poly_pts, strict=False))

    # round coordinates to upper-left corner
    xs = (
        np.around(np.array(xs) / x_pixel_size, decimals=DECIMALS).astype(int)
        * x_pixel_size
    )
    ys = (
        np.ceil(np.around(np.array(ys) / y_pixel_size, decimals=DECIMALS)).astype(int)
        * y_pixel_size
    )  # use ceil to round to upper corner

    # define extent of the polygon
    if extent is None:
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    else:
        x_min = int(round(extent[0] / x_pixel_size, DECIMALS)) * x_pixel_size
        x_max = int(round(extent[2] / x_pixel_size, DECIMALS)) * x_pixel_size
        y_min = int(np.ceil(round(extent[1] / y_pixel_size, DECIMALS))) * y_pixel_size
        y_max = int(np.ceil(round(extent[3] / y_pixel_size, DECIMALS))) * y_pixel_size

    # number of columns and rows (+1 to include last pixel row and column,
    # which is lost when computing the difference)
    n_rows = int(round((y_max - y_min) / y_pixel_size, DECIMALS)) + 1
    n_cols = int(round((x_max - x_min) / x_pixel_size, DECIMALS)) + 1

    # raster with zeros
    mask_img = Image.new("1", (n_cols, n_rows), 0)
    rows = (np.around(np.abs(ys - y_max) / y_pixel_size, decimals=DECIMALS)).astype(int)
    cols = (np.around(np.abs(xs - x_min) / x_pixel_size, decimals=DECIMALS)).astype(int)
    ImageDraw.Draw(mask_img).polygon(
        list(zip(cols, rows, strict=False)), outline=1, fill=1
    )
    return np.array(mask_img).astype(np.uint8)


def split_polygon_by_antimeridian(
    geog_geom: GeogGeom, *, great_circle: bool = False
) -> GeogGeom:
    """Split a polygon or multi-polygon at the antimeridian.

    Parameters
    ----------
    geog_geom : GeogGeom
        Projected polygon or multi-polygon object in LonLat space to be
        split by the antimeridian.
    great_circle: bool, optional
        True, if a great circle on the sphere should be used to split
        segments crossing the antimeridian. Defaults to false.

    Returns
    -------
    ProjGeom
        Multi-polygon comprising east and west parts of `geog_poly`.
        It contains only one polygon if no intersect with the antimeridian is given.

    """
    if geog_geom.crs.to_epsg() != GEOG_EPSG:
        err_msg = "Geometry is not in the LonLat projection."
        raise ValueError(err_msg)

    geom_type = geog_geom.geom.geom_type
    if geom_type == "MultiPolygon":
        geog_poly_am = fix_multi_polygon(geog_geom.geom, great_circle=great_circle)
    elif geom_type == "Polygon":
        geog_poly_am = fix_polygon(geog_geom.geom, great_circle=great_circle)
    else:
        err_msg = f"Geometry type {geom_type} not supported."
        raise ValueError(err_msg)

    return GeogGeom(geom=geog_poly_am)


def transform_geometry(
    proj_geom: ProjGeom,
    crs: Any,  # noqa: ANN401
    segment: float | None = None,
) -> ProjGeom | GeogGeom:
    """Transform a geometry to the given target spatial reference system.

    Parameters
    ----------
    proj_geom : ProjGeom
        Projected geometry object to transform.
    crs : Any
        Target CRS of the geometry. A projection definition
        pyproj.CRS can handle.
    segment : float, optional
        For precision: distance in units of the geometry projection
        defining longest segment of the geometry.

    Returns
    -------
    ProjGeom
        Geometry represented in the target spatial reference.

    Notes
    -----
    When working with geographic projections, this function also
    takes the antimeridian into account.

    """
    src_geom = proj_geom.geom

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        src_geom = shapely.segmentize(src_geom, max_segment_length=segment)

    transformer = pyproj.Transformer.from_crs(proj_geom.crs, crs, always_xy=True)
    dst_geom = shapely.transform(src_geom, transformer.transform, interleaved=False)
    dst_crs = pyproj.CRS.from_user_input(crs)

    if dst_crs.to_epsg() == GEOG_EPSG:
        dst_geom = split_polygon_by_antimeridian(GeogGeom(geom=dst_geom))
    else:
        dst_geom = ProjGeom(geom=dst_geom, crs=dst_crs)

    return dst_geom


def transform_geom_to_geog(proj_geom: ProjGeom) -> GeogGeom:
    """Transform geometry to the LonLat system.

    Parameters
    ----------
    proj_geom: ProjGeom
        Projected geometry object to transform.

    Returns
    -------
    ProjGeom
        Geometry object in the LonLat system.

    """
    return cast(
        "GeogGeom", transform_geometry(proj_geom, GEOG_EPSG, segment=DEF_SEG_LEN_M)
    )


def convert_any_to_geog_geom(
    arg: Path | shapely.Geometry | ProjGeom | str | dict | None,
) -> ProjGeom | None:
    """Convert an arbitrary input to a projected geometry in the LonLat system.

    Parameters
    ----------
    arg: Path | shapely.Polygon | ProjGeom | None
        Input representing a geometry object. It can be one of:
            - a path to a GeoJSON file
            - a shapely.Geometry
            - a ProjGeom
            - str
            - None

    Returns
    -------
    ProjGeom | None
        Input converted to an projected polygon or multi-polygon.
        If the input is None, then None will be returned.

    """
    if isinstance(arg, Path):
        if arg.suffix == ".geojson":
            with arg.open() as f:
                geom = shapely.from_geojson(f.read())
        elif arg.suffix == ".parquet":
            if not GEO_INSTALLED:
                err_msg = "It is required to install the 'geo' extension."
                raise ImportError(err_msg)
            geom = gpd.read_parquet(arg)["geometry"][0]
        else:
            err_msg = (
                "File format is not supported "
                f"(only 'geojson' and 'parquet'): {arg.suffix}"
            )
            raise OSError(err_msg)
        proj_geom = GeogGeom(geom=geom)
    elif isinstance(arg, shapely.Geometry):
        proj_geom = GeogGeom(geom=arg)
    elif isinstance(arg, str):
        proj_geom = GeogGeom(geom=swkt.loads(arg))
    elif isinstance(arg, dict):
        proj_geom = ProjGeom(**arg)
    elif isinstance(arg, ProjGeom):
        proj_geom = arg
    else:
        proj_geom = None

    return proj_geom
