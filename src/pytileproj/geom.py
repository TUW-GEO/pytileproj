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

"""Utilities for handling OGR and shapely geometries."""

import json
import warnings
from pathlib import Path

import numpy as np
import shapely
import shapely.wkt as swkt
from antimeridian import fix_multi_polygon, fix_polygon
from osgeo import ogr, osr
from PIL import Image, ImageDraw

from pytileproj._const import DECIMALS, DEFAULT_SEG_NUM
from pytileproj.proj import get_geog_sref

__all__ = [
    "convert_any_to_geog_ogr_geom",
    "get_geog_intersection",
    "ij2xy",
    "rasterise_polygon",
    "round_polygon_vertices",
    "segmentize_geometry",
    "shapely_to_ogr_polygon",
    "split_polygon_by_antimeridian",
    "transform_geom_to_geog",
    "transform_geometry",
    "xy2ij",
]


def xy2ij(
    x: float, y: float, geotrans: tuple, origin: str = "ul"
) -> tuple[int | np.ndarray, int | np.ndarray]:
    """Transform global/world system coordinates to pixel coordinates/indexes.

    Parameters
    ----------
    x : float or np.array
        World system coordinate(s) in X direction.
    y : float or np.array
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
    i : int or np.ndarray
        Column number(s) in pixels.
    j : int or np.ndarray
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


def ij2xy(
    i: int, j: int, geotrans: tuple, origin: str = "ul"
) -> tuple[float | np.ndarray, float | np.ndarray]:
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
    i += px_shift[0]
    j += px_shift[1]

    # applying affine model: https://gdal.org/user/raster_data_model.html
    x = geotrans[0] + i * geotrans[1] + j * geotrans[2]
    y = geotrans[3] + i * geotrans[4] + j * geotrans[5]

    return x, y


def round_polygon_vertices(poly: ogr.Geometry, decimals: int) -> ogr.Geometry:
    """Clean coordinates of an OGR polygon, so that it has rounded coordinates.

    Parameters
    ----------
    poly : ogr.Geometry
        An OGR polygon object.
    decimals : int
        Number of significant digits to round to.

    Returns
    -------
    ogr.Geometry
        An OGR polygon with rounded coordinates.

    """
    ring = poly.GetGeometryRef(0)

    rounded_ring = ogr.Geometry(ogr.wkbLinearRing)

    n_points = ring.GetPointCount()

    for p in range(n_points):
        x, y, z = ring.GetPoint(p)
        rx, ry, rz = [
            np.round(x, decimals=decimals),
            np.round(y, decimals=decimals),
            np.round(z, decimals=decimals),
        ]
        rounded_ring.AddPoint(rx, ry, rz)

    poly_out = ogr.Geometry(ogr.wkbPolygon)
    poly_out.AddGeometry(rounded_ring)

    return poly_out


def shapely_to_ogr_polygon(poly: shapely.Polygon, epsg: int) -> ogr.Geometry:
    """Convert a shapely to an OGR polygon and assigns the given projection.

    Parameters
    ----------
    poly: shapely.Polygon
        Shapely polygon object.
    epsg: int
        EPSG code for the polygon.

    Returns
    -------
    ogr.Geometry
        OGR polygon object.

    """
    # doing a double WKT conversion to prevent precision
    # issues nearby machine epsilon
    poly_ogr = ogr.CreateGeometryFromWkt(
        ogr.CreateGeometryFromWkt(poly.wkt).ExportToWkt()
    )
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)
    poly_ogr.AssignSpatialReference(sref)

    return poly_ogr


def rasterise_polygon(
    poly: shapely.Polygon,
    x_pixel_size: float,
    y_pixel_size: float,
    extent: tuple | None = None,
) -> np.ndarray:
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


def segmentize_geometry(geom: ogr.Geometry, segment: float = 0.5) -> ogr.Geometry:
    """Segmentizes the lines of a geometry.

    Parameters
    ----------
    geom : ogr.Geometry
        OGR geometry object.
    segment : float, optional
        For precision: distance in units of the geometry projection
        defining longest segment of the geometry.

    Returns
    -------
    ogr.Geometry
        A congruent geometry realised by more vertices along its shape.

    """
    geom_seg = geom.Clone()
    geom_seg.Segmentize(segment)

    return geom_seg


def get_geog_intersection(poly_1: ogr.Geometry, poly_2: ogr.Geometry) -> bool:
    """Intersect in the LonLat space.

    'poly_1' is split at the antimeridian (i.e. the 180 degree dateline).

    Parameters
    ----------
    poly_1 : ogr.Geometry
        OGR polygon object in LonLat space. It is split by the antimeridian.
    poly_2 : ogr.Geometry
        Other OGR polygon object to intersect with.

    Returns
    -------
    boolean
        True if both polygons intersect, false otherwise.

    """
    poly_1c = poly_1.Clone()
    if poly_1.GetGeometryName() == "MULTIPOLYGON":
        poly_1c = ogr.ForceToPolygon(poly_1c)
        warnings.warn("Take care: multi-polygon is forced to a polygon!", stacklevel=1)

    polygons = split_polygon_by_antimeridian(poly_1c)

    return polygons.Intersection(poly_2.Clone())


def split_polygon_by_antimeridian(
    geog_poly: ogr.Geometry, *, great_circle: bool = False
) -> ogr.Geometry:
    """Split a polygon or multi-polygon at the antimeridian.

    Parameters
    ----------
    geog_poly : ogr.Geometry
        OGR polygon or multi-polygon object in LonLat space to be
        split by the antimeridian.
    great_circle: bool, optional
        True, if a great circle on the sphere should be used to split
        segments crossing the antimeridian. Defaults to false.

    Returns
    -------
    ogr.Geometry
        Multi-polygon comprising east and west parts of `geog_poly`.
        It contains only one polygon if no intersect with the antimeridian is given.

    """
    geom_type = geog_poly.GetGeometryName()
    if geom_type == "MULTIPOLYGON":
        geog_poly_am = fix_multi_polygon(
            swkt.loads(geog_poly.ExportToWkt()), great_circle=great_circle
        )
    elif geom_type == "POLYGON":
        geog_poly_am = fix_polygon(
            swkt.loads(geog_poly.ExportToWkt()), great_circle=great_circle
        )
    else:
        err_msg = f"Geometry type {geom_type} not supported."
        raise ValueError(err_msg)

    geog_poly_am = ogr.CreateGeometryFromWkt(geog_poly_am.wkt)
    geog_poly_am.AssignSpatialReference(geog_poly.GetSpatialReference())

    return geog_poly_am


def transform_geometry(
    geom: ogr.Geometry, sref: osr.SpatialReference, segment: float | None = None
) -> ogr.Geometry:
    """Transform an OGR geometry to the given target spatial reference system.

    Parameters
    ----------
    geom : ogr.Geometry
        OGR geometry object to transform.
    sref : osr.SpatialReference
        OSR spatial reference to what the geometry should be transformed to.
    segment : float, optional
        For precision: distance in units of the geometry projection
        defining longest segment of the geometry.

    Returns
    -------
    ogr.Geometry
        An OGR geometry represented in the target spatial reference.

    """
    trans_geom = geom.Clone()

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        trans_geom = segmentize_geometry(trans_geom, segment=segment)

    trans_geom.TransformTo(sref)

    if sref.ExportToProj4().startswith(
        "+proj=longlat"
    ) and trans_geom.GetGeometryName() in ["POLYGON", "MULTIPOLYGON"]:
        trans_geom = split_polygon_by_antimeridian(trans_geom)

    return trans_geom


def transform_geom_to_geog(geom: ogr.Geometry) -> ogr.Geometry:
    """Transform geometry to the LonLat system.

    Parameters
    ----------
    geom: ogr.Geometry
        OGR geometry object to transform.

    Returns
    -------
    ogr.Geometry
        OGR geometry object in the LonLat system.

    """
    return transform_geometry(geom, get_geog_sref(), segment=DEFAULT_SEG_NUM)


def convert_any_to_geog_ogr_geom(
    arg: Path | shapely.Geometry | ogr.Geometry | None,
) -> ogr.Geometry | None:
    """Convert an arbitrary input to an OGR geometry in the LonLat system.

    Parameters
    ----------
    arg: Path | shapely.Polygon | ogr.Geometry | None
        Input representing a geometry object. It can be one of:
            - a path to a GeoJSON file
            - a shapely.Geometry
            - an ogr.Geometry
            - None

    Returns
    -------
    ogr.Geometry | None
        Input converted to an OGR polygon or multi-polygon.
        If the input is None, then None will be returned.

    """
    if isinstance(arg, Path):
        with arg.open() as f:
            geojson = json.load(f)
        proj_zone_geog = ogr.CreateGeometryFromJson(geojson)
        proj_zone_geog.AssignSpatialReference(get_geog_sref())
    elif isinstance(arg, shapely.Polygon):
        proj_zone_geog = ogr.CreateGeometryFromWkt(arg.wkt)
        proj_zone_geog.AssignSpatialReference(get_geog_sref())
    elif isinstance(arg, ogr.Geometry):
        proj_zone_geog = arg
    else:
        proj_zone_geog = None

    return proj_zone_geog
