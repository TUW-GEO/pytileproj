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
Code for osgeo geometry operations.
"""

from copy import deepcopy
import warnings
import numpy as np
import shapely
from PIL import Image, ImageDraw

from osgeo import ogr
from osgeo import osr
from osgeo.gdal import __version__ as gdal_version
from typing import Tuple
import pyproj

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.ops import polygonize

from pytileproj._const import DEFAULT_TILE_SEG_NUM, DECIMALS

def xy2ij(x: float, y: float, geotrans: Tuple, origin: str = "ul") -> Tuple[int | np.ndarray, int | np.ndarray]:
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

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

    px_shift_map = {"ul": (0, 0),
                    "ur": (1, 0),
                    "lr": (1, 1),
                    "ll": (0, 1),
                    "c": (.5, .5)}

    px_shift = px_shift_map.get(origin, None)
    if px_shift is None:
        wrng_msg = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead.".format(origin)
        warnings.warn(wrng_msg)
        px_shift = (0, 0)

    # shift world system coordinates to the desired pixel origin
    x -= px_shift[0] * geotrans[1]
    y -= px_shift[1] * geotrans[5]

    # solved equation system describing an affine model: https://gdal.org/user/raster_data_model.html
    i = np.around((-1.0 * (geotrans[2] * geotrans[3] - geotrans[0] * geotrans[5] + geotrans[5] * x - geotrans[2] * y)/
                   (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])), decimals=DECIMALS).astype(int)
    j = np.around((-1.0 * (-1 * geotrans[1] * geotrans[3] + geotrans[0] * geotrans[4] - geotrans[4] * x + geotrans[1] * y)/
                   (geotrans[2] * geotrans[4] - geotrans[1] * geotrans[5])), decimals=DECIMALS).astype(int)

    return i, j


def ij2xy(i: int, j: int, geotrans: Tuple, origin: str = "ul") -> Tuple[float | np.ndarray, float | np.ndarray]:
    """
    Transforms global/world system coordinates to pixel coordinates/indexes.

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
    x : float or np.ndarray
        World system coordinate(s) in X direction.
    y : float or np.ndarray
        World system coordinate(s) in Y direction.

    """

    px_shift_map = {"ul": (0, 0),
                    "ur": (1, 0),
                    "lr": (1, 1),
                    "ll": (0, 1),
                    "c": (.5, .5)}

    px_shift = px_shift_map.get(origin, None)
    if px_shift is None:
        wrng_msg = "Pixel origin '{}' unknown. Upper left origin 'ul' will be taken instead".format(origin)
        warnings.warn(wrng_msg)
        px_shift = (0, 0)

    # shift pixel coordinates to the desired pixel origin
    i += px_shift[0]
    j += px_shift[1]

    # applying affine model: https://gdal.org/user/raster_data_model.html
    x = geotrans[0] + i * geotrans[1] + j * geotrans[2]
    y = geotrans[3] + i * geotrans[4] + j * geotrans[5]

    return x, y


def transform_coords(x: float, y: float, this_crs: pyproj.CRS, other_crs: pyproj.CRS) -> Tuple[float, float]:
    traffo = pyproj.Transformer.from_crs(this_crs, other_crs, always_xy=True)
    return traffo.transform(x, y)


def round_polygon_vertices(polygon: ogr.Geometry, decimals: int) -> ogr.Geometry:
    """
    'Cleans' the coordinates of an OGR polygon, so that it has rounded coordinates.

    Parameters
    ----------
    polygon : ogr.wkbPolygon
        An OGR polygon.
    decimals : int
        Number of significant digits to round to.

    Returns
    -------
    geometry_out : ogr.wkbPolygon
        An OGR polygon with rounded coordinates.

    """
    ring = polygon.GetGeometryRef(0)

    rounded_ring = ogr.Geometry(ogr.wkbLinearRing)

    n_points = ring.GetPointCount()

    for p in range(n_points):
        x, y, z = ring.GetPoint(p)
        rx, ry, rz = [np.round(x, decimals=decimals),
                      np.round(y, decimals=decimals),
                      np.round(z, decimals=decimals)]
        rounded_ring.AddPoint(rx, ry, rz)

    geometry_out = ogr.Geometry(ogr.wkbPolygon)
    geometry_out.AddGeometry(rounded_ring)

    return geometry_out


def transform_geometry(geometry: ogr.Geometry, sref: osr.SpatialReference, segment=None):
    """
    returns the reprojected geometry - in the specified spatial reference

    Parameters
    ----------
    geometry : OGRGeometry
        geometry object
    osr_spref : OGRSpatialReference
        spatial reference to what the geometry should be transformed to
    segment : float, optional
        for precision: distance in units of input osr_spref of longest
        segment of the geometry polygon

    Returns
    -------
    OGRGeometry
        a geometry represented in the target spatial reference

    """
    geometry_out = geometry.Clone()

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        geometry_out = segmentize_geometry(geometry_out, segment=segment)

    geometry_out.TransformTo(sref)

    if sref.ExportToProj4().startswith('+proj=longlat'):
        if geometry_out.GetGeometryName() in ['POLYGON', 'MULTIPOLYGON']:
            geometry_out = split_polygon_by_antimeridian(geometry_out)

    geometry = None
    return geometry_out


def get_lonlat_sref() -> osr.SpatialReference:
    sref = osr.SpatialReference()
    sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    sref.ImportFromEPSG(4326)

    return sref


def transform_geom_to_geog(geom: ogr.Geometry) -> ogr.Geometry:
    return transform_geometry(geom, get_lonlat_sref(), segment=DEFAULT_TILE_SEG_NUM)


def rasterise_polygon(geom: shapely.Polygon, x_pixel_size: int | float, y_pixel_size: int | float, 
                      extent: Tuple = None) -> np.ndarray:
    """
    Rasterises a Shapely polygon defined by a clockwise list of points.

    Parameters
    ----------
    geom : shapely.geometry.Polygon
        Clockwise list of x and y coordinates defining a polygon.
    x_pixel_size : float
        Absolute pixel size in X direction.
    y_pixel_size : float
        Absolute pixel size in Y direction.
    extent : 4-tuple, optional
        Output extent of the raster (x_min, y_min, x_max, y_max). If it is not set the output extent is taken from the
        given geometry.

    Returns
    -------
    raster : np.ndarray
        Binary array where zeros are background pixels and ones are foreground (polygon) pixels. Its shape is defined by
        the coordinate extent of the input polygon or by the specified `extent` parameter.

    Notes
    -----
    The coordinates are always expected to refer to the upper-left corner of a pixel, in a right-hand coordinate system.
    If the coordinates do not match the sampling, they are automatically aligned to upper-left.

    For rasterising the actual polygon, PIL's `ImageDraw` class is used.

    """

    # retrieve polygon points
    geom_pts = list(geom.exterior.coords)

    # split tuple points into x and y coordinates
    xs, ys = list(zip(*geom_pts))

    # round coordinates to upper-left corner
    xs = np.around(np.array(xs)/x_pixel_size, decimals=DECIMALS).astype(int) * x_pixel_size
    ys = np.ceil(np.around(np.array(ys)/y_pixel_size, decimals=DECIMALS)).astype(int) * y_pixel_size # use ceil to round to upper corner

    # define extent of the polygon
    if extent is None:
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    else:
        x_min = int(round(extent[0] / x_pixel_size, DECIMALS)) * x_pixel_size
        x_max = int(round(extent[2] / x_pixel_size, DECIMALS)) * x_pixel_size
        y_min = int(np.ceil(round(extent[1] / y_pixel_size, DECIMALS))) * y_pixel_size
        y_max = int(np.ceil(round(extent[3] / y_pixel_size, DECIMALS))) * y_pixel_size

    # number of columns and rows (+1 to include last pixel row and column, which is lost when computing the difference)
    n_rows = int(round((y_max - y_min) / y_pixel_size, DECIMALS)) + 1
    n_cols = int(round((x_max - x_min) / x_pixel_size, DECIMALS)) + 1

    # raster with zeros
    mask_img = Image.new('1', (n_cols, n_rows), 0)
    rows = (np.around(np.abs(ys - y_max) / y_pixel_size, decimals=DECIMALS)).astype(int)
    cols = (np.around(np.abs(xs - x_min) / x_pixel_size, decimals=DECIMALS)).astype(int)
    ImageDraw.Draw(mask_img).polygon(list(zip(cols, rows)), outline=1, fill=1)
    mask_ar = np.array(mask_img).astype(np.uint8)

    return mask_ar


def segmentize_geometry(geometry, segment=0.5):
    """
    segmentizes the lines of a geometry

    Parameters
    ----------
    geometry : OGRGeometry
        geometry object
    segment : float, optional
        for precision: distance in units of input osr_spref of longest
        segment of the geometry polygon

    Returns
    -------
    OGRGeometry
        a congruent geometry realised by more vertices along its shape
    """

    geometry_out = geometry.Clone()

    geometry_out.Segmentize(segment)

    geometry = None
    return geometry_out


def get_lonlat_intersection(geometry1, geometry2):
    """
    gets the intersect in lonlat space.
    geometry1 is split at the antimeridian
    (i.e. the 180 degree dateline)

    Parameters
    ----------
    geometry1 : OGRGeometry
        polygon geometry object in lonlat space
        is split by the antimeridian
    geometry2 : OGRGeometry
        geometry object
        should be the large one

    Returns
    -------
    boolean
        does geometry1 intersect with geometry2?
    """

    geometry1c = geometry1.Clone()
    geometry2c = geometry2.Clone()
    geometry1 = None
    geometry2 = None

    if geometry1c.GetGeometryName() == 'MULTIPOLYGON':
        geometry1c = ogr.ForceToPolygon(geometry1c)
        print(
            'Warning: get_lonlat_intersection(): Take care: Multipolygon is forced to Polygon!'
        )

    polygons = split_polygon_by_antimeridian(geometry1c)

    return polygons.Intersection(geometry2c)


def split_polygon_by_antimeridian(lonlat_polygon, split_limit=150.0):
    """
    Function that splits a polygon at the antimeridian
    (i.e. the 180 degree dateline)

    Parameters
    ----------
    lonlat_polygon : OGRGeometry
        geometry object in lonlat space to be split by the antimeridian
    split_limit : float, optional
        longitude that determines what is split and what not. default is 150.0
        e.g. a polygon with a centre east of 150E or west of 150W will be split!
    Returns
    -------
    splitted_polygons : OGRGeometry
        MULTIPOLYGON comprising east and west parts of lonlat_polygon
        contains only one POLYGON if no intersect with antimeridian is given

    """

    # prepare the input polygon
    in_points = lonlat_polygon.GetGeometryRef(0).GetPoints()
    lons = [p[0] for p in in_points]

    # case of very long polygon in east-west direction,
    # crossing the Greenwich meridian, but not the antimeridian,
    # which is most probably a wrong interpretion.
    # --> wrapping longitudes to the eastern Hemisphere (adding 360°)
    if (len(np.unique(np.sign(lons))) == 2) and (np.mean(np.abs(lons))
                                                 > split_limit):
        new_points = [(y[0] + 360, y[1]) if y[0] < 0 else y for y in in_points]
        lonlat_polygon = create_polygon_geometry(
            new_points, lonlat_polygon.GetSpatialReference(), segment=0.5)

    # return input polygon if not cross anti-meridian
    max_lon = np.max(
        [p[0] for p in lonlat_polygon.GetGeometryRef(0).GetPoints()])
    if max_lon <= 180:
        return lonlat_polygon

    # define the antimeridian
    antimeridian = LineString([(180, -90), (180, 90)])

    # use shapely for the splitting
    merged = linemerge([
        Polygon(lonlat_polygon.GetBoundary().GetPoints()).boundary,
        antimeridian
    ])
    borders = unary_union(merged)
    polygons = polygonize(borders)

    # setup OGR multipolygon
    splitted_polygons = ogr.Geometry(ogr.wkbMultiPolygon)
    geo_sr = get_lonlat_sref()
    splitted_polygons.AssignSpatialReference(geo_sr)

    # wrap the longitude coordinates
    # to get only longitudes out out [0, 180] or [-180, 0]
    for p in polygons:

        point_coords = p.exterior.coords[:]
        lons = [p[0] for p in point_coords]

        # all greater than 180° longitude (Western Hemisphere)
        if (len(np.unique(np.sign(lons))) == 1) and (np.greater_equal(
                lons, 180).all()):
            wrapped_points = [(y[0] - 360, y[1], y[2]) for y in point_coords]

        # all less than 180° longitude (Eastern Hemisphere)
        elif (len(np.unique(np.sign(lons))) == 1) and (np.less_equal(
                lons, 180).all()):
            wrapped_points = point_coords

        # crossing the Greenwhich-meridian
        elif (len(np.unique(np.sign(lons))) >= 2) and (np.mean(np.abs(lons))
                                                       < split_limit):
            wrapped_points = point_coords

        # crossing the Greenwhich-meridian, but should cross the antimeridian
        # (should not be happen actually)
        else:
            continue

        new_poly = Polygon(wrapped_points)
        splitted_polygons.AddGeometry(ogr.CreateGeometryFromWkt(new_poly.wkt))

    return splitted_polygons
