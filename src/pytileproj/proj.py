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

"""Utility module for projections."""

import json
import sys
import warnings

import pyproj
import requests
from osgeo import ogr, osr
from shapely.geometry import MultiPolygon, Polygon

from pytileproj._const import TIMEOUT

if "cartopy" in sys.modules:
    import cartopy.crs as ccrs


def fetch_proj_zone(epsg: int) -> ogr.Geometry | None:
    """Fetch the zone polygon of the given projection from the EPSG database.

    Parameters
    ----------
    epsg: int
        EPSG code representing the projection.

    Returns
    -------
    ogr.Geometry | None
        OGR polygon or multi-polygon object representing the projection zone.

    Notes
    -----
    This function requires a internet connection.

    """
    epsg_code_url = "https://apps.epsg.org/api/v1/ProjectedCoordRefSystem/"
    epsg_extent_url = "https://apps.epsg.org/api/v1/Extent/"

    zone_geom = None
    code_resp = requests.get(f"{epsg_code_url}/{epsg}/", timeout=TIMEOUT)
    if code_resp.ok:
        code_data = json.loads(code_resp.content)
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
                extent_data = json.loads(extent_resp.content)
                geom_type = extent_data["type"]
                coords = extent_data["coordinates"]
                if geom_type == "Polygon":
                    zone_geom = Polygon(coords[0])
                elif geom_type == "MultiPolygon":
                    zone_geom = MultiPolygon(coords)
                else:
                    err_msg = f"Geometry type '{geom_type}' not supported."
                    raise ValueError(err_msg)
                zone_geom = ogr.CreateGeometryFromWkt(zone_geom.wkt)
                zone_geom.AssignSpatialReference(get_geog_sref())

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
    x: float, y: float, this_crs: pyproj.CRS, other_crs: pyproj.CRS
) -> tuple[float, float]:
    """Transform coordinate tuple from a given to another projection.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.
    this_crs: pyproj.CRS
        CRS of the input coordinates.
    other_crs: pyproj.CRS
        CRS of the target projection.

    Returns
    -------
    float
        X coordinate in the target projection.
    float
        Y coordinate in the target projection.

    """
    traffo = pyproj.Transformer.from_crs(this_crs, other_crs, always_xy=True)
    return traffo.transform(x, y)


def get_geog_sref() -> osr.SpatialReference:
    """Create OSR spatial reference object representing the LonLat projection.

    Returns
    -------
    osr.SpatialReference
        Spatial reference representing the LonLat projection.

    """
    sref = osr.SpatialReference()
    sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    sref.ImportFromEPSG(4326)

    return sref
