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

import json
import warnings

import cartopy.crs as ccrs
import pyproj
import requests
from osgeo import ogr, osr
from shapely.geometry import MultiPolygon, Polygon


def fetch_proj_zone(epsg: int) -> ogr.Geometry | None:
    """
    Fetches the zone polygon of the given projection from the EPSG database.

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
    code_resp = requests.get(f"{epsg_code_url}/{epsg}/")
    if code_resp.ok:
        code_data = json.loads(code_resp.content)
        code_usages = code_data["Usage"]
        if len(code_usages):
            if len(code_usages) != 1:
                warnings.warn("Multiple EPSG code usages found!", stacklevel=1)
            code_usage = code_usages[-1]
            extent_resp = requests.get(
                f'{epsg_extent_url}/{code_usage["Extent"]["Code"]}/polygon'
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
                    raise ValueError(f"Geometry type '{geom_type}' not supported.")
                zone_geom = ogr.CreateGeometryFromWkt(zone_geom.wkt)
                zone_geom.AssignSpatialReference(get_geog_sref())

    return zone_geom


def pyproj_to_cartopy_crs(crs: pyproj.CRS) -> ccrs.CRS:
    """
    Converts a pyproj to a cartopy CRS object.

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

    if proj4_name == "longlat":
        ccrs_proj = ccrs.PlateCarree(central_longitude)
    elif proj4_name == "aeqd":
        ccrs_proj = ccrs.AzimuthalEquidistant(
            central_longitude, central_latitude, false_easting, false_northing
        )
    elif proj4_name == "merc":
        ccrs_proj = ccrs.Mercator(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            scale_factor=scale_factor,
        )
    elif proj4_name == "eck1":
        ccrs_proj = ccrs.EckertI(central_longitude, false_easting, false_northing)
    elif proj4_name == "eck2":
        ccrs_proj = ccrs.EckertII(central_longitude, false_easting, false_northing)
    elif proj4_name == "eck3":
        ccrs_proj = ccrs.EckertIII(central_longitude, false_easting, false_northing)
    elif proj4_name == "eck4":
        ccrs_proj = ccrs.EckertIV(central_longitude, false_easting, false_northing)
    elif proj4_name == "eck5":
        ccrs_proj = ccrs.EckertV(central_longitude, false_easting, false_northing)
    elif proj4_name == "eck6":
        ccrs_proj = ccrs.EckertVI(central_longitude, false_easting, false_northing)
    elif proj4_name == "aea":
        ccrs_proj = ccrs.AlbersEqualArea(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels,
        )
    elif proj4_name == "eqdc":
        ccrs_proj = ccrs.EquidistantConic(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels,
        )
    elif proj4_name == "gnom":
        ccrs_proj = ccrs.Gnomonic(central_longitude, central_latitude)
    elif proj4_name == "laea":
        ccrs_proj = ccrs.LambertAzimuthalEqualArea(
            central_longitude, central_latitude, false_easting, false_northing
        )
    elif proj4_name == "lcc":
        ccrs_proj = ccrs.LambertConformal(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            standard_parallels=standard_parallels,
        )
    elif proj4_name == "mill":
        ccrs_proj = ccrs.Miller(central_longitude)
    elif proj4_name == "moll":
        ccrs_proj = ccrs.Mollweide(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
        )
    elif proj4_name == "stere":
        ccrs_proj = ccrs.Stereographic(
            central_latitude,
            central_longitude,
            false_easting,
            false_northing,
            scale_factor=scale_factor,
        )
    elif proj4_name == "ortho":
        ccrs_proj = ccrs.Orthographic(central_longitude, central_latitude)
    elif proj4_name == "robin":
        ccrs_proj = ccrs.Robinson(
            central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
        )
    elif proj4_name == "sinus":
        ccrs_proj = ccrs.Sinusoidal(central_longitude, false_easting, false_northing)
    elif proj4_name == "tmerc":
        ccrs_proj = ccrs.TransverseMercator(
            central_longitude,
            central_latitude,
            false_easting,
            false_northing,
            scale_factor,
        )
    else:
        err_msg = f"Projection '{proj4_name}' is not supported."
        raise ValueError(err_msg)

    return ccrs_proj


def transform_coords(
    x: float, y: float, this_crs: pyproj.CRS, other_crs: pyproj.CRS
) -> tuple[float, float]:
    """
    Transforms coordinate tuple from a given to another projection.

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
    """
    Creates OSR spatial reference object representing the LonLat projection.

    Returns
    -------
    osr.SpatialReference
        Spatial reference representing the LonLat projection.

    """
    sref = osr.SpatialReference()
    sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    sref.ImportFromEPSG(4326)

    return sref
