import json
import warnings

import cartopy.crs as ccrs
import pyproj
import requests
from osgeo import ogr, osr
from shapely.geometry import MultiPolygon, Polygon


def fetch_proj_zone(epsg: int) -> ogr.Geometry | None:
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
                zone_sref = osr.SpatialReference()
                zone_sref.ImportFromEPSG(4326)
                zone_sref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                zone_geom.AssignSpatialReference(zone_sref)

    return zone_geom


def pyproj_to_cartopy_crs(crs: pyproj.CRS) -> ccrs.CRS:
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
