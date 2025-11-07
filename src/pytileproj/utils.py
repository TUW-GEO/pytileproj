import requests
import warnings
import json
from shapely.geometry import Polygon, MultiPolygon
from osgeo import ogr, osr


def fetch_proj_zone(epsg: int) -> ogr.Geometry | None:
    epsg_code_url = "https://apps.epsg.org/api/v1/ProjectedCoordRefSystem/"
    epsg_extent_url = "https://apps.epsg.org/api/v1/Extent/"

    zone_geom = None
    code_resp = requests.get(f'{epsg_code_url}/{epsg}/')
    if code_resp.ok:
        code_data = json.loads(code_resp.content)
        code_usages = code_data["Usage"]
        if len(code_usages):
            if len(code_usages) != 1:
                warnings.warn("Multiple EPSG code usages found!")
            code_usage = code_usages[-1]
            extent_resp = requests.get(f'{epsg_extent_url}/{code_usage["Extent"]["Code"]}/polygon')
            if extent_resp.ok:
                extent_data = json.loads(extent_resp.content)
                geom_type = extent_data['type']
                coords = extent_data['coordinates']
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