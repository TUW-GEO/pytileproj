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
import numpy as np

from osgeo import ogr
from osgeo import osr
from osgeo.gdal import __version__ as gdal_version

from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import linemerge
from shapely.ops import unary_union
from shapely.ops import polygonize


def get_geog_spatial_ref():
    """
    Small function to generate an OSR.SpatialReference() for the geographic lonlat-space.
    This was necessary because GDAL>=3.0.0 is sensitive to
        the axis order (or was changed the order, whatever, who really knows?) ;-)

    Returns
    -------
    SpatialReference
        osgeo spatial reference defining lonlat-space

    """

    geo_sr = osr.SpatialReference()
    geo_sr.SetWellKnownGeogCS("EPSG:{}".format(str(4326)))

    if gdal_version[0] == '3':
        # "0" is "OAMS_TRADITIONAL_GIS_ORDER" which is lon-lat-order (like in GDAL 2.x)
        # "1" is "OAMS_AUTHORITY_COMPLIANT " which is lat-lon-order (like in GDAL 3.x)
        #               (following ISO:19111)
        # please check https://gdal.org/tutorials/osr_api_tut.html
        geo_sr.SetAxisMappingStrategy(0)

    # nothing to do, as everything in pytileproj is using the lon-lat-order
    elif gdal_version[0] == '2':
        pass

    # check for future releases
    else:
        return

    return geo_sr


def uv2xy(u, v, src_ref, dst_ref):
    """
    wrapper; reprojects a pair of point coordinates
    Parameters
    ----------
    u : number
        input coordinate ("Rechtswert")
    v : number
        input coordinate ("Hochwert")
    src_ref : SpatialReference
        osgeo spatial reference defining the input u, v coordinates
    dst_ref : SpatialReference
        osgeo spatial reference defining the output x, y coordinates

    Returns
    -------
    x : number
        output coordinate ("Rechtswert")
    y : : number
        output coordinate ("Hochwert")
    """
    tx = osr.CoordinateTransformation(src_ref, dst_ref)
    x, y, _ = tx.TransformPoint(u, v)
    return x, y


def create_multipoint_geometry(u, v, osr_spref):
    """
    wrapper; creates multipoint geometry in given projection
    Parameters
    ----------
    u : list of numbers
        input coordinates ("Rechtswert")
    v : list of numbers
        input coordinates ("Hochwert")
    osr_spref : OGRSpatialReference
        spatial reference of given coordinates

    Returns
    -------
    OGRGeometry
        a geometry holding all points defined by (u, v)

    """
    point_geom = ogr.Geometry(ogr.wkbMultiPoint)
    point_geom.AssignSpatialReference(osr_spref)
    for p, _ in enumerate(u):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(u[p], v[p], 0)
        point_geom.AddGeometry(point)

    return point_geom


def create_point_geometry(u, v, osr_spref):
    """
    wrapper; creates single point geometry in given projection

    Parameters
    ----------
    u : list of numbers
        input coordinates ("Rechtswert")
    v : list of numbers
        input coordinates ("Hochwert")
    osr_spref : OGRSpatialReference
        spatial reference of given coordinates

    Returns
    -------
    OGRGeometry
        a geometry holding point defined by (u, v)

    """
    point_geom = ogr.Geometry(ogr.wkbPoint)
    point_geom.AddPoint(u, v)
    point_geom.AssignSpatialReference(osr_spref)

    return point_geom


def create_geometry_from_wkt(wkt_multipolygon, epsg=4326, segment=None):
    """
    return extent geometry from multipolygon defined by wkt string

    Parameters
    ----------
    wkt_multipolygon : str
        WKT text containing points of geometry (e.g. polygon)
    epsg : int
        EPSG code of spatial reference of the points.
    segment : float
        for precision: distance of longest segment of the geometry polygon
        in units of input osr_spref (defined by epsg)

    Returns
    -------
    OGRGeometry
        a geometry holding the multipolygon and spatial reference

    """
    geom = ogr.CreateGeometryFromWkt(wkt_multipolygon)

    if epsg == 4326:
        geo_sr = get_geog_spatial_ref()
    else:
        geo_sr = osr.SpatialReference()
        geo_sr.SetWellKnownGeogCS("EPSG:{}".format(str(epsg)))

    geom.AssignSpatialReference(geo_sr)

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        geom = segmentize_geometry(geom, segment=segment)

    return geom


def open_geometry(fname, feature=0, format='shapefile'):
    '''
    opens a geometry from a vector file.

    fname : str
        full path of the output file name
    feature : int
        optional; order number of feature that should be returned;
        default is 0 for the first feature/object/geometry.
    format : str
        optional; format name. currently only shapefile is supported
    def __getitem__(key):
        return geoms[key]

    Returns
    -------
    OGRGeometry
        a geometry from the input file as indexed by the feature number
    '''

    if format == 'shapefile':
        drivername = "ESRI Shapefile"

    driver = ogr.GetDriverByName(drivername)
    ds = driver.Open(fname, 0)
    feature = ds.GetLayer(0).GetFeature(feature)
    geom = feature.GetGeometryRef()

    out = geom.Clone()
    ds, feature, geom = None, None, None
    return out


def write_geometry(geom, fname, format="shapefile", segment=None):
    """
    writes a geometry to a vector file.

    parameters
    ----------
    geom : OGRGeometry
        geometry object
    fname : str
        full path of the output file name
    format : str
        optional; format name. currently only shapefile is supported
    segment : float
        for precision: distance of longest segment of the geometry polygon
        in units of input osr_spref
    """

    if format == 'shapefile':
        drivername = "ESRI Shapefile"

    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(fname)
    srs = geom.GetSpatialReference()

    dst_layer = dst_ds.CreateLayer("out", srs=srs)
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(fd)

    feature = ogr.Feature(dst_layer.GetLayerDefn())
    feature.SetField("DN", 1)

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        geom = segmentize_geometry(geom, segment=segment)

    feature.SetGeometry(geom)
    dst_layer.CreateFeature(feature)

    dst_ds, feature, geom = None, None, None
    return


def bbox2polygon(bbox, osr_spref, segment=None):
    """
    create a polygon geometry from bounding-box bbox, given by
    a set of two points, spanning a polygon area

    bbox : list
        list of coordinates representing the rectangle-region-of-interest
        in the format of [(left, lower), (right, upper)]
    osr_spref : OGRSpatialReference
        spatial reference of the coordinates in bbox
    segment : float
        for precision: distance of longest segment of the geometry polygon
        in units of input osr_spref

    Returns
    -------
    geom_area : OGRGeometry
        a geometry representing the input bbox as
        a) polygon-geometry when defined by a rectangle bbox
        b) point-geometry when defined by bbox through tuples of coordinates
    """

    bbox2 = deepcopy(bbox)

    # wrap around dateline (considering left-lower and right-upper logic).
    if bbox2[0][0] > bbox2[1][0]:
        bbox2[1] = (bbox2[1][0] + 360, bbox2[1][1])

    corners = [(float(bbox2[0][0]), float(bbox2[0][1])),
               (float(bbox2[0][0]), float(bbox2[1][1])),
               (float(bbox2[1][0]), float(bbox2[1][1])),
               (float(bbox2[1][0]), float(bbox2[0][1]))]

    return create_polygon_geometry(corners, osr_spref, segment=segment)


def create_polygon_geometry(points, osr_spref, segment=None):
    """
    returns polygon geometry defined by list of points

    Parameters
    ----------
    points : list
        points defining the polygon, either...
        2D: [(x1, y1), (x2, y2), ...]
        3D: [(x1, y1, z1), (x2, y2, z2), ...]
    osr_spref : OGRSpatialReference
        spatial reference to what the geometry should be transformed to
    segment : float, optional
        for precision: distance in units of input osr_spref of longest
        segment of the geometry polygon

    Returns
    -------
    OGRGeometry
        a geometry projected in the target spatial reference

    """
    # create ring from all points
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for p in points:
        if len(p) == 2:
            p += (0.0, )
        ring.AddPoint(*p)
    ring.CloseRings()

    # create the geometry
    polygon_geometry = ogr.Geometry(ogr.wkbPolygon)
    polygon_geometry.AddGeometry(ring)

    # assign spatial reference
    polygon_geometry.AssignSpatialReference(osr_spref)

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        polygon_geometry = segmentize_geometry(polygon_geometry,
                                               segment=segment)

    return polygon_geometry


def transform_geometry(geometry, osr_spref, segment=None):
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
    # to count the number of points of the polygon
    # print(geometry_out.GetGeometryRef(0).GetGeometryRef(0).GetPointCount())

    geometry_out = geometry.Clone()

    # modify the geometry such it has no segment longer then the given distance
    if segment is not None:
        geometry_out = segmentize_geometry(geometry_out, segment=segment)

    # transform geometry to new spatial reference system.
    geometry_out.TransformTo(osr_spref)

    # split polygons at antimeridian
    if osr_spref.ExportToProj4().startswith('+proj=longlat'):
        if geometry_out.GetGeometryName() in ['POLYGON', 'MULTIPOLYGON']:
            geometry_out = split_polygon_by_antimeridian(geometry_out)

    geometry = None
    return geometry_out


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


def intersect_geometry(geometry1, geometry2):
    """
    returns the intersection of two point or polygon geometries

    Parameters
    ----------
    geometry1, geometry2 : OGRGeometry
        geometry objects

    Returns
    -------
    intersection : OGRGeometry
        a geometry representing the intersection area
    """

    geometry1c = geometry1.Clone()
    geometry2c = geometry2.Clone()
    geometry1 = None
    geometry2 = None

    intersection = geometry1c.Intersection(geometry2c)

    return intersection


def check_lonlat_intersection(geometry1, geometry2):
    """
    checks if two polygon geometries intersect in lonlat space.
    geometry1 is split at the antimeridian
    (i.e. the 180 degree dateline)

    Parameters
    ----------
    geometry1 : OGRGeometry
        polygon geometry object in lonlat space
        is split by the antimeridian
    geometry2 : OGRGeometry
        polygon geometry object
        should be the large one

    Returns
    -------
    boolean
        does geometry1 intersect with geometry2?
    """

    area = get_lonlat_intersection(geometry1, geometry2)

    if area.Area() == 0.0:
        return False
    if area.Area() != 0.0:
        return True


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
    geo_sr = get_geog_spatial_ref()
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


def get_geometry_envelope(geometry, rounding=1.0):
    """
    returns the envelope (= the axis-parallel bounding box) of the geometry

    Parameters
    ----------
    geometry : Geometry
        geometry object
    rounding : float
        precision
    Returns
    -------
    tuple
        rounded coordinates of geometry-envelope
        as (xmin, ymin, xmax, ymax)

    """

    # get the "envelope" of a POINT geometry
    if geometry.ExportToWkt().startswith('POINT'):
        out = tuple(
            [int(x / rounding) * rounding
             for x in geometry.GetPoint()[0:2]]) * 2

    # get the envelope for each sub-geometry
    # works for MULTIPOLYGON; POLYGON; MULTIPOINT
    else:
        n_geometries = geometry.GetGeometryCount()
        li = np.zeros((n_geometries, 4))
        for g in range(n_geometries):
            env = geometry.GetGeometryRef(g).GetEnvelope()
            li[g] = [int(x / rounding) * rounding for x in env]

        # shuffle order to [xmin, ymin, xmax, ymax]
        # BBM: please change if you know a better way!
        envelope = np.zeros_like(li)
        envelope[:, 0] = li[:, 0]
        envelope[:, 1] = li[:, 2]
        envelope[:, 2] = li[:, 1]
        envelope[:, 3] = li[:, 3]

        # exclude antimeridian as potential limit (experimential)
        if geometry.GetSpatialReference().ExportToProj4().startswith(
                '+proj=longlat') and (n_geometries >= 2):
            envelope[envelope == -180.0] = np.nan
            envelope[envelope == 180.0] = np.nan

        # get the extreme values as tuple
        out = tuple((*np.nanmin(envelope, axis=0)[0:2],
                     *np.nanmax(envelope, axis=0)[2:4]))

    return out


def round_vertices_of_polygon(geometry, decimals=0):
    """
    'Cleans' the vertices of a polygon, so that it has rounded coordinates.
    Attention: MULTIPOLYGON is not yet implemented!
    Parameters
    ----------
    geometry : OGRGeometry
        a polygon geometry
    decimals : int
        optional. rounding precision. default is 0.

    Returns
    -------
    geometry_out : OGRGeometry
        a polygon geometry with rounded vertices
    """

    ring = geometry.GetGeometryRef(0)

    rounded_ring = ogr.Geometry(ogr.wkbLinearRing)

    n_points = ring.GetPointCount()

    for p in range(n_points):
        lon, lat, z = ring.GetPoint(p)
        rlon, rlat, rz = [
            np.round(lon, decimals=decimals),
            np.round(lat, decimals=decimals),
            np.round(z, decimals=decimals)
        ]
        rounded_ring.AddPoint(rlon, rlat, rz)

    geometry_out = ogr.Geometry(ogr.wkbPolygon)
    geometry_out.AddGeometry(rounded_ring)

    return geometry_out


def points2geometry(coords, osr_spref):
    """create a point geometry from a list of coordinate-tuples

    coords : list of tuples
        point-coords in terms of [(x1, y1), (x2, y2), ...]
    osr_spref : OGRSpatialReference
        spatial reference of the coordinates in extent

    Returns
    -------
    geom_area : OGRGeometry
        a point-geometry representing the input tuples of coordinates
    """

    u = []
    v = []
    for co in coords:
        if len(co) == 2:
            u.append(co[0])
            v.append(co[1])

    return create_multipoint_geometry(u, v, osr_spref)


def setup_test_geom_spitzbergen():
    """
    Routine providing a geometry for testing

    Returns
    -------
    poly_spitzbergen : OGRGeometry
        4-corner polygon over high latitudes (is much curved on the globe)
    """

    osr_spref = get_geog_spatial_ref()

    points = [(8.391827331539572, 77.35762113396143),
              (16.87007957357446, 81.59290885863483),
              (40.50119498304080, 79.73786853853339),
              (25.43098663332705, 75.61353436967198)]

    poly_spitzbergen = create_polygon_geometry(points, osr_spref, segment=None)

    return poly_spitzbergen


def setup_geom_kamchatka():
    """
    Routine providing a geometry for testing

    Returns
    -------
    poly_kamchatka : OGRGeometry
        4-corner polygon close to the dateline at Kamchatka peninsula.

    """

    osr_spref = get_geog_spatial_ref()

    points = [(165.6045170932673, 59.05482187690058),
              (167.0124744118732, 55.02758744559601),
              (175.9512099050924, 54.36588084375806),
              (179.4591330039386, 56.57634572271662)]

    poly_kamchatka = create_polygon_geometry(points, osr_spref, segment=None)

    return poly_kamchatka


def setup_test_geom_siberia_antimeridian_180plus():
    """
    Routine providing a geometry for testing

    Returns
    -------
    poly_siberia_antim_180plus : OGRGeometry
        4-corner polygon in Siberia, crossing the antimeridian

    """

    osr_spref = get_geog_spatial_ref()

    points = [(177.6584965942706, 67.04864900747906),
              (179.0142461506587, 65.34233852520839),
              (184.1800038679373, 65.74423313395079),
              (183.1741580487398, 67.46683765736415)]

    poly_siberia_antim_180plus = create_polygon_geometry(points,
                                                         osr_spref,
                                                         segment=None)

    return poly_siberia_antim_180plus


def setup_test_geom_siberia_alaska():
    """
    Routine providing a geometry for testing

    Returns
    -------
        poly_siberia_alaska : OGRGeometry
        4-corner polygon over Siberia and Alaska, crossing antimeridian
        and covering two Equi7Grid subgrids.

    """

    osr_spref = get_geog_spatial_ref()

    points = [(177.6545884597184, 67.05574774066811),
              (179.0195867605756, 65.33232820668778),
              (198.4723636216472, 66.06909015550372),
              (198.7828129097253, 68.14247939909886)]

    poly_siberia_alaska = create_polygon_geometry(points,
                                                  osr_spref,
                                                  segment=None)

    return poly_siberia_alaska
