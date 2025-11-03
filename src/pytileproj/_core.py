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
Code for Tiled Projection Systems.
"""

from morecantile.models import TileMatrix, TileMatrixSet
from pydantic import BaseModel, AfterValidator, PositiveInt, PositiveFloat
from typing import Annotated, Literal, Optional, Tuple, Union
import pyproj
from osgeo import ogr, osr
import requests
import warnings
import json
from shapely.geometry import Polygon, MultiPolygon


def translate_proj_zone(input: str | ogr.Geometry | None) -> ogr.Geometry | None:
    pass

def _fetch_proj_zone(epsg: int) -> ogr.Geometry | None:
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
                    zone_geom = Polygon(coords)
                elif geom_type == "MultiPolygon":
                    zone_geom = MultiPolygon(coords)
                else:
                    raise ValueError(f"Geometry type '{geom_type}' not supported.")
                zone_geom = ogr.CreateGeometryFromWkt(zone_geom.wkt)
                zone_sref = osr.SpatialReference()
                zone_sref.ImportFromEPSG(4326)
                zone_geom.AssignSpatialReference(zone_sref)
            
    return zone_geom


class ProjGrid(BaseModel, arbitrary_types_allowed=True):
    name: str
    crs: pyproj.CRS
    extent:  Tuple[float, float, float, float]
    sampling: PositiveFloat
    origin_xy: Tuple[float, float]
    tile_width_px: PositiveInt
    tile_height_px: PositiveInt
    tiling_level: Optional[PositiveInt] = 0
    x_direction: Optional[Literal["E", "W"]] = "E"
    y_direction: Optional[Literal["N", "S"]] = "N"
    proj_zone: Annotated[str | ogr.Geometry | None, AfterValidator(translate_proj_zone)] = None 
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        corner_of_ori = 'bottomLeft' if self.y_direction == 'N' else 'topLeft'
        matrix_width = int((self.extent[2] - self.extent[0]) // (self.tile_width_px * self.sampling))
        matrix_height = int((self.extent[3] - self.extent[1]) // (self.tile_height_px * self.sampling))

        self._tm = TileMatrix(scaleDenominator= self.sampling/ 0.28e-3,  # per OGC definition
                              cellSize=self.sampling,
                              cornerOfOrigin=corner_of_ori,
                              pointOfOrigin=self.origin_xy,
                              tileWidth=self.tile_width_px,
                              tileHeight=self.tile_height_px,
                              matrixWidth=matrix_width,
                              matrixHeight=matrix_height,
                              id=str(self.tiling_level), title=f"{self.name} at tiling level {self.tiling_level}")
        
        self.proj_zone = _fetch_proj_zone(self._epsg) if self._epsg else None

    @property
    def _epsg(self) -> int:
        return self.crs.to_epsg()

    def __eq__(self, other: Union['ProjGrid', object]) -> bool:
        return True

class ProjPyramid(BaseModel):
    pass

'''
class TPSCoreProperty(object):
    """
    Class holding information needed at every level of `TiledProjectionSystem`,
    the alltime-valid "core properties".
    With this, core parameters are everywhere accessible via the same name.
    """

    def __init__(self, tag, projection, sampling, tiletype, tile_xsize_m,
                 tile_ysize_m):
        """
        Initialises a TPSCoreProperty.

        Parameters
        ----------
        tag : str
            identifier of the object holding the TPSCoreProperty
            e.g. 'EU' or 'Equi7'.
        projection : None or TPSProjection
            A TPSProjection() holding info on the spatial reference
        sampling : int
            the grid sampling = size of pixels; in metres.
        tiletype : str
            tilecode (related the tile size of the grid)
        tile_xsize_m : int
            tile size in x direction defined for the grid's sampling
        tile_ysize_m :
            tile size in y direction defined for the grid's sampling
        """

        self.tag = tag
        self.projection = projection
        self.sampling = sampling
        self.tiletype = tiletype
        self.tile_xsize_m = tile_xsize_m
        self.tile_ysize_m = tile_ysize_m


class TPSProjection():
    """
    Projection class holding and translating the definitions of a projection when initialising.
    """

    def __init__(self, epsg=None, proj4=None, wkt=None):
        """
        Initialises a TPSProjection().

        Parameters
        ----------
        epsg : int
            The EPSG-code of the spatial reference.
            As from http://www.epsg-registry.org
            Not all reference do have a EPSG code.
        proj4 : str
            The proj4-string defining the spatial reference.
        wkt : str
            The wkt-string (well-know-text) defining the spatial reference.

        Notes
        -----
        Either one of epsg, proj4, or wkt must be given.
        """

        checker = {epsg, proj4, wkt}
        checker.discard(None)
        if len(checker) == 0:
            raise ValueError('Projection is not defined!')

        if len(checker) != 1:
            raise ValueError('Projection is defined ambiguously!')

        spref = osr.SpatialReference()

        if epsg is not None:

            if epsg == 4326:
                spref = ptpgeometry.get_geog_spatial_ref()
            else:
                spref.ImportFromEPSG(epsg)

            self.osr_spref = spref
            self.proj4 = spref.ExportToProj4()
            self.wkt = spref.ExportToWkt()
            self.epsg = epsg

        if proj4 is not None:
            spref.ImportFromProj4(proj4)
            self.osr_spref = spref
            self.proj4 = proj4
            self.wkt = spref.ExportToWkt()
            self.epsg = self.extract_epsg(self.wkt)

        if wkt is not None:
            spref.ImportFromWkt(wkt)
            self.osr_spref = spref
            self.proj4 = spref.ExportToProj4()
            self.wkt = wkt
            self.epsg = self.extract_epsg(self.wkt)

    def extract_epsg(self, wkt):
        """
        Checks if the WKT contains an EPSG code for the spatial reference,
        and returns it, if found.

        Parameters
        ----------
        wkt : string
            The wkt-string (well-know-text) defining the spatial reference.

        Returns
        -------
        epsg : integer, None
            the EPSG code of the spatial reference (if found). Else: None
        """

        pos_last_code = wkt.rfind('EPSG')
        pos_end = len(wkt)
        if pos_end - pos_last_code < 16:
            epsg = int(wkt[pos_last_code + 7:pos_last_code + 11])
        else:
            epsg = None

        return epsg


class TiledProjectionSystem(object):

    __metaclass__ = abc.ABCMeta

    # placeholders for static data defining the grid
    # static attribute
    _static_data = None
    # sub grid IDs
    _static_subgrid_ids = ['SG']
    # supported tile widths (linked to grid sampling)
    _static_tilecodes = ['T1']
    # supported grid spacing ( = the pixel sampling)
    _static_sampling = [1]

    def __init__(self, sampling, tag='TPS'):
        """
        Initialises a TiledProjectionSystem().

        Parameters
        ----------
        sampling : int
            the grid sampling = size of pixels; in metres.
        tag : str
            identifier of the object holding the TPSCoreProperty
            e.g. 'EU' or 'Equi7'.
        """

        tiletype = self.get_tiletype(sampling)
        tile_xsize_m, tile_ysize_m = self.get_tilesize(sampling)

        self.core = TPSCoreProperty(tag, None, sampling, tiletype,
                                    tile_xsize_m, tile_ysize_m)

        self.subgrids = self.define_subgrids()

    def __getattr__(self, item):
        """
        short link for items of subgrids and core
        """
        if item in self.subgrids:
            return self.subgrids[item]
        elif item in self.core.__dict__:
            return self.core.__dict__[item]
        else:
            return self.__dict__[item]

    @abc.abstractmethod
    def define_subgrids(self):
        pass

    def locate_geometry_in_subgrids(self, geometry):
        """
        finds overlapping subgrids of given geometry.
        checks for crossing the antimeridian

        Attributes
        ----------
        geometry : OGRGeometry
            a geometry to be located

        Returns
        -------
        list of TiledProjection()
            all subgrids that overlap with geom
        """

        covering_subgrid = list()

        if geometry.GetGeometryName() in ['POLYGON', 'MULTIPOLYGON']:
            for x in self.subgrids.keys():
                if ptpgeometry.check_lonlat_intersection(
                        geometry,
                        self.subgrids.get(x).polygon_geog):
                    covering_subgrid.append(x)

        if geometry.GetGeometryName() in ['POINT', 'MULTIPOINT']:
            for x in self.subgrids.keys():
                if geometry.Intersects(self.subgrids.get(x).polygon_geog):
                    covering_subgrid.append(x)

        return covering_subgrid

    def lonlat2xy(self, lon, lat, subgrid=None):
        """
        converts latitude and longitude coordinates to TPS grid coordinates

        Parameters
        ----------
        lon : list of numbers
            longitude coordinates
        lat : list of numbers
            latitude coordinates
        subgrid : str, optional
             acronym / subgrid ID to search within (speeding up)
             forces to find coordinates in given subgrid
             --> can return outlying or negative coordinates!

        Returns
        -------
        subgrid : str
            subgrid ID in which the returned x, y coordinates are defined
        x, y : list of float
            TPS grid coordinates
        """

        lon = np.array(lon, dtype=np.float64)
        lat = np.array(lat, dtype=np.float64)

        if subgrid is None:
            vfunc = np.vectorize(self._lonlat2xy)
            return vfunc(lon, lat)
        else:
            return self._lonlat2xy_subgrid(lon, lat, subgrid)

    def _lonlat2xy(self, lon, lat):
        """
        finds overlapping subgrid of a given point in lon-lat-space
        and computes the projected coordinates referring to that subgrid.

        Parameters
        ----------
        lon : number
            longitude coordinate
        lat : number
            latitude coordinate

        Returns
        -------
        subgrid : str
            subgrid ID
        x, y : float
            TPS grid coordinates
        """

        # create point geometry
        lonlatprojection = TPSProjection(epsg=4326)
        point_geom = ptpgeometry.create_point_geometry(
            lon, lat, lonlatprojection.osr_spref)

        # search for co-locating subgrid
        subgrid = self.locate_geometry_in_subgrids(point_geom)[0]

        x, y, = ptpgeometry.uv2xy(
            lon, lat, lonlatprojection.osr_spref,
            self.subgrids[subgrid].core.projection.osr_spref)

        return np.full_like(x, subgrid, dtype=(str, len(subgrid))), x, y

    def _lonlat2xy_subgrid(self, lon, lat, subgrid):
        """
        computes the projected coordinates in given subgrid.

        Parameters
        ----------
        lon : list of numbers
            longitude coordinates
        lat : list of numbers
            latitude coordinates
        subgrid : str, optional
             acronym / subgrid ID to search within (speeding up)
             forces to find coordinates in given subgrid
             --> can return outlying or negative coordinates!

        Returns
        -------
        subgrid : str
            acronym / subgrid ID
        x, y : int
            TPS grid coordinates
        """

        tf = Transformer.from_crs(('epsg', '4326'),
                                  self.subgrids[subgrid].core.projection.proj4,
                                  always_xy=True)
        x, y = tf.transform(lon, lat)

        return subgrid, x, y

    @abc.abstractmethod
    def create_tile(self, name):
        pass

    def get_tile_bbox_geog(self, tilename):
        """
        returns the envelope of the tile in the lonlat-space

        Parameters
        ----------
        tilename : str
            name of the tile; e.g EU500M_E012N018T6 or E012N018T6

        Returns
        -------
        tuple
            bounding box of subgrid
            as (lonmin, latmin, lonmax, latmax)

        """
        return self.create_tile(tilename).bbox_geog

    def get_tile_bbox_proj(self, tilename):
        """
        returns the envelope of the tile in the projected space

        Parameters
        ----------
        tilename : str
            name of the tile; e.g EU500M_E012N018T6 or E012N018T6

        Returns
        -------
        tuple
            bounding box of subgrid
            as (xmin, ymin, xmax, ymax)

        """
        return self.create_tile(tilename).bbox_proj

    @abc.abstractmethod
    def get_tiletype(self, sampling=None):
        pass

    def _get_tiletype(self):
        """
        Internal function to get the tile code of the grid instance.

        Returns
        -------
        tilecode : str
            tilecode (related the tile size of the grid.

        """
        return self.get_tiletype(self.core.sampling)

    @abc.abstractmethod
    def get_tilesize(self, sampling):
        pass

    def search_tiles_in_roi(self,
                            roi_geometry=None,
                            bbox=None,
                            points=None,
                            osr_spref=None,
                            subgrid_ids=None,
                            coverland=False):
        """
        Search the tiles of the grid which intersect by the given area.

        Parameters
        ----------
        roi_geometry : geometry
            a polygon or multipolygon geometry object representing the ROI
        bbox : list
            a list of coordinate-tuples representing a rectangle-shape
            region-of-interest in the format of
                [(left, lower), (right, upper)]
        points : list
            a list of points-of-interest as tuples in the format of
                [(x1, y1), (x2, y2), ...]
        osr_spref : OGRSpatialReference
            spatial reference of input coordinates in extent
        sgrid_ids : string or list of strings
            subgrid IDs, e.g. specifying over which continent
            you want to search.
            Default value is None for searching all subgrids.
        coverland : Boolean
            option to search for tiles covering land at any point in the tile

        Returns
        -------
        list
            return a list of  the overlapped tiles' name.
            If not found, return empty list.
        """

        # check input grids
        if subgrid_ids is None:
            subgrid_ids = self.subgrids.keys()
        if isinstance(subgrid_ids, str):
            subgrid_ids = [subgrid_ids]
        if set(subgrid_ids).issubset(set(self.subgrids.keys())):
            subgrid_ids = list(subgrid_ids)
        else:
            raise ValueError("Invalid argument: grid must one of [ %s ]." %
                             " ".join(self.subgrids.keys()))

        if roi_geometry is None and bbox is None and points is None:
            print("Error: Either roi_geometry, bbox, or points must be given "
                  "as the region-of-interest!")
            return list()

        # obtain the ROI
        if roi_geometry is None:

            if osr_spref is None:
                projection = TPSProjection(epsg=4326)
                osr_spref = projection.osr_spref

            if points is not None:
                roi_geometry = ptpgeometry.points2geometry(points, osr_spref)

            elif bbox is not None:
                segment = 50000 if osr_spref.IsProjected() else 0.5
                roi_geometry = ptpgeometry.bbox2polygon(bbox,
                                                        osr_spref,
                                                        segment=segment)

        # switch for ROI defined by a single polygon or point(s)
        if roi_geometry.GetGeometryName() in [
                'POLYGON', 'MULTIPOINT', 'POINT'
        ]:

            tiles = self._search_tiles_in_roi(roi_geometry=roi_geometry,
                                              subgrid_ids=subgrid_ids,
                                              coverland=coverland)

        # switch for ROI defined by multiple polygons
        if roi_geometry.GetGeometryName() == 'MULTIPOLYGON':

            tiles = []

            # search tiles for each polygon individually
            for i_polygon in list(range(roi_geometry.GetGeometryCount())):

                geometry = roi_geometry.GetGeometryRef(i_polygon)

                i_tiles = self._search_tiles_in_roi(roi_geometry=geometry,
                                                    subgrid_ids=subgrid_ids,
                                                    coverland=coverland)

                tiles += i_tiles

            # reduce to unique list of tiles
            tiles = list(set(tiles))

        return tiles

    def _search_tiles_in_roi(self,
                             roi_geometry=None,
                             subgrid_ids=None,
                             coverland=False):
        """
        Internal function: Search the tiles of the grid which intersect by the given area.

        Parameters
        ----------
        roi_geometry : geometry
            a polygon or multipolygon geometry object representing the ROI
        sgrid_ids : string or list of strings
            subgrid IDs, e.g. specifying over which continent
            you want to search.
            Default value is None for searching all subgrids.
        coverland : Boolean
            option to search for tiles covering land at any point in the tile

        Returns
        -------
        list
            return a list of  the overlapped tiles' name.
            If not found, return empty list.
        """

        # load lat-lon spatial reference as the default
        geog_sr = TPSProjection(epsg=4326).osr_spref

        geom_sr = roi_geometry.GetSpatialReference()
        if geom_sr is None:
            roi_geometry.AssignSpatialReference(geog_sr)
        elif not geom_sr.IsSame(geog_sr):
            projected = roi_geometry.GetSpatialReference().IsProjected()
            if projected == 0:
                max_segment = 0.5
            elif projected == 1:
                max_segment = 50000
            else:
                raise Warning(
                    'Please check unit of geometry before reprojection!')
            roi_geometry = ptpgeometry.transform_geometry(roi_geometry,
                                                          geog_sr,
                                                          segment=max_segment)

        if roi_geometry.GetGeometryName() == 'MULTIPOLYGON':
            roi_polygons = []
            for i in range(roi_geometry.GetGeometryCount()):
                poly = roi_geometry.GetGeometryRef(i).Clone()
                poly.AssignSpatialReference(geog_sr)
                roi_polygons.append(poly)
        else:
            roi_polygons = [roi_geometry]

        overlapped_tiles = list()
        for roi_polygon in roi_polygons:
            # intersect the given grid ids and the overlapped ids
            overlapped_grids = self.locate_geometry_in_subgrids(roi_polygon)
            subgrid_ids = list(set(subgrid_ids) & set(overlapped_grids))

            # finding tiles
            for sgrid_id in subgrid_ids:
                overlapped_tiles.extend(
                    self.subgrids[sgrid_id].search_tiles_over_geometry(
                        roi_polygon, coverland=coverland))
        return list(set(overlapped_tiles))


class TiledProjection(object):
    """
    Class holding the projection and tiling definition of a
    tiled projection space.

    Parameters
    ----------
    Projection : Projection()
        A Projection object defining the spatial reference.
    tile_definition: TilingSystem()
        A TilingSystem object defining the tiling system.
        If None, the whole space is one single tile.
    """

    __metaclass__ = abc.ABCMeta

    staticdata = None

    def __init__(self, core, polygon_geog, tilingsystem=None):
        """
        Initialises a TiledProjection().

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the (sub-) grid
        polygon_geog : OGRGeometry
            geometry defining the extent/outline of the subgrid.
            if not given, a single global subgrid is assigned to the grid.
        tilingsystem : TilingSystem
            optional; an instance of TilingSystem()
            if not given, a single global tile is assigned to the grid.
        """

        self.core = core
        self.polygon_geog = ptpgeometry.segmentize_geometry(polygon_geog,
                                                            segment=0.5)
        self.polygon_proj = ptpgeometry.transform_geometry(
            self.polygon_geog, self.core.projection.osr_spref)
        self.bbox_proj = ptpgeometry.get_geometry_envelope(
            self.polygon_proj, rounding=self.core.sampling)

        # does this use anybody? its justs makes problems with the antimeridian!
        # self.bbox_geog = ptpgeometry.get_geometry_envelope(
        #     self.polygon_geog, rounding=self.core.sampling / 1000000.0)

        if tilingsystem is None:
            tilingsystem = GlobalTile(self.core, 'TG', self.get_bbox_proj())
        self.tilesys = tilingsystem

    def __getattr__(self, item):
        """
        short link for items of core
        """
        if item in self.core.__dict__:
            return self.core.__dict__[item]
        else:
            return self.__dict__[item]

    def get_bbox_geog(self):
        """
        Returns the limits of the subgrid in the lon-lat-space.

        Returns
        -------
        tuple
            bounding box of subgrid
            as (lonmin, latmin, lonmax, latmax)

        """
        return ptpgeometry.get_geometry_envelope(self.polygon_geog,
                                                 rounding=0.0001)

    def get_bbox_proj(self):
        """
        Returns the limits of the subgrid in the projected space.

        Returns
        -------
        tuple
            boundind box of subgrid
            as (xmin, ymin, xmax, ymax)

        """
        return self.polygon_proj.GetEnvelope()

    def xy2lonlat(self, x, y):
        """
        Converts projected coordinates to longitude and latitude coordinates
        Parameters

        ----------
        x : number or list of numbers
            projected x coordinate(s) in metres
        y : number or list of numbers
            projected y coordinate(s) in metres

        Returns
        -------
        lon : float or list of floats
            longitude coordinate(s)
        lat : float or list of floats
            latitude coordinate(s)

        """
        # set up spatial references

        tf = Transformer.from_crs(self.core.projection.proj4, ('epsg', '4326'),
                                  always_xy=True)
        lon, lat = tf.transform(x, y)

        return lon, lat

    def search_tiles_over_geometry(self, geometry, coverland=True):
        """
        Search tiles of the subgrid that are overlapping with the geometry.

        Parameters
        ----------
        geometry : OGRGeometry
            A point or polygon geometry representing the region of interest.
        coverland : Boolean
            option to search for tiles covering land at any point in the tile

        Returns
        -------
        overlapped_tiles : list
            Return a list of the overlapped tiles' name.
            If not found, return empty list.

        """
        overlapped_tiles = list()

        if geometry.GetGeometryName() in ['MULTIPOINT', 'POINT']:
            if geometry.Intersects(self.polygon_geog):
                # get intersect area with subgrid in latlon
                intersect_geometry = geometry.Intersection(self.polygon_geog)
                intersect_geometry = ptpgeometry.transform_geometry(
                    intersect_geometry, self.projection.osr_spref)
            else:
                return overlapped_tiles

        if geometry.GetGeometryName() in ['POLYGON', 'MULTIPOLYGON']:

            # get intersect area with subgrid in latlon
            intersect = ptpgeometry.get_lonlat_intersection(
                geometry, self.polygon_geog)

            # check if geom intersects subgrid
            if intersect.Area() == 0.0:
                return overlapped_tiles

            # transform intersection geometry back to the spatial ref system of the subgrid.
            # segmentise for high precision during reprojection.
            projected = intersect.GetSpatialReference().IsProjected()
            if projected == 0:
                max_segment = 0.5
            elif projected == 1:
                max_segment = 50000
            else:
                raise Warning(
                    'Please check unit of geometry before reprojection!')

            _ = ptpgeometry.transform_geometry(intersect,
                                               self.projection.osr_spref,
                                               segment=max_segment)
            intersect_geometry = _.MakeValid()

        # get envelope of the geometry
        envelope = ptpgeometry.get_geometry_envelope(intersect_geometry)

        # get overlapped tiles
        tiles = self.tilesys.identify_tiles_overlapping_xybbox(envelope)

        for tile in tiles:

            # get tile object
            t = self.tilesys.create_tile(tile)

            # get only tile that overlaps with intersect_geometry
            if t.polygon_proj.Intersects(intersect_geometry):

                # get only tile if coverland is satisfied
                if not coverland or self.tilesys.check_tile_covers_land(
                        t.name):
                    overlapped_tiles.append(t.name)

        return overlapped_tiles


class TilingSystem(object):
    """
    Class defining the tiling system and providing methods for queries and handling.

    Parameters (BBM: init(stuff))
    ----------
    projection : :py:class:`Projection`
        A Projection object defining the spatial reference.
    tile_definition: TilingSystem
        A TilingSystem object defining the tiling system.
        If None, the whole space is one single tile.

    Attributes (BBM: stuff that needs to be explained)
    ----------
    extent_geog:
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, core, polygon_geog, x0, y0):
        """
        Initialises an TilingSystem class for a specified subgrid.

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the (sub-) grid
        polygon_geog : OGRGeometry
            geometry defining the extent/outline of the subgrid
        x0 : int
            lower-left x (right) coordinates of the subgrid
        y0 : int
            lower-left y (up) coordinates of the subgrid
        """

        self.core = core
        self.x0 = x0
        self.y0 = y0
        self.xstep = self.core.tile_xsize_m
        self.ystep = self.core.tile_ysize_m
        self.polygon_proj = ptpgeometry.transform_geometry(
            polygon_geog, self.core.projection.osr_spref)
        self.bbox_proj = ptpgeometry.get_geometry_envelope(
            self.polygon_proj, rounding=self.core.sampling)

    def __getattr__(self, item):
        """
        short link for items of core
        """
        if item in self.core.__dict__:
            return self.core.__dict__[item]
        else:
            return self.__dict__[item]

    @abc.abstractmethod
    def create_tile(self, name=None, x=None, y=None):
        """
        Returns a Tile object of the grid.

        Parameters
        ----------
        name : str
            name of the tile
        x : int
            x (right) coordinate of a pixel located in the desired tile
            must to given together with y
        y : int
            y (up) coordinate of a pixel located in the desired tile
            must to given together with x

        Returns
        -------
        Tile
            object containing info of the specified tile.

        Notes
        -----
        either name, or x and y, must be given.
        """
        return

    def xy2ij_in_tile(self, x, y, lowerleft=False):
        """
        finds the tile and the pixel indices for a point given in projected coords.
        pixel indices comprise the column and row number (i, j)

        columns go from left to right (easting)
        rows go either
            top to bottom (lowerleft=False)
            bottom to top (lowerleft=True)

        Parameters
        ----------
        x : number
            projected x coordinate(s) in metres
        y : number
            projected y coordinate(s) in metres
        lowerleft : bool, optional
            should the row numbering start at the bottom?
            If yes, it returns lowerleft indices.

        Returns
        -------
        tilename : str
            long form of the tilename containing the lon-lat position
        i : integer
            pixel column number; starts with 0
        j : integer
            pixel row number; starts with 0

        """
        # get the overlapping tile
        tile = self.create_tile(x=x, y=y)

        i, j = tile.xy2ij(x=x, y=y, lowerleft=lowerleft)
        tilename = tile.name

        return tilename, i, j

    def round_xy2lowerleft(self, x, y):
        """
        Returns the lower-left coordinates of the tile in which the point,
        defined by x and y coordinates (in metres), is located.

        Parameters
        ----------
        x : int
            x (right) coordinate in the desired tile
            must to given together with y
        y : int
            y (up) coordinate in the desired tile
            must to given together with x

        Returns
        -------
        llx, lly: int
            lower-left coordinates of the tile
        """

        llx = x // self.core.tile_xsize_m * self.core.tile_xsize_m
        lly = y // self.core.tile_ysize_m * self.core.tile_ysize_m
        return llx, lly

    @abc.abstractmethod
    def point2tilename(self, x, y):
        """
        Returns the name string of an Tile() in which the point,
        defined by x and y coordinates (in metres), is located.

        Parameters
        ----------
        x : int
            x (right) coordinate in the desired tile
            must to given together with y
        y : int
            y (up) coordinate in the desired tile
            must to given together with x

        Returns
        -------
        str
            the tilename

        """
        return

    @abc.abstractmethod
    def _encode_tilename(self, llx, lly):
        """
        Encodes a tilename defined by the lower-left coordinates of the tile,
        using inherent information

        Parameters
        ----------
        llx : int
            Lower-left x coordinate.
        lly : int
            Lower-left y coordinate.

        Returns
        -------
        str
            the tilename
        """
        return

    @abc.abstractmethod
    def decode_tilename(self, tilename):
        """
        Returns the information assigned to the tilename

        Parameters
        ----------
        tilename : str
            the tilename

        Returns
        -------
        various
            features of the tiles
        """
        a = None
        return a

    def identify_tiles_overlapping_xybbox(self, bbox, flatten=True):
        """Light-weight routine that returns
           the name of tiles overlapping the bounding box.

        Parameters
        ----------
        bbox : list
            list of projected coordinates limiting the bounding box.
            scheme: [xmin, ymin, xmax, ymax]
        flatten : bool
            should the output be a list, or a 2D-array?
            default is a list

        Return
        ------
        tilenames : list or array
            tilenames overlapping the bounding box
        """

        xmin, ymin, xmax, ymax = [int(round(x)) for x in bbox]
        if (xmin > xmax) or (ymin > ymax):
            raise ValueError("Check order of coordinates of bbox! "
                             "Scheme: [xmin, ymin, xmax, ymax]")

        tsize_x = self.core.tile_xsize_m
        factor_x = tsize_x
        tsize_y = self.core.tile_ysize_m
        factor_y = tsize_y

        llxs = list(
            range(xmin // tsize_x * factor_x, xmax // tsize_x * factor_x + 1,
                  factor_x))
        llys = list(
            reversed(
                range(ymin // tsize_y * factor_y,
                      ymax // tsize_y * factor_y + 1, factor_y)))

        nx = len(llxs)
        ny = len(llys)

        tilenames = np.zeros((ny, nx), dtype=object)

        for x in range(nx):
            for y in range(ny):
                tilenames[y, x] = self._encode_tilename(llxs[x], llys[y])

        if flatten:
            return list(tilenames.flatten())
        else:
            return tilenames

    def create_tiles_overlapping_xybbox(self, bbox):
        """Light-weight routine that returns an 2D-array arranging the
         tiles intersecting the bounding box.

        Parameters
        ----------
        bbox : list of numbers
            list of projected coordinates limiting the bounding box.
            scheme: [xmin, ymin, xmax, ymax]

        Return
        ------
        tiles : array of Tiles()
            Array of Tiles() intersecting the bounding box, arranged in the correct 2D-topology
            With .active_subset_px() holding indices of the tile that cover the bounding box.

        """
        tilenames = self.identify_tiles_overlapping_xybbox(bbox, flatten=False)
        tiles = np.zeros_like(tilenames, dtype=object)

        for x in range(tilenames.shape[0]):
            for y in range(tilenames.shape[1]):

                tile = self.create_tile(name=tilenames[x, y])
                le, be, re, te = tile.active_subset_px
                extent = tile.bbox_proj

                # left_edge
                if extent[0] <= bbox[0]:
                    le = int((bbox[0] - extent[0]) // tile.core.sampling)
                # bottom_edge
                if extent[1] <= bbox[1]:
                    be = int((bbox[1] - extent[1]) // tile.core.sampling)
                # right_edge
                if extent[2] > bbox[2]:
                    re = int((bbox[2] - extent[2] + self.core.tile_xsize_m) //
                             tile.core.sampling)
                # top_edge
                if extent[3] > bbox[3]:
                    te = int((bbox[3] - extent[3] + self.core.tile_ysize_m) //
                             tile.core.sampling)

                # subset holding indices of the tile that cover the bounding box.
                tile.active_subset_px = le, be, re, te

                tiles[x, y] = tile

        return tiles

    @abc.abstractmethod
    def get_congruent_tiles_from_tilename(self,
                                          tilename,
                                          target_sampling=None,
                                          target_tiletype=None):
        """
        finds the "family tiles", which share a congruent or partial overlap,
        but with different sampling and tilecode

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'EU500M_E012N018T6'
            or in shortform e.g. 'E012N018T6'.
        target_sampling : int
            the sampling of the target grid system
        target_tiletype : string
            tilecode string

        Returns
        -------
        list
            list of found tiles
            for smaller tiles: tiles contained in tile with 'tilename'
            for larger tiles: the tile overlap the with 'tilename'

        Notes
        -----
        Either the sampling or tilecode should be given.
        But if both are given, the sampling will be used.
        """

        return []

    def collect_congruent_tiles(self,
                                tiles,
                                target_sampling=None,
                                target_tiletype=None):
        """
        Collects all tiles of other_tile_type covering the list of given tiles.

        Parameters
        ----------
        tiles : list of str
            list of tilenames
        target_sampling : int
            sampling related to target tile type
        target_tiletype : str
            string defining target tile type

        Returns
        -------
        list of str
            list of co-locating tiles of other_tile_type
            e.g. ['E000N054T6', 'E000N060T6']
        """

        cover_tiles = []
        for t in tiles:
            cover_tiles += self.get_congruent_tiles_from_tilename(
                t,
                target_sampling=target_sampling,
                target_tiletype=target_tiletype)

        return list(set(cover_tiles))


class Tile(object):
    """
    A tile in the TiledProjectedSystem, holding characteristics of the tile.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, core, name, xll, yll):
        """
        Initialises a Tile().

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the tile
        name : str
            name of the tile
        llx : int
            lower-left x (right) coordinate of the tile
        lly : int
            lower-left y (up) coordinate of the tile
        """

        self.core = core

        self.name = name
        self.typename = core.tiletype
        self.llx = xll
        self.lly = yll
        self.x_size_px = int(self.core.tile_xsize_m / self.core.sampling)
        self.y_size_px = int(self.core.tile_ysize_m / self.core.sampling)
        self._subset_px = (0, 0, self.x_size_px, self.y_size_px)

        self.polygon_proj = self.get_extent_geometry_proj()
        self.polygon_geog = self.get_extent_geometry_geog()
        self.bbox_proj = ptpgeometry.get_geometry_envelope(
            self.polygon_proj, rounding=self.core.sampling)
        self.bbox_geog = ptpgeometry.get_geometry_envelope(self.polygon_geog,
                                                           rounding=0.000001)

    def __getattr__(self, item):
        """
        short link for items of core
        """
        if item in self.core.__dict__:
            return self.core.__dict__[item]
        else:
            return self.__dict__[item]

    def shape_px(self):
        """
        Returns the shape of the pixel array

        Returns
        -------
        tuple
            shape of the tile's pixel array as (samples_x, samples_y)
        """

        return (self.x_size_px, self.y_size_px)

    def _limits_m(self):
        """
        returns the limits of the tile in projected coordinates (in metres)

        Returns
        -------
        tuple
            limits in the terms of (xmin, ymin, xmax, ymax)
        """

        return (self.llx, self.lly, self.llx + self.core.tile_xsize_m,
                self.lly + self.core.tile_ysize_m)

    def get_extent_geometry_proj(self):
        """
        returns the extent-geometry of the tile in the projected space.

        Returns
        -------
        OGRGeometry

        """
        return ptpgeometry.bbox2polygon(
            (self._limits_m()[0:2], self._limits_m()[2:4]),
            self.core.projection.osr_spref,
            segment=self.x_size_px * self.core.sampling / 4)

    def get_extent_geometry_geog(self):
        """
        returns the extent-geometry of the tile in the lon-lat-space.

        Returns
        -------
        OGRGeometry

        """
        tile_geom = self.polygon_proj

        geo_sr = ptpgeometry.get_geog_spatial_ref()

        return ptpgeometry.transform_geometry(tile_geom, geo_sr, segment=25000)

    @property
    def active_subset_px(self):
        """
        holds indices of the active_subset_px-of-interest

        Returns
        -------
        tuple
            active subset as
            (xmin, ymin, xmax, ymax) =
            (left edge, bottom edge, right edge, top edge)
        """

        return self._subset_px

    @active_subset_px.setter
    def active_subset_px(self, limits):
        """
        changes the indices of the active_subset_px-of-interest,
        mostly to a smaller extent, for efficient reading

        Parameters
        ----------
        limits : tuple
            the limits of subsets as
            (xmin, ymin, xmax, ymax) =
            (left edge, bottom edge, right edge, top edge)

        """

        string = ['xmin', 'ymin', 'xmax', 'ymax']
        if len(limits) != 4:
            raise ValueError('Limits are not properly set!')

        _max = [self.x_size_px, self.y_size_px, self.x_size_px, self.y_size_px]

        for l, limit in enumerate(limits):
            if (limit < 0) or (limit > _max[l]):
                raise ValueError('{} is out of bounds!'.format(string[l]))

        xmin, ymin, xmax, ymax = limits

        if xmin >= xmax:
            raise ValueError('xmin >= xmax!')
        if ymin >= ymax:
            raise ValueError('ymin >= ymax!')

        self._subset_px = limits

    def get_active_subset_px_upperleft(self):
        """
        returns the .active_subset_px after conversion
        to upperleft indices (numpy-style from top to bottom)

        """

        x = self._subset_px
        return tuple(
            (x[0], self.y_size_px - x[1], x[2], self.y_size_px - x[3]))

    def geotransform(self):
        """
        returns the GDAL geotransform list

        Returns
        -------
        list
            a list contain the geotransform elements (no rotation specified)
            as (llx, x pixel spacing, 0, lly, 0, y pixel spacing)
        """

        geot = [
            self.llx, self.core.sampling, 0, self.lly + self.core.tile_ysize_m,
            0, -self.core.sampling
        ]

        return geot

    def geotransform_lowerleft(self):
        """
        returns the geographic geotransform list,
        which is lowerleft-defined

        Returns
        -------
        list
            a list contain the geotransform elements (no rotation specified)
            as (llx, x pixel spacing, 0, lly, 0, y pixel spacing)
        """

        geot = [
            self.llx, self.core.sampling, 0, self.lly, 0, self.core.sampling
        ]

        return geot

    def ij2xy(self, i, j, lowerleft=False, offset='ul'):
        """
        Returns the projected coordinates of a tile pixel in the TilingSystem
        for a given pixel pair defined by column and row (pixel indices)

        By default, the pixels center is returned (this can be changed
        by setting 'offset' to one of 'll', 'lr', 'ul', 'ur')

        columns go from left to right
        rows go either
            top to bottom (lowerleft=False)
            bottom to top (lowerleft=True)

        Parameters
        ----------
        i : number
            pixel column number
        j : number
            pixel row number
        lowerleft : bool, optional
            should the row numbering start at the bottom?
            If yes, it returns lowerleft indices.
        offset : str, optional
            location of the returned coordinates.
            possible values are: ('ll', 'lr', 'ul', 'ur', 'center')
            The default is 'ul'.

        Returns
        -------
        x : number
            x coordinate in the projection
        y : number
            y coordinate in the projection
        """

        if lowerleft:
            gt = self.geotransform_lowerleft()
        else:
            gt = self.geotransform()

        x = gt[0] + i * gt[1] + j * gt[2]
        y = gt[3] + i * gt[4] + j * gt[5]

        assert offset in [
            'll', 'lr', 'ul', 'ur', 'center'
        ], ("offset must be one of ['ll', 'lr', 'ul', 'ur', 'center']")

        if offset == 'center':
            xcenterpos = gt[1] / 2
            ycenterpos = gt[5] / 2
            # use integers if possible
            if xcenterpos.is_integer(): xcenterpos = int(xcenterpos)
            if ycenterpos.is_integer(): ycenterpos = int(ycenterpos)

            x += xcenterpos
            y += ycenterpos

        else:
            if lowerleft:
                if offset == 'ul':
                    y += gt[5]
                if offset == 'ur':
                    y += gt[5]
                    x += gt[1]
                if offset == 'll':
                    pass
                if offset == 'lr':
                    x += gt[1]
            else:
                if offset == 'ul':
                    pass
                if offset == 'ur':
                    x += gt[1]
                if offset == 'll':
                    y += gt[5]
                if offset == 'lr':
                    x += gt[1]
                    y += gt[5]

        if self.core.sampling <= 1.0:
            precision = len(str(int(1.0 / self.core.sampling))) + 1
            return round(x, precision), round(y, precision)
        else:
            return x, y

    def xy2ij(self, x, y, lowerleft=False):
        """
        returns the column and row number (i, j)
        of a projection coordinate pair (x, y)

        columns go from left to right
        rows go either
            top to bottom (lowerleft=False)
            bottom to top (lowerleft=True)

        Parameters
        ----------
        x : number
            x coordinate in the projection
        y : number
            y coordinate in the projection
        lowerleft : bool, optional
            should the row numbering start at the bottom?
            If yes, it returns lowerleft indices.

        Returns
        -------
        i : integer
            pixel column number; starts with 0
        j : integer
            pixel row number; starts with 0
        """

        # get the geotransform
        if lowerleft:
            gt = self.geotransform_lowerleft()
        else:
            gt = self.geotransform()

        # get the indices
        i = (-1.0 * (gt[2] * gt[3] - gt[0] * gt[5] + gt[5] * x - gt[2] * y) /
             (gt[2] * gt[4] - gt[1] * gt[5]))
        j = (-1.0 *
             (-1 * gt[1] * gt[3] + gt[0] * gt[4] - gt[4] * x + gt[1] * y) /
             (gt[2] * gt[4] - gt[1] * gt[5]))

        # round to lower-closest integer
        i = math.floor(i)
        j = math.floor(j)

        return i, j

    def get_geotags(self):
        """
        returns the geotags for given tile used as geo-information for GDAL

        Returns
        -------
        geotags : dict
            dict containing the geotransform and the spatial reference in WKT
            format
        """

        geotags = {
            'geotransform': self.geotransform(),
            'spatialreference': self.core.projection.wkt
        }

        return geotags


class GlobalTile(Tile):

    __metaclass__ = abc.ABCMeta

    def __init__(self, core, name, bbox_polygon_proj):
        """
        Initialising a GlobalTile(), covering the whole extent of the subgrid

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the tile/subgrid
        name : str
            defining the name of the GlobalTile()
        bbox_polygon_proj : tuple
            limits in projection spacve of the subgrid
            as (xmin, xmax, ymin, ymax)
        """

        super(GlobalTile, self).__init__(core, name, 0, 0)
        self.typename = 'TG'
        self.core.tiletype = self.typename
        self.core.tile_xsize_m = np.int(
            (np.floor(bbox_polygon_proj[1] / self.core.sampling) *
             self.core.sampling) -
            (np.ceil(bbox_polygon_proj[0] / self.core.sampling) *
             self.core.sampling))
        self.core.tile_ysize_m = np.int(
            (np.floor(bbox_polygon_proj[3] / self.core.sampling) *
             self.core.sampling) -
            (np.ceil(bbox_polygon_proj[2] / self.core.sampling) *
             self.core.sampling))
        self.x_size_px = int(self.core.tile_xsize_m / self.core.sampling)
        self.y_size_px = int(self.core.tile_ysize_m / self.core.sampling)
        self._subset_px = (0, 0, self.x_size_px, self.y_size_px)
'''

if __name__ == "__main__":
    pass