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


'''
Created on July 12, 2018

Code for the Universal Transverse Mercator / Universal Polar Stereographic
(UTM/UPS) conformal projection system, consisting of 60 + 4 zones.

@author: Bernhard Bauer-Marschallinger, bbm@geo.tuwien.ac.at

'''

import os
import pickle
import copy
import itertools
import warnings

import numpy as np

from pytileproj.base import TiledProjectionSystem
from pytileproj.base import TiledProjection
from pytileproj.base import TPSProjection
from pytileproj.base import TilingSystem
from pytileproj.base import Tile
from pytileproj.geometry import create_geometry_from_wkt


def _load_static_data(module_path):
    """
    load the data, raise the error if failed to load utmgrid.dat

    Parameters
    ----------
    module_path : string
        mainpath of the UTMGrid module

    Returns
    -------
    utm_data : dict
        dictionary containing for each subgrid...
            a) the multipolygon 'zone_extent'
            b) the WKT-string 'projection'

    """
    utm_data = None
    fname = os.path.join(os.path.dirname(module_path), "data", "utm",
                         "utmgrid.dat")
    if not os.path.isfile(fname):
        warnings.warn("utmgrid.dat is not available! UTMGrid won't work!")
        return
    with open(fname, "rb") as f:
        utm_data = pickle.load(f)
    return utm_data


def create_UTM_zone_names():
    """
    small routine to build all the UTM/UPS zone names

    Returns
    -------
    subgrids : list of str
        all acronyms for the 124 zones, e.g. Z17S

    """
    helper = [str(x).zfill(2) for x in range(0, 61)]
    subgrids = ['Z' + x + 'N' for x in helper[1:]] + \
               ['Z' + x + 'S' for x in helper[1:]] + \
               ['Z00A', 'Z00B', 'Z00Y', 'Z00Z']
    subgrids.sort()

    return subgrids


class UTMGrid(TiledProjectionSystem):
    """
    UTMGrid class object, inheriting TiledProjectionSystem() from pytileproj.

    Attributes
    ----------
    _static_data  : dict
        dictionary containing for each subgrid...
            a) the multipolygon 'zone_extent'
            b) the WKT-string 'projection'
    _static_subgrid_ids : list of strings
        lists the acronyms of the 124 (zonal) subgrids
    _static_tilecodes : list of strings
        lists the 3 tile acronyms
    _static_sampling : list of int
        lists all allowed grid samplings
    """

    # static attribute
    _static_data = _load_static_data(__file__)
    # sub grid IDs
    _static_subgrid_ids = create_UTM_zone_names()
    # TODO: supported tile widths (linked to the grid sampling)
    _static_tilecodes = ["T6", "T3", "T1"]
    # supported grid spacing ( = the pixel sampling)
    _static_sampling = [1000, 800, 750, 600, 500, 400, 300, 250, 200,
                        150, 125, 100, 96, 80, 75, 64, 60, 50, 48, 40,
                        32, 30, 25, 24, 20, 16, 10, 8, 5, 4, 2, 1]

    def __init__(self, sampling):
        """
        Initialises an UTMGrid class for a specified sampling.

        Parameters
        ----------
        sampling : int
            the grid sampling = size of pixels; in metres.

        """
        # check if the utmgrid.data have been loaded successfully
        if UTMGrid._static_data is None:
            raise ValueError("cannot load UTMGrid ancillary data!")
        # check if sampling is allowed
        if sampling not in UTMGrid._static_sampling:
            raise ValueError("Sampling {}m is not supported!".format(sampling))

        # initializing
        super(UTMGrid, self).__init__(sampling, tag='UTM')
        self.core.projection = 'multiple'

    @staticmethod
    def encode_sampling(sampling):
        """
        provides a string representing the sampling (e.g. for the tilenames)

        Parameters
        ----------
        sampling : int
            the grid sampling = size of pixels; in metres.

        Returns
        -------
        sampling_str : str
            string representing the sampling

        """
        if sampling <= 999:
            sampling_str = str(sampling).rjust(3, '0')
        if sampling >= 1000:
            sampling_str = "".join(
                (str(sampling / 1000.0)[0], 'K', str(sampling / 1000.0)[2]))
        return sampling_str

    @staticmethod
    def decode_sampling(sampling_str):
        """
        converts the string representing the sampling (e.g. from the tilenames)
        to an integer value in metres

        Parameters
        ----------
        sampling_str : str
            string representing the sampling

        Returns
        -------
        sampling : int
            the grid sampling = size of pixels; in metres.

        """
        if len(sampling_str) != 3:
            raise ValueError('Resolution is badly defined!')
        if sampling_str[1] == 'K':
            sampling = int(sampling_str[0]) * 1000 + int(sampling_str[2]) * 100
        else:
            sampling = int(sampling_str)
        return sampling

    def define_subgrids(self):
        """
        Builds the grid's subgrids from a static file.

        Returns
        -------
        subgrids : dict of UTMSubgrid
            dict of all subgrids of the grid
        """
        subgrids = dict()
        for sg in self._static_subgrid_ids:
            subgrids[sg] = UTMSubgrid(self.core, sg)
        return subgrids

    def get_tiletype(self, sampling=None):
        """
        Returns the tilecode defined for the grid's sampling

        Parameters
        ----------
        sampling : int, optional
            the grid sampling = size of pixels; in metres.

        Returns
        -------
        tilecode : str
            tilecode (related the tile size of the grid)
        """

        # get the tile code of the grid instance
        if sampling is None:
            return self._get_tiletype()

        sampling = int(sampling)

        # allowing sampling of [1000, 800, 750, 600, 500, 400, 300, 250, 200,
        # 150, 125, 100, 96, 80, 75, 64] metres
        if ((sampling in range(64, 1001)) and (600000 % sampling == 0)):
            tilecode = "T6"
        # allowing sampling of [60, 50, 48, 40, 32, 30, 25, 24, 20] metres
        elif ((sampling in range(20, 61)) and (300000 % sampling == 0)):
            tilecode = "T3"
        # allowing sampling of [16, 10, 8, 5, 4, 2, 1] metres
        elif ((sampling in range(1, 17)) and (100000 % sampling == 0)):
            tilecode = "T1"
        else:
            msg = "Error: Given resolution %d is not supported!" % sampling
            msg += " Supported resolutions: {}".format(
                str(UTMGrid._static_sampling))
            raise ValueError(msg)

        return tilecode

    def get_tilesize(self, sampling):
        """
        Return the tile size in metres defined for the grid's sampling

        Parameters
        ----------
        sampling : int
            the grid sampling = size of pixels; in metres.

        Returns
        -------
        xsize, ysize : int
            tile size in x and y direction defined for the grid's sampling

        """
        xsize = {'T6': 600000, 'T3': 300000, 'T1': 100000}[
            self.get_tiletype(sampling)]
        ysize = {'T6': 600000, 'T3': 300000, 'T1': 100000}[
            self.get_tiletype(sampling)]
        return xsize, ysize

    def create_tile(self, name):
        """
        shortcut to create_tile, returning a UTMTile object

        Parameters
        ----------
        name : str
            name of the tile; e.g Z17S500M_E012N018T6

        Returns
        -------
        UTMTile
            object containing info of the specified tile

        """
        return self.subgrids[name[0:2]].tilesys.create_tile(name)

    def lonlat2xy_MGRS(self, lon, lat, subgrid=None):
        """
        converts latitude and longitude coordinates to MGRS grid coordinates

        Parameters
        ----------
        lon : list of numbers
            longitude coordinates
        lat : list of numbers
            latitude coordinates
        subgrid : str
            optional: acronym / subgrid ID to search within (speeding up)
        return_lat_band : bool
            optional: if set to True, the zone code with the latitude letter
            is returned (used e.g. for the MGRS coordinates)

        Returns
        -------
        zone : str
            subgrid ID; the zone code with the latitude band letter
        x, y : list of float
            TPS grid coordinates
        """

        zone, x, y = super(UTMGrid, self).lonlat2xy(lon, lat, subgrid=subgrid)

        vfunc = np.vectorize(self._return_latitude_band)
        zone = vfunc(zone, lon, lat)

        return zone, x, y

    def _return_latitude_band(self, subgrid, lon, lat):
        """
        Returns the UTM zone code with the latitude letter
        used e.g. for the MGRS coordinates
        [33T lon lat] instead of [33N lon lat]

        Parameters
        ----------
        subgrid : str
            acronym / subgrid ID to be translated
        lon : number
            longitude coordinate
        lat : number
            latitude coordinate

        Returns
        -------
        zone : str
            zone identifier with latitudinal band letter
        """

        if lat < -80:
            letter = 'A' if lon < 0 else 'B'
        if lat < 0.0 and lat >= -80.0:
            letter = ['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M'][
                int(np.floor((lat + 80) / 8))]
        if lat >= 0.0 and lat <= 84.0:
            letter = ['N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'X'][
                int(np.floor((lat) / 8))]
        if lat > 84.0:
            letter = 'X' if lon < 0 else 'Z'

        zone_code = subgrid[0:3] + letter

        return zone_code

    def lonlat2ij_in_tile(self, lon, lat, lowerleft=False):
        """
        finds the tile and the pixel indices of a given point in lon-lat-space.
        pixel indices comprise the column and row number (i, j)

        columns go from left to right (easting)
        rows go either
            top to bottom (lowerleft=False)
            bottom to top (lowerleft=True)

        Parameters
        ----------
        lon : number
            longitude coordinate
        lat : number
            latitude coordinate)
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
        # get the xy-coordinates
        subgrid, x, y = self._lonlat2xy(lon, lat)

        tilename, i, j = self.subgrids[str(subgrid)].tilesys.xy2ij_in_tile(
            x, y, lowerleft=lowerleft)

        return tilename, i, j


class UTMSubgrid(TiledProjection):
    """
    UTMSubgrid class object, inheriting TiledProjection() from pytileproj.

    """

    def __init__(self, core, zone):
        """
        Initialises an UTMSubgrid class for a specified continent.

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the (sub-) grid
        zone : str
            acronym of the continent, e.g. '01N' or '17S'.
        """

        # load WKT string and extent shape
        data = UTMGrid._static_data[zone]

        _core = copy.copy(core)
        _core.tag = zone
        _core.projection = TPSProjection(proj4=data['proj4'])

        # holds core parameters of the (sub-) grid
        self.core = _core

        # holds name of the subgrid
        self.name = ''.join(
            ('UTMG_', zone, UTMGrid.encode_sampling(core.sampling), 'M'))

        # holds the extent of the subgrid in the latlon-space
        self.polygon_geog = create_geometry_from_wkt(data['zone_extent'])

        # defines the tilingsystem of the subgrid
        self.tilesys = UTMTilingSystem(self.core, self.polygon_geog)

        super(UTMSubgrid, self).__init__(
            self.core, self.polygon_geog, self.tilesys)


class UTMTilingSystem(TilingSystem):
    """
    UTMTilingSystem class, inheriting TilingSystem() from pytileproj.
    provides methods for queries and handling.
    """

    def __init__(self, core, polygon_geog):
        """
        Initialises an UTMTilingSystem class for a specified continent.

        Parameters
        ----------
        core : TPSCoreProperty
            defines core parameters of the (sub-) grid
        polygon_geog : OGRGeometry
            geometry defining the extent/outline of the subgrid
        """

        super(UTMTilingSystem, self).__init__(core, polygon_geog, 0, 0)

        self.msg1 = '"tilename" is not properly defined! Examples: ' \
                    '"{0}{1:03d}M_E012N036{2}" ' \
                    'or "E012N036{2}"'.format(
                        self.core.tag, self.core.sampling, self.core.tiletype)
        self.msg2 = 'East and North coordinates of lower-left-pixel ' \
                    'must be multiples of {}00km!'.format(
                        self.core.tile_ysize_m // 100000)
        self.msg3 = 'Tilecode must be one of T6, T3, T1!'

    def create_tile(self, name=None, x=None, y=None):
        """
        Returns a UTMTile object

        Parameters
        ----------
        name : str
            name of the tile; e.g Z17S500M_E012N018T6 or E012N018T6
        x : int
            x (right) coordinate of a pixel located in the desired tile
            must to given together with y
        y : int
            y (up) coordinate of a pixel located in the desired tile
            must to given together with x

        Returns
        -------
        UTMTile
            object containing info of the specified tile

        Notes
        -----
        either name, or x and y, must be given.
        """

        # use the x and y coordinates for specifing the tile
        if x is not None and y is not None and name is None:
            llx, lly = self.round_xy2lowerleft(x, y)
        # use the tile name for specifing the tile
        elif name is not None and x is None and y is None:
            llx, lly = self.tilename2lowerleft(name)
        else:
            raise AttributeError('"name" or "x"&"y" must be defined!')

        # get name of tile (assures long-form of tilename, even if short-form
        # is given)
        name = self._encode_tilename(llx, lly)
        # set True if land in the tile
        covers_land = self.check_tile_covers_land(tilename=name)

        return UTMTile(self.core, name, llx, lly, covers_land=covers_land)

    def point2tilename(self, x, y, shortform=False):
        """
        Returns the name string of an UTMTile in which the point,
        defined by x and y coordinates (in metres), is located.

        Parameters
        ----------
        x : int
            x (right) coordinate in the desired tile
            must to given together with y
        y : int
            y (up) coordinate in the desired tile
            must to given together with x
        shortform : Boolean
            option for giving back the shortform of the tile name
            e.g. 'E000N018T6'.

        Returns
        -------
        str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.

        """
        llx, lly = self.round_xy2lowerleft(x, y)
        return self._encode_tilename(llx, lly, shortform=shortform)

    def encode_tilename(self, llx, lly, sampling, tilecode, shortform=False):
        """
        Encodes a tilename

        Parameters
        ----------
        llx : int
            Lower-left x coordinate.
        lly : int
            Lower-left y coordinate.
        sampling : int
            the grid sampling = size of pixels; in metres.
        tilecode : str
            tilecode
        shortform : boolean, optional
            return shortform of tilename (default: False).

        Returns
        -------
        str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.
        """

        # gives long-form of tilename (e.g. "Z17S500M_E012N018T6")
        tilename = "{}{}M_E{:03d}N{:03d}{}".format(
            self.core.tag, UTMGrid.encode_sampling(sampling),
            int(llx) // 100000, int(lly) // 100000, tilecode)

        if shortform:
            tilename = self.tilename2short(tilename)

        return tilename

    def _encode_tilename(self, llx, lly, shortform=False):
        """
        Encodes a tilename defined by the lower-left coordinates of the tile,
        using inherent information

        Parameters
        ----------
        llx : int
            lower-left x coordinate.
        lly : int
            lower-left y coordinate.
        shortform : boolean, optional
            return shortform of tilename (default: False).

        Returns
        -------
        str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.

        """
        return self.encode_tilename(llx, lly, self.core.sampling,
                                    self.core.tiletype, shortform=shortform)

    def tilename2short(self, longform):
        """
        Converts a tilename in longform to shortform
        e.g. 'Z17S500M_E000N018T6' --> 'E000N018T6'

        Parameters
        ----------
        longform : str
            longform of the tilename

        Returns
        -------
        str
            shortform of the tilename

        """
        self.check_tilename(longform)
        if len(longform) == 19:
            shortform = longform[9:]
        return shortform

    def tilename2lowerleft(self, tilename):
        """
        Return the lower-left coordinates of the tile

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.

        Returns
        -------
        llx, lly: int
            lower-left coordinates of the tile
        """
        _, _, _, llx, lly, _ = self.decode_tilename(tilename)
        return llx, lly

    def check_tilename(self, tilename):
        """
        checks if the given tilename is valid

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.

        Returns
        -------
        Boolean
            result of the check
        """

        check = False
        self.decode_tilename(tilename)
        check = True
        return check

    def decode_tilename(self, tilename):
        """
        Returns the information assigned to the tilename

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.

        Returns
        -------
        subgrid_id : str
            ID acronym of the subgrid, e.g. 'Z17S'
        sampling : int
            the grid sampling = size of pixels; in metres.
        tile_size_m : int
            extent/size of the tile; in metres.
        llx : int
            lower-left x coordinate.
        lly : int
            lower-left y coordinate.
        tilecode : str
            tilecode (related the tile size of the grid)
        """
        tf = self.core.tile_ysize_m // 100000

        # allow short-form of tilename (e.g. "E012N018T6")
        if len(tilename) == 10:
            tile_size_m = int(tilename[-1]) * 100000
            if tile_size_m != self.core.tile_xsize_m:
                raise ValueError(self.msg1)
            llx = int(tilename[1:4])
            if llx % tf:
                raise ValueError(self.msg2)
            lly = int(tilename[5:8])
            if lly % tf:
                raise ValueError(self.msg2)
            tilecode = tilename[-2:]
            if tilecode != self.core.tiletype:
                raise ValueError(self.msg1)
            subgrid_id = self.core.tag
            sampling = self.core.sampling

        # allow long-form of tilename (e.g. "Z17S500M_E012N018T6")
        elif len(tilename) == 19:
            subgrid_id = tilename[0:4]
            if subgrid_id != self.core.tag:
                raise ValueError(self.msg1)
            sampling = UTMGrid.decode_sampling(tilename[4:7])
            if sampling != self.core.sampling:
                raise ValueError(self.msg1)
            tile_size_m = int(tilename[-1]) * 100000
            if tile_size_m != self.core.tile_xsize_m:
                raise ValueError(self.msg1)
            llx = int(tilename[10:13])
            if llx % tf:
                raise ValueError(self.msg2)
            lly = int(tilename[14:17])
            if lly % tf:
                raise ValueError(self.msg2)
            tilecode = tilename[-2:]
            if tilecode != self.core.tiletype:
                raise ValueError(self.msg1)

        # wrong length
        else:
            raise ValueError(self.msg1)

        return subgrid_id, sampling, tile_size_m, llx * 100000, lly * 100000, tilecode

    def get_congruent_tiles_from_tilename(self, tilename,
                                          target_sampling=None,
                                          target_tiletype=None):
        """
        finds the "family tiles", which share a congruent or partial overlap,
        but with different resolution and tilecode

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'
            or in shortform e.g. 'E000N018T6'.
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

        # return tilenames in shortform or longform?
        if target_sampling is None:
            shortform = True
        else:
            shortform = False

        # check search input
        if target_sampling is not None and target_tiletype is None:
            sampling = target_sampling
        if target_sampling is None and target_tiletype is not None:
            if target_tiletype == 'T1':
                sampling = 10
            elif target_tiletype == 'T3':
                sampling = 20
            elif target_tiletype == 'T6':
                sampling = 500
            else:
                raise ValueError(self.msg3)
        if target_sampling is not None and target_tiletype is not None:
            sampling = target_sampling

        # grid of the searched tiles
        target_grid = UTMGrid(sampling=sampling)
        target_tiletype = target_grid.core.tiletype
        target_tilesize = target_grid.core.tile_xsize_m

        # features of the input tile(name)
        _, src_sampling, src_tile_size_m, src_llx, src_lly, src_tiletype = \
            self.decode_tilename(tilename)

        # overlapping tiles
        family_tiles = list()

        # for larger tiles
        if target_tiletype >= src_tiletype:
            t_east = (src_llx // target_tilesize) * target_tilesize
            t_north = (src_lly // target_tilesize) * target_tilesize
            name = target_grid.subgrids[self.core.tag]. \
                tilesys.encode_tilename(t_east, t_north,
                                        sampling,
                                        target_tiletype,
                                        shortform=shortform)
            family_tiles.append(name)

        # for smaller tiles
        else:
            n = int(src_tile_size_m // target_tilesize)
            for x, y in itertools.product(range(n), range(n)):
                s_east = (src_llx + x * target_tilesize)
                s_north = (src_lly + y * target_tilesize)
                name = target_grid.subgrids[self.core.tag]. \
                    tilesys.encode_tilename(s_east, s_north,
                                            sampling,
                                            target_tiletype,
                                            shortform=shortform)
                family_tiles.append(name)

        return family_tiles

    def check_tile_covers_land(self, tilename=None):
        """
        checks if a tile covers land

        Parameters
        ----------
        tilename : str
            the tilename in longform e.g. 'Z17S500M_E000N018T6'

        Returns
        -------
        Boolean
        """

        """
        land_tiles = self.list_tiles_covering_land()
        if self.check_tilename(tilename):
            tilename = self.tilename2short(tilename)
            return tilename in land_tiles
        """
        # TODO check_tile_covers_land() not implemented!"
        return True

    def list_tiles_covering_land(self):
        """
        Returns a list of all tiles in the subgrid covering land

        Returns
        -------
        list
            list containing land tiles
        """

        """
        land_tiles = UTMGrid._static_data[
            self.core.tag]["coverland"][self.core.tiletype]
        return list(land_tiles)
        """
        # TODO list_tiles_covering_land() not implemented!"
        return True


class UTMTile(Tile):
    """
    The UTMTile class, inheriting Tile() from pytileproj.

    A tile in the UTMGrid system, holding characteristics of the tile.
    """

    def __init__(self, core, name, xll, yll, covers_land):
        super(UTMTile, self).__init__(core, name, xll, yll)
        self.covers_land = covers_land

    @property
    def shortname(self):
        return self.name[7:]
