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

import os
import argparse
import pickle

from osgeo import osr

from pytileproj import geometry
from pytileproj.utmgrid import create_UTM_zone_names


def make_utmdata(outpath, version="V10"):
    """ Make the utmgrid.dat file

    Parameters
    ----------
    outpath : string
        output file directory path.

    Returns
    -------
    int
        0 if succeeded, otherwise error code

    Notes
    -----
    utmgrid.dat is a dictionary including necessary information required by
    utmgrid.py class in the following structure.
    { ...
      "14S": { "projection": "projection in wkt format",
               "zone_extent": "zone geometry of subgrid with ID=14S
                                in wkt format" }
      "15N": { ... }
    }

    """

    outfile = os.path.join(outpath, "utmgrid.dat")
    if os.path.exists(outfile):
        raise IOError("Error: File Already Exist!")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    utm_data = dict()

    subgrids = create_UTM_zone_names()

    for subgrid in subgrids:
        subgrid_data = dict()

        zone_extent = load_zone_extent(subgrid)
        subgrid_data["zone_extent"] = zone_extent.ExportToWkt()

        str_proj4 = load_spatial_reference(subgrid)
        subgrid_data["proj4"] = str_proj4

        utm_data[subgrid] = subgrid_data

    # Serialize utm_data by pickle with protocal=2
    with open(outfile, "wb") as f:
        pickle.dump(utm_data, f, protocol=2)

    return 0


def load_zone_extent(subgrid):

    module_path = os.path.dirname(os.path.abspath(__file__))
    shape_file = os.path.join(os.path.dirname(module_path), "utm", "grids",
                              subgrid, 'GEOG',
                              'UTM_UPS_V10_{}_GEOG_ZONE.shp'.format(subgrid))

    return geometry.open_geometry(shape_file)


def load_spatial_reference(subgrid):
    # EPSG codes for WGS84-datum zones
    # N : 326##
    # S : 327##
    if subgrid[3] == 'S':
        epsg = '327' + subgrid[1:3]
    if subgrid[3] == 'N':
        epsg = '326' + subgrid[1:3]
    if subgrid[3] in ['A', 'B']:
        epsg = '32761'
    if subgrid[3] in ['Y', 'Z']:
        epsg = '32661'
    spref = osr.SpatialReference()
    spref.ImportFromEPSG(int(epsg))
    str_proj4 = spref.ExportToProj4()
    return str_proj4


def load_coverland_tiles(tile_fpath):
    tiles_coversland = None

    return tiles_coversland


def main():
    parser = argparse.ArgumentParser(description='Make UTMgrid Data File')
    parser.add_argument("outpath", help="output folder")
    parser.add_argument("-v",
                        "--version",
                        dest="version",
                        nargs=1,
                        metavar="",
                        help="UTM Grid Version. Default is V10.")
    args = parser.parse_args()

    outpath = os.path.abspath(args.outpath)
    version = args.version[0] if args.version else "V10"
    return make_utmdata(outpath, version)


if __name__ == "__main__":
    import sys
    sys.argv.append(r"C:\code\TPS\pytileproj\pytileproj\data\utm")
    main()
