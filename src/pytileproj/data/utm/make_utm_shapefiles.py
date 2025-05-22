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
A script for creating a shapefile that contains all zones of the UTM/UPS grid
system.
"""

import os
import shutil

import numpy as np
from osgeo import ogr

from pytileproj import geometry

# from https://gis.ucla.edu/geodata/dataset/world_utm_zones/resource/
# fc8af8e2-0818-4f50-baf0-2cc04cbaa541
infile = r"D:\Arbeit\bfix\gis\utm_ups_zones\utmzone.shp"

driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open(infile, 0)
layer = ds.GetLayer(0)
srs = layer.GetSpatialRef()
n_features = layer.GetFeatureCount()

gridpath = r"D:\Arbeit\bfix\gis\utm_ups_zones\grids"
shapepath = r"D:\Arbeit\bfix\gis\utm_ups_zones\grids\000_grids_combined"

if os.path.exists(shapepath):
    shutil.rmtree(shapepath)
if os.path.exists(gridpath):
    shutil.rmtree(gridpath)

os.mkdir(gridpath)
os.mkdir(shapepath)

drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(
    os.path.join(shapepath, 'UTM_UPS_V10_ALL_ZONES_GEOG_ZONE.shp'))
dst_layer = dst_ds.CreateLayer("out", srs=srs)
fd = ogr.FieldDefn('ZONE', ogr.OFTInteger)
dst_layer.CreateField(fd)
fd = ogr.FieldDefn('WEST_LON', ogr.OFTInteger)
dst_layer.CreateField(fd)
fd = ogr.FieldDefn('CM_LON', ogr.OFTInteger)
dst_layer.CreateField(fd)
fd = ogr.FieldDefn('EAST_LON', ogr.OFTInteger)
dst_layer.CreateField(fd)
fd = ogr.FieldDefn('ROW', ogr.OFTString)
dst_layer.CreateField(fd)

all_features = dict()

ns = range(n_features)
geometries = []
zones = []
wests = []
cms = []
easts = []
rows = []

for n in ns:

    feature = layer.GetFeature(n)
    zone = int(feature.GetField('ZONE'))
    west = feature.GetField('WEST_VALUE')
    cm = feature.GetField('CM_VALUE')
    east = feature.GetField('EAST_VALUE')
    row = feature.GetField('ROW_')

    nice_geometry = geometry.round_vertices_of_polygon(
        feature.geometry().Clone())
    geometries.append(nice_geometry)
    zones.append(zone)
    wests.append(west)
    cms.append(cm)
    easts.append(east)
    rows.append(row)

    feature = None

all_zones = np.unique(zones)

polar_rows = ['A', 'B', 'Y', 'Z']
south_rows = ['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M']
north_rows = ['N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']

for zone_number in all_zones:
    if zone_number == 0:
        polar_wests = [-180, 0, -180, 0]
        for l, letter in enumerate(polar_rows):
            ind = list(np.where(np.array(rows) == letter)[0])

            geometries_in_zone = np.array(geometries)[ind]

            geometries_merged = geometries_in_zone[0].Clone()

            # modify the geometry such it has no segment longer then the given distance
            geometries_merged = geometry.segmentize_geometry(geometries_merged,
                                                             segment=0.5)

            xfeature = ogr.Feature(dst_layer.GetLayerDefn())

            xfeature.SetField('ZONE', zone_number)
            xfeature.SetField('WEST_LON', polar_wests[l])
            xfeature.SetField('CM_LON', 0)
            xfeature.SetField('EAST_LON', polar_wests[l] + 180)
            xfeature.SetField('ROW', letter)

            xfeature.SetGeometry(geometries_merged)
            dst_layer.CreateFeature(xfeature)

            geometries_merged.AssignSpatialReference(dst_layer.GetSpatialRef())
            zone_string = 'Z' + str(zone_number).zfill(2) + letter
            xpath = os.path.join(gridpath, zone_string, 'GEOG')
            if not os.path.exists(xpath):
                os.makedirs(xpath)
            filename = os.path.join(
                xpath, 'UTM_UPS_V10_{}_GEOG_ZONE.shp'.format(zone_string))
            geometry.write_geometry(geometries_merged, filename)

            xfeature = None

    else:
        ind = np.where(zones == zone_number)[0]

        west_lon = np.array(wests)[ind][-1]
        cm_lon = np.array(cms)[ind][-1]
        east_lon = np.array(easts)[ind][-1]

        if '0E/W' in west_lon:
            west = 0
        else:
            west = int(west_lon[:-1])
            if 'W' in west_lon:
                west *= -1
        if '0E/W' in cm_lon:
            cm = 0
        else:
            cm = int(cm_lon[:-1])
            if 'W' in cm_lon:
                cm *= -1
        if '0E/W' in east_lon:
            east = 0
        else:
            east = int(east_lon[:-1])
            if 'W' in east_lon:
                east *= -1

        si = [
            i for i in range(len(ind)) if np.array(rows)[ind][i] in south_rows
        ]
        geometries_in_south_zone = np.array(geometries)[ind][si]
        n_si = len(si)

        geometries_merged_south = geometries_in_south_zone[0].Clone()
        for f in range(n_si - 1):
            geometries_merged_south = geometries_merged_south.Union(
                geometries_in_south_zone[f + 1])

        ni = [
            i for i in range(len(ind)) if np.array(rows)[ind][i] in north_rows
        ]
        geometries_in_north_zone = np.array(geometries)[ind][ni]
        n_ni = len(ni)

        geometries_merged_north = geometries_in_north_zone[0].Clone()
        for f in range(n_ni - 1):
            geometries_merged_north = geometries_merged_north.Union(
                geometries_in_north_zone[f + 1])

        # modify the geometry such it has no segment longer than the given
        # distance
        geometries_merged_south = geometry.segmentize_geometry(
            geometries_merged_south, segment=0.5)
        xfeature = ogr.Feature(dst_layer.GetLayerDefn())

        xfeature.SetField('ZONE', zone_number)
        xfeature.SetField('WEST_LON', west)
        xfeature.SetField('CM_LON', cm)
        xfeature.SetField('EAST_LON', east)
        xfeature.SetField('ROW', 'C-M')

        xfeature.SetGeometry(geometries_merged_south)
        dst_layer.CreateFeature(xfeature)

        geometries_merged_south.AssignSpatialReference(
            dst_layer.GetSpatialRef())
        zone_string = 'Z' + str(zone_number).zfill(2) + 'S'
        xpath = os.path.join(gridpath, zone_string, 'GEOG')
        if not os.path.exists(xpath):
            os.makedirs(xpath)
        filename = os.path.join(
            xpath, 'UTM_UPS_V10_{}_GEOG_ZONE.shp'.format(zone_string))
        geometry.write_geometry(geometries_merged_south, filename)

        xfeature = None

        # modify the geometry such it has no segment longer than the given
        # distance
        geometries_merged_north = geometry.segmentize_geometry(
            geometries_merged_north, segment=0.5)
        xfeature = ogr.Feature(dst_layer.GetLayerDefn())

        xfeature.SetField('ZONE', zone_number)
        xfeature.SetField('WEST_LON', west)
        xfeature.SetField('CM_LON', cm)
        xfeature.SetField('EAST_LON', east)
        xfeature.SetField('ROW', 'N-X')

        xfeature.SetGeometry(geometries_merged_north)
        dst_layer.CreateFeature(xfeature)

        geometries_merged_north.AssignSpatialReference(
            dst_layer.GetSpatialRef())
        zone_string = 'Z' + str(zone_number).zfill(2) + 'N'
        xpath = os.path.join(gridpath, zone_string, 'GEOG')
        if not os.path.exists(xpath):
            os.makedirs(xpath)
        filename = os.path.join(
            xpath, 'UTM_UPS_V10_{}_GEOG_ZONE.shp'.format(zone_string))
        geometry.write_geometry(geometries_merged_north, filename)

        xfeature = None

dst_ds = None
