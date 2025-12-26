import math

import numpy as np
import pyproj
import pytest
import shapely

from pytileproj._const import GEOG_EPSG
from pytileproj.projgeom import (
    ProjGeom,
    rasterise_polygon,
    split_polygon_by_antimeridian,
)


@pytest.fixture
def poly_siberia_alaska() -> ProjGeom:
    points = [
        (177.6545884597184, 67.05574774066811),
        (179.0195867605756, 65.33232820668778),
        (198.4723636216472, 66.06909015550372),
        (198.7828129097253, 68.14247939909886),
    ]
    return ProjGeom(geom=shapely.Polygon(points), crs=pyproj.CRS.from_epsg(GEOG_EPSG))


@pytest.fixture
def poly_spitzbergen() -> ProjGeom:
    points = [
        (8.391827331539572, 77.35762113396143),
        (16.87007957357446, 81.59290885863483),
        (40.50119498304080, 79.73786853853339),
        (25.43098663332705, 75.61353436967198),
    ]
    return ProjGeom(geom=shapely.Polygon(points), crs=pyproj.CRS.from_epsg(GEOG_EPSG))


def test_split_polygon_by_am_siberia_alaska(poly_siberia_alaska: ProjGeom):
    result = split_polygon_by_antimeridian(poly_siberia_alaska, great_circle=False)

    assert math.isclose(
        poly_siberia_alaska.geom.area,
        result.geom.geoms[0].area + result.geom.geoms[1].area,
        rel_tol=1e-6,
    )


def test_split_polygon_by_am_spitzbergen(poly_spitzbergen: ProjGeom):
    result = split_polygon_by_antimeridian(poly_spitzbergen)

    assert math.isclose(
        poly_spitzbergen.geom.area,
        result.geom.area,
        rel_tol=1e-6,
    )


def test_rasterise_polygon():
    ref_raster = np.array(
        [
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    ref_raster = np.array(ref_raster)
    poly_pts = [(1, 1), (1, 4), (5, 8), (6, 8), (6, 5), (8, 3), (6, 1), (1, 1)]
    geom = shapely.Polygon(poly_pts)
    raster = rasterise_polygon(geom, 1, 1)
    assert np.all(raster == ref_raster)

    ref_raster = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    ref_raster = np.array(ref_raster)
    poly_pts = [(1, 1), (1, 7), (5, 3), (8, 6), (8, 1), (1, 1)]
    geom = shapely.Polygon(poly_pts)
    raster = rasterise_polygon(geom, 1, 1)
    assert np.all(raster == ref_raster)


if __name__ == "__main__":
    pass
