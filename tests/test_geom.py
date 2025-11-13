import math

import pytest
import shapely
from osgeo import ogr

from pytileproj.geom import get_geog_sref, split_polygon_by_antimeridian


@pytest.fixture
def poly_siberia_alaska() -> ogr.Geometry:
    points = [
        (177.6545884597184, 67.05574774066811),
        (179.0195867605756, 65.33232820668778),
        (198.4723636216472, 66.06909015550372),
        (198.7828129097253, 68.14247939909886),
    ]
    poly = ogr.CreateGeometryFromWkt(shapely.Polygon(points).wkt)
    poly.AssignSpatialReference(get_geog_sref())

    return poly


@pytest.fixture
def poly_spitzbergen() -> ogr.Geometry:
    points = [
        (8.391827331539572, 77.35762113396143),
        (16.87007957357446, 81.59290885863483),
        (40.50119498304080, 79.73786853853339),
        (25.43098663332705, 75.61353436967198),
    ]
    poly = ogr.CreateGeometryFromWkt(shapely.Polygon(points).wkt)
    poly.AssignSpatialReference(get_geog_sref())

    return poly


def test_split_polygon_by_am_siberia_alaska(poly_siberia_alaska):
    result = split_polygon_by_antimeridian(poly_siberia_alaska)

    assert math.isclose(
        poly_siberia_alaska.Area() * 2,
        result.GetGeometryRef(0).Area()
        + result.GetGeometryRef(1).Area()
        + result.Area(),
        rel_tol=1e-6,
    )


def test_split_polygon_by_am_spitzbergen(poly_spitzbergen):
    result = split_polygon_by_antimeridian(poly_spitzbergen)

    assert math.isclose(
        poly_spitzbergen.Area() * 2,
        result.GetGeometryRef(0).Area() + result.Area(),
        rel_tol=1e-6,
    )
