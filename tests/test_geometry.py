# Copyright (c) 2019,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Tests for the geometry module of pytielproj.
"""
import unittest

from pytileproj.geom import split_polygon_by_antimeridian
from pytileproj.geom import setup_test_geom_siberia_alaska
from pytileproj.geom import setup_test_geom_spitzbergen


class TestGeometry(unittest.TestCase):

    def test_split_polygon_by_antimeridian(self):

        # intersect with antimeridian
        poly_siberia_alaska = setup_test_geom_siberia_alaska()

        result = split_polygon_by_antimeridian(poly_siberia_alaska)

        self.assertAlmostEqual(poly_siberia_alaska.Area() * 2,
                               result.GetGeometryRef(0).Area() +
                               result.GetGeometryRef(1).Area() +
                               result.Area(),
                               places=6)

        # no intersect with antimeridian
        geom_spitzbergen = setup_test_geom_spitzbergen()

        result = split_polygon_by_antimeridian(geom_spitzbergen)

        self.assertAlmostEqual(geom_spitzbergen.Area() * 2,
                               result.GetGeometryRef(0).Area() +
                               result.Area(),
                               places=6)