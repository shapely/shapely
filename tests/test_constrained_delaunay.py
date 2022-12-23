import unittest

from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import constrained_triangulate
from shapely.geos import geos_version
import pytest


@pytest.mark.skipif(geos_version < (3, 10, 0), reason="GEOS < 3.10")
class ConstrainedDelaunayTriangulation(unittest.TestCase):
    """
    Only testing the number of triangles and their type here.
    This doesn't actually test the points in the resulting geometries.

    """
    def test_poly(self):
        polys = constrained_triangulate(Polygon([(10, 10), (20, 40), (90, 90), (90, 10), (10, 10)]))
        self.assertEqual(len(polys), 2)
        for p in polys:
            self.assertIsInstance(p, Polygon)

    def test_multi_polygon(self):
        multipoly = MultiPolygon([
            Polygon(((50, 30), (60, 30), (100, 100), (50, 30))),
            Polygon(((10, 10), (20, 40), (90, 90), (90, 10), (10, 10))),
        ])
        polys = constrained_triangulate(multipoly)
        self.assertEqual(len(polys), 3)
        for p in polys:
            self.assertIsInstance(p, Polygon)

    def test_point(self):
        p = Point(1, 1)
        polys = constrained_triangulate(p)
        self.assertEqual(len(polys), 0)

    def test_empty_poly(self):
        polys = constrained_triangulate(Polygon())
        self.assertEqual(len(polys), 0)
