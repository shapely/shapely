from . import unittest
from shapely.geos import geos_version
from shapely import prepared
from shapely import geometry


class PreparedGeometryTestCase(unittest.TestCase):

    @unittest.skipIf(geos_version < (3, 1, 0), 'GEOS 3.1.0 required')
    def test_prepared(self):
        polygon = geometry.Polygon([
            (0, 0), (1, 0), (1, 1), (0, 1)
        ])
        p = prepared.PreparedGeometry(polygon)
        self.assertTrue(p.contains(geometry.Point(0.5, 0.5)))
        self.assertFalse(p.contains(geometry.Point(0.5, 1.5)))

    @unittest.skipIf(geos_version < (3, 1, 0), 'GEOS 3.1.0 required')
    def test_op_not_allowed(self):
        p = prepared.PreparedGeometry(geometry.Point(0.0, 0.0).buffer(1.0))
        self.assertRaises(ValueError, geometry.Point(0.0, 0.0).union, p)

    @unittest.skipIf(geos_version < (3, 1, 0), 'GEOS 3.1.0 required')
    def test_predicate_not_allowed(self):
        p = prepared.PreparedGeometry(geometry.Point(0.0, 0.0).buffer(1.0))
        self.assertRaises(ValueError, geometry.Point(0.0, 0.0).contains, p)

    @unittest.skipIf(geos_version < (3, 1, 0), 'GEOS 3.1.0 required')
    def test_prepared_predicates(self):
        # check prepared predicates give the same result as regular predicates
        polygon1 = geometry.Polygon([
            (0, 0), (0, 1), (1, 1), (1, 0), (0, 0)
        ])
        polygon2 = geometry.Polygon([
            (0.5, 0.5), (1.5, 0.5), (1.0, 1.0), (0.5, 0.5)
        ])
        point2 = geometry.Point(0.5, 0.5)
        polygon_empty = geometry.Polygon()
        prepared_polygon1 = prepared.PreparedGeometry(polygon1)
        for geom2 in (polygon2, point2, polygon_empty):
            self.assertTrue(polygon1.disjoint(geom2) == prepared_polygon1.disjoint(geom2))
            self.assertTrue(polygon1.touches(geom2) == prepared_polygon1.touches(geom2))
            self.assertTrue(polygon1.intersects(geom2) == prepared_polygon1.intersects(geom2))
            self.assertTrue(polygon1.crosses(geom2) == prepared_polygon1.crosses(geom2))
            self.assertTrue(polygon1.within(geom2) == prepared_polygon1.within(geom2))
            self.assertTrue(polygon1.contains(geom2) == prepared_polygon1.contains(geom2))
            self.assertTrue(polygon1.overlaps(geom2) == prepared_polygon1.overlaps(geom2))

def test_suite():
    loader = unittest.TestLoader()
    return loader.loadTestsFromTestCase(PreparedGeometryTestCase)
