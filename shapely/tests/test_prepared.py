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


def test_suite():
    loader = unittest.TestLoader()
    return loader.loadTestsFromTestCase(PreparedGeometryTestCase)
