from . import unittest
from shapely.geometry import Point, Polygon, MultiPoint, GeometryCollection
from shapely.wkt import loads


class OperationsTestCase(unittest.TestCase):

    def test_operations(self):
        point = Point(0.0, 0.0)

        # General geometry
        self.assertEqual(point.area, 0.0)
        self.assertEqual(point.length, 0.0)
        self.assertAlmostEqual(point.distance(Point(-1.0, -1.0)),
                               1.4142135623730951)

        # Topology operations

        # Envelope
        self.assertIsInstance(point.envelope, Point)

        # Intersection
        self.assertIsInstance(point.intersection(Point(-1, -1)),
                              GeometryCollection)

        # Buffer
        self.assertIsInstance(point.buffer(10.0), Polygon)
        self.assertIsInstance(point.buffer(10.0, 32), Polygon)

        # Simplify
        p = loads('POLYGON ((120 120, 121 121, 122 122, 220 120, 180 199, '
                  '160 200, 140 199, 120 120))')
        expected = loads('POLYGON ((120 120, 140 199, 160 200, 180 199, '
                         '220 120, 120 120))')
        s = p.simplify(10.0, preserve_topology=False)
        self.assertTrue(s.equals_exact(expected, 0.001))

        p = loads('POLYGON ((80 200, 240 200, 240 60, 80 60, 80 200),'
                  '(120 120, 220 120, 180 199, 160 200, 140 199, 120 120))')
        expected = loads(
            'POLYGON ((80 200, 240 200, 240 60, 80 60, 80 200),'
            '(120 120, 220 120, 180 199, 160 200, 140 199, 120 120))')
        s = p.simplify(10.0, preserve_topology=True)
        self.assertTrue(s.equals_exact(expected, 0.001))

        # Convex Hull
        self.assertIsInstance(point.convex_hull, Point)

        # Differences
        self.assertIsInstance(point.difference(Point(-1, 1)), Point)

        self.assertIsInstance(point.symmetric_difference(Point(-1, 1)),
                              MultiPoint)

        # Boundary
        self.assertIsInstance(point.boundary, GeometryCollection)

        # Union
        self.assertIsInstance(point.union(Point(-1, 1)), MultiPoint)

        self.assertIsInstance(point.representative_point(), Point)

        self.assertIsInstance(point.centroid, Point)

        # Relate
        self.assertEqual(point.relate(Point(-1, -1)), 'FF0FFF0F2')


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(OperationsTestCase)
