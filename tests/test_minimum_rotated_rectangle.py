from . import unittest
from shapely import geometry

class MinimumRotatedRectangleTestCase(unittest.TestCase):

    def test_minimum_rectangle(self):
        poly = geometry.Polygon([(0,1), (1, 2), (2, 1), (1, 0), (0, 1)])
        rect = poly.minimum_rotated_rectangle
        self.assertIsInstance(rect, geometry.Polygon)
        self.assertEqual(rect.area - poly.area < 0.1, True)
        self.assertEqual(len(rect.exterior.coords), 5)

        ls = geometry.LineString([(0,1), (1, 2), (2, 1), (1, 0)])
        rect = ls.minimum_rotated_rectangle
        self.assertIsInstance(rect, geometry.Polygon)
        self.assertEqual(rect.area - ls.convex_hull.area < 0.1, True)
        self.assertEqual(len(rect.exterior.coords), 5)

    def test_degenerate(self):
        # point
        rect1 = geometry.Point((0,1)).minimum_rotated_rectangle
        self.assertIsInstance(rect1, geometry.Point)
        self.assertEqual(len(rect1.coords), 1)
        self.assertEqual(rect1.coords[0], (0,1))

        # invalid linestring: point
        rect2 = geometry.LineString([(0,1), (0,1), (0,1)]).minimum_rotated_rectangle
        self.assertIsInstance(rect2, geometry.Point)
        self.assertEqual(len(rect2.coords), 1)
        self.assertEqual(rect2.coords[0], (0,1))

        # linestring
        rect3 = geometry.LineString([(0,0), (2,2)]).minimum_rotated_rectangle
        self.assertIsInstance(rect3, geometry.LineString)
        self.assertEqual(len(rect3.coords), 2)
        self.assertSetEqual(set(rect3.coords), set([(0,0), (2,2)]))

        # complex linestring
        rect4 = geometry.LineString([(2,2), (0,0), (-2,-2)]).minimum_rotated_rectangle
        self.assertIsInstance(rect4, geometry.LineString)
        self.assertEqual(len(rect4.coords), 2)
        self.assertSetEqual(set(rect4.coords), set([(2,2), (-2,-2)]))
