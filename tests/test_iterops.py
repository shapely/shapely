"""Test operator iterations
"""
from . import unittest
from shapely import iterops
from shapely.geometry import Point, Polygon


class IterOpsTestCase(unittest.TestCase):

    def test_iterops(self):

        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        polygon = Polygon(coords)
        points = [Point(0.5, 0.5), Point(2.0, 2.0)]

        # List of the points contained by the polygon
        self.assertTrue(
            all([isinstance(x, Point)
                 for x in iterops.contains(polygon, points, True)]))

        # 'True' is the default value
        self.assertTrue(
            all([isinstance(x, Point)
                 for x in iterops.contains(polygon, points)]))

        # Test a false value
        self.assertTrue(
            all([isinstance(x, Point)
                 for x in iterops.contains(polygon, points, False)]))

        # If the provided iterator yields tuples, the second value will be
        # yielded
        self.assertEqual(
            list(iterops.contains(polygon, [(p, p.coords[:])
                 for p in points], False)),
            [[(2.0, 2.0)]])

        # Just to demonstrate that the important thing is that the second
        # parameter is an iterator:
        self.assertEqual(
            list(iterops.contains(polygon, iter((p, p.coords[:])
                 for p in points))),
            [[(0.5, 0.5)]])


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(IterOpsTestCase)
