from . import unittest
from shapely.geometry import Point


class OperatorsTestCase(unittest.TestCase):

    def test_point(self):
        point = Point(0, 0)
        point2 = Point(-1, 1)
        self.assertTrue(point.union(point2).equals(point | point2))
        self.assertTrue((point & point2).is_empty)
        self.assertTrue(point.equals(point - point2))
        self.assertTrue(
            point.symmetric_difference(point2).equals(point ^ point2))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(OperatorsTestCase)
