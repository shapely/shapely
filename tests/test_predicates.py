"""Test GEOS predicates
"""
from . import unittest
from shapely.geometry import Point


class PredicatesTestCase(unittest.TestCase):

    def test_binary_predicates(self):

        point = Point(0.0, 0.0)

        self.assertTrue(point.disjoint(Point(-1.0, -1.0)))
        self.assertFalse(point.touches(Point(-1.0, -1.0)))
        self.assertFalse(point.crosses(Point(-1.0, -1.0)))
        self.assertFalse(point.within(Point(-1.0, -1.0)))
        self.assertFalse(point.contains(Point(-1.0, -1.0)))
        self.assertFalse(point.equals(Point(-1.0, -1.0)))
        self.assertFalse(point.touches(Point(-1.0, -1.0)))
        self.assertTrue(point.equals(Point(0.0, 0.0)))

    def test_unary_predicates(self):

        point = Point(0.0, 0.0)

        self.assertFalse(point.is_empty)
        self.assertTrue(point.is_valid)
        self.assertTrue(point.is_simple)
        self.assertFalse(point.is_ring)
        self.assertFalse(point.has_z)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PredicatesTestCase)
