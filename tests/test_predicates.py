"""Test GEOS predicates
"""
from . import unittest
from shapely.geometry import Point, Polygon
from shapely.geos import TopologicalError


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
        self.assertTrue(point.covers(Point(0.0, 0.0)))
        self.assertFalse(point.covers(Point(-1.0, -1.0)))

    def test_unary_predicates(self):

        point = Point(0.0, 0.0)

        self.assertFalse(point.is_empty)
        self.assertTrue(point.is_valid)
        self.assertTrue(point.is_simple)
        self.assertFalse(point.is_ring)
        self.assertFalse(point.has_z)

    def test_binary_predicate_exceptions(self):

        p1 = [(339, 346), (459,346), (399,311), (340, 277), (399, 173),
              (280, 242), (339, 415), (280, 381), (460, 207), (339, 346)]
        p2 = [(339, 207), (280, 311), (460, 138), (399, 242), (459, 277),
              (459, 415), (399, 381), (519, 311), (520, 242), (519, 173),
              (399, 450), (339, 207)]
        self.assertRaises(TopologicalError, Polygon(p1).within, Polygon(p2))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PredicatesTestCase)
