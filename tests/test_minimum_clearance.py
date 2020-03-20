"""
Tests for the minimum clearance property.
"""

import math

import unittest

from shapely.wkt import loads as load_wkt
from shapely.geos import geos_version


@unittest.skipIf(geos_version < (3, 9, 0),
                 "GEOS > 3.6.0 is required.")
class TestMinimumClearance(unittest.TestCase):

    def test_point(self):
        point = load_wkt("POINT (0 0)")
        self.assertEqual(math.inf, point.minimum_clearance)

    def test_linestring(self):
        line = load_wkt('LINESTRING (0 0, 1 1, 2 2)')
        self.assertEqual(1.414214, round(line.minimum_clearance, 6))

    def test_simple_polygon(self):
        poly = load_wkt('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')
        self.assertEqual(1.0, poly.minimum_clearance)

    def test_more_complicated_polygon(self):
        poly = load_wkt('POLYGON ((20 20, 34 124, 70 140, 130 130, 70 100, 110 70, 170 20, 90 10, 20 20))')
        self.assertEqual(35.777088, round(poly.minimum_clearance, 6))
