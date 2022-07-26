"""Test recovery from operation on invalid geometries
"""

from . import unittest
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

import shapely


class InvalidGeometriesTestCase(unittest.TestCase):
    def test_invalid_intersection(self):
        # Make a self-intersecting polygon
        polygon_invalid = Polygon([(0, 0), (1, 1), (1, -1), (0, 1), (0, 0)])
        self.assertFalse(polygon_invalid.is_valid)

        # Intersect with a valid polygon
        polygon = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -5)])
        self.assertTrue(polygon.is_valid)
        self.assertTrue(polygon_invalid.intersects(polygon))
        self.assertRaises(
            (TopologicalError, shapely.GEOSException),
            polygon_invalid.intersection,
            polygon,
        )
        self.assertRaises(
            (TopologicalError, shapely.GEOSException),
            polygon.intersection,
            polygon_invalid,
        )
        return
