"""Test recovery from operation on invalid geometries
"""

from . import unittest
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

import shapely
import pytest


class InvalidGeometriesTestCase(unittest.TestCase):
    def test_invalid_intersection(self):
        # Make a self-intersecting polygon
        polygon_invalid = Polygon([(0, 0), (1, 1), (1, -1), (0, 1), (0, 0)])
        assert not polygon_invalid.is_valid

        # Intersect with a valid polygon
        polygon = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -5)])
        assert polygon.is_valid
        assert polygon_invalid.intersects(polygon)

        with pytest.raises((TopologicalError, shapely.GEOSException)):
            polygon_invalid.intersection(polygon)
        with pytest.raises((TopologicalError, shapely.GEOSException)):
            polygon.intersection(polygon_invalid)
        return
