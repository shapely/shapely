from . import unittest
from shapely.geometry import Polygon

import pytest

from tests.conftest import shapely20_todo

class PolygonTestCase(unittest.TestCase):

    # TODO(shapely-2.0) LinearRing doesn't do "ring closure" with all-equal coords
    @shapely20_todo
    def test_polygon_3(self):
        p = (1.0, 1.0)
        poly = Polygon([p, p, p])
        self.assertEqual(poly.bounds, (1.0, 1.0, 1.0, 1.0))

    def test_polygon_5(self):
        p = (1.0, 1.0)
        poly = Polygon([p, p, p, p, p])
        self.assertEqual(poly.bounds, (1.0, 1.0, 1.0, 1.0))
