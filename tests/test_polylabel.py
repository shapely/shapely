from . import unittest
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import asShape, Point
import json
import os


class PolylabelTestCase(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__),
                  'polylabel_lake_superior.json'), 'r') as f:
            self.lake_superior_polygon = asShape(json.load(f))

    def test_polylabel(self):
        """
        finds pole of inaccessibility for polygons with varying levels of
        precision.

        """
        # Find the polylabel for the lake superior polygon with 10000 meter
        # precision
        ls_polylabel = polylabel(self.lake_superior_polygon, precision=10000)
        expected = Point(918926.3475031244, 5311759.390540625)
        self.assertEqual(ls_polylabel, expected)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PolylabelTestCase)
