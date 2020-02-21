from . import unittest
from shapely.geometry import shape, Polygon
import numpy as np


class ShapeTestCase(unittest.TestCase):

    def test_polygon_no_coords(self):
        # using None
        d = {"type": "Polygon", "coordinates": None}
        p = shape(d)
        self.assertEqual(p, Polygon())

        # using empty list
        d = {"type": "Polygon", "coordinates": []}
        p = shape(d)
        self.assertEqual(p, Polygon())

        # using empty array
        d = {"type": "Polygon", "coordinates": np.array([])}
        p = shape(d)
        self.assertEqual(p, Polygon())

    def test_polygon_with_coords_list(self):
        # list
        d = {"type": "Polygon", "coordinates": [[[5, 10], [10, 10], [10, 5]]]}
        p = shape(d)
        self.assertEqual(p, Polygon([(5, 10), (10, 10), (10, 5)]))

        # numpy array
        d = {"type": "Polygon", "coordinates": np.array([[[5, 10],
                                                          [10, 10],
                                                          [10, 5]]])}
        p = shape(d)
        self.assertEqual(p, Polygon([(5, 10), (10, 10), (10, 5)]))


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ShapeTestCase)
