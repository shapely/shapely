from . import unittest, numpy
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon
from shapely.vectorized import contains

try:
    import numpy as np
except ImportError:
    pass


@unittest.skipIf(not numpy, 'numpy required')
class VectorizedContainsTestCase(unittest.TestCase):
    def assertContainsResults(self, geom, x, y):
        result = contains(geom, x, y)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.bool)

        # Do the equivalent operation, only slowly, comparing the result
        # as we go.
        for idx in range(len(x)):
            self.assertEqual(result[idx], geom.contains(Point(x[idx], y[idx])))
        return result

    def construct_torus(self):
        point = Point(0, 0)
        return point.buffer(5).symmetric_difference(point.buffer(2.5))

    def test_contains_poly(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_contains_point(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)
    
    def test_contains_linestring(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)
    
    def test_contains_multipoly(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        # Construct a geometry of the torus cut in half vertically.
        cut_poly = box(-1, -10, -2.5, 10)
        geom = self.construct_torus().difference(cut_poly)
        self.assertIsInstance(geom, MultiPolygon)
        self.assertContainsResults(geom, x, y)

    def test_array_order(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        y = y.copy(order='f')
        self.assertContainsResults(self.construct_torus(), x, y)
    
    def test_array_dtype(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        x = x.astype(np.int16)
        msg = "Buffer dtype mismatch *"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults(self.construct_torus(), x, y)
    
    def test_array_ndim(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        msg = "Buffer has wrong number of dimensions \(expected 1, got 2\)"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults(self.construct_torus(), x, y)

    def test_shapely_xy_attr_contains(self):
        g = Point(0, 0).buffer(10.0)
        self.assertContainsResults(self.construct_torus(), *g.exterior.xy)
        x, y = g.exterior.xy
        self.assertContainsResults(self.construct_torus(), x, y)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(VectorizedContainsTestCase)
