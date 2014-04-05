from . import unittest, numpy
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon
from shapely.vectorized import contains, contains2d

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
        for index, _ in enumerate(x):
            self.assertEqual(result[index],
                             geom.contains(Point(x[index], y[index])))
        return result

    def assertContainsResults2D(self, geom, x, y):
        result = contains2d(geom, x, y)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.bool)

        for index in np.ndindex(x.shape):
            self.assertEqual(result[index],
                             geom.contains(Point(x[index], y[index])))
        return result
    
    def construct_torus(self):
        point = Point(0, 0)
        return point.buffer(5).symmetric_difference(point.buffer(2.5))

    def test_contains_poly_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_contains_point_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)
    
    def test_contains_linestring_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        self.assertContainsResults(Point(x[0], y[0]), x, y)
    
    def test_contains_multipoly_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        # Construct a geometry of the torus cut in half vertically.
        cut_poly = box(-1, -10, -2.5, 10)
        geom = self.construct_torus().difference(cut_poly)
        self.assertIsInstance(geom, MultiPolygon)
        self.assertContainsResults(geom, x, y)

    def test_array_order_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        y = y.copy(order='f')
        self.assertContainsResults(self.construct_torus(), x, y)
    
    def test_array_dtype_1d(self):
        y, x = np.mgrid[-10:10:5j], np.mgrid[-5:15:5j]
        x = x.astype(np.int16)
        msg = "Buffer dtype mismatch, expected 'float64' but got 'short'"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults(self.construct_torus(), x, y)
    
    def test_array_ndim_1d(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        msg = "Buffer has wrong number of dimensions \(expected 1, got 2\)"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults(self.construct_torus(),
                                       x, y)

    def test_shapely_xy_attr_contains(self):
        g = Point(0, 0).buffer(10.0)
        self.assertContainsResults(self.construct_torus(), *g.exterior.xy)
        x, y = g.exterior.xy
        self.assertContainsResults(self.construct_torus(), x, y)

    def test_contains_poly_2d(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        self.assertContainsResults2D(self.construct_torus(), x, y)

    def test_contains_point_2d(self):
        y, x = np.mgrid[-10:10:5j, -5:15:6j]
        self.assertContainsResults2D(Point(x[0, 0], y[0, 0]), x, y)
    
    def test_contains_linestring_2d(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        self.assertContainsResults2D(Point(x[0, 0], y[0, 0]), x, y)
    
    def test_contains_multipoly_2d(self):
        y, x = np.mgrid[-10:10:15j, -5:15:16j]
        # Construct a geometry of the torus cut in half vertically.
        cut_poly = box(-1, -10, -2.5, 10)
        geom = self.construct_torus().difference(cut_poly)
        self.assertIsInstance(geom, MultiPolygon)
        self.assertContainsResults2D(geom, x, y)

    def test_array_order_2d(self):
        y, x = np.mgrid[-10:10:5j, -5:15:6j]
        y = y.copy(order='f')
        self.assertContainsResults2D(self.construct_torus(), x, y)
    
    def test_array_dtype_2d(self):
        y, x = np.mgrid[-10:10:5j, -5:15:6j]
        x = x.astype(np.int16)
        msg = "Buffer dtype mismatch, expected 'float64' but got 'short'"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults2D(self.construct_torus(), x, y)
    
    def test_array_ndim_2d(self):
        y, x = np.mgrid[-10:10:5j, -5:15:6j]
        msg = "Buffer has wrong number of dimensions \(expected 2, got 1\)"
        with self.assertRaisesRegexp(ValueError, msg):
            self.assertContainsResults2D(self.construct_torus(),
                                       x[:, 0], y[:, 0])


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(VectorizedContainsTestCase)
