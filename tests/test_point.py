from . import unittest, numpy, shapely20_deprecated
from shapely.geometry import Point, asPoint
from shapely.errors import DimensionError, ShapelyDeprecationWarning

import pytest


class LineStringTestCase(unittest.TestCase):

    def test_point(self):

        # Test 2D points
        p = Point(1.0, 2.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.coords[:], [(1.0, 2.0)])
        self.assertEqual(str(p), p.wkt)
        self.assertFalse(p.has_z)
        with self.assertRaises(DimensionError):
            p.z

        # Check 3D
        p = Point(1.0, 2.0, 3.0)
        self.assertEqual(p.coords[:], [(1.0, 2.0, 3.0)])
        self.assertEqual(str(p), p.wkt)
        self.assertTrue(p.has_z)
        self.assertEqual(p.z, 3.0)

        # From coordinate sequence
        p = Point((3.0, 4.0))
        self.assertEqual(p.coords[:], [(3.0, 4.0)])

        # From another point
        q = Point(p)
        self.assertEqual(q.coords[:], [(3.0, 4.0)])

        # Coordinate access
        self.assertEqual(p.x, 3.0)
        self.assertEqual(p.y, 4.0)
        self.assertEqual(tuple(p.coords), ((3.0, 4.0),))
        self.assertEqual(p.coords[0], (3.0, 4.0))
        with self.assertRaises(IndexError):  # index out of range
            p.coords[1]

        # Bounds
        self.assertEqual(p.bounds, (3.0, 4.0, 3.0, 4.0))

        # Geo interface
        self.assertEqual(p.__geo_interface__,
                         {'type': 'Point', 'coordinates': (3.0, 4.0)})

    @shapely20_deprecated
    def test_point_mutate(self):
        # Modify coordinates
        p = Point(3.0, 4.0)
        p.coords = (2.0, 1.0)
        self.assertEqual(p.__geo_interface__,
                         {'type': 'Point', 'coordinates': (2.0, 1.0)})

        # Alternate method
        p.coords = ((0.0, 0.0),)
        self.assertEqual(p.__geo_interface__,
                         {'type': 'Point', 'coordinates': (0.0, 0.0)})

    @shapely20_deprecated
    def test_point_adapter(self):
        p = Point(0.0, 0.0)
        # Adapt a coordinate list to a point
        coords = [3.0, 4.0]
        pa = asPoint(coords)
        self.assertEqual(pa.coords[0], (3.0, 4.0))
        self.assertEqual(pa.distance(p), 5.0)

        # Move the coordinates and watch the distance change
        coords[0] = 1.0
        self.assertEqual(pa.coords[0], (1.0, 4.0))
        self.assertAlmostEqual(pa.distance(p), 4.123105625617661)

    def test_point_empty(self):
        # Test Non-operability of Null geometry
        p_null = Point()
        self.assertEqual(p_null.wkt, 'GEOMETRYCOLLECTION EMPTY')
        self.assertEqual(p_null.coords[:], [])
        self.assertEqual(p_null.area, 0.0)

    @shapely20_deprecated
    def test_point_empty_mutate(self):
        # Check that we can set coordinates of a null geometry
        p_null = Point()
        p_null.coords = (1, 2)
        self.assertEqual(p_null.coords[:], [(1.0, 2.0)])

        # Passing > 3 arguments to Point is erroneous
        with self.assertRaises(TypeError):
            Point(1.0, 2.0, 3.0, 4.0)

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Construct from a numpy array
        p = Point(array([1.0, 2.0]))
        self.assertEqual(p.coords[:], [(1.0, 2.0)])

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_adapter(self):
        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Adapt a Numpy array to a point
        a = array([1.0, 2.0])
        pa = asPoint(a)
        assert_array_equal(pa.context, array([1.0, 2.0]))
        self.assertEqual(pa.coords[:], [(1.0, 2.0)])

        # Now, the inverse
        self.assertEqual(pa.__array_interface__,
                         pa.context.__array_interface__)

        pas = asarray(pa)
        assert_array_equal(pas, array([1.0, 2.0]))

        # Adapt a coordinate list to a point
        coords = [3.0, 4.0]
        pa = asPoint(coords)
        coords[0] = 1.0

        # Now, the inverse (again?)
        self.assertIsNotNone(pa.__array_interface__)
        pas = asarray(pa)
        assert_array_equal(pas, array([1.0, 4.0]))

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_asarray(self):
        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # From Array.txt
        p = Point(0.0, 0.0, 1.0)
        coords = p.coords[0]
        self.assertEqual(coords, (0.0, 0.0, 1.0))
        self.assertIsNotNone(p.ctypes)

        # Convert to Numpy array, passing through Python sequence
        a = asarray(coords)
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.size, 3)
        self.assertEqual(a.shape, (3,))

        # Convert to Numpy array, passing through a ctypes array
        b = asarray(p)
        self.assertEqual(b.size, 3)
        self.assertEqual(b.shape, (3,))
        assert_array_equal(b, array([0.0, 0.0, 1.0]))

        # Make a point from a Numpy array
        a = asarray([1.0, 1.0, 0.0])
        p = Point(*list(a))
        self.assertEqual(p.coords[:], [(1.0, 1.0, 0.0)])

        # Test array interface of empty geometry
        pe = Point()
        a = asarray(pe)
        self.assertEqual(a.shape[0], 0)


def test_empty_point_bounds():
    """The bounds of an empty point is an empty tuple"""
    p = Point()
    assert p.bounds == ()


def test_point_mutability_deprecated():
    p = Point(3.0, 4.0)
    with pytest.warns(ShapelyDeprecationWarning, match="Setting"):
        p.coords = (2.0, 1.0)


def test_point_adapter_deprecated():
    coords = [3.0, 4.0]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asPoint(coords)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(LineStringTestCase)
