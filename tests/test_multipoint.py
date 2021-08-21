from . import unittest, numpy, shapely20_deprecated
from .test_multi import MultiGeometryTestCase

import pytest

from shapely.errors import EmptyPartError, ShapelyDeprecationWarning
from shapely.geometry import Point, MultiPoint, asMultiPoint
from shapely.geometry.base import dump_coords


class MultiPointTestCase(MultiGeometryTestCase):

    def test_multipoint(self):

        # From coordinate tuples
        geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(dump_coords(geom), [[(1.0, 2.0)], [(3.0, 4.0)]])

        # From points
        geom = MultiPoint((Point(1.0, 2.0), Point(3.0, 4.0)))
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(dump_coords(geom), [[(1.0, 2.0)], [(3.0, 4.0)]])

        # From another multi-point
        geom2 = MultiPoint(geom)
        self.assertEqual(len(geom2.geoms), 2)
        self.assertEqual(dump_coords(geom2), [[(1.0, 2.0)], [(3.0, 4.0)]])

        # Sub-geometry Access
        self.assertIsInstance(geom.geoms[0], Point)
        self.assertEqual(geom.geoms[0].x, 1.0)
        self.assertEqual(geom.geoms[0].y, 2.0)
        with self.assertRaises(IndexError):  # index out of range
            geom.geoms[2]

        # Geo interface
        self.assertEqual(geom.__geo_interface__,
                         {'type': 'MultiPoint',
                          'coordinates': ((1.0, 2.0), (3.0, 4.0))})

    @shapely20_deprecated
    def test_multipoint_adapter(self):
        # Adapt a coordinate list to a line string
        coords = [[5.0, 6.0], [7.0, 8.0]]
        geoma = asMultiPoint(coords)
        self.assertEqual(dump_coords(geoma), [[(5.0, 6.0)], [(7.0, 8.0)]])

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_multipoint_from_numpy(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Construct from a numpy array
        geom = MultiPoint(array([[0.0, 0.0], [1.0, 2.0]]))
        self.assertIsInstance(geom, MultiPoint)
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(dump_coords(geom), [[(0.0, 0.0)], [(1.0, 2.0)]])

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Geo interface (cont.)
        geom = MultiPoint((Point(1.0, 2.0), Point(3.0, 4.0)))
        assert_array_equal(array(geom), array([[1., 2.], [3., 4.]]))

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_adapter(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Adapt a Numpy array to a multipoint
        a = array([[1.0, 2.0], [3.0, 4.0]])
        geoma = asMultiPoint(a)
        assert_array_equal(geoma.context, array([[1., 2.], [3., 4.]]))
        self.assertEqual(dump_coords(geoma), [[(1.0, 2.0)], [(3.0, 4.0)]])

        # Now, the inverse
        self.assertEqual(geoma.__array_interface__,
                         geoma.context.__array_interface__)

        pas = asarray(geoma)
        assert_array_equal(pas, array([[1., 2.], [3., 4.]]))

    @shapely20_deprecated
    def test_subgeom_access(self):
        p0 = Point(1.0, 2.0)
        p1 = Point(3.0, 4.0)
        self.subgeom_access_test(MultiPoint, [p0, p1])

    def test_create_multi_with_empty_component(self):
        with self.assertRaises(EmptyPartError) as exc:
            wkt = MultiPoint([Point(0, 0), Point()]).wkt

        self.assertEqual(str(exc.exception), "Can't create MultiPoint with empty component")


def test_multipoint_adapter_deprecated():
    coords = [[5.0, 6.0], [7.0, 8.0]]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asMultiPoint(coords)


def test_multipoint_ctypes_deprecated():
    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="ctypes"):
        geom.ctypes


def test_multipoint_array_interface_deprecated():
    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="array_interface"):
        geom.array_interface()


@unittest.skipIf(not numpy, 'Numpy required')
def test_multipoint_array_interface_numpy_deprecated():
    import numpy as np

    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="array interface"):
        np.array(geom)


@shapely20_deprecated
@pytest.mark.filterwarnings("error:An exception was ignored")  # NumPy 1.21
def test_numpy_object_array():
    np = pytest.importorskip("numpy")

    geom = MultiPoint(((1.0, 2.0), (3.0, 4.0)))
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom


def test_iteration_deprecated():
    geom = MultiPoint([[5.0, 6.0], [7.0, 8.0]])
    with pytest.warns(ShapelyDeprecationWarning, match="Iteration"):
        for g in geom:
            pass


def test_getitem_deprecated():
    geom = MultiPoint([[5.0, 6.0], [7.0, 8.0]])
    with pytest.warns(ShapelyDeprecationWarning, match="__getitem__"):
        part = geom[0]


def test_len_deprecated():
    geom = MultiPoint([[5.0, 6.0], [7.0, 8.0]])
    with pytest.warns(ShapelyDeprecationWarning, match="__len__"):
        assert len(geom) == 2
