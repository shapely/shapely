from . import unittest, numpy, shapely20_deprecated
import pytest

from shapely.coords import CoordinateSequence
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString, asLineString, Point, LinearRing


def test_from_coordinate_sequence():
    # From coordinate tuples
    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    assert len(line.coords) == 2
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]

    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_coordinate_sequence_3D():
    line = LineString(((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)))
    assert line.has_z
    assert line.coords[:] == [(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)]


def test_from_points():
    # From Points
    line = LineString((Point(1.0, 2.0), Point(3.0, 4.0)))
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]

    line = LineString([Point(1.0, 2.0), Point(3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_mix():
    # From mix of tuples and Points
    line = LineString((Point(1.0, 2.0), (2.0, 3.0), Point(3.0, 4.0)))
    assert line.coords[:] == [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def test_from_linestring():
    # From another linestring
    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    copy = LineString(line)
    assert copy.coords[:] == [(1.0, 2.0), (3.0, 4.0)]
    assert copy.geom_type == 'LineString'


def test_from_linearring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)
    copy = LineString(ring)
    assert copy.coords[:] == coords
    assert copy.geom_type == 'LineString'


def test_from_linestring_z():
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    line = LineString(coords)
    copy = LineString(line)
    assert copy.coords[:] == coords
    assert copy.geom_type == 'LineString'


def test_from_generator():
    gen = (coord for coord in [(1.0, 2.0), (3.0, 4.0)])
    line = LineString(gen)
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


def test_from_empty():
    line = LineString()
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []

    line = LineString([])
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []


def test_from_numpy():
    # Construct from a numpy array
    np = pytest.importorskip("numpy")

    line = LineString(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]


@pytest.mark.filterwarnings("error:An exception was ignored")  # NumPy 1.21
def test_numpy_empty_linestring_coords():
    np = pytest.importorskip("numpy")

    # Check empty
    line = LineString([])
    la = np.asarray(line.coords)

    assert la.shape == (0,)


@shapely20_deprecated
@pytest.mark.filterwarnings("error:An exception was ignored")  # NumPy 1.21
def test_numpy_object_array():
    np = pytest.importorskip("numpy")

    geom = LineString([(0.0, 0.0), (0.0, 1.0)])
    ar = np.empty(1, object)
    ar[:] = [geom]
    assert ar[0] == geom


def test_from_invalid_dim():
    with pytest.raises(ValueError, match="at least 2 coordinate tuples"):
        LineString([(1, 2)])

    with pytest.raises(ValueError, match="Inconsistent coordinate dimensionality"):
        LineString([(1, 2, 3), (4, 5)])

    # TODO this does not fail
    # with pytest.raises(ValueError, match="Inconsistent coordinate dimensionality"):
    #     LineString([(1, 2), (3, 4, 5)]))

    # TODO better error, right now raises AssertionError
    with pytest.raises(Exception):
        LineString([(1, 2, 3, 4), (4, 5, 6, 7)])


def test_from_single_coordinate():
    """Test for issue #486"""
    coords = [[-122.185933073564, 37.3629353839073]]
    with pytest.raises(ValueError):
        ls = LineString(coords)
        ls.geom_type  # caused segfault before fix


class LineStringTestCase(unittest.TestCase):

    def test_linestring(self):

        # From coordinate tuples
        line = LineString(((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(len(line.coords), 2)
        self.assertEqual(line.coords[:], [(1.0, 2.0), (3.0, 4.0)])

        # Bounds
        self.assertEqual(line.bounds, (1.0, 2.0, 3.0, 4.0))

        # Coordinate access
        self.assertEqual(tuple(line.coords), ((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(line.coords[0], (1.0, 2.0))
        self.assertEqual(line.coords[1], (3.0, 4.0))
        with self.assertRaises(IndexError):
            line.coords[2]  # index out of range

        # Geo interface
        self.assertEqual(line.__geo_interface__,
                         {'type': 'LineString',
                          'coordinates': ((1.0, 2.0), (3.0, 4.0))})

    @shapely20_deprecated
    def test_linestring_mutate(self):
        line = LineString(((1.0, 2.0), (3.0, 4.0)))

        # Coordinate modification
        line.coords = ((-1.0, -1.0), (1.0, 1.0))
        self.assertEqual(line.__geo_interface__,
                         {'type': 'LineString',
                          'coordinates': ((-1.0, -1.0), (1.0, 1.0))})

    @shapely20_deprecated
    def test_linestring_adapter(self):
        # Adapt a coordinate list to a line string
        coords = [[5.0, 6.0], [7.0, 8.0]]
        la = asLineString(coords)
        self.assertEqual(la.coords[:], [(5.0, 6.0), (7.0, 8.0)])

    def test_linestring_empty(self):
        # Test Non-operability of Null geometry
        l_null = LineString()
        self.assertEqual(l_null.wkt, 'GEOMETRYCOLLECTION EMPTY')
        self.assertEqual(l_null.length, 0.0)

    @shapely20_deprecated
    def test_linestring_empty_mutate(self):
        # Check that we can set coordinates of a null geometry
        l_null = LineString()
        l_null.coords = [(0, 0), (1, 1)]
        self.assertAlmostEqual(l_null.length, 1.4142135623730951)

    def test_equals_argument_order(self):
        """
        Test equals predicate functions correctly regardless of the order
        of the inputs. See issue #317.
        """
        coords = ((0, 0), (1, 0), (1, 1), (0, 0))
        ls = LineString(coords)
        lr = LinearRing(coords)

        self.assertFalse(ls.__eq__(lr))  # previously incorrectly returned True
        self.assertFalse(lr.__eq__(ls))
        self.assertFalse(ls == lr)
        self.assertFalse(lr == ls)

        ls_clone = LineString(coords)
        lr_clone = LinearRing(coords)

        self.assertTrue(ls.__eq__(ls_clone))
        self.assertTrue(lr.__eq__(lr_clone))
        self.assertTrue(ls == ls_clone)
        self.assertTrue(lr == lr_clone)

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Construct from a numpy array
        line = LineString(array([[0.0, 0.0], [1.0, 2.0]]))
        self.assertEqual(len(line.coords), 2)
        self.assertEqual(line.coords[:], [(0.0, 0.0), (1.0, 2.0)])

        line = LineString(((1.0, 2.0), (3.0, 4.0)))
        la = asarray(line)
        expected = array([[1.0, 2.0], [3.0, 4.0]])
        assert_array_equal(la, expected)

    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_linestring_coords(self):
        from numpy.testing import assert_array_equal

        line = LineString(((1.0, 2.0), (3.0, 4.0)))
        expected = numpy.array([[1.0, 2.0], [3.0, 4.0]])

        # Coordinate sequences can be adapted as well
        la = numpy.asarray(line.coords)
        assert_array_equal(la, expected)

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_adapter(self):
        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Adapt a Numpy array to a line string
        a = array([[1.0, 2.0], [3.0, 4.0]])
        la = asLineString(a)
        assert_array_equal(la.context, a)
        self.assertEqual(la.coords[:], [(1.0, 2.0), (3.0, 4.0)])

        # Now, the inverse
        self.assertEqual(la.__array_interface__,
                         la.context.__array_interface__)

        pas = asarray(la)
        assert_array_equal(pas, array([[1.0, 2.0], [3.0, 4.0]]))

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_asarray(self):
        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # From Array.txt
        a = asarray([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
        line = LineString(a)
        self.assertEqual(line.coords[:], [(0.0, 0.0), (2.0, 2.0), (1.0, 1.0)])

        data = line.ctypes
        self.assertEqual(data[0], 0.0)
        self.assertEqual(data[5], 1.0)

        b = asarray(line)
        assert_array_equal(b, array([[0., 0.], [2., 2.], [1., 1.]]))

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_numpy_empty(self):
        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        # Test array interface of empty linestring
        le = LineString()
        a = asarray(le)
        self.assertEqual(a.shape[0], 0)


def test_linestring_mutability_deprecated():
    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="Setting"):
        line.coords = ((-1.0, -1.0), (1.0, 1.0))


def test_linestring_adapter_deprecated():
    coords = [[5.0, 6.0], [7.0, 8.0]]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asLineString(coords)


def test_linestring_ctypes_deprecated():
    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="ctypes"):
        line.ctypes


def test_linestring_array_interface_deprecated():
    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="array_interface"):
        line.array_interface()


@unittest.skipIf(not numpy, 'Numpy required')
def test_linestring_array_interface_numpy_deprecated():
    import numpy as np

    line = LineString(((1.0, 2.0), (3.0, 4.0)))
    with pytest.warns(ShapelyDeprecationWarning, match="array interface"):
        np.array(line)
