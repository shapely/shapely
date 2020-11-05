"""Polygons and Linear Rings
"""

from . import unittest, numpy, shapely20_deprecated

import pytest

from shapely.coords import CoordinateSequence
from shapely.wkb import loads as load_wkb
from shapely.errors import TopologicalError, ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry import Point, Polygon, asPolygon
from shapely.geometry.polygon import LinearRing, LineString, asLinearRing
from shapely.geometry.base import dump_coords


def test_linearring_from_coordinate_sequence():
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    assert ring.coords[:] == expected_coords

    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert ring.coords[:] == expected_coords


def test_linearring_from_points():
    # From Points
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing([Point(0.0, 0.0), Point(0.0, 1.0), Point(1.0, 1.0)])
    assert ring.coords[:] == expected_coords


def test_linearring_from_closed_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords)
    ring = LinearRing(line)
    assert len(ring.coords) == 4
    assert ring.coords[:] == coords
    assert ring.geom_type == 'LinearRing'


def test_linearring_from_unclosed_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords[:-1])  # Pass in unclosed line
    ring = LinearRing(line)
    assert len(ring.coords) == 4
    assert ring.coords[:] == coords
    assert ring.geom_type == 'LinearRing'


def test_linearring_from_invalid():
    coords = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    line = LineString(coords)
    assert not line.is_valid
    with pytest.raises(TopologicalError):
        LinearRing(line)


def test_linearring_from_too_short_linestring():
    # Creation of LinearRing request at least 3 coordinates (unclosed) or
    # 4 coordinates (closed)
    coords = [(0.0, 0.0), (1.0, 1.0)]
    line = LineString(coords)
    with pytest.raises(ValueError, match="at least 3 coordinate tuple"):
        LinearRing(line)


def test_linearring_from_generator():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    gen = (coord for coord in coords)
    ring = LinearRing(gen)
    assert ring.coords[:] == coords


def test_linearring_from_empty():
    ring = LinearRing()
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []

    ring = LinearRing([])
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []


def test_linearring_from_numpy():
    # Construct from a numpy array
    np = pytest.importorskip("numpy")
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]

    ring = LinearRing(np.array(coords))
    assert ring.coords[:] == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]


def test_numpy_linearring_coords():
    np = pytest.importorskip("numpy")
    from numpy.testing import assert_array_equal

    ring = LinearRing(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    ra = np.asarray(ring.coords)
    expected = np.asarray([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)])
    assert_array_equal(ra, expected)


def test_numpy_empty_linearring_coords():
    np = pytest.importorskip("numpy")

    ring = LinearRing()
    assert np.asarray(ring.coords).shape == (0,)


def test_polygon_from_coordinate_sequence():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    # Construct a polygon, exterior ring only
    polygon = Polygon(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0

    polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0


def test_polygon_from_coordinate_sequence_with_holes():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]

    # Interior rings (holes)
    polygon = Polygon(coords, [((0.25, 0.25), (0.25, 0.5),
                                (0.5, 0.5), (0.5, 0.25))])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 1
    assert len(polygon.interiors[0].coords) == 5


def test_polygon_from_linearring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)

    polygon = Polygon(ring)
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0


def test_polygon_from_polygon():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    polygon = Polygon(coords, [((0.25, 0.25), (0.25, 0.5),
                                (0.5, 0.5), (0.5, 0.25))])

    # Test from another Polygon
    copy = Polygon(polygon)
    assert len(polygon.exterior.coords) == 5
    assert len(polygon.interiors) == 1
    assert len(polygon.interiors[0].coords) == 5


def test_polygon_from_invalid():
    # Error handling
    with pytest.raises(ValueError):
        # A LinearRing must have at least 3 coordinate tuples
        Polygon([[1, 2], [2, 3]])


def test_polygon_from_empty():
    polygon = Polygon()
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []

    polygon = Polygon([])
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []


def test_polygon_from_numpy():
    np = pytest.importorskip("numpy")

    a = np.array(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
    polygon = Polygon(a)
    assert len(polygon.exterior.coords) == 5
    assert (polygon.exterior.coords[:]
            == [(0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)])
    assert len(polygon.interiors) == 0


class PolygonTestCase(unittest.TestCase):

    def test_linearring(self):

        # Initialization
        # Linear rings won't usually be created by users, but by polygons
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        ring = LinearRing(coords)
        self.assertEqual(len(ring.coords), 5)
        self.assertEqual(ring.coords[0], ring.coords[4])
        self.assertEqual(ring.coords[0], ring.coords[-1])
        self.assertTrue(ring.is_ring)

    @shapely20_deprecated
    def test_linearring_mutate(self):
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        ring = LinearRing(coords)

        # Coordinate modification
        ring.coords = ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0))
        self.assertEqual(
            ring.__geo_interface__,
            {'type': 'LinearRing',
             'coordinates': ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0),
                             (0.0, 0.0))})

    @shapely20_deprecated
    def test_linearring_adapter(self):
        # Test ring adapter
        coords = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        ra = asLinearRing(coords)
        self.assertTrue(ra.wkt.upper().startswith('LINEARRING'))
        self.assertEqual(dump_coords(ra),
                         [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0),
                          (0.0, 0.0)])
        coords[3] = [2.0, -1.0]
        self.assertEqual(dump_coords(ra),
                         [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0),
                          (0.0, 0.0)])

    def test_polygon(self):
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))

        # Construct a polygon, exterior ring only
        polygon = Polygon(coords)
        self.assertEqual(len(polygon.exterior.coords), 5)

        # Ring Access
        self.assertIsInstance(polygon.exterior, LinearRing)
        ring = polygon.exterior
        self.assertEqual(len(ring.coords), 5)
        self.assertEqual(ring.coords[0], ring.coords[4])
        self.assertEqual(ring.coords[0], (0., 0.))
        self.assertTrue(ring.is_ring)
        self.assertEqual(len(polygon.interiors), 0)

        # Create a new polygon from WKB
        data = polygon.wkb
        polygon = None
        ring = None
        polygon = load_wkb(data)
        ring = polygon.exterior
        self.assertEqual(len(ring.coords), 5)
        self.assertEqual(ring.coords[0], ring.coords[4])
        self.assertEqual(ring.coords[0], (0., 0.))
        self.assertTrue(ring.is_ring)
        polygon = None

        # Interior rings (holes)
        polygon = Polygon(coords, [((0.25, 0.25), (0.25, 0.5),
                                    (0.5, 0.5), (0.5, 0.25))])
        self.assertEqual(len(polygon.exterior.coords), 5)
        self.assertEqual(len(polygon.interiors[0].coords), 5)
        with self.assertRaises(IndexError):  # index out of range
            polygon.interiors[1]

        # Coordinate getters and setters raise exceptions
        self.assertRaises(NotImplementedError, polygon._get_coords)
        with self.assertRaises(NotImplementedError):
            polygon.coords

        # Geo interface
        self.assertEqual(
            polygon.__geo_interface__,
            {'type': 'Polygon',
             'coordinates': (((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0),
                             (0.0, 0.0)), ((0.25, 0.25), (0.25, 0.5),
                             (0.5, 0.5), (0.5, 0.25), (0.25, 0.25)))})

    @shapely20_deprecated
    def test_polygon_adapter(self):
        # Adapter
        coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        hole_coords = [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
        pa = asPolygon(coords, hole_coords)
        self.assertEqual(len(pa.exterior.coords), 5)
        self.assertEqual(len(pa.interiors), 1)
        self.assertEqual(len(pa.interiors[0].coords), 5)

    def test_linearring_empty(self):
        # Test Non-operability of Null rings
        r_null = LinearRing()
        self.assertEqual(r_null.wkt, 'GEOMETRYCOLLECTION EMPTY')
        self.assertEqual(r_null.length, 0.0)

    @shapely20_deprecated
    def test_linearring_empty_mutate(self):
        # Check that we can set coordinates of a null geometry
        r_null = LinearRing()
        r_null.coords = [(0, 0), (1, 1), (1, 0)]
        self.assertAlmostEqual(r_null.length, 3.414213562373095)

    @shapely20_deprecated
    @unittest.skipIf(not numpy, 'Numpy required')
    def test_polygon_exterior_array_interface(self):

        from numpy import array, asarray
        from numpy.testing import assert_array_equal

        a = asarray(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
        polygon = Polygon(a)

        b = asarray(polygon.exterior)
        self.assertEqual(b.shape, (5, 2))
        assert_array_equal(
            b, array([(0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)]))

    def test_dimensions(self):

        # Background: see http://trac.gispython.org/lab/ticket/168
        # http://lists.gispython.org/pipermail/community/2008-August/001859.html

        coords = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
                  (1.0, 0.0, 0.0))
        polygon = Polygon(coords)
        self.assertEqual(polygon._ndim, 3)
        gi = polygon.__geo_interface__
        self.assertEqual(
            gi['coordinates'],
            (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
              (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),))

        e = polygon.exterior
        self.assertEqual(e._ndim, 3)
        gi = e.__geo_interface__
        self.assertEqual(
            gi['coordinates'],
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
             (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def test_attribute_chains(self):

        # Attribute Chaining
        # See also ticket #151.
        p = Polygon(((0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)))
        self.assertEqual(
            list(p.boundary.coords),
            [(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (0.0, 0.0)])

        ec = list(Point(0.0, 0.0).buffer(1.0, 1).exterior.coords)
        self.assertIsInstance(ec, list)  # TODO: this is a poor test

        # Test chained access to interiors
        p = Polygon(
            ((0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)),
            [((-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25))]
        )
        self.assertEqual(p.area, 0.75)

        """Not so much testing the exact values here, which are the
        responsibility of the geometry engine (GEOS), but that we can get
        chain functions and properties using anonymous references.
        """
        self.assertEqual(
            list(p.interiors[0].coords),
            [(-0.25, 0.25), (-0.25, 0.75), (-0.75, 0.75), (-0.75, 0.25),
             (-0.25, 0.25)])
        xy = list(p.interiors[0].buffer(1).exterior.coords)[0]
        self.assertEqual(len(xy), 2)

        # Test multiple operators, boundary of a buffer
        ec = list(p.buffer(1).boundary.coords)
        self.assertIsInstance(ec, list)  # TODO: this is a poor test

    def test_empty_equality(self):
        # Test equals operator, including empty geometries
        # see issue #338

        point1 = Point(0, 0)
        polygon1 = Polygon(((0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)))
        polygon2 = Polygon(((0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)))
        polygon_empty1 = Polygon()
        polygon_empty2 = Polygon()

        self.assertNotEqual(point1, polygon1)
        self.assertEqual(polygon_empty1, polygon_empty2)
        self.assertNotEqual(polygon1, polygon_empty1)
        self.assertEqual(polygon1, polygon2)
        self.assertNotEqual(None, polygon_empty1)

    def test_from_bounds(self):
        xmin, ymin, xmax, ymax = -180, -90, 180, 90
        coords = [
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin)]
        self.assertEqual(
            Polygon(coords),
            Polygon.from_bounds(xmin, ymin, xmax, ymax))

    def test_empty_polygon_exterior(self):
        p = Polygon()
        assert p.exterior == LinearRing()


def test_linearring_adapter_deprecated():
    coords = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asLinearRing(coords)


def test_polygon_adapter_deprecated():
    coords = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    hole_coords = [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
    with pytest.warns(ShapelyDeprecationWarning, match="proxy geometries"):
        asPolygon(coords, hole_coords)


def test_ctypes_deprecated():
    coords = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    hole_coords = [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))]
    ring = LinearRing(coords)
    polygon = Polygon(coords, hole_coords)

    with pytest.warns(ShapelyDeprecationWarning, match="ctypes"):
        ring.ctypes

    with pytest.warns(ShapelyDeprecationWarning, match="ctypes"):
        polygon.ctypes
