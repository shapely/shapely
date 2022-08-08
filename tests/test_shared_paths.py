import pytest

from . import unittest

from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import Point, LineString, Polygon, MultiLineString, \
                             GeometryCollection
from shapely.ops import shared_paths


class SharedPaths(unittest.TestCase):
    def test_shared_paths_forward(self):
        g1 = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])
        g2 = LineString([(5, 0), (15, 0)])
        result = shared_paths(g1, g2)

        self.assertTrue(isinstance(result, GeometryCollection))
        self.assertTrue(len(result.geoms) == 2)
        a, b = result.geoms
        self.assertTrue(isinstance(a, MultiLineString))
        self.assertTrue(len(a.geoms) == 1)
        self.assertEqual(a.geoms[0].coords[:], [(5, 0), (10, 0)])
        self.assertTrue(b.is_empty)

    def test_shared_paths_forward2(self):
        g1 = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])
        g2 = LineString([(15, 0), (5, 0)])
        result = shared_paths(g1, g2)

        self.assertTrue(isinstance(result, GeometryCollection))
        self.assertTrue(len(result.geoms) == 2)
        a, b = result.geoms
        self.assertTrue(isinstance(b, MultiLineString))
        self.assertTrue(len(b.geoms) == 1)
        self.assertEqual(b.geoms[0].coords[:], [(5, 0), (10, 0)])
        self.assertTrue(a.is_empty)

    def test_wrong_type(self):
        g1 = Point(0, 0)
        g2 = LineString([(5, 0), (15, 0)])

        with pytest.warns(ShapelyDeprecationWarning):
            with self.assertRaises(GeometryTypeError):
                result = shared_paths(g1, g2)

        with pytest.warns(ShapelyDeprecationWarning):
            with self.assertRaises(GeometryTypeError):
                result = shared_paths(g2, g1)
