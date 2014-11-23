'''Test recovery from operation on invalid geometries
'''

from . import unittest
from shapely.geos import TopologicalError, DimensionError
from shapely.geometry import Polygon, LineString, asPolygon, asLineString
from shapely.topology import TopologicalError


class InvalidGeometriesTestCase(unittest.TestCase):

    def test_invalid_intersection(self):
        # Make a self-intersecting polygon
        polygon_invalid = Polygon(((0, 0), (1, 1), (1, -1), (0, 1), (0, 0)))
        self.assertFalse(polygon_invalid.is_valid)

        # Intersect with a valid polygon
        polygon = Polygon(((-.5, -.5), (-.5, .5), (.5, .5), (.5, -5)))
        self.assertTrue(polygon.is_valid)
        self.assertTrue(polygon_invalid.intersects(polygon))
        self.assertRaises(TopologicalError,
                          polygon_invalid.intersection, polygon)
        self.assertRaises(TopologicalError,
                          polygon.intersection, polygon_invalid)
        return

    def test_polygon_inconsistent_dimensionality(self):
        invalid_arr = [(0, 0, 0), (1, 1), (2, 2)]
        with self.assertRaises(DimensionError):
            Polygon(invalid_arr)
        p = asPolygon(invalid_arr)
        self.assertFalse(p.is_valid)

    def test_polygon_missing_coordinates(self):
        invalid_arr = [(0, 0), (1, 1)]
        with self.assertRaises(TopologicalError):
            Polygon(invalid_arr)
        p = asPolygon(invalid_arr)
        self.assertFalse(p.is_valid)

    def test_linestring_inconsistent_dimensionality(self):
        invalid_arr = [(0, 0, 0), (1, 1)]
        with self.assertRaises(DimensionError):
            LineString(invalid_arr)
        l = asLineString(invalid_arr)
        self.assertFalse(l.is_valid)

    def test_linestring_missing_coordinates(self):
        invalid_arr = [(0, 0)]
        with self.assertRaises(TopologicalError):
            LineString(invalid_arr)
        l = asLineString(invalid_arr)
        self.assertFalse(l.is_valid)

def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        loader.loadTestsFromTestCase(InvalidGeometriesTestCase)])
