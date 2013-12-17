from . import unittest
from shapely import geometry
from shapely.geometry.base import dump_coords
from shapely.algorithms import longitude


class LongitudeShiftTestCase(unittest.TestCase):
    def test_shift_point(self):
        # Test point that will be shifted
        point = geometry.Point(-179, 0)
        shifted_point = longitude.shift(point)
        self.assertEqual(dump_coords(shifted_point), [(181.0, 0.0)])
        deshifted_point = longitude.shift(shifted_point)
        self.assertEqual(dump_coords(deshifted_point), dump_coords(point))

        # Test point that will not be shifted
        point = geometry.Point(179, 0)
        shifted_point = longitude.shift(point)
        self.assertEqual(dump_coords(shifted_point), dump_coords(point))

    def test_shift_multi_point(self):
        # Test multi-point that contains 2 points
        # One that will contain shifted coords and one that does not
        p1 = geometry.Point(-179, 0)
        p2 = geometry.Point(179, 0)
        multi_point = geometry.MultiPoint([p1, p2])
        shifted_multi_point = longitude.shift(multi_point)
        self.assertEqual(dump_coords(shifted_multi_point), [[(181.0, 0.0)], [(179.0, 0.0)]])
        deshifted_multi_point = longitude.shift(shifted_multi_point)
        self.assertEqual(dump_coords(deshifted_multi_point), dump_coords(multi_point))

    def test_shift_line_string(self):
        # Test line that will be shifted
        line = geometry.LineString(((179, -1), (180, 0), (-179, 1)))
        shifted_line = longitude.shift(line)
        self.assertEqual(dump_coords(shifted_line), [(179.0, -1.0), (180.0, 0.0), (181.0, 1.0)])
        deshifted_line = longitude.shift(shifted_line)
        self.assertEqual(dump_coords(deshifted_line), dump_coords(line))

        # Test line that will not be shifted
        line = geometry.LineString(((1, -1), (2, 0), (3, 1)))
        shifted_line = longitude.shift(line)
        self.assertEqual(dump_coords(shifted_line), dump_coords(line))

    def test_shift_multi_line_string(self):
        # Test multi-line that contains 2 linestrings
        # One that will contain shifted coords and one that does not
        l1 = geometry.LineString(((179, -1), (180, 0), (-179, 1)))
        l2 = geometry.LineString(((1, -1), (2, 0), (3, 1)))
        multi_line_string = geometry.MultiLineString([l1, l2])
        shifted_multi_line = longitude.shift(multi_line_string)
        self.assertEqual(dump_coords(shifted_multi_line),
                         [[(179.0, -1.0), (180.0, 0.0), (181.0, 1.0)],
                          [(1.0, -1.0), (2.0, 0.0), (3.0, 1.0)]])
        deshifted_multi_line = longitude.shift(shifted_multi_line)
        self.assertEqual(dump_coords(deshifted_multi_line),
                         dump_coords(multi_line_string))

    def test_shift_polygon(self):
        # Test polygon that will be shifted
        poly = geometry.Polygon([(-179, 0), (180, 1), (179, 0), (180, -1), (-179, 0)])
        shifted_poly = longitude.shift(poly)
        self.assertEqual(dump_coords(shifted_poly),
                         [(181.0, 0.0), (180.0, 1.0), (179.0, 0.0), (180.0, -1.0), (181.0, 0.0)])
        deshifted_poly = longitude.shift(shifted_poly)
        self.assertEqual(dump_coords(deshifted_poly), dump_coords(poly))

        # Test polygon that does not need to be shifted
        poly = geometry.Polygon([(1, 0), (2, 1), (3, 0), (2, -1), (1, 0)])
        shifted_poly = longitude.shift(poly)
        self.assertEqual(dump_coords(shifted_poly),
                         dump_coords(poly))

    def test_shift_multi_polygon(self):
        # Test multi-polygon with 2 polygons
        # One that will contain shifted coords and one that does not
        p1 = geometry.Polygon([(-179, 0), (180, 1), (179, 0), (180, -1), (-179, 0)])
        p2 = geometry.Polygon([(1, 0), (2, 1), (3, 0), (2, -1), (1, 0)])

        multi_poly = geometry.MultiPolygon([p1, p2])
        shifted_poly = longitude.shift(multi_poly)
        self.assertEqual(dump_coords(shifted_poly),
                         [[(181.0, 0.0), (180.0, 1.0), (179.0, 0.0), (180.0, -1.0), (181.0, 0.0)],
                          [(1, 0), (2, 1), (3, 0), (2, -1), (1, 0)]])
        deshifted_poly = longitude.shift(shifted_poly)
        self.assertEqual(dump_coords(deshifted_poly), dump_coords(multi_poly))


class LongitudeResolveIdlIntersectionTestCase(unittest.TestCase):
    def test_idl_intersecting_polygon(self):
        poly = geometry.Polygon([(-179, 0), (180, 1), (179, 0), (180, -1), (-179, 0)])
        corrected_poly = longitude.idl_resolve(poly)
        self.assertEqual(dump_coords(corrected_poly),
                         [
                             [(-179.9999999, 0.9999999000000059), (-179.0, 0.0),
                              (-179.9999999, -0.9999999000000059), (-179.9999999, 0.9999999000000059)],
                             [(179.9999999, -0.9999999000000059), (179.0, 0.0),
                              (179.9999999, 0.9999999000000059), (179.9999999, -0.9999999000000059)]
                         ])

    def test_idl_intersecting_linestring(self):
        line = geometry.LineString(((179, -1), (180, 0), (-179, 1)))
        corrected_line = longitude.idl_resolve(line)
        self.assertEquals(dump_coords(corrected_line),
                          [
                              [(179.0, -1.0), (179.9999999, -9.999999406318238e-08)],
                              [(-179.9999999, 9.999999406318238e-08), (-179.0, 1.0)]
                          ])

    def test_idl_not_intersecting_polygon(self):
        poly = geometry.Polygon([(1, 0), (2, 1), (3, 0), (2, -1), (1, 0)])
        corrected_poly = longitude.idl_resolve(poly)
        self.assertEqual(dump_coords(corrected_poly),
                         dump_coords(poly))

    def test_idl_not_intersecting_linestring(self):
        line = geometry.LineString(((1, -1), (2, 0), (3, 1)))
        corrected_line = longitude.idl_resolve(line)
        self.assertEquals(dump_coords(corrected_line),
                          dump_coords(line))


def test_suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([
        loader.loadTestsFromTestCase(LongitudeShiftTestCase),
        loader.loadTestsFromTestCase(LongitudeResolveIdlIntersectionTestCase)])
