from . import unittest
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import LineString, Point, Polygon
from shapely.errors import TopologicalError


class PolylabelTestCase(unittest.TestCase):
    def test_polylabel(self):
        """
        Finds pole of inaccessibility for a polygon with a tolerance of 10

        """
        polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50),
                              (-100, -20), (-150, -200)]).buffer(100)
        label = polylabel(polygon, tolerance=10)
        expected = Point(59.35615556364569, 121.8391962974644)
        self.assertTrue(expected.equals_exact(label, 1e-6))

    def test_concave_polygon(self):
        """
        Finds pole of inaccessibility for a concave polygon and ensures that
        the point is inside.

        """
        concave_polygon = LineString([(500, 0), (0, 0), (0, 500),
                                      (500, 500)]).buffer(100)
        label = polylabel(concave_polygon)
        self.assertTrue(concave_polygon.contains(label))

    def test_rectangle_special_case(self):
        """
        The centroid algorithm used is vulnerable to floating point errors
        and can give unexpected results for rectangular polygons. Test
        that this special case is handled correctly.
        https://github.com/mapbox/polylabel/issues/3
        """
        polygon = Polygon([(32.71997,-117.19310), (32.71997,-117.21065),
                           (32.72408,-117.21065), (32.72408,-117.19310)])
        label = polylabel(polygon)
        self.assertEqual(label.coords[:], [(32.722025, -117.208595)])

    def test_polygon_with_hole(self):
        """
        Finds pole of inaccessibility for a polygon with a hole
        https://github.com/shapely/shapely/issues/817
        """
        polygon = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]],
        )
        label = polylabel(polygon, 0.05)
        self.assertAlmostEqual(label.x, 7.65625)
        self.assertAlmostEqual(label.y, 7.65625)
