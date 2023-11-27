import unittest

import pytest

import shapely
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import LineString, Point, Polygon


class PolylabelTestCase(unittest.TestCase):
    def test_polylabel(self):
        """
        Finds pole of inaccessibility for a polygon with a tolerance of 10

        """
        polygon = LineString(
            [(0, 0), (50, 200), (100, 100), (20, 50), (-100, -20), (-150, -200)]
        ).buffer(100)
        label = polylabel(polygon, tolerance=0.001)
        expected = Point(59.733, 111.330)
        assert expected.equals_exact(label, 1e-3)

    def test_concave_polygon(self):
        """
        Finds pole of inaccessibility for a concave polygon and ensures that
        the point is inside.

        """
        concave_polygon = LineString([(500, 0), (0, 0), (0, 500), (500, 500)]).buffer(
            100
        )
        label = polylabel(concave_polygon)
        assert concave_polygon.contains(label)

    def test_rectangle_special_case(self):
        """
        The centroid algorithm used is vulnerable to floating point errors
        and can give unexpected results for rectangular polygons. Test
        that this special case is handled correctly.
        https://github.com/mapbox/polylabel/issues/3
        """
        polygon = Polygon(
            [
                (32.71997, -117.19310),
                (32.71997, -117.21065),
                (32.72408, -117.21065),
                (32.72408, -117.19310),
            ]
        )
        label = polylabel(polygon)
        if (shapely.geos_version < (3, 9, 0)) or (shapely.geos_version >= (3, 12, 0)):
            # our own implementation corrects for this
            assert label.coords[:] == [(32.722025, -117.201875)]
        else:
            # the upstream GEOS implementation does not
            assert label.coords[:] == [(32.722025, -117.208595)]

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
        assert label.x == pytest.approx(7.65625)
        assert label.y == pytest.approx(7.65625)
