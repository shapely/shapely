import unittest

import pytest
from numpy.testing import assert_array_equal

import shapely
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import orient


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
class OrientTestCase(unittest.TestCase):
    def test_point(self):
        point = Point(0, 0)
        assert orient(point, 1) == point
        assert orient(point, -1) == point

    def test_multipoint(self):
        multipoint = MultiPoint([(0, 0), (1, 1)])
        assert orient(multipoint, 1) == multipoint
        assert orient(multipoint, -1) == multipoint

    def test_linestring(self):
        linestring = LineString([(0, 0), (1, 1)])
        assert orient(linestring, 1) == linestring
        assert orient(linestring, -1) == linestring

    def test_multilinestring(self):
        multilinestring = MultiLineString([[(0, 0), (1, 1)], [(1, 0), (0, 1)]])
        assert orient(multilinestring, 1) == multilinestring
        assert orient(multilinestring, -1) == multilinestring

    def test_linearring(self):
        linearring = LinearRing([(0, 0), (0, 1), (1, 0)])
        linearring_reversed = LinearRing(linearring.coords[::-1])
        assert orient(linearring, 1) == linearring_reversed
        assert orient(linearring, -1) == linearring

    def test_empty_polygon(self):
        polygon = Polygon()
        assert orient(polygon) == polygon

    def test_polygon(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 0)])
        polygon_reversed = Polygon(polygon.exterior.coords[::-1])
        assert (orient(polygon, 1)) == polygon_reversed
        assert (orient(polygon, -1)) == polygon

    def test_multipolygon(self):
        polygon1 = Polygon([(0, 0), (0, 1), (1, 0)])
        polygon2 = Polygon([(1, 0), (2, 0), (2, 1)])
        polygon1_reversed = Polygon(polygon1.exterior.coords[::-1])
        polygon2_reversed = Polygon(polygon2.exterior.coords[::-1])
        multipolygon = MultiPolygon([polygon1, polygon2])
        assert not polygon1.exterior.is_ccw
        assert polygon2.exterior.is_ccw
        assert orient(multipolygon, 1) == MultiPolygon([polygon1_reversed, polygon2])
        assert orient(multipolygon, -1) == MultiPolygon([polygon1, polygon2_reversed])

    def test_geometrycollection(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 0)])
        polygon_reversed = Polygon(polygon.exterior.coords[::-1])
        collection = GeometryCollection([polygon])
        assert orient(collection, 1) == GeometryCollection([polygon_reversed])
        assert orient(collection, -1) == GeometryCollection([polygon])

    def test_vectorized(self):
        ring_cw = LinearRing([(0, 0), (0, 1), (1, 1), (0, 0)])
        ring_cw2 = LinearRing([(0, 0), (0, 3), (3, 3), (0, 0)])
        ring_ccw = LinearRing([(0, 0), (1, 1), (0, 1), (0, 0)])
        ring_ccw2 = LinearRing([(0, 0), (2, 2), (0, 2), (0, 0)])

        polygon_cw = Polygon(ring_cw)
        polygon_with_holes_mixed = Polygon(
            ring_ccw, [ring_cw, ring_ccw2, ring_cw2, ring_ccw]
        )
        polygon_with_holes_ccw = Polygon(
            ring_ccw, [ring_cw, ring_ccw2.reverse(), ring_cw2, ring_ccw.reverse()]
        )

        assert_array_equal(orient(polygon_with_holes_ccw, 1), polygon_with_holes_ccw)
        assert_array_equal(
            orient(polygon_with_holes_ccw, -1), polygon_with_holes_ccw.reverse()
        )
        assert_array_equal(orient(polygon_with_holes_mixed, 1), polygon_with_holes_ccw)
        assert_array_equal(
            orient(polygon_with_holes_mixed, -1), polygon_with_holes_ccw.reverse()
        )

        # vectorized with scalar sign parameter
        assert_array_equal(
            orient([ring_ccw, ring_cw, polygon_cw, polygon_with_holes_mixed], 1),
            [ring_ccw, ring_cw.reverse(), polygon_cw.reverse(), polygon_with_holes_ccw],
        )

        # vectorized with scalar sign parameter
        assert_array_equal(
            orient([ring_ccw, ring_cw, polygon_cw, polygon_with_holes_mixed], -1),
            [ring_ccw.reverse(), ring_cw, polygon_cw, polygon_with_holes_ccw.reverse()],
        )

        # vectorized with vector sign parameter
        assert_array_equal(
            orient(
                [ring_ccw, ring_cw, polygon_cw, polygon_with_holes_mixed, polygon_cw],
                [1, -1, 1, -1, -1],
            ),
            [
                ring_ccw,
                ring_cw,
                polygon_cw.reverse(),
                polygon_with_holes_ccw.reverse(),
                polygon_cw,
            ],
        )

        # vectorized with single geom and vector sign parameter
        assert_array_equal(orient(ring_ccw, [1, -1]), [ring_ccw, ring_ccw.reverse()])
