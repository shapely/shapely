import unittest
from itertools import product

import pytest

from shapely import geometry
from shapely.ops import transform


class IdentityTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def func(self, x, y, z=None):
        return tuple(c for c in [x, y, z] if c is not None)

    def test_empty(self):
        g = geometry.Point()
        h = transform(self.func, g)
        assert h.is_empty

    def test_point(self):
        g = geometry.Point(0, 1)
        for include_z, rebuild in product([False], [False, True]):
            h = transform(self.func, g, include_z=include_z, rebuild=rebuild)
            assert h.geom_type == "Point"
            assert list(h.coords) == [(0, 1)]

    def test_two_point(self):
        g = [geometry.Point(0, 1), geometry.Point(2, 3, 4)]
        h = transform(self.func, g)
        assert h[0].geom_type == "Point"
        assert h[1].geom_type == "Point"
        assert list(h[0].coords) == [(0, 1)]
        expected_coords = [(2, 3, 4)]
        assert list(h[1].coords) == expected_coords

    def test_two_point_non_default(self):
        g = [geometry.Point(0, 1), geometry.Point(2, 3, 4)]
        for include_z, rebuild in product([False, None], [False, True]):
            h = transform(self.func, g, include_z=include_z, rebuild=rebuild)
            assert h[0].geom_type == "Point"
            assert h[1].geom_type == "Point"
            assert list(h[0].coords) == [(0, 1)]
            expected_coords = [(2, 3)] if include_z is False else [(2, 3, 4)]
            assert list(h[1].coords) == expected_coords

    def test_point_z(self):
        g = geometry.Point(0, 1, 2)
        for include_z, rebuild in product([False, True], [False, True]):
            h = transform(self.func, g, include_z=include_z, rebuild=rebuild)
            assert h.geom_type == "Point"
            expected_coords = [(0, 1, 2) if include_z else (0, 1)]
            assert list(h.coords) == expected_coords

    def test_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        h = transform(self.func, g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(0, 1), (2, 3)]

    def test_linearring(self):
        g = geometry.LinearRing([(0, 1), (2, 3), (2, 2), (0, 1)])
        h = transform(self.func, g)
        assert h.geom_type == "LinearRing"
        assert list(h.coords) == [(0, 1), (2, 3), (2, 2), (0, 1)]

    def test_linearring_z(self):
        g = geometry.LinearRing([(0, 1, 2), (2, 3, 4), (2, 2, 2), (0, 1, 2)])
        for include_z, rebuild in product([False, True], [False, True]):
            h = transform(self.func, g, include_z=include_z, rebuild=rebuild)
            assert h.geom_type == "LinearRing"
            expected_coords = (
                [(0, 1, 2), (2, 3, 4), (2, 2, 2), (0, 1, 2)]
                if include_z
                else [(0, 1), (2, 3), (2, 2), (0, 1)]
            )
            assert list(h.coords) == expected_coords

    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        h = transform(self.func, g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(h.area)

    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        h = transform(self.func, g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(h.area)


class LambdaTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_point(self):
        g = geometry.Point(0, 1)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "Point"
        assert list(h.coords) == [(1.0, 2.0)]

    def test_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(1.0, 2.0), (3.0, 4.0)]

    def test_line_single_values(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        h = transform(lambda x, y, z=None: (float(x) + 1.0, float(y) + 1.0), g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(1.0, 2.0), (3.0, 4.0)]

    def test_line_z(self):
        g = geometry.LineString([(0, 1, 2), (2, 3, 4)])
        for include_z, rebuild in product([True], [False, True]):
            h = transform(
                lambda x, y, z=None: (x + 1.0, y + 1.0, z + 1.0),
                g,
                include_z=include_z,
                rebuild=rebuild,
            )
            assert h.geom_type == "LineString"
            expected_coords = (
                [(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)]
                if include_z
                else [(1.0, 2.0), (3.0, 4.0)]
            )
            assert list(h.coords) == expected_coords

    def test_insert_z_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        h = transform(
            lambda x, y, z=None: (x + 1.0, y + 1.0, x + y),
            g,
            include_z=True,
            rebuild=True,
        )
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(1.0, 2.0, 1.0), (3.0, 4.0, 5.0)]

    def test_insert_z_line_and_point(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        p2 = geometry.Point(5, 6)
        p3 = geometry.Point(4, 5, 7)
        h = transform(
            lambda x, y, z=None: (x + 1.0, y + 1.0, z if z is not None else x + y),
            [g, p2, p3],
            rebuild=True,
        )
        assert h[0].geom_type == "LineString"
        assert list(h[0].coords) == [(1.0, 2.0, 1.0), (3.0, 4.0, 5.0)]
        assert h[1].geom_type == "Point"
        assert list(h[1].coords) == [(6, 7, 11)]
        assert h[2].geom_type == "Point"
        assert list(h[2].coords) == [(5, 6, 7)]

    def test_linestring_remove_last_coord(self):
        g = geometry.LineString([(0, 1), (2, 3), (4, 5)])
        h = transform(lambda x, y, z=None: (x[:-1], y[:-1]), g, rebuild=2)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(0, 1), (2, 3)]

    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(2.0)

    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(3.5)
