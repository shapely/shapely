import unittest

import pytest

from shapely import geometry
from shapely.errors import ShapelyDeprecationWarning
from shapely.ops import transform


class IdentityTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def func(self, x, y, z=None):
        return tuple(c for c in [x, y, z] if c)

    def test_empty(self):
        g = geometry.Point()
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.is_empty

    def test_point(self):
        g = geometry.Point(0, 1)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.geom_type == "Point"
        assert list(h.coords) == [(0, 1)]

    def test_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(0, 1), (2, 3)]

    def test_linearring(self):
        g = geometry.LinearRing([(0, 1), (2, 3), (2, 2), (0, 1)])
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.geom_type == "LinearRing"
        assert list(h.coords) == [(0, 1), (2, 3), (2, 2), (0, 1)]

    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(h.area)

    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(self.func, g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(h.area)


class LambdaTestCase(unittest.TestCase):
    """New geometry/coordseq method 'xy' makes numpy interop easier"""

    def test_point(self):
        g = geometry.Point(0, 1)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "Point"
        assert list(h.coords) == [(1.0, 2.0)]

    def test_line(self):
        g = geometry.LineString([(0, 1), (2, 3)])
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "LineString"
        assert list(h.coords) == [(1.0, 2.0), (3.0, 4.0)]

    def test_polygon(self):
        g = geometry.Point(0, 1).buffer(1.0)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "Polygon"
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(2.0)

    def test_multipolygon(self):
        g = geometry.MultiPoint([(0, 1), (0, 4)]).buffer(1.0)
        with pytest.warns(ShapelyDeprecationWarning):
            h = transform(lambda x, y, z=None: (x + 1.0, y + 1.0), g)
        assert h.geom_type == "MultiPolygon"
        assert g.area == pytest.approx(h.area)
        assert h.centroid.x == pytest.approx(1.0)
        assert h.centroid.y == pytest.approx(3.5)


class TypeTestCase(unittest.TestCase):
    """With the shapely 2.1 implemenation func gets the expected types"""

    def test_func_gets_tuples(self):
        def func(x, y):
            for coord in (x, y):
                assert type(coord) is tuple
                assert all(type(elem) is float for elem in coord)
            return (x, y)

        with pytest.warns(ShapelyDeprecationWarning):
            transform(func, geometry.LineString([(0, 1), (2, 3)]))

    def test_func_fallback_gets_floats(self):
        def func(x, y):
            if isinstance(x, tuple):
                raise TypeError("float expected")
            for coord in (x, y):
                assert type(coord) is float
            return (x, y)

        with pytest.warns(ShapelyDeprecationWarning):
            transform(func, geometry.LineString([(0, 1), (2, 3)]))
