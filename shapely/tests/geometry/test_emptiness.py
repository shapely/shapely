import numpy as np

from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    mapping,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.geometry.base import BaseGeometry, EmptyGeometry


def empty_generator():
    return iter([])


class TestEmptiness:
    def test_empty_class(self):
        g = EmptyGeometry()
        assert g.is_empty

    def test_empty_base(self):
        g = BaseGeometry()
        assert g.is_empty

    def test_empty_point(self):
        assert Point().is_empty

    def test_empty_multipoint(self):
        assert MultiPoint().is_empty

    def test_empty_geometry_collection(self):
        assert GeometryCollection().is_empty

    def test_empty_linestring(self):
        assert LineString().is_empty
        assert LineString(None).is_empty
        assert LineString([]).is_empty
        assert LineString(empty_generator()).is_empty

    def test_empty_multilinestring(self):
        assert MultiLineString([]).is_empty

    def test_empty_polygon(self):
        assert Polygon().is_empty
        assert Polygon(None).is_empty
        assert Polygon([]).is_empty
        assert Polygon(empty_generator()).is_empty

    def test_empty_multipolygon(self):
        assert MultiPolygon([]).is_empty

    def test_empty_linear_ring(self):
        assert LinearRing().is_empty
        assert LinearRing(None).is_empty
        assert LinearRing([]).is_empty
        assert LinearRing(empty_generator()).is_empty


def test_numpy_object_array():
    geoms = [BaseGeometry(), EmptyGeometry()]
    arr = np.empty(2, object)
    arr[:] = geoms


def test_shape_empty():
    empty_mp = MultiPolygon()
    empty_json = mapping(empty_mp)
    empty_shape = shape(empty_json)
    assert empty_shape.is_empty
