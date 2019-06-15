import pytest

import numpy as np
import pygeos

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.geometry import \
    Point, LineString, LinearRing, Polygon, MultiPoint, MultiLineString,\
    MultiPolygon, GeometryCollection


def horizontal_slider(n):
    return np.array([box(i / n, 0, 1 + i / n, 1) for i in range(n)])


def vertical_slider(n):
    return np.array([box(0, i / n, 1, 1 + i / n) for i in range(n)])


slider_testdata = (
    (horizontal_slider(1)[0], vertical_slider(1)[0]),
    (horizontal_slider(10), vertical_slider(1)[0]),
    (horizontal_slider(1)[0], vertical_slider(10)),
    (horizontal_slider(10), vertical_slider(10)),
)

point_polygon_testdata = (
    (Point(2, 2), box(2, 2, 4, 4)),
    ([Point(i, i) for i in range(6)], box(2, 2, 4, 4)),
)

unary_testdata = ((
    Point(2, 2),
    LineString([[0, 0], [1, 0], [1, 1]]),
    LinearRing(((0, 0), (0, 1), (1, 1), (1, 0))),
    Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))),
    MultiPoint([[0.0, 0.0], [1.0, 2.0]]),
    MultiLineString([[[0.0, 0.0], [1.0, 2.0]]]),
    MultiPolygon([
        Polygon(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))),
        Polygon(((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)))
    ]),
    GeometryCollection([Point(51, -1), LineString([(52, -1), (49, 2)])])
),)

linestring_testdata = (
    LineString([[0, 0], [1, 0], [1, 1]]),
    LineString([[i, i] for i in range(100)]),
)


distance_testdata = (
    (Point(1, 1), Point(2, 1)),
)

def _shp_to_arr(x):
    # util for converting the test geoms to ndarrays
    if isinstance(x, BaseGeometry):
        x = [x]
    return np.array(x, dtype=np.object)


@pytest.mark.parametrize("a", unary_testdata)
def test_G_b(a):
    actual = pygeos.is_ring(a)
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a),
    ):
        assert _actual == _a.is_ring


@pytest.mark.parametrize("a", unary_testdata)
def test_G_u1(a):
    actual = pygeos.geom_type_id(a)
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a),
    ):
        assert _actual == pygeos.GEOM_CLASSES.index(_a.__class__)


@pytest.mark.parametrize("a", linestring_testdata)
def test_G_i(a):
    actual = pygeos.get_num_points(a)
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a),
    ):
        assert _actual == len(_a.coords)


@pytest.mark.parametrize("a", linestring_testdata)
def test_Gi_G(a):
    inds = [2, 1]
    actual = pygeos.get_point_n(a, inds)
    for _actual, ind in zip(np.atleast_1d(actual), inds):
        expected = a.coords[ind]
        assert list(_actual['obj'].coords[0]) == list(expected)


@pytest.mark.parametrize("a, b", slider_testdata)
def test_G_d(a, b):
    actual = pygeos.area(a)
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a),
    ):
        assert _actual == pytest.approx(_a.area)


@pytest.mark.parametrize("a, b", point_polygon_testdata)
def test_GG_b(a, b):
    actual = pygeos.contains(a, b)
    for _actual, _a, _b in zip(
            np.atleast_1d(actual), _shp_to_arr(a), _shp_to_arr(b)
    ):
        assert _actual == _a.contains(_b)


@pytest.mark.parametrize("a, b", distance_testdata)
def test_GG_d(a, b):
    actual = pygeos.distance(a, b)
    for _actual, _a, _b in zip(
            np.atleast_1d(actual), _shp_to_arr(a), _shp_to_arr(b)
    ):
        assert _actual == pytest.approx(_a.distance(_b))


@pytest.mark.parametrize("a, b", slider_testdata)
def test_G_G(a, b):
    actual = pygeos.get_centroid(a)['obj']
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a)
    ):
        assert _actual.equals(_a.centroid)


@pytest.mark.parametrize("a, b", slider_testdata)
def test_Gd_G(a, b):
    actual = pygeos.simplify(a, 1.)['obj']
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a)
    ):
        assert _actual.equals(_a.simplify(1.))


@pytest.mark.parametrize("a, b", slider_testdata)
def test_GG_G(a, b):
    actual = pygeos.intersection(a, b)['obj']
    for _actual, _a, _b in zip(
            np.atleast_1d(actual), _shp_to_arr(a), _shp_to_arr(b)
    ):
        assert _actual.equals(_a.intersection(_b))


@pytest.mark.parametrize("a", unary_testdata)
def test_buffer(a):
    actual = pygeos.buffer(a, 1.4, 8)
    for _actual, _a in zip(
            np.atleast_1d(actual), _shp_to_arr(a),
    ):
        assert _actual['obj'].equals(_a.buffer(1.4, 8))


def test_garr_from_shapely():
    geoms = np.array(horizontal_slider(10))
    garr = pygeos.garr_from_shapely(geoms)
    for actual, expected in zip(garr['obj'], geoms):
        assert actual is expected
    for actual, expected in zip(garr['_ptr'], geoms):
        assert actual == expected.__geom__


def test_finalize_garr():
    geoms = np.array(horizontal_slider(10))
    garr = np.empty_like(geoms, dtype=pygeos.GEOM_DTYPE)
    garr['_ptr'] = [obj.__geom__ for obj in geoms]
    pygeos.garr_finalize(garr)

    for actual, expected in zip(garr['obj'], geoms):
        assert actual.equals(expected)

    # this is to not have a SIGABRT; actual and expected share to the same
    # underlying C GEOSGeometries therefore they will both try to free it.
    for actual in geoms:
        actual.__geom__ = None
