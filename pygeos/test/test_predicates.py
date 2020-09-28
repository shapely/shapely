import pytest
import pygeos
from pygeos import Geometry
import numpy as np

from .common import point, all_types, polygon, geometry_collection

UNARY_PREDICATES = (
    pygeos.is_empty,
    pygeos.is_simple,
    pygeos.is_ring,
    pygeos.is_closed,
    pygeos.is_valid,
    pygeos.is_missing,
    pygeos.is_geometry,
    pygeos.is_valid_input,
    pytest.param(pygeos.is_ccw, marks=pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")),
)

BINARY_PREDICATES = (
    pygeos.disjoint,
    pygeos.touches,
    pygeos.intersects,
    pygeos.crosses,
    pygeos.within,
    pygeos.contains,
    pygeos.overlaps,
    pygeos.equals,
    pygeos.covers,
    pygeos.covered_by,
    pygeos.equals_exact,
)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_unary_array(geometry, func):
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual.dtype == np.bool


@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_unary_with_kwargs(func):
    out = np.empty((), dtype=np.uint8)
    actual = func(point, out=out)
    assert actual is out
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_unary_missing(func):
    if func in (pygeos.is_valid_input, pygeos.is_missing):
        assert func(None)
    else:
        assert not func(None)


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", BINARY_PREDICATES)
def test_binary_array(a, func):
    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert actual.dtype == np.bool


@pytest.mark.parametrize("func", BINARY_PREDICATES)
def test_binary_with_kwargs(func):
    out = np.empty((), dtype=np.uint8)
    actual = func(point, point, out=out)
    assert actual is out
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("func", BINARY_PREDICATES)
def test_binary_missing(func):
    actual = func(np.array([point, None, None]), np.array([None, point, None]))
    assert (~actual).all()


def test_equals_exact_tolerance():
    # specifying tolerance
    p1 = pygeos.points(50, 4)
    p2 = pygeos.points(50.1, 4.1)
    actual = pygeos.equals_exact([p1, p2, None], p1, tolerance=0.05)
    np.testing.assert_allclose(actual, [True, False, False])
    assert actual.dtype == np.bool
    actual = pygeos.equals_exact([p1, p2, None], p1, tolerance=0.2)
    np.testing.assert_allclose(actual, [True, True, False])
    assert actual.dtype == np.bool

    # default value for tolerance
    assert pygeos.equals_exact(p1, p1).item() is True
    assert pygeos.equals_exact(p1, p2).item() is False


def test_relate():
    p1 = pygeos.points(0, 0)
    p2 = pygeos.points(1, 1)
    actual = pygeos.relate(p1, p2)
    assert isinstance(actual, str)
    assert actual == "FF0FFF0F2"


@pytest.mark.parametrize("g1, g2", [(point, None), (None, point), (None, None)])
def test_relate_none(g1, g2):
    assert pygeos.relate(g1, g2) is None


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize("geom, expected", [
    (Geometry("LINEARRING (0 0, 0 1, 1 1, 0 0)"), False),
    (Geometry("LINEARRING (0 0, 1 1, 0 1, 0 0)"), True),
    (Geometry("LINESTRING (0 0, 0 1, 1 1, 0 0)"), False),
    (Geometry("LINESTRING (0 0, 1 1, 0 1, 0 0)"), True),
    (Geometry("LINESTRING (0 0, 1 1, 0 1)"), False),
    (Geometry("LINESTRING (0 0, 0 1, 1 1)"), False),
    (point, False),
    (polygon, False),
    (geometry_collection, False),
    (None, False),
])
def test_is_ccw(geom, expected):
    assert pygeos.is_ccw(geom) == expected
