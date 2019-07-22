import pytest
import pygeos
import numpy as np

from .common import point, all_types, point_polygon_testdata

UNARY_PREDICATES = (
    pygeos.is_empty,
    pygeos.is_simple,
    pygeos.is_ring,
    pygeos.is_closed,
    pygeos.is_valid,
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
)


def test_disjoint():
    actual = pygeos.disjoint(*point_polygon_testdata)
    expected = [True, True, False, False, False, True]
    np.testing.assert_equal(actual, expected)


def test_touches():
    actual = pygeos.touches(*point_polygon_testdata)
    expected = [False, False, True, False, True, False]
    np.testing.assert_equal(actual, expected)


def test_intersects():
    actual = pygeos.intersects(*point_polygon_testdata)
    expected = [False, False, True, True, True, False]
    np.testing.assert_equal(actual, expected)


def test_within():
    actual = pygeos.within(*point_polygon_testdata)
    expected = [False, False, False, True, False, False]
    np.testing.assert_equal(actual, expected)


def test_contains():
    actual = pygeos.contains(*reversed(point_polygon_testdata))
    expected = [False, False, False, True, False, False]
    np.testing.assert_equal(actual, expected)


def test_equals_exact():
    point1 = pygeos.points(0, 0)
    point2 = pygeos.points(0, 0.1)
    actual = pygeos.equals_exact(point1, point2, [0.01, 1.0])
    expected = [False, True]
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_unary_broadcasting(geometry, func):
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual.dtype == np.bool


@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_unary_nan(func):
    actual = func(np.array([np.nan, None]))
    if func in [pygeos.is_empty, pygeos.is_valid]:
        assert actual.all()
    else:
        assert (~actual).all()


@pytest.mark.parametrize("func", BINARY_PREDICATES)
def test_binary_nan(func):
    actual = func(
        np.array([point, np.nan, np.nan, point, None, None]),
        np.array([np.nan, point, np.nan, None, point, None]),
    )
    if func is pygeos.disjoint:
        assert actual.all()
    elif func is pygeos.equals:
        # an empty set equals an empty set. behaviour is different from NaN
        expected = [False, False, True, False, False, True]
        np.testing.assert_equal(actual, expected)
    else:
        assert (~actual).all()
