import pytest
import pygeos
import numpy as np

from .common import point, all_types

UNARY_PREDICATES = (
    pygeos.is_empty,
    pygeos.is_simple,
    pygeos.is_ring,
    pygeos.is_closed,
    pygeos.is_valid,
    pygeos.is_missing,
    pygeos.is_geometry,
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
    if func is pygeos.is_missing:
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
