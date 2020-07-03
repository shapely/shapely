import pytest
import pygeos
from pygeos import Geometry

from .common import point, all_types

SET_OPERATIONS = (
    pygeos.difference,
    pygeos.intersection,
    pygeos.symmetric_difference,
    pygeos.union,
)

REDUCE_SET_OPERATIONS = (
    (pygeos.intersection_all, pygeos.intersection),
    (pygeos.symmetric_difference_all, pygeos.symmetric_difference),
    (pygeos.union_all, pygeos.union),
)

reduce_test_data = [
    pygeos.box(0, 0, 5, 5),
    pygeos.box(2, 2, 7, 7),
    pygeos.box(4, 4, 9, 9),
    pygeos.box(5, 5, 10, 10),
]


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_array(a, func):
    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_1dim(n, func, related_func):
    actual = func(reduce_test_data[:n])
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i])
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_axis(func, related_func):
    data = [[point] * 2] * 3  # shape = (3, 2)
    actual = func(data)
    assert actual.shape == (2,)
    actual = func(data, axis=0)  # default
    assert actual.shape == (2,)
    actual = func(data, axis=1)
    assert actual.shape == (3,)
    actual = func(data, axis=-1)
    assert actual.shape == (3,)
