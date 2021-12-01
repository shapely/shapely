import numpy as np
import pytest

import pygeos
from pygeos import Geometry
from pygeos.decorators import UnsupportedGEOSOperation
from pygeos.testing import assert_geometries_equal

from .common import all_types, multi_polygon, point, polygon

# fixed-precision operations raise GEOS exceptions on mixed dimension geometry collections
all_single_types = [g for g in all_types if not pygeos.get_type_id(g) == 7]

SET_OPERATIONS = (
    pygeos.difference,
    pygeos.intersection,
    pygeos.symmetric_difference,
    pygeos.union,
    # pygeos.coverage_union is tested seperately
)

REDUCE_SET_OPERATIONS = (
    (pygeos.intersection_all, pygeos.intersection),
    (pygeos.symmetric_difference_all, pygeos.symmetric_difference),
    (pygeos.union_all, pygeos.union),
    # (pygeos.coverage_union_all, pygeos.coverage_union) is tested seperately
)

# operations that support fixed precision
REDUCE_SET_OPERATIONS_PREC = ((pygeos.union_all, pygeos.union),)


reduce_test_data = [
    pygeos.box(0, 0, 5, 5),
    pygeos.box(2, 2, 7, 7),
    pygeos.box(4, 4, 9, 9),
    pygeos.box(5, 5, 10, 10),
]

non_polygon_types = [
    geom
    for geom in all_types
    if (not pygeos.is_empty(geom) and geom not in (polygon, multi_polygon))
]


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_array(a, func):
    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.skipif(pygeos.geos_version >= (3, 9, 0), reason="GEOS >= 3.9")
@pytest.mark.parametrize("func", SET_OPERATIONS)
@pytest.mark.parametrize("grid_size", [0, 1])
def test_set_operations_prec_not_supported(func, grid_size):
    with pytest.raises(
        UnsupportedGEOSOperation, match="grid_size parameter requires GEOS >= 3.9.0"
    ):
        func(point, point, grid_size)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_prec_nonscalar_grid_size(func):
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func(point, point, grid_size=[1])


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("a", all_single_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
@pytest.mark.parametrize("grid_size", [0, 1, 2])
def test_set_operation_prec_array(a, func, grid_size):
    actual = func([a, a], point, grid_size=grid_size)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)

    # results should match the operation when the precision is previously set
    # to same grid_size
    b = pygeos.set_precision(a, grid_size=grid_size)
    point2 = pygeos.set_precision(point, grid_size=grid_size)
    expected = func([b, b], point2)

    assert pygeos.equals(pygeos.normalize(actual), pygeos.normalize(expected)).all()


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
    actual = func(data, axis=None)  # default
    assert isinstance(actual, Geometry)  # scalar output
    actual = func(data, axis=0)
    assert actual.shape == (2,)
    actual = func(data, axis=1)
    assert actual.shape == (3,)
    actual = func(data, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_one_none(func, related_func, none_position):
    # API change: before, intersection_all and symmetric_difference_all returned
    # None if any input geometry was None.
    # The new behaviour is to ignore None values.
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    actual = func(test_data)
    expected = related_func(reduce_test_data[0], reduce_test_data[1])
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data)
    expected = related_func(reduce_test_data[0], reduce_test_data[1])
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none(n, func, related_func):
    # API change: before, union_all([None]) yielded EMPTY GEOMETRYCOLLECTION
    # The new behaviour is that it returns None if all inputs are None.
    assert func([None] * n) is None


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none_arr(n, func, related_func):
    # API change: before, union_all([None]) yielded EMPTY GEOMETRYCOLLECTION
    # The new behaviour is that it returns None if all inputs are None.
    assert func([[None] * n] * 2, axis=1).tolist() == [None, None]


@pytest.mark.skipif(pygeos.geos_version >= (3, 9, 0), reason="GEOS >= 3.9")
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
@pytest.mark.parametrize("grid_size", [0, 1])
def test_set_operation_prec_reduce_not_supported(func, related_func, grid_size):
    with pytest.raises(
        UnsupportedGEOSOperation, match="grid_size parameter requires GEOS >= 3.9.0"
    ):
        func([point, point], grid_size)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_nonscalar_grid_size(func, related_func):
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func([point, point], grid_size=[1])


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_grid_size_nan(func, related_func):
    actual = func([point, point], grid_size=np.nan)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
@pytest.mark.parametrize("grid_size", [0, 1])
def test_set_operation_prec_reduce_1dim(n, func, related_func, grid_size):
    actual = func(reduce_test_data[:n], grid_size=grid_size)
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i], grid_size=grid_size)

    assert pygeos.equals(actual, expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_axis(func, related_func):
    data = [[point] * 2] * 3  # shape = (3, 2)
    actual = func(data, grid_size=1, axis=None)  # default
    assert isinstance(actual, Geometry)  # scalar output
    actual = func(data, grid_size=1, axis=0)
    assert actual.shape == (2,)
    actual = func(data, grid_size=1, axis=1)
    assert actual.shape == (3,)
    actual = func(data, grid_size=1, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_one_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_all_none(n, func, related_func):
    # API change: before, union_all([None]) yielded EMPTY GEOMETRYCOLLECTION
    # The new behaviour is that it returns None if all inputs are None.
    assert func([None] * n, grid_size=1) is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize("n", range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested seperately from other set operations as it differs in two ways:
      1. It expects only non-overlapping polygons
      2. It expects GEOS 3.8.0+
    """
    test_data = [pygeos.box(0, 0, 1, 1), pygeos.box(1, 0, 2, 1), pygeos.box(2, 0, 3, 1)]
    actual = pygeos.coverage_union_all(test_data[:n])
    # perform the reduction in a python loop and compare
    expected = test_data[0]
    for i in range(1, n):
        expected = pygeos.coverage_union(expected, test_data[i])
    assert_geometries_equal(actual, expected, normalize=True)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_coverage_union_reduce_axis():
    # shape = (3, 2), all polygons - none of them overlapping
    data = [[pygeos.box(i, j, i + 1, j + 1) for i in range(2)] for j in range(3)]
    actual = pygeos.coverage_union_all(data, axis=None)  # default
    assert isinstance(actual, Geometry)
    actual = pygeos.coverage_union_all(data, axis=0)
    assert actual.shape == (2,)
    actual = pygeos.coverage_union_all(data, axis=1)
    assert actual.shape == (3,)
    actual = pygeos.coverage_union_all(data, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_coverage_union_overlapping_inputs():
    polygon = Geometry("POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))")

    # Overlapping polygons raise an error
    with pytest.raises(
        pygeos.GEOSException,
        match="CoverageUnion cannot process incorrectly noded inputs.",
    ):
        pygeos.coverage_union(
            polygon, Geometry("POLYGON ((1 0, 0.9 1, 2 1, 2 0, 1 0))")
        )


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom_1, geom_2",
    # All possible polygon, non_polygon combinations
    [[polygon, non_polygon] for non_polygon in non_polygon_types]
    # All possible non_polygon, non_polygon combinations
    + [
        [non_polygon_1, non_polygon_2]
        for non_polygon_1 in non_polygon_types
        for non_polygon_2 in non_polygon_types
    ],
)
def test_coverage_union_non_polygon_inputs(geom_1, geom_2):
    # Non polygon geometries raise an error
    with pytest.raises(
        pygeos.GEOSException, match="Unhandled geometry type in CoverageUnion."
    ):
        pygeos.coverage_union(geom_1, geom_2)


@pytest.mark.skipif(pygeos.geos_version < (3, 9, 0), reason="GEOS < 3.9")
@pytest.mark.parametrize(
    "geom,grid_size,expected",
    [
        # floating point precision, expect no change
        (
            [pygeos.box(0.1, 0.1, 5, 5), pygeos.box(0, 0.2, 5.1, 10)],
            0,
            pygeos.Geometry(
                "POLYGON ((0 0.2, 0 10, 5.1 10, 5.1 0.2, 5 0.2, 5 0.1, 0.1 0.1, 0.1 0.2, 0 0.2))"
            ),
        ),
        # grid_size is at effective precision, expect no change
        (
            [pygeos.box(0.1, 0.1, 5, 5), pygeos.box(0, 0.2, 5.1, 10)],
            0.1,
            pygeos.Geometry(
                "POLYGON ((0 0.2, 0 10, 5.1 10, 5.1 0.2, 5 0.2, 5 0.1, 0.1 0.1, 0.1 0.2, 0 0.2))"
            ),
        ),
        # grid_size forces rounding to nearest integer
        (
            [pygeos.box(0.1, 0.1, 5, 5), pygeos.box(0, 0.2, 5.1, 10)],
            1,
            pygeos.Geometry("POLYGON ((0 5, 0 10, 5 10, 5 5, 5 0, 0 0, 0 5))"),
        ),
        # grid_size much larger than effective precision causes rounding to nearest
        # multiple of 10
        (
            [pygeos.box(0.1, 0.1, 5, 5), pygeos.box(0, 0.2, 5.1, 10)],
            10,
            pygeos.Geometry("POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))"),
        ),
        # grid_size is so large that polygons collapse to empty
        (
            [pygeos.box(0.1, 0.1, 5, 5), pygeos.box(0, 0.2, 5.1, 10)],
            100,
            pygeos.Geometry("POLYGON EMPTY"),
        ),
    ],
)
def test_union_all_prec(geom, grid_size, expected):
    actual = pygeos.union_all(geom, grid_size=grid_size)
    assert pygeos.equals(actual, expected)
