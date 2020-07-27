import pytest
import pygeos
from pygeos import Geometry

from .common import point, all_types, polygon, multi_polygon

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


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize("n", range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested seperately from other set operations as it differs in two ways:
      1. It expects only non-overlapping polygons
      2. It expects GEOS 3.8.0+
    """
    test_data = [
        pygeos.box(0, 0, 1, 1),
        pygeos.box(1, 0, 2, 1),
        pygeos.box(2, 0, 3, 1),
    ]
    actual = pygeos.coverage_union_all(test_data[:n])
    # perform the reduction in a python loop and compare
    expected = test_data[0]
    for i in range(1, n):
        expected = pygeos.coverage_union(expected, test_data[i])
    assert pygeos.equals(actual, expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_coverage_union_reduce_axis():
    # shape = (3, 2), all polygons - none of them overlapping
    data = [[pygeos.box(i, j, i + 1, j + 1) for i in range(2)] for j in range(3)]
    actual = pygeos.coverage_union_all(data)
    assert actual.shape == (2,)
    actual = pygeos.coverage_union_all(data, axis=0)  # default
    assert actual.shape == (2,)
    actual = pygeos.coverage_union_all(data, axis=1)
    assert actual.shape == (3,)
    actual = pygeos.coverage_union_all(data, axis=-1)
    assert actual.shape == (3,)


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_coverage_union_overlapping_inputs():
    polygon = Geometry("POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))")

    # Overlapping polygons raise an error
    with pytest.raises(pygeos.GEOSException, match="CoverageUnion cannot process incorrectly noded inputs."):
        pygeos.coverage_union(polygon, Geometry("POLYGON ((1 0, 0.9 1, 2 1, 2 0, 1 0))"))


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom_1, geom_2",
    # All possible polygon, non_polygon combinations
    [
        [polygon, non_polygon]
        for non_polygon in non_polygon_types
    ] +
    # All possible non_polygon, non_polygon combinations
    [
        [non_polygon_1, non_polygon_2]
        for non_polygon_1 in non_polygon_types
        for non_polygon_2 in non_polygon_types
    ],
)
def test_coverage_union_non_polygon_inputs(geom_1, geom_2):
    # Non polygon geometries raise an error
    with pytest.raises(pygeos.GEOSException, match="Unhandled geometry type in CoverageUnion."):
        pygeos.coverage_union(geom_1, geom_2)
