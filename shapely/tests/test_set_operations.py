import numpy as np
import pytest

import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty, ignore_invalid, point, polygon

pytestmark = pytest.mark.filterwarnings(
    "ignore:The symmetric_difference_all function:DeprecationWarning"
)

# fixed-precision operations raise GEOS exceptions on mixed dimension geometry
# collections
all_single_types = np.array(all_types)[
    ~shapely.is_empty(all_types)
    & (shapely.get_type_id(all_types) != shapely.GeometryType.GEOMETRYCOLLECTION)
]

SET_OPERATIONS = (
    shapely.difference,
    shapely.intersection,
    shapely.symmetric_difference,
    shapely.union,
    # shapely.coverage_union is tested separately
)

REDUCE_SET_OPERATIONS = (
    (shapely.intersection_all, shapely.intersection),
    (shapely.symmetric_difference_all, shapely.symmetric_difference),
    (shapely.union_all, shapely.union),
    #  shapely.coverage_union_all, shapely.coverage_union) is tested separately
)

# operations that support fixed precision
REDUCE_SET_OPERATIONS_PREC = ((shapely.union_all, shapely.union),)

if shapely.geos_version >= (3, 12, 0):
    SET_OPERATIONS += (shapely.disjoint_subset_union,)
    REDUCE_SET_OPERATIONS += (
        (shapely.disjoint_subset_union_all, shapely.disjoint_subset_union),
    )

reduce_test_data = [
    shapely.box(0, 0, 5, 5),
    shapely.box(2, 2, 7, 7),
    shapely.box(4, 4, 9, 9),
    shapely.box(5, 5, 10, 10),
]

non_polygon_types = np.array(all_types)[
    ~shapely.is_empty(all_types) & (shapely.get_dimensions(all_types) != 2)
]


@pytest.mark.parametrize("a", all_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_array(a, func):
    actual = func(a, point)
    assert isinstance(actual, Geometry)

    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", SET_OPERATIONS)
def test_set_operation_prec_nonscalar_grid_size(func):
    if func is shapely.disjoint_subset_union:
        pytest.skip("disjoint_subset_union does not support grid_size")
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func(point, point, grid_size=[1])


@pytest.mark.parametrize("a", all_single_types)
@pytest.mark.parametrize("func", SET_OPERATIONS)
@pytest.mark.parametrize("grid_size", [0, 1, 2])
def test_set_operation_prec_array(a, func, grid_size):
    if func is shapely.disjoint_subset_union:
        pytest.skip("disjoint_subset_union does not support grid_size")
    actual = func([a, a], point, grid_size=grid_size)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)

    # results should match the operation when the precision is previously set
    # to same grid_size
    b = shapely.set_precision(a, grid_size=grid_size)
    point2 = shapely.set_precision(point, grid_size=grid_size)
    expected = func([b, b], point2)

    assert shapely.equals(shapely.normalize(actual), shapely.normalize(expected)).all()


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_1dim(n, func, related_func):
    actual = func(reduce_test_data[:n])
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i])
    assert shapely.equals(actual, expected)


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_single_geom(func, related_func):
    geom = shapely.Point(1, 1)
    actual = func([geom, None, None])
    assert shapely.equals(actual, geom)


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


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_empty(func, related_func):
    assert func(np.empty((0,), dtype=object)) == empty
    arr_empty_2D = np.empty((0, 2), dtype=object)
    assert func(arr_empty_2D) == empty
    assert func(arr_empty_2D, axis=0).tolist() == [empty] * 2
    assert func(arr_empty_2D, axis=1).tolist() == []


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


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_some_none_len2(func, related_func):
    # in a previous implementation, this would take a different code path
    # and return wrong result
    assert func([empty, None]) == empty


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none(n, func, related_func):
    assert_geometries_equal(func([None] * n), GeometryCollection([]))


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_all_none_arr(n, func, related_func):
    assert func([[None] * n] * 2, axis=1).tolist() == [empty, empty]
    assert func([[None] * 2] * n, axis=0).tolist() == [empty, empty]


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_nonscalar_grid_size(func, related_func):
    with pytest.raises(
        ValueError, match="grid_size parameter only accepts scalar values"
    ):
        func([point, point], grid_size=[1])


@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_grid_size_nan(func, related_func):
    actual = func([point, point], grid_size=np.nan)
    assert actual is None


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
@pytest.mark.parametrize("grid_size", [0, 1])
def test_set_operation_prec_reduce_1dim(n, func, related_func, grid_size):
    actual = func(reduce_test_data[:n], grid_size=grid_size)
    # perform the reduction in a python loop and compare
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i], grid_size=grid_size)

    assert shapely.equals(actual, expected)


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


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_one_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("none_position", range(3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_two_none(func, related_func, none_position):
    test_data = reduce_test_data[:2]
    test_data.insert(none_position, None)
    test_data.insert(none_position, None)
    actual = func(test_data, grid_size=1)
    expected = related_func(reduce_test_data[0], reduce_test_data[1], grid_size=1)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("n", range(1, 3))
@pytest.mark.parametrize("func, related_func", REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_all_none(n, func, related_func):
    assert_geometries_equal(func([None] * n, grid_size=1), GeometryCollection([]))


@pytest.mark.parametrize("n", range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested separately from other set operations as it expects only
    non-overlapping polygons
    """
    test_data = [
        shapely.box(0, 0, 1, 1),
        shapely.box(1, 0, 2, 1),
        shapely.box(2, 0, 3, 1),
    ]
    actual = shapely.coverage_union_all(test_data[:n])
    # perform the reduction in a python loop and compare
    expected = test_data[0]
    for i in range(1, n):
        expected = shapely.coverage_union(expected, test_data[i])
    assert_geometries_equal(actual, expected, normalize=True)


def test_coverage_union_reduce_axis():
    # shape = (3, 2), all polygons - none of them overlapping
    data = [[shapely.box(i, j, i + 1, j + 1) for i in range(2)] for j in range(3)]
    actual = shapely.coverage_union_all(data, axis=None)  # default
    assert isinstance(actual, Geometry)
    actual = shapely.coverage_union_all(data, axis=0)
    assert actual.shape == (2,)
    actual = shapely.coverage_union_all(data, axis=1)
    assert actual.shape == (3,)
    actual = shapely.coverage_union_all(data, axis=-1)
    assert actual.shape == (3,)


def test_coverage_union_overlapping_inputs():
    polygon = Polygon([(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)])
    other = Polygon([(1, 0), (0.9, 1), (2, 1), (2, 0), (1, 0)])

    if shapely.geos_version >= (3, 14, 0):
        # Overlapping polygons raise an error again
        with pytest.raises(shapely.GEOSException, match="TopologyException"):
            shapely.coverage_union(polygon, other)
    elif shapely.geos_version >= (3, 12, 0):
        # Return mostly unchanged output
        result = shapely.coverage_union(polygon, other)
        expected = shapely.multipolygons([polygon, other])
        assert_geometries_equal(result, expected, normalize=True)
    else:
        # Overlapping polygons raise an error
        with pytest.raises(
            shapely.GEOSException,
            match="CoverageUnion cannot process incorrectly noded inputs.",
        ):
            shapely.coverage_union(polygon, other)


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
    if shapely.geos_version >= (3, 12, 0):

        def effective_geom_types(geom):
            if hasattr(geom, "geoms") and not geom.is_empty:
                gts = set()
                for part in geom.geoms:
                    gts |= effective_geom_types(part)
                return gts
            return {geom.geom_type.lstrip("Multi").replace("LinearRing", "LineString")}

        geom_types_1 = effective_geom_types(geom_1)
        geom_types_2 = effective_geom_types(geom_2)
        if len(geom_types_1) == 1 and geom_types_1 == geom_types_2:
            with ignore_invalid():
                # these show "invalid value encountered in coverage_union"
                result = shapely.coverage_union(geom_1, geom_2)
            assert geom_types_1 == effective_geom_types(result)
        else:
            with pytest.raises(
                shapely.GEOSException, match="Overlay input is mixed-dimension"
            ):
                shapely.coverage_union(geom_1, geom_2)
    else:
        # Non polygon geometries raise an error
        with pytest.raises(
            shapely.GEOSException, match="Unhandled geometry type in CoverageUnion."
        ):
            shapely.coverage_union(geom_1, geom_2)


@pytest.mark.parametrize(
    "geom,grid_size,expected",
    [
        # floating point precision, expect no change
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            0,
            Polygon(
                (
                    (0, 0.2),
                    (0, 10),
                    (5.1, 10),
                    (5.1, 0.2),
                    (5, 0.2),
                    (5, 0.1),
                    (0.1, 0.1),
                    (0.1, 0.2),
                    (0, 0.2),
                )
            ),
        ),
        # grid_size is at effective precision, expect no change
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            0.1,
            Polygon(
                (
                    (0, 0.2),
                    (0, 10),
                    (5.1, 10),
                    (5.1, 0.2),
                    (5, 0.2),
                    (5, 0.1),
                    (0.1, 0.1),
                    (0.1, 0.2),
                    (0, 0.2),
                )
            ),
        ),
        # grid_size forces rounding to nearest integer
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            1,
            Polygon([(0, 5), (0, 10), (5, 10), (5, 5), (5, 0), (0, 0), (0, 5)]),
        ),
        # grid_size much larger than effective precision causes rounding to nearest
        # multiple of 10
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            10,
            Polygon([(0, 10), (10, 10), (10, 0), (0, 0), (0, 10)]),
        ),
        # grid_size is so large that polygons collapse to empty
        (
            [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)],
            100,
            Polygon(),
        ),
    ],
)
def test_union_all_prec(geom, grid_size, expected):
    actual = shapely.union_all(geom, grid_size=grid_size)
    assert shapely.equals(actual, expected)


def test_union_all_error():
    polygons = np.array(
        [
            [
                [-9.107924729871753, -49.578770022602505],
                [-11.477396284004609, -55.48211427005445],
                [-8.779906989660152, -68.10055885244654],
                [-9.107924729871753, -49.578770022602505],
            ],
            [
                [-10.944553152533185, -68.40940481609661],
                [-8.779906989660152, -68.10055885244654],
                [-11.477396284004609, -55.48211427005445],
                [-10.944553152533185, -68.40940481609661],
            ],
            [
                [-9.665730626080167, -46.63520657856243],
                [-9.107924729871753, -49.578770022602505],
                [-6.1606394792661385, -39.56368325939373],
                [-9.665730626080167, -46.63520657856243],
            ],
            [
                [-9.107924729871753, -49.578770022602505],
                [-9.665730626080167, -46.63520657856243],
                [-11.477396284004609, -55.48211427005445],
                [-9.107924729871753, -49.578770022602505],
            ],
            [
                [-5.60266871361742, -43.06086475203977],
                [-9.107924729871753, -49.578770022602505],
                [-4.725701460041062, -67.88249984010983],
                [-5.60266871361742, -43.06086475203977],
            ],
            [
                [-8.779906989660152, -68.10055885244654],
                [-4.725701460041062, -67.88249984010983],
                [-9.107924729871753, -49.578770022602505],
                [-8.779906989660152, -68.10055885244654],
            ],
            [
                [-6.1606394792661385, -39.56368325939373],
                [-5.60266871361742, -43.06086475203977],
                [-4.414116785691194, -23.00105208875605],
                [-6.1606394792661385, -39.56368325939373],
            ],
            [
                [-5.60266871361742, -43.06086475203977],
                [-6.1606394792661385, -39.56368325939373],
                [-9.107924729871753, -49.578770022602505],
                [-5.60266871361742, -43.06086475203977],
            ],
            [
                [2.1655564768683804, -59.48149581506282],
                [-1.2937332149450005, -65.40092352724034],
                [-0.47975546014527914, -67.79429254345098],
                [2.1655564768683804, -59.48149581506282],
            ],
            [
                [-4.725701460041062, -67.88249984010983],
                [-0.47975546014527914, -67.79429254345098],
                [-1.2937332149450005, -65.40092352724034],
                [-4.725701460041062, -67.88249984010983],
            ],
            [
                [-2.053526421384758, -39.38388953414288],
                [-5.60266871361742, -43.06086475203977],
                [-1.2937332149450005, -65.40092352724034],
                [-2.053526421384758, -39.38388953414288],
            ],
            [
                [-4.725701460041062, -67.88249984010983],
                [-1.2937332149450005, -65.40092352724034],
                [-5.60266871361742, -43.06086475203977],
                [-4.725701460041062, -67.88249984010983],
            ],
            [
                [-2.053526421384758, -39.38388953414288],
                [2.1655564768683804, -59.48149581506282],
                [2.5949143307751905, -38.432078492985355],
                [-2.053526421384758, -39.38388953414288],
            ],
            [
                [2.1655564768683804, -59.48149581506282],
                [-2.053526421384758, -39.38388953414288],
                [-1.2937332149450005, -65.40092352724034],
                [2.1655564768683804, -59.48149581506282],
            ],
            [
                [2.5949143307751905, -38.432078492985355],
                [4.045017485226917, -44.591813884340915],
                [3.6619129070589773, -17.61026441148638],
                [2.5949143307751905, -38.432078492985355],
            ],
            [
                [4.045017485226917, -44.591813884340915],
                [2.5949143307751905, -38.432078492985355],
                [2.1655564768683804, -59.48149581506282],
                [4.045017485226917, -44.591813884340915],
            ],
            [
                [-3.7536959540663597, -8.859098859659706],
                [-5.923720686905879, -16.235821209122424],
                [-2.053526421384758, -39.38388953414288],
                [-3.7536959540663597, -8.859098859659706],
            ],
            [
                [-5.60266871361742, -43.06086475203977],
                [-2.053526421384758, -39.38388953414288],
                [-5.923720686905879, -16.235821209122424],
                [-5.60266871361742, -43.06086475203977],
            ],
            [
                [-4.04425967854734, -1.346139278400533],
                [-3.7536959540663597, -8.859098859659706],
                [-1.5732562941774, 7.904434391775184],
                [-4.04425967854734, -1.346139278400533],
            ],
            [
                [-3.7536959540663597, -8.859098859659706],
                [-4.04425967854734, -1.346139278400533],
                [-5.923720686905879, -16.235821209122424],
                [-3.7536959540663597, -8.859098859659706],
            ],
            [
                [4.843141390976509, -25.008359188703416],
                [0.4110324192521997, -31.95495216694697],
                [2.5949143307751905, -38.432078492985355],
                [4.843141390976509, -25.008359188703416],
            ],
            [
                [-2.053526421384758, -39.38388953414288],
                [2.5949143307751905, -38.432078492985355],
                [0.4110324192521997, -31.95495216694697],
                [-2.053526421384758, -39.38388953414288],
            ],
            [
                [-2.170090738900679, 1.6419854605789586],
                [-3.7536959540663597, -8.859098859659706],
                [0.4110324192521997, -31.95495216694697],
                [-2.170090738900679, 1.6419854605789586],
            ],
            [
                [-2.053526421384758, -39.38388953414288],
                [0.4110324192521997, -31.95495216694697],
                [-3.7536959540663597, -8.859098859659706],
                [-2.053526421384758, -39.38388953414288],
            ],
            [
                [-1.5732562941774, 7.904434391775184],
                [-2.170090738900679, 1.6419854605789586],
                [2.8024433800252715, 7.181446815562015],
                [-1.5732562941774, 7.904434391775184],
            ],
            [
                [-2.170090738900679, 1.6419854605789586],
                [-1.5732562941774, 7.904434391775184],
                [-3.7536959540663597, -8.859098859659706],
                [-2.170090738900679, 1.6419854605789586],
            ],
            [
                [2.8024433800252715, 7.181446815562015],
                [5.381901308363052, -5.870328981043782],
                [4.681265179678338, 8.397375883356942],
                [2.8024433800252715, 7.181446815562015],
            ],
            [
                [5.381901308363052, -5.870328981043782],
                [2.8024433800252715, 7.181446815562015],
                [4.843141390976509, -25.008359188703416],
                [5.381901308363052, -5.870328981043782],
            ],
            [
                [2.580837628690858, 7.786696889308906],
                [2.8024433800252715, 7.181446815562015],
                [4.681265179678338, 8.397375883356942],
                [2.580837628690858, 7.786696889308906],
            ],
            [
                [2.8024433800252715, 7.181446815562015],
                [2.580837628690858, 7.786696889308906],
                [-1.5732562941774, 7.904434391775184],
                [2.8024433800252715, 7.181446815562015],
            ],
            [
                [-0.6817281672365398, -10.358260061109213],
                [-2.319196218072649, -18.323526770995862],
                [0.6062394536853688, -36.171064870529165],
                [-0.6817281672365398, -10.358260061109213],
            ],
            [
                [-0.8479686316326367, -38.51013375163708],
                [0.6062394536853688, -36.171064870529165],
                [-2.319196218072649, -18.323526770995862],
                [-0.8479686316326367, -38.51013375163708],
            ],
            [
                [-0.6817281672365398, -10.358260061109213],
                [4.566295472788307, -31.108901689146226],
                [3.0243411341241897, -3.518368608349892],
                [-0.6817281672365398, -10.358260061109213],
            ],
            [
                [4.566295472788307, -31.108901689146226],
                [-0.6817281672365398, -10.358260061109213],
                [0.6062394536853688, -36.171064870529165],
                [4.566295472788307, -31.108901689146226],
            ],
            [
                [-2.141304908669232, 18.810314416253544],
                [-4.104516608506552, 12.112119256727777],
                [-0.6817281672365398, -10.358260061109213],
                [-2.141304908669232, 18.810314416253544],
            ],
            [
                [-2.319196218072649, -18.323526770995862],
                [-0.6817281672365398, -10.358260061109213],
                [-4.104516608506552, 12.112119256727777],
                [-2.319196218072649, -18.323526770995862],
            ],
            [
                [-2.141304908669232, 18.810314416253544],
                [3.0243411341241897, -3.518368608349892],
                [1.2563100802122635, 29.327045820015645],
                [-2.141304908669232, 18.810314416253544],
            ],
            [
                [3.0243411341241897, -3.518368608349892],
                [-2.141304908669232, 18.810314416253544],
                [-0.6817281672365398, -10.358260061109213],
                [3.0243411341241897, -3.518368608349892],
            ],
            [
                [-2.1982913417356875, 26.994559772208923],
                [-2.141304908669232, 18.810314416253544],
                [1.2563100802122635, 29.327045820015645],
                [-2.1982913417356875, 26.994559772208923],
            ],
            [
                [-2.141304908669232, 18.810314416253544],
                [-2.1982913417356875, 26.994559772208923],
                [-4.104516608506552, 12.112119256727777],
                [-2.141304908669232, 18.810314416253544],
            ],
            [
                [1.3464458329117068, 33.01070476711907],
                [1.2563100802122635, 29.327045820015645],
                [4.532507979820731, 35.61904346954334],
                [1.3464458329117068, 33.01070476711907],
            ],
            [
                [1.2563100802122635, 29.327045820015645],
                [1.3464458329117068, 33.01070476711907],
                [-2.1982913417356875, 26.994559772208923],
                [1.2563100802122635, 29.327045820015645],
            ],
            [
                [6.776547203732532, 2.2865090915982376],
                [3.0243411341241897, -3.518368608349892],
                [8.143645356366674, -24.8074378190185],
                [6.776547203732532, 2.2865090915982376],
            ],
            [
                [4.566295472788307, -31.108901689146226],
                [8.143645356366674, -24.8074378190185],
                [3.0243411341241897, -3.518368608349892],
                [4.566295472788307, -31.108901689146226],
            ],
            [
                [6.776547203732532, 2.2865090915982376],
                [9.854949279150409, -14.031791236751758],
                [8.051902692875965, 16.1745493529877],
                [6.776547203732532, 2.2865090915982376],
            ],
            [
                [9.854949279150409, -14.031791236751758],
                [6.776547203732532, 2.2865090915982376],
                [8.143645356366674, -24.8074378190185],
                [9.854949279150409, -14.031791236751758],
            ],
            [
                [5.3340444513882135, 30.498982366146663],
                [1.2563100802122635, 29.327045820015645],
                [6.776547203732532, 2.2865090915982376],
                [5.3340444513882135, 30.498982366146663],
            ],
            [
                [3.0243411341241897, -3.518368608349892],
                [6.776547203732532, 2.2865090915982376],
                [1.2563100802122635, 29.327045820015645],
                [3.0243411341241897, -3.518368608349892],
            ],
            [
                [5.3340444513882135, 30.498982366146663],
                [8.051902692875965, 16.1745493529877],
                [6.612777301917237, 36.063185154347444],
                [5.3340444513882135, 30.498982366146663],
            ],
            [
                [8.051902692875965, 16.1745493529877],
                [5.3340444513882135, 30.498982366146663],
                [6.776547203732532, 2.2865090915982376],
                [8.051902692875965, 16.1745493529877],
            ],
            [
                [4.532507979820731, 35.61904346954334],
                [5.3340444513882135, 30.498982366146663],
                [6.612777301917237, 36.063185154347444],
                [4.532507979820731, 35.61904346954334],
            ],
            [
                [5.3340444513882135, 30.498982366146663],
                [4.532507979820731, 35.61904346954334],
                [1.2563100802122635, 29.327045820015645],
                [5.3340444513882135, 30.498982366146663],
            ],
        ]
    )

    shapely.union_all([Polygon(p) for p in polygons])


def test_uary_union_alias():
    geoms = [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)]
    actual = shapely.unary_union(geoms, grid_size=1)
    expected = shapely.union_all(geoms, grid_size=1)
    assert shapely.equals(actual, expected)


def test_difference_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `difference` is deprecated"
    ):
        shapely.difference(point, point, None)


def test_intersection_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `intersection` is deprecated"
    ):
        shapely.intersection(point, point, None)


def test_intersection_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `axis` for `intersection_all` is deprecated"
    ):
        shapely.intersection_all([point, point], None)


def test_symmetric_difference_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `symmetric_difference` is deprecated"
    ):
        shapely.symmetric_difference(point, point, None)


def test_symmetric_difference_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `axis` for `symmetric_difference_all` is deprecated"
    ):
        shapely.symmetric_difference_all([point, point], None)


def test_union_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `union` is deprecated"
    ):
        shapely.union(point, point, None)


def test_union_all_deprecate_positional():
    with pytest.deprecated_call(
        match="positional argument `grid_size` for `union_all` is deprecated"
    ):
        shapely.union_all([point, point], None)
    with pytest.deprecated_call(
        match="positional arguments `grid_size` and `axis` for `union_all` "
        "are deprecated"
    ):
        shapely.union_all([point, point], None, None)


def test_coverage_union_all_deprecate_positional():
    data = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    with pytest.deprecated_call(
        match="positional argument `axis` for `coverage_union_all` is deprecated"
    ):
        shapely.coverage_union_all(data, None)
