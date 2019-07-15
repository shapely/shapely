import numpy as np
import pygeos
import pytest

from .common import point_polygon_testdata
from .common import point
from .common import line_string
from .common import polygon
from .common import point_z

from .common import UNARY_PREDICATES
from .common import BINARY_PREDICATES
from .common import all_types

# Y_b


def test_has_z():
    actual = pygeos.has_z([point, point_z])
    expected = [False, True]
    np.testing.assert_equal(actual, expected)


# YY_b


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


# Y_Y


def test_get_centroid():
    actual = pygeos.get_centroid(polygon)
    assert pygeos.equals(actual, pygeos.points(1, 1))


# Yi_Y


def test_get_point_n():
    actual = pygeos.get_point_n(line_string, 1)
    assert pygeos.equals(actual, pygeos.points(1, 0))


def test_set_srid():
    actual = pygeos.set_srid(point, 4326)
    assert pygeos.get_srid(point) == 0
    assert pygeos.get_srid(actual) == 4326


# Yd_Y


def test_simplify():
    line = pygeos.linestrings([[0, 0], [0.1, 1], [0, 2]])
    actual = pygeos.simplify(line, [0, 1.0])
    assert pygeos.get_num_points(actual).tolist() == [3, 2]


# YY_Y


def test_intersection():
    poly1, poly2 = pygeos.box(0, 0, 10, 10), pygeos.box(5, 5, 20, 20)
    actual = pygeos.intersection(poly1, poly2)
    expected = pygeos.box(5, 5, 10, 10)
    assert pygeos.equals(actual, expected)


def test_union():
    poly1, poly2 = pygeos.box(0, 0, 10, 10), pygeos.box(10, 0, 20, 10)
    actual = pygeos.union(poly1, poly2)
    expected = pygeos.box(0, 0, 20, 10)
    assert pygeos.equals(actual, expected)


# Y_d


def test_area():
    assert pygeos.area(polygon) == 4.0


# Y_B


def test_geom_type_id():
    assert pygeos.geom_type_id(all_types).tolist() == list(range(8))


# Y_i


def test_get_num_points():
    assert pygeos.get_num_points(line_string) == 3


# YY_d


def test_distance():
    actual = pygeos.distance(*point_polygon_testdata)
    expected = [2 * 2 ** 0.5, 2 ** 0.5, 0, 0, 0, 2 ** 0.5]
    np.testing.assert_allclose(actual, expected)


def test_haussdorf_distance():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    actual = pygeos.hausdorff_distance(a, b)
    assert actual == pytest.approx(22.360679775, abs=1e-7)


# YY_d_2


def test_project():
    line = pygeos.linestrings([[0, 0], [1, 1], [2, 2]])
    points = pygeos.points([1, 3], [0, 3])
    actual = pygeos.project(line, points)
    expected = [0.5 * 2 ** 0.5, 2 * 2 ** 0.5]
    np.testing.assert_allclose(actual, expected)


# specials


def test_buffer():
    radii = np.array([1.0, 2.0])
    actual = pygeos.buffer(point, radii, 16)
    assert pygeos.area(actual) == pytest.approx(np.pi * radii ** 2, rel=0.01)


def test_buffer_with_style():
    radii = np.array([1.0, 2.0])
    actual = pygeos.buffer_with_style(point, radii, 16, 1, 1, 5)
    assert pygeos.area(actual) == pytest.approx(np.pi * radii ** 2, rel=0.01)


def test_snap():
    line = pygeos.linestrings([[0, 0], [1, 0], [2, 0]])
    points = pygeos.points([0, 1], [1, 0.1])
    actual = pygeos.snap(points, line, 0.5)
    expected = pygeos.points([0, 1], [1, 0])
    assert pygeos.equals(actual, expected).all()


def test_equals_exact():
    point1 = pygeos.points(0, 0)
    point2 = pygeos.points(0, 0.1)
    actual = pygeos.equals_exact(point1, point2, [0.01, 1.0])
    expected = [False, True]
    np.testing.assert_equal(actual, expected)


def test_haussdorf_distance_densify():
    # example from GEOS docs
    a = pygeos.linestrings([[0, 0], [100, 0], [10, 100], [10, 100]])
    b = pygeos.linestrings([[0, 100], [0, 10], [80, 10]])
    actual = pygeos.haussdorf_distance_densify(a, b, 0.001)
    assert actual == pytest.approx(47.8, abs=0.1)

# NaN / None handling


@pytest.mark.parametrize("func", UNARY_PREDICATES)
def test_Y_b_nan(func):
    actual = func(np.array([np.nan, None]))
    if func is pygeos.is_empty:
        assert actual.all()
    else:
        assert (~actual).all()

@pytest.mark.parametrize("func", BINARY_PREDICATES)
def test_YY_b_nan(func):
    actual = func(
        np.array([point, np.nan, np.nan, point, None, None]),
        np.array([np.nan, point, np.nan, None, point, None]),
    )
    if func is pygeos.disjoint:
        assert actual.all()
    else:
        assert (~actual).all()


def test_Y_Y_nan():
    actual = pygeos.clone(np.array([point, np.nan, None]))
    assert pygeos.equals(actual[0], point)
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])


def test_Y_d_nan():
    actual = pygeos.area(np.array([polygon, np.nan, None]))
    assert actual[0] == pygeos.area(polygon)
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])


def test_YY_Y_nan():
    actual = pygeos.intersection(
        np.array([point, np.nan, np.nan, point, None, None, point]),
        np.array([np.nan, point, np.nan, None, point, None, point]),
    )
    assert pygeos.equals(actual[-1], point)
    assert np.isnan(actual[:-1].astype(np.float)).all()


def test_Yd_Y_nan():
    actual = pygeos.simplify(
        np.array([point, np.nan, np.nan, None, point]),
        np.array([np.nan, 1.0, np.nan, 1.0, 1.0]),
    )
    assert pygeos.equals(actual[-1], point)
    assert np.isnan(actual[:-1].astype(np.float)).all()


def test_YY_d_nan():
    actual = pygeos.distance(
        np.array([point, np.nan, np.nan, point, None, None, point]),
        np.array([np.nan, point, np.nan, None, point, None, point]),
    )
    assert actual[-1] == 0.0
    assert np.isnan(actual[:-1].astype(np.float)).all()


def test_create_collection_only_nan():
    actual = pygeos.multipoints(np.array([np.nan], dtype=object))
    assert actual.to_wkt() == "MULTIPOINT EMPTY"


def test_create_collection_skips_nan():
    actual = pygeos.multipoints([point, np.nan, np.nan, point])
    assert actual.to_wkt() == "MULTIPOINT (2 2, 2 2)"
