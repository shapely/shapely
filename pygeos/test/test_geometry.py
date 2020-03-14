import numpy as np
import pygeos
import pytest

from .common import point
from .common import point_nan
from .common import line_string
from .common import linear_ring
from .common import polygon
from .common import polygon_with_hole
from .common import multi_point
from .common import multi_line_string
from .common import multi_polygon
from .common import geometry_collection
from .common import point_z
from .common import all_types


def test_get_num_points():
    actual = pygeos.get_num_points(all_types).tolist()
    assert actual == [0, 3, 5, 0, 0, 0, 0, 0, 0]


def test_get_num_interior_rings():
    actual = pygeos.get_num_interior_rings(all_types + (polygon_with_hole,)).tolist()
    assert actual == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def test_get_num_geometries():
    actual = pygeos.get_num_geometries(all_types).tolist()
    assert actual == [1, 1, 1, 1, 2, 1, 2, 2, 0]


@pytest.mark.parametrize(
    "geom",
    [
        point,
        polygon,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_point_non_linestring(geom):
    actual = pygeos.get_point(geom, [0, 2, -1])
    assert pygeos.is_missing(actual).all()


@pytest.mark.parametrize("geom", [line_string, linear_ring])
def test_get_point(geom):
    n = pygeos.get_num_points(geom)
    actual = pygeos.get_point(geom, [0, -n, n, -(n + 1)])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_missing(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_exterior_ring_non_polygon(geom):
    actual = pygeos.get_exterior_ring(geom)
    assert pygeos.is_missing(actual).all()


def test_get_exterior_ring():
    actual = pygeos.get_exterior_ring([polygon, polygon_with_hole])
    assert (pygeos.get_type_id(actual) == 2).all()


@pytest.mark.parametrize(
    "geom",
    [
        point,
        line_string,
        linear_ring,
        multi_point,
        multi_line_string,
        multi_polygon,
        geometry_collection,
    ],
)
def test_get_interior_ring_non_polygon(geom):
    actual = pygeos.get_interior_ring(geom, [0, 2, -1])
    assert pygeos.is_missing(actual).all()


def test_get_interior_ring():
    actual = pygeos.get_interior_ring(polygon_with_hole, [0, -1, 1, -2])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_missing(actual[2:4]).all()


@pytest.mark.parametrize("geom", [point, line_string, linear_ring, polygon])
def test_get_geometry_simple(geom):
    actual = pygeos.get_geometry(geom, [0, -1, 1, -2])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_missing(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom", [multi_point, multi_line_string, multi_polygon, geometry_collection]
)
def test_get_geometry_collection(geom):
    n = pygeos.get_num_geometries(geom)
    actual = pygeos.get_geometry(geom, [0, -n, n, -(n + 1)])
    assert pygeos.equals(actual[0], actual[1]).all()
    assert pygeos.is_missing(actual[2:4]).all()


def test_get_type_id():
    actual = pygeos.get_type_id(all_types).tolist()
    assert actual == [0, 1, 2, 3, 4, 5, 6, 7, 7]


def test_get_dimensions():
    actual = pygeos.get_dimensions(all_types).tolist()
    assert actual == [0, 1, 1, 2, 0, 1, 2, 1, -1]


def test_get_coordinate_dimensions():
    actual = pygeos.get_coordinate_dimensions([point, point_z]).tolist()
    assert actual == [2, 3]


def test_get_num_coordinates():
    actual = pygeos.get_num_coordinates(all_types).tolist()
    assert actual == [1, 3, 5, 5, 2, 2, 10, 3, 0]


def test_get_set_srid():
    actual = pygeos.set_srid(point, 4326)
    assert pygeos.get_srid(point) == 0
    assert pygeos.get_srid(actual) == 4326


@pytest.mark.parametrize("func", [pygeos.get_x, pygeos.get_y])
@pytest.mark.parametrize("geom", all_types[1:])
def test_get_xy_no_point(func, geom):
    assert np.isnan(func(geom))


def test_get_x():
    assert pygeos.get_x([point, point_z]).tolist() == [2.0, 1.0]


def test_get_y():
    assert pygeos.get_y([point, point_z]).tolist() == [3.0, 1.0]


@pytest.mark.parametrize("geom", all_types)
def test_new_from_wkt(geom):
    actual = pygeos.Geometry(str(geom))
    assert pygeos.equals(actual, geom)


def test_adapt_ptr_raises():
    point = pygeos.Geometry("POINT (2 2)")
    with pytest.raises(AttributeError):
        point._ptr += 1


@pytest.mark.parametrize("geom", all_types + (pygeos.points(np.nan, np.nan),))
def test_hash_same_equal(geom):
    assert hash(geom) == hash(pygeos.apply(geom, lambda x: x))


@pytest.mark.parametrize("geom", all_types[:-1])
def test_hash_same_not_equal(geom):
    assert hash(geom) != hash(pygeos.apply(geom, lambda x: x + 1))


@pytest.mark.parametrize("geom", all_types)
def test_eq(geom):
    assert geom == pygeos.apply(geom, lambda x: x)


@pytest.mark.parametrize("geom", all_types[:-1])
def test_neq(geom):
    assert geom != pygeos.apply(geom, lambda x: x + 1)


@pytest.mark.parametrize("geom", all_types)
def test_set_unique(geom):
    a = {geom, pygeos.apply(geom, lambda x: x)}
    assert len(a) == 1


def test_eq_nan():
    assert point_nan != point_nan


def test_neq_nan():
    assert not (point_nan == point_nan)


def test_set_nan():
    # As NaN != NaN, you can have multiple "NaN" points in a set
    # set([float("nan"), float("nan")]) also returns a set with 2 elements
    a = set(pygeos.points([[np.nan, np.nan]] * 10))
    assert len(a) == 10  # different objects: NaN != NaN


def test_set_nan_same_objects():
    # You can't put identical objects in a set.
    # x = float("nan"); set([x, x]) also retuns a set with 1 element
    a = set([point_nan] * 10)
    assert len(a) == 1
