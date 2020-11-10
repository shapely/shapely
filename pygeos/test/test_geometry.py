import sys
import numpy as np
import pygeos
import pytest

from .common import point
from .common import point_nan
from .common import empty_point
from .common import line_string
from .common import empty_line_string
from .common import linear_ring
from .common import polygon
from .common import polygon_with_hole
from .common import empty_polygon
from .common import multi_point
from .common import multi_line_string
from .common import multi_polygon
from .common import geometry_collection
from .common import empty as empty_geometry_collection
from .common import point_z
from .common import all_types


def test_get_num_points():
    actual = pygeos.get_num_points(all_types + (None,)).tolist()
    assert actual == [0, 3, 5, 0, 0, 0, 0, 0, 0, 0]


def test_get_num_interior_rings():
    actual = pygeos.get_num_interior_rings(all_types + (polygon_with_hole, None))
    assert actual.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_get_num_geometries():
    actual = pygeos.get_num_geometries(all_types + (None,)).tolist()
    assert actual == [1, 1, 1, 1, 2, 1, 2, 2, 0, 0]


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


def test_get_coordinate_dimension():
    actual = pygeos.get_coordinate_dimension([point, point_z, None]).tolist()
    assert actual == [2, 3, -1]


def test_get_num_coordinates():
    actual = pygeos.get_num_coordinates(all_types + (None,)).tolist()
    assert actual == [1, 3, 5, 5, 2, 2, 10, 3, 0, 0]


def test_get_srid():
    """All geometry types have no SRID by default; None returns -1"""
    actual = pygeos.get_srid(all_types + (None,)).tolist()
    assert actual == [0, 0, 0, 0, 0, 0, 0, 0, 0, -1]


def test_get_set_srid():
    actual = pygeos.set_srid(point, 4326)
    assert pygeos.get_srid(point) == 0
    assert pygeos.get_srid(actual) == 4326


@pytest.mark.parametrize(
    "func",
    [
        pygeos.get_x,
        pygeos.get_y,
        pytest.param(
            pygeos.get_z,
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7"
            ),
        ),
    ],
)
@pytest.mark.parametrize("geom", all_types[1:])
def test_get_xyz_no_point(func, geom):
    assert np.isnan(func(geom))


def test_get_x():
    assert pygeos.get_x([point, point_z]).tolist() == [2.0, 1.0]


def test_get_y():
    assert pygeos.get_y([point, point_z]).tolist() == [3.0, 1.0]


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_get_z():
    assert pygeos.get_z([point_z]).tolist() == [1.0]


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_get_z_2d():
    assert np.isnan(pygeos.get_z(point))


@pytest.mark.parametrize("geom", all_types)
def test_new_from_wkt(geom):
    actual = pygeos.Geometry(str(geom))
    assert pygeos.equals(actual, geom)


def test_adapt_ptr_raises():
    point = pygeos.Geometry("POINT (2 2)")
    with pytest.raises(AttributeError):
        point._ptr += 1


@pytest.mark.parametrize(
    "geom", all_types + (pygeos.points(np.nan, np.nan), empty_point)
)
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


@pytest.mark.parametrize(
    "geom",
    [
        point,
        multi_point,
        line_string,
        multi_line_string,
        polygon,
        multi_polygon,
        geometry_collection,
        empty_point,
        empty_line_string,
        empty_polygon,
        empty_geometry_collection,
    ],
)
def test_get_parts(geom):
    expected_num_parts = pygeos.get_num_geometries(geom)
    expected_parts = pygeos.get_geometry(geom, range(0, expected_num_parts))

    parts = pygeos.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert np.all(pygeos.equals_exact(parts, expected_parts))


def test_get_parts_array():
    # note: this also verifies that None is handled correctly
    # in the mix; internally it returns -1 for count of geometries
    geom = np.array([None, empty_line_string, multi_point, point, multi_polygon])
    expected_parts = []
    for g in geom:
        for i in range(0, pygeos.get_num_geometries(g)):
            expected_parts.append(pygeos.get_geometry(g, i))

    parts = pygeos.get_parts(geom)
    assert len(parts) == len(expected_parts)
    assert np.all(pygeos.equals_exact(parts, expected_parts))


def test_get_parts_geometry_collection_multi():
    """On the first pass, the individual Multi* geometry objects are returned
    from the collection.  On the second pass, the individual singular geometry
    objects within those are returned.
    """
    geom = pygeos.geometrycollections([multi_point, multi_line_string, multi_polygon])
    expected_num_parts = pygeos.get_num_geometries(geom)
    expected_parts = pygeos.get_geometry(geom, range(0, expected_num_parts))

    parts = pygeos.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert np.all(pygeos.equals_exact(parts, expected_parts))

    expected_subparts = []
    for g in np.asarray(expected_parts):
        for i in range(0, pygeos.get_num_geometries(g)):
            expected_subparts.append(pygeos.get_geometry(g, i))

    subparts = pygeos.get_parts(parts)
    assert len(subparts) == len(expected_subparts)
    assert np.all(pygeos.equals_exact(subparts, expected_subparts))


def test_get_parts_return_index():
    geom = np.array([multi_point, point, multi_polygon])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        for j in range(0, pygeos.get_num_geometries(g)):
            expected_parts.append(pygeos.get_geometry(g, j))
            expected_index.append(i)

    parts, index = pygeos.get_parts(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert np.all(pygeos.equals_exact(parts, expected_parts))
    assert np.array_equal(index, expected_index)


@pytest.mark.parametrize(
    "geom",
    ([[None]], [[empty_point]], [[multi_point]], [[multi_point, multi_line_string]]),
)
def test_get_parts_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match="Array should be one dimensional"):
        pygeos.get_parts(geom)


@pytest.mark.parametrize(
    "geom", [point, line_string, polygon],
)
def test_get_parts_non_multi(geom):
    """Non-multipart geometries should be returned identical to inputs"""
    assert np.all(pygeos.equals_exact(np.asarray(geom), pygeos.get_parts(geom)))


@pytest.mark.parametrize(
    "geom", [None, [None], []],
)
def test_get_parts_None(geom):
    assert len(pygeos.get_parts(geom)) == 0


@pytest.mark.parametrize(
    "geom", ["foo", ["foo"], 42],
)
def test_get_parts_invalid_geometry(geom):
    with pytest.raises(
        TypeError, match="One of the arguments is of incorrect type.",
    ):
        pygeos.get_parts(geom)

