import warnings

import numpy as np
import pytest

import pygeos
from pygeos.testing import assert_geometries_equal

from .common import all_types
from .common import empty as empty_geometry_collection
from .common import (
    empty_line_string,
    empty_line_string_z,
    empty_point,
    empty_point_z,
    empty_polygon,
    geometry_collection,
    geometry_collection_z,
    line_string,
    line_string_nan,
    line_string_z,
    linear_ring,
    multi_line_string,
    multi_line_string_z,
    multi_point,
    multi_point_z,
    multi_polygon,
    multi_polygon_z,
    point,
    point_z,
    polygon,
    polygon_with_hole,
    polygon_with_hole_z,
    polygon_z,
)


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
    assert_geometries_equal(actual[0], actual[1])
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
    assert_geometries_equal(actual[0], actual[1])
    assert pygeos.is_missing(actual[2:4]).all()


@pytest.mark.parametrize("geom", [point, line_string, linear_ring, polygon])
def test_get_geometry_simple(geom):
    actual = pygeos.get_geometry(geom, [0, -1, 1, -2])
    assert_geometries_equal(actual[0], actual[1])
    assert pygeos.is_missing(actual[2:4]).all()


@pytest.mark.parametrize(
    "geom", [multi_point, multi_line_string, multi_polygon, geometry_collection]
)
def test_get_geometry_collection(geom):
    n = pygeos.get_num_geometries(geom)
    actual = pygeos.get_geometry(geom, [0, -n, n, -(n + 1)])
    assert_geometries_equal(actual[0], actual[1])
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
    assert pygeos.get_x([point, point_z]).tolist() == [2.0, 2.0]


def test_get_y():
    assert pygeos.get_y([point, point_z]).tolist() == [3.0, 3.0]


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_get_z():
    assert pygeos.get_z([point_z]).tolist() == [4.0]


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_get_z_2d():
    assert np.isnan(pygeos.get_z(point))


@pytest.mark.parametrize("geom", all_types)
def test_new_from_wkt(geom):
    actual = pygeos.Geometry(str(geom))
    assert_geometries_equal(actual, geom)


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
    assert line_string_nan != line_string_nan


def test_neq_nan():
    assert not (line_string_nan == line_string_nan)


def test_set_nan():
    # As NaN != NaN, you can have multiple "NaN" points in a set
    # set([float("nan"), float("nan")]) also returns a set with 2 elements
    a = set(pygeos.linestrings([[[np.nan, np.nan], [np.nan, np.nan]]] * 10))
    assert len(a) == 10  # different objects: NaN != NaN


def test_set_nan_same_objects():
    # You can't put identical objects in a set.
    # x = float("nan"); set([x, x]) also retuns a set with 1 element
    a = set([line_string_nan] * 10)
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
        np.array([None]),
        np.empty_like(np.array([None])),
    ],
)
def test_get_parts(geom):
    expected_num_parts = pygeos.get_num_geometries(geom)
    if expected_num_parts == 0:
        expected_parts = []
    else:
        expected_parts = pygeos.get_geometry(geom, range(0, expected_num_parts))

    parts = pygeos.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert_geometries_equal(parts, expected_parts)


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
    assert_geometries_equal(parts, expected_parts)


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
    assert_geometries_equal(parts, expected_parts)

    expected_subparts = []
    for g in np.asarray(expected_parts):
        for i in range(0, pygeos.get_num_geometries(g)):
            expected_subparts.append(pygeos.get_geometry(g, i))

    subparts = pygeos.get_parts(parts)
    assert len(subparts) == len(expected_subparts)
    assert_geometries_equal(subparts, expected_subparts)


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
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)


@pytest.mark.parametrize(
    "geom",
    ([[None]], [[empty_point]], [[multi_point]], [[multi_point, multi_line_string]]),
)
def test_get_parts_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match="Array should be one dimensional"):
        pygeos.get_parts(geom)


@pytest.mark.parametrize("geom", [point, line_string, polygon])
def test_get_parts_non_multi(geom):
    """Non-multipart geometries should be returned identical to inputs"""
    assert_geometries_equal(geom, pygeos.get_parts(geom))


@pytest.mark.parametrize("geom", [None, [None], []])
def test_get_parts_None(geom):
    assert len(pygeos.get_parts(geom)) == 0


@pytest.mark.parametrize("geom", ["foo", ["foo"], 42])
def test_get_parts_invalid_geometry(geom):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type."):
        pygeos.get_parts(geom)


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
        None,
    ],
)
def test_get_rings(geom):
    if (pygeos.get_type_id(geom) != pygeos.GeometryType.POLYGON) or pygeos.is_empty(
        geom
    ):
        rings = pygeos.get_rings(geom)
        assert len(rings) == 0
    else:
        rings = pygeos.get_rings(geom)
        assert len(rings) == 1
        assert rings[0] == pygeos.get_exterior_ring(geom)


def test_get_rings_holes():
    rings = pygeos.get_rings(polygon_with_hole)
    assert len(rings) == 2
    assert rings[0] == pygeos.get_exterior_ring(polygon_with_hole)
    assert rings[1] == pygeos.get_interior_ring(polygon_with_hole, 0)


def test_get_rings_return_index():
    geom = np.array([polygon, None, empty_polygon, polygon_with_hole])
    expected_parts = []
    expected_index = []
    for i, g in enumerate(geom):
        if g is None or pygeos.is_empty(g):
            continue
        expected_parts.append(pygeos.get_exterior_ring(g))
        expected_index.append(i)
        for j in range(0, pygeos.get_num_interior_rings(g)):
            expected_parts.append(pygeos.get_interior_ring(g, j))
            expected_index.append(i)

    parts, index = pygeos.get_rings(geom, return_index=True)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)
    assert np.array_equal(index, expected_index)


@pytest.mark.parametrize("geom", [[[None]], [[polygon]]])
def test_get_rings_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match="Array should be one dimensional"):
        pygeos.get_parts(geom)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_get_precision():
    geometries = all_types + (point_z, empty_point, empty_line_string, empty_polygon)
    # default is 0
    actual = pygeos.get_precision(geometries).tolist()
    assert actual == [0] * len(geometries)

    geometry = pygeos.set_precision(geometries, 1)
    actual = pygeos.get_precision(geometry).tolist()
    assert actual == [1] * len(geometries)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_get_precision_none():
    assert np.all(np.isnan(pygeos.get_precision([None])))


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision(mode):
    initial_geometry = pygeos.Geometry("POINT (0.9 0.9)")
    assert pygeos.get_precision(initial_geometry) == 0

    geometry = pygeos.set_precision(initial_geometry, 0, mode=mode)
    assert pygeos.get_precision(geometry) == 0
    assert_geometries_equal(geometry, initial_geometry)

    geometry = pygeos.set_precision(initial_geometry, 1, mode=mode)
    assert pygeos.get_precision(geometry) == 1
    assert_geometries_equal(geometry, pygeos.Geometry("POINT (1 1)"))
    # original should remain unchanged
    assert_geometries_equal(initial_geometry, pygeos.Geometry("POINT (0.9 0.9)"))


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_set_precision_drop_coords():
    # setting precision of 0 will not drop duplicated points in original
    geometry = pygeos.set_precision(
        pygeos.Geometry("LINESTRING (0 0, 0 0, 0 1, 1 1)"), 0
    )
    assert_geometries_equal(
        geometry, pygeos.Geometry("LINESTRING (0 0, 0 0, 0 1, 1 1)")
    )

    # setting precision will remove duplicated points
    geometry = pygeos.set_precision(geometry, 1)
    assert_geometries_equal(geometry, pygeos.Geometry("LINESTRING (0 0, 0 1, 1 1)"))


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision_z(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GEOS <= 3.9 emits warning for 'pointwise'
        geometry = pygeos.set_precision(
            pygeos.Geometry("POINT Z (0.9 0.9 0.9)"), 1, mode=mode
        )
        assert pygeos.get_precision(geometry) == 1
        assert_geometries_equal(geometry, pygeos.Geometry("POINT Z (1 1 0.9)"))


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("mode", ("valid_output", "pointwise", "keep_collapsed"))
def test_set_precision_nan(mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # GEOS <= 3.9 emits warning for 'pointwise'
        actual = pygeos.set_precision(line_string_nan, 1, mode=mode)
        assert_geometries_equal(actual, line_string_nan)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_set_precision_none():
    assert pygeos.set_precision(None, 0) is None


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_set_precision_grid_size_nan():
    assert pygeos.set_precision(pygeos.Geometry("POINT (0.9 0.9)"), np.nan) is None


@pytest.mark.parametrize(
    "geometry,mode,expected",
    [
        (
            pygeos.Geometry("POLYGON((2 2,4 2,3.2 3,4 4, 2 4, 2.8 3, 2 2))"),
            "valid_output",
            pygeos.Geometry(
                "MULTIPOLYGON (((4 2, 2 2, 3 3, 4 2)), ((2 4, 4 4, 3 3, 2 4)))"
            ),
        ),
        pytest.param(
            pygeos.Geometry("POLYGON((2 2,4 2,3.2 3,4 4, 2 4, 2.8 3, 2 2))"),
            "pointwise",
            pygeos.Geometry("POLYGON ((2 2, 4 2, 3 3, 4 4, 2 4, 3 3, 2 2))"),
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            pygeos.Geometry("POLYGON((2 2,4 2,3.2 3,4 4, 2 4, 2.8 3, 2 2))"),
            "keep_collapsed",
            pygeos.Geometry(
                "MULTIPOLYGON (((4 2, 2 2, 3 3, 4 2)), ((2 4, 4 4, 3 3, 2 4)))"
            ),
        ),
        (
            pygeos.Geometry("LINESTRING (0 0, 0.1 0.1)"),
            "valid_output",
            pygeos.Geometry("LINESTRING EMPTY"),
        ),
        pytest.param(
            pygeos.Geometry("LINESTRING (0 0, 0.1 0.1)"),
            "pointwise",
            pygeos.Geometry("LINESTRING (0 0, 0 0)"),
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            pygeos.Geometry("LINESTRING (0 0, 0.1 0.1)"),
            "keep_collapsed",
            pygeos.Geometry("LINESTRING (0 0, 0 0)"),
        ),
        pytest.param(
            pygeos.Geometry("LINEARRING (0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)"),
            "valid_output",
            pygeos.Geometry("LINEARRING EMPTY"),
            marks=pytest.mark.skipif(
                pygeos.geos_version == (3, 10, 0), reason="Segfaults on GEOS 3.10.0"
            ),
        ),
        pytest.param(
            pygeos.Geometry("LINEARRING (0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)"),
            "pointwise",
            pygeos.Geometry("LINEARRING (0 0, 0 0, 0 0, 0 0, 0 0)"),
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        pytest.param(
            pygeos.Geometry("LINEARRING (0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)"),
            "keep_collapsed",
            # See https://trac.osgeo.org/geos/ticket/1135#comment:5
            pygeos.Geometry("LINESTRING (0 0, 0 0, 0 0)"),
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 10, 0),
                reason="this collapsed into an invalid linearring pre-GEOS 3.10",
            ),
        ),
        (
            pygeos.Geometry("POLYGON ((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))"),
            "valid_output",
            pygeos.Geometry("POLYGON EMPTY"),
        ),
        pytest.param(
            pygeos.Geometry("POLYGON ((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))"),
            "pointwise",
            pygeos.Geometry("POLYGON ((0 0, 0 0, 0 0, 0 0, 0 0))"),
            marks=pytest.mark.skipif(
                pygeos.geos_version < (3, 10, 0),
                reason="pointwise does not work pre-GEOS 3.10",
            ),
        ),
        (
            pygeos.Geometry("POLYGON ((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))"),
            "keep_collapsed",
            pygeos.Geometry("POLYGON EMPTY"),
        ),
    ],
)
def test_set_precision_collapse(geometry, mode, expected):
    """Lines and polygons collapse to empty geometries if vertices are too close"""
    actual = pygeos.set_precision(geometry, 1, mode=mode)
    if pygeos.geos_version < (3, 9, 0):
        # pre GEOS 3.9 has difficulty comparing empty geometries exactly
        # normalize and compare by WKT instead
        assert pygeos.to_wkt(pygeos.normalize(actual)) == pygeos.to_wkt(
            pygeos.normalize(expected)
        )
    else:
        # force to 2D because GEOS 3.10 yields 3D geometries when they are empty.
        assert_geometries_equal(pygeos.force_2d(actual), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_set_precision_intersection():
    """Operations should use the most precise presision grid size of the inputs"""

    box1 = pygeos.normalize(pygeos.box(0, 0, 0.9, 0.9))
    box2 = pygeos.normalize(pygeos.box(0.75, 0, 1.75, 0.75))

    assert pygeos.get_precision(pygeos.intersection(box1, box2)) == 0

    # GEOS will use and keep the most precise precision grid size
    box1 = pygeos.set_precision(box1, 0.5)
    box2 = pygeos.set_precision(box2, 1)
    out = pygeos.intersection(box1, box2)
    assert pygeos.get_precision(out) == 0.5
    assert_geometries_equal(out, pygeos.Geometry("LINESTRING (1 1, 1 0)"))


@pytest.mark.parametrize("preserve_topology", [False, True])
def set_precision_preserve_topology(preserve_topology):
    # the preserve_topology kwarg is deprecated (ignored)
    with pytest.warns(UserWarning):
        actual = pygeos.set_precision(
            pygeos.Geometry("LINESTRING (0 0, 0.1 0.1)"),
            1.0,
            preserve_topology=preserve_topology,
        )
    assert_geometries_equal(
        pygeos.force_2d(actual), pygeos.Geometry("LINESTRING EMPTY")
    )


@pytest.mark.skipif(pygeos.geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
def set_precision_pointwise_pre_310():
    # using 'pointwise' emits a warning
    with pytest.warns(UserWarning):
        actual = pygeos.set_precision(
            pygeos.Geometry("LINESTRING (0 0, 0.1 0.1)"),
            1.0,
            mode="pointwise",
        )
    assert_geometries_equal(
        pygeos.force_2d(actual), pygeos.Geometry("LINESTRING EMPTY")
    )


@pytest.mark.parametrize("flags", [np.array([0, 1]), 4, "foo"])
def set_precision_illegal_flags(flags):
    # the preserve_topology kwarg is deprecated (ignored)
    with pytest.raises((ValueError, TypeError)):
        pygeos.lib.set_precision(line_string, 1.0, flags)


def test_empty():
    """Compatibility with empty_like, see GH373"""
    g = np.empty_like(np.array([None, None]))
    assert pygeos.is_missing(g).all()


# corresponding to geometry_collection_z:
geometry_collection_2 = pygeos.geometrycollections([point, line_string])
empty_geom_mark = pytest.mark.skipif(
    pygeos.geos_version < (3, 9, 0),
    reason="Empty points don't have a dimensionality before GEOS 3.9",
)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),
        (point_z, point),
        pytest.param(empty_point, empty_point, marks=empty_geom_mark),
        pytest.param(empty_point_z, empty_point, marks=empty_geom_mark),
        (line_string, line_string),
        (line_string_z, line_string),
        pytest.param(empty_line_string, empty_line_string, marks=empty_geom_mark),
        pytest.param(empty_line_string_z, empty_line_string, marks=empty_geom_mark),
        (polygon, polygon),
        (polygon_z, polygon),
        (polygon_with_hole, polygon_with_hole),
        (polygon_with_hole_z, polygon_with_hole),
        (multi_point, multi_point),
        (multi_point_z, multi_point),
        (multi_line_string, multi_line_string),
        (multi_line_string_z, multi_line_string),
        (multi_polygon, multi_polygon),
        (multi_polygon_z, multi_polygon),
        (geometry_collection_2, geometry_collection_2),
        (geometry_collection_z, geometry_collection_2),
    ],
)
def test_force_2d(geom, expected):
    actual = pygeos.force_2d(geom)
    assert pygeos.get_coordinate_dimension(actual) == 2
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point_z),
        (point_z, point_z),
        pytest.param(empty_point, empty_point_z, marks=empty_geom_mark),
        pytest.param(empty_point_z, empty_point_z, marks=empty_geom_mark),
        (line_string, line_string_z),
        (line_string_z, line_string_z),
        pytest.param(empty_line_string, empty_line_string_z, marks=empty_geom_mark),
        pytest.param(empty_line_string_z, empty_line_string_z, marks=empty_geom_mark),
        (polygon, polygon_z),
        (polygon_z, polygon_z),
        (polygon_with_hole, polygon_with_hole_z),
        (polygon_with_hole_z, polygon_with_hole_z),
        (multi_point, multi_point_z),
        (multi_point_z, multi_point_z),
        (multi_line_string, multi_line_string_z),
        (multi_line_string_z, multi_line_string_z),
        (multi_polygon, multi_polygon_z),
        (multi_polygon_z, multi_polygon_z),
        (geometry_collection_2, geometry_collection_z),
        (geometry_collection_z, geometry_collection_z),
    ],
)
def test_force_3d(geom, expected):
    actual = pygeos.force_3d(geom, z=4)
    assert pygeos.get_coordinate_dimension(actual) == 3
    assert_geometries_equal(actual, expected)
