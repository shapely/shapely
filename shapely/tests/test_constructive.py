import numpy as np
import pytest

import shapely
from shapely import Geometry, GeometryCollection, GEOSException, Polygon
from shapely.testing import assert_geometries_equal

from .common import (
    all_types,
    empty,
    empty_line_string,
    empty_point,
    empty_polygon,
    ignore_invalid,
    line_string,
    multi_point,
    point,
    point_z,
)

CONSTRUCTIVE_NO_ARGS = (
    shapely.boundary,
    shapely.centroid,
    shapely.convex_hull,
    shapely.envelope,
    shapely.extract_unique_points,
    shapely.normalize,
    shapely.point_on_surface,
)

CONSTRUCTIVE_FLOAT_ARG = (
    shapely.buffer,
    shapely.offset_curve,
    shapely.delaunay_triangles,
    shapely.simplify,
    shapely.voronoi_polygons,
)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_array(geometry, func):
    actual = func([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    if (
        func is shapely.offset_curve
        and shapely.get_type_id(geometry) not in [1, 2]
        and shapely.geos_version < (3, 11, 0)
    ):
        with pytest.raises(GEOSException, match="only accept linestrings"):
            func([geometry, geometry], 0.0)
        return
    # voronoi_polygons emits an "invalid" warning when supplied with an empty
    # point (see https://github.com/libgeos/geos/issues/515)
    with ignore_invalid(
        func is shapely.voronoi_polygons and shapely.get_type_id(geometry) == 0
    ):
        actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("reference", all_types)
def test_snap_array(geometry, reference):
    actual = shapely.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("func", CONSTRUCTIVE_NO_ARGS)
def test_no_args_missing(func):
    actual = func(None)
    assert actual is None


@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_missing(func):
    actual = func(None, 1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("func", CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float("nan"))
    assert actual is None


def test_buffer_cap_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, cap_style="invalid")


def test_buffer_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, join_style="invalid")


def test_snap_none():
    actual = shapely.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = shapely.snap(geometry, point, tolerance=np.nan)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_build_area_none():
    actual = shapely.build_area(None)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, empty),  # a point has no area
        (line_string, empty),  # a line string has no area
        # geometry collection of two polygons are combined into one
        (
            Geometry(
                "GEOMETRYCOLLECTION(POLYGON((0 0, 3 0, 3 3, 0 3, 0 0)), POLYGON((1 1, 1 2, 2 2, 1 1)))"
            ),
            Geometry("POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 2 2, 1 2, 1 1))"),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_build_area(geom, expected):
    actual = shapely.build_area(geom)
    assert actual is not expected
    assert actual == expected


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_make_valid_none():
    actual = shapely.make_valid(None)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to a multilinestring
        (
            Polygon(((0, 0), (1, 1), (1, 2), (1, 1), (0, 0))),
            Geometry("MULTILINESTRING ((1 1, 1 2), (0 0, 1 1))"),
        ),
        # a polygon with self-intersection (bowtie) is converted into polygons
        (
            Geometry("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))"),
            Geometry("MULTIPOLYGON (((1 1, 2 2, 2 0, 1 1)), ((0 0, 0 2, 1 1, 0 0)))"),
        ),
        (empty, empty),
        ([empty], [empty]),
    ],
)
def test_make_valid(geom, expected):
    actual = shapely.make_valid(geom)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert shapely.normalize(actual) == expected


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (all_types, all_types),
        # first polygon is valid, second polygon has self-intersection
        (
            [
                Geometry("POLYGON((0 0, 2 2, 0 2, 0 0))"),
                Geometry("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))"),
            ],
            [
                Geometry("POLYGON((0 0, 2 2, 0 2, 0 0))"),
                Geometry(
                    "MULTIPOLYGON (((1 1, 0 0, 0 2, 1 1)), ((1 1, 2 2, 2 0, 1 1)))"
                ),
            ],
        ),
        ([point, None, empty], [point, None, empty]),
    ],
)
def test_make_valid_1d(geom, expected):
    actual = shapely.make_valid(geom)
    # normalize needed to handle variation in output across GEOS versions
    assert np.all(shapely.normalize(actual) == shapely.normalize(expected))


@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a point is always in normalized form
        # order coordinates of linestrings and parts of multi-linestring
        (
            Geometry("MULTILINESTRING ((1 1, 0 0), (1 1, 1 2))"),
            Geometry("MULTILINESTRING ((1 1, 1 2), (0 0, 1 1))"),
        ),
    ],
)
def test_normalize(geom, expected):
    actual = shapely.normalize(geom)
    assert actual == expected


def test_offset_curve_empty():
    with ignore_invalid():
        # Empty geometries emit an "invalid" warning
        # (see https://github.com/libgeos/geos/issues/515)
        actual = shapely.offset_curve(empty_line_string, 2.0)
    assert shapely.is_empty(actual)


def test_offset_curve_distance_array():
    # check that kwargs are passed through
    result = shapely.offset_curve([line_string, line_string], [-2.0, -3.0])
    assert result[0] == shapely.offset_curve(line_string, -2.0)
    assert result[1] == shapely.offset_curve(line_string, -3.0)


def test_offset_curve_kwargs():
    # check that kwargs are passed through
    result1 = shapely.offset_curve(
        line_string, -2.0, quadsegs=2, join_style="mitre", mitre_limit=2.0
    )
    result2 = shapely.offset_curve(line_string, -2.0)
    assert result1 != result2


def test_offset_curve_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve([line_string, line_string], 1, quadsegs=np.array([8, 9]))

    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve(
            [line_string, line_string], 1, join_style=["round", "bevel"]
        )

    with pytest.raises(TypeError, match=msg):
        shapely.offset_curve([line_string, line_string], 1, mitre_limit=[5.0, 6.0])


def test_offset_curve_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.offset_curve(line_string, 1.0, join_style="invalid")


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (
            shapely.Geometry("LINESTRING (0 0, 1 2)"),
            shapely.Geometry("LINESTRING (1 2, 0 0)"),
        ),
        (
            shapely.Geometry("LINEARRING (0 0, 1 2, 1 3, 0 0)"),
            shapely.Geometry("LINEARRING (0 0, 1 3, 1 2, 0 0)"),
        ),
        (
            shapely.Geometry("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            shapely.Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0))),
        ),
        (
            shapely.Geometry(
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"
            ),
            shapely.Geometry(
                "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))"
            ),
        ),
        pytest.param(
            shapely.Geometry("MULTILINESTRING ((0 0, 1 2), (3 3, 4 4))"),
            shapely.Geometry("MULTILINESTRING ((1 2, 0 0), (4 4, 3 3))"),
            marks=pytest.mark.skipif(
                shapely.geos_version < (3, 8, 1), reason="GEOS < 3.8.1"
            ),
        ),
        (
            shapely.Geometry(
                "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"
            ),
            shapely.Geometry(
                "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
            ),
        ),
        # points are unchanged
        (point, point),
        (point_z, point_z),
        (multi_point, multi_point),
        # empty geometries are unchanged
        (empty_point, empty_point),
        (empty_line_string, empty_line_string),
        (empty, empty),
        (empty_polygon, empty_polygon),
    ],
)
def test_reverse(geom, expected):
    assert_geometries_equal(shapely.reverse(geom), expected)


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_reverse_none():
    assert shapely.reverse(None) is None
    assert shapely.reverse([None]).tolist() == [None]

    geometry = shapely.Geometry("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    expected = shapely.Geometry("POLYGON ((0 0,  0 1, 1 1, 1 0, 0 0))")
    result = shapely.reverse([None, geometry])
    assert result[0] is None
    assert_geometries_equal(result[1], expected)


@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize("geom", ["Not a geometry", 1])
def test_reverse_invalid_type(geom):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type"):
        shapely.reverse(geom)


@pytest.mark.parametrize(
    "geom,expected",
    [
        # Point outside
        ("POINT (0 0)", "GEOMETRYCOLLECTION EMPTY"),
        # Point inside
        ("POINT (15 15)", "POINT (15 15)"),
        # Point on boundary
        ("POINT (15 10)", "GEOMETRYCOLLECTION EMPTY"),
        # Line outside
        ("LINESTRING (0 0, -5 5)", "GEOMETRYCOLLECTION EMPTY"),
        # Line inside
        ("LINESTRING (15 15, 16 15)", "LINESTRING (15 15, 16 15)"),
        # Line on boundary
        ("LINESTRING (10 15, 10 10, 15 10)", "GEOMETRYCOLLECTION EMPTY"),
        # Line splitting rectangle
        ("LINESTRING (10 5, 25 20)", "LINESTRING (15 10, 20 15)"),
    ],
)
def test_clip_by_rect(geom, expected):
    geom, expected = shapely.Geometry(geom), shapely.Geometry(expected)
    actual = shapely.clip_by_rect(geom, 10, 10, 20, 20)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize(
    "geom, rect, expected",
    [
        # Polygon hole (CCW) fully on rectangle boundary"""
        (
            "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))",
            (10, 10, 20, 20),
            "GEOMETRYCOLLECTION EMPTY",
        ),
        # Polygon hole (CW) fully on rectangle boundary"""
        (
            "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 10 20, 20 20, 20 10, 10 10))",
            (10, 10, 20, 20),
            "GEOMETRYCOLLECTION EMPTY",
        ),
        # Polygon fully within rectangle"""
        (
            "POLYGON ((1 1, 1 30, 30 30, 30 1, 1 1), (10 10, 20 10, 20 20, 10 20, 10 10))",
            (0, 0, 40, 40),
            "POLYGON ((1 1, 1 30, 30 30, 30 1, 1 1), (10 10, 20 10, 20 20, 10 20, 10 10))",
        ),
        # Polygon overlapping rectangle
        (
            "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))",
            (5, 5, 15, 15),
            "POLYGON ((5 5, 5 15, 10 15, 10 10, 15 10, 15 5, 5 5))",
        ),
    ],
)
def test_clip_by_rect_polygon(geom, rect, expected):
    geom, expected = shapely.Geometry(geom), shapely.Geometry(expected)
    actual = shapely.clip_by_rect(geom, *rect)
    assert_geometries_equal(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
def test_clip_by_rect_array(geometry):
    actual = shapely.clip_by_rect([geometry, geometry], 0.0, 0.0, 1.0, 1.0)
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


def test_clip_by_rect_missing():
    actual = shapely.clip_by_rect(None, 0, 0, 1, 1)
    assert actual is None


@pytest.mark.parametrize("geom", [empty, empty_line_string, empty_polygon])
def test_clip_by_rect_empty(geom):
    # TODO empty point
    actual = shapely.clip_by_rect(geom, 0, 0, 1, 1)
    assert actual == GeometryCollection()


def test_clip_by_rect_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        shapely.clip_by_rect([line_string, line_string], 0, 0, 1, np.array([0, 1]))


def test_polygonize():
    lines = [
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
        shapely.Geometry("LINESTRING (1 1, 1 0)"),
        shapely.Geometry("LINESTRING (1 0, 0 0)"),
        shapely.Geometry("LINESTRING (5 5, 6 6)"),
        shapely.Point(0, 0),
        None,
    ]
    result = shapely.polygonize(lines)
    assert shapely.get_type_id(result) == 7  # GeometryCollection
    expected = shapely.Geometry(
        "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 1, 1 0, 0 0)), POLYGON ((1 1, 0 0, 0 1, 1 1)))"
    )
    assert result == expected


def test_polygonize_array():
    lines = [
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
    ]
    expected = shapely.Geometry("GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))")
    result = shapely.polygonize(np.array(lines))
    assert isinstance(result, shapely.Geometry)
    assert result == expected

    result = shapely.polygonize(np.array([lines]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == expected

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] == expected
    assert result[1] == expected

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    for res in result.flatten():
        assert res == expected


@pytest.mark.skipif(
    np.__version__ < "1.15",
    reason="axis keyword for generalized ufunc introduced in np 1.15",
)
def test_polygonize_array_axis():
    lines = [
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize(arr, axis=1)
    assert result.shape == (2,)
    result = shapely.polygonize(arr, axis=0)
    assert result.shape == (3,)


def test_polygonize_missing():
    # set of geometries that is all missing
    result = shapely.polygonize([None, None])
    assert result == shapely.GeometryCollection()


def test_polygonize_full():
    lines = [
        None,
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
        shapely.Geometry("LINESTRING (1 1, 1 0)"),
        None,
        shapely.Geometry("LINESTRING (1 0, 0 0)"),
        shapely.Geometry("LINESTRING (5 5, 6 6)"),
        shapely.Geometry("LINESTRING (1 1, 100 100)"),
        shapely.Point(0, 0),
        None,
    ]
    result = shapely.polygonize_full(lines)
    assert len(result) == 4
    assert all(shapely.get_type_id(geom) == 7 for geom in result)  # GeometryCollection
    polygons, cuts, dangles, invalid = result
    expected_polygons = shapely.Geometry(
        "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 1, 1 0, 0 0)), POLYGON ((1 1, 0 0, 0 1, 1 1)))"
    )
    assert polygons == expected_polygons
    assert cuts == shapely.GeometryCollection()
    expected_dangles = shapely.Geometry(
        "GEOMETRYCOLLECTION (LINESTRING (1 1, 100 100), LINESTRING (5 5, 6 6))"
    )
    assert dangles == expected_dangles
    assert invalid == shapely.GeometryCollection()


def test_polygonize_full_array():
    lines = [
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
    ]
    expected = shapely.Geometry("GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))")
    result = shapely.polygonize_full(np.array(lines))
    assert len(result) == 4
    assert all(isinstance(geom, shapely.Geometry) for geom in result)
    assert result[0] == expected
    assert all(geom == shapely.GeometryCollection() for geom in result[1:])

    result = shapely.polygonize_full(np.array([lines]))
    assert len(result) == 4
    assert all(isinstance(geom, np.ndarray) for geom in result)
    assert all(geom.shape == (1,) for geom in result)
    assert result[0][0] == expected
    assert all(geom[0] == shapely.GeometryCollection() for geom in result[1:])

    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (2,) for arr in result)
    assert result[0][0] == expected
    assert result[0][1] == expected
    assert all(g == shapely.GeometryCollection() for geom in result[1:] for g in geom)

    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert all(arr.shape == (3, 2) for arr in result)
    for res in result[0].flatten():
        assert res == expected
    for arr in result[1:]:
        for res in arr.flatten():
            assert res == shapely.GeometryCollection()


@pytest.mark.skipif(
    np.__version__ < "1.15",
    reason="axis keyword for generalized ufunc introduced in np 1.15",
)
def test_polygonize_full_array_axis():
    lines = [
        shapely.Geometry("LINESTRING (0 0, 1 1)"),
        shapely.Geometry("LINESTRING (0 0, 0 1)"),
        shapely.Geometry("LINESTRING (0 1, 1 1)"),
    ]
    arr = np.array([lines, lines])  # shape (2, 3)
    result = shapely.polygonize_full(arr, axis=1)
    assert len(result) == 4
    assert all(arr.shape == (2,) for arr in result)
    result = shapely.polygonize_full(arr, axis=0)
    assert len(result) == 4
    assert all(arr.shape == (3,) for arr in result)


def test_polygonize_full_missing():
    # set of geometries that is all missing
    result = shapely.polygonize_full([None, None])
    assert len(result) == 4
    assert all(geom == shapely.GeometryCollection() for geom in result)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("tolerance", [-1, 0])
def test_segmentize_invalid_tolerance(geometry, tolerance):
    with pytest.raises(GEOSException, match="IllegalArgumentException"):
        shapely.segmentize(geometry, tolerance=tolerance)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", all_types)
def test_segmentize_tolerance_nan(geometry):
    actual = shapely.segmentize(geometry, tolerance=np.nan)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry", [empty, empty_point, empty_line_string, empty_polygon]
)
def test_segmentize_empty(geometry):
    actual = shapely.segmentize(geometry, tolerance=5)
    assert_geometries_equal(actual, geometry)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize("geometry", [point, point_z, multi_point])
def test_segmentize_no_change(geometry):
    actual = shapely.segmentize(geometry, tolerance=5)
    assert_geometries_equal(actual, geometry)


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
def test_segmentize_none():
    assert shapely.segmentize(None, tolerance=5) is None


@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,tolerance, expected",
    [
        # tolerance greater than max edge length, no change
        (
            shapely.Geometry("LINESTRING (0 0, 0 10)"),
            20,
            shapely.Geometry("LINESTRING (0 0, 0 10)"),
        ),
        (
            shapely.Geometry("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
            20,
            shapely.Geometry("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
        ),
        # tolerance causes one vertex per segment
        (
            shapely.Geometry("LINESTRING (0 0, 0 10)"),
            5,
            shapely.Geometry("LINESTRING (0 0, 0 5, 0 10)"),
        ),
        (
            Geometry("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
            5,
            shapely.Geometry(
                "POLYGON ((0 0, 5 0, 10 0, 10 5, 10 10, 5 10, 0 10, 0 5, 0 0))"
            ),
        ),
        # ensure input arrays are broadcast correctly
        (
            [
                shapely.Geometry("LINESTRING (0 0, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 2)"),
            ],
            5,
            [
                shapely.Geometry("LINESTRING (0 0, 0 5, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 2)"),
            ],
        ),
        (
            [
                shapely.Geometry("LINESTRING (0 0, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 2)"),
            ],
            [5],
            [
                shapely.Geometry("LINESTRING (0 0, 0 5, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 2)"),
            ],
        ),
        (
            [
                shapely.Geometry("LINESTRING (0 0, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 2)"),
            ],
            [5, 1.5],
            [
                shapely.Geometry("LINESTRING (0 0, 0 5, 0 10)"),
                shapely.Geometry("LINESTRING (0 0, 0 1, 0 2)"),
            ],
        ),
    ],
)
def test_segmentize(geometry, tolerance, expected):
    actual = shapely.segmentize(geometry, tolerance)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize("geometry", all_types)
def test_minimum_bounding_circle_all_types(geometry):
    actual = shapely.minimum_bounding_circle([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.minimum_bounding_circle(None)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            shapely.Geometry("POLYGON ((0 5, 5 10, 10 5, 5 0, 0 5))"),
            shapely.buffer(shapely.Geometry("POINT (5 5)"), 5),
        ),
        (
            shapely.Geometry("LINESTRING (1 0, 1 10)"),
            shapely.buffer(shapely.Geometry("POINT (1 5)"), 5),
        ),
        (
            shapely.Geometry("MULTIPOINT (2 2, 4 2)"),
            shapely.buffer(shapely.Geometry("POINT (3 2)"), 1),
        ),
        (
            shapely.Point(2, 2),
            shapely.Point(2, 2),
        ),
        (
            shapely.GeometryCollection(),
            shapely.Polygon(),
        ),
    ],
)
def test_minimum_bounding_circle(geometry, expected):
    actual = shapely.minimum_bounding_circle(geometry)
    assert_geometries_equal(actual, expected)


@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("geometry", all_types)
def test_oriented_envelope_all_types(geometry):
    actual = shapely.oriented_envelope([geometry, geometry])
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)

    actual = shapely.oriented_envelope(None)
    assert actual is None


@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            shapely.Geometry("MULTIPOINT (0 0, 10 0, 10 10)"),
            shapely.Geometry("POLYGON ((0 0, 5 -5, 15 5, 10 10, 0 0))"),
        ),
        (
            shapely.Geometry("LINESTRING (1 1, 5 1, 10 10)"),
            shapely.Geometry("POLYGON ((1 1, 3 -1, 12 8, 10 10, 1 1))"),
        ),
        (
            shapely.Geometry("POLYGON ((1 1, 15 1, 5 10, 1 1))"),
            shapely.Geometry("POLYGON ((15 1, 15 10, 1 10, 1 1, 15 1))"),
        ),
        (
            shapely.Geometry("LINESTRING (1 1, 10 1)"),
            shapely.Geometry("LINESTRING (1 1, 10 1)"),
        ),
        (
            shapely.Point(2, 2),
            shapely.Point(2, 2),
        ),
        (
            shapely.GeometryCollection(),
            shapely.Polygon(),
        ),
    ],
)
def test_oriented_envelope(geometry, expected):
    actual = shapely.oriented_envelope(geometry)
    assert shapely.equals(actual, expected).all()


@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry, expected",
    [
        (
            shapely.Geometry("MULTIPOINT (0 0, 10 0, 10 10)"),
            shapely.Geometry("POLYGON ((0 0, 5 -5, 15 5, 10 10, 0 0))"),
        ),
        (
            shapely.Geometry("LINESTRING (1 1, 5 1, 10 10)"),
            shapely.Geometry("POLYGON ((1 1, 3 -1, 12 8, 10 10, 1 1))"),
        ),
        (
            shapely.Geometry("POLYGON ((1 1, 15 1, 5 10, 1 1))"),
            shapely.Geometry("POLYGON ((15 1, 15 10, 1 10, 1 1, 15 1))"),
        ),
        (
            shapely.Geometry("LINESTRING (1 1, 10 1)"),
            shapely.Geometry("LINESTRING (1 1, 10 1)"),
        ),
        (
            shapely.Point(2, 2),
            shapely.Point(2, 2),
        ),
        (
            shapely.GeometryCollection(),
            shapely.Polygon(),
        ),
    ],
)
def test_minimum_rotated_rectangle(geometry, expected):
    actual = shapely.minimum_rotated_rectangle(geometry)
    assert shapely.equals(actual, expected).all()
