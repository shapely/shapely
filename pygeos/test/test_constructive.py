import pygeos
import numpy as np
import pytest

from pygeos import Geometry, GEOSException

from .common import (
    point,
    point_z,
    line_string,
    all_types,
    empty,
    empty_point,
    empty_line_string,
    empty_polygon,
    geometry_collection,
    multi_point,
)

CONSTRUCTIVE_NO_ARGS = (
    pygeos.boundary,
    pygeos.centroid,
    pygeos.convex_hull,
    pygeos.envelope,
    pygeos.extract_unique_points,
    pygeos.normalize,
    pygeos.point_on_surface,
)

CONSTRUCTIVE_FLOAT_ARG = (
    pygeos.buffer,
    pygeos.offset_curve,
    pygeos.delaunay_triangles,
    pygeos.simplify,
    pygeos.voronoi_polygons,
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
    if func is pygeos.offset_curve and pygeos.get_type_id(geometry) not in [1, 2]:
        with pytest.raises(GEOSException, match="only accept linestrings"):
            func([geometry, geometry], 0.0)
        return
    actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)


@pytest.mark.parametrize("geometry", all_types)
@pytest.mark.parametrize("reference", all_types)
def test_snap_array(geometry, reference):
    actual = pygeos.snap([geometry, geometry], [reference, reference], tolerance=1.0)
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


def test_snap_none():
    actual = pygeos.snap(None, point, tolerance=1.0)
    assert actual is None


@pytest.mark.parametrize("geometry", all_types)
def test_snap_nan_float(geometry):
    actual = pygeos.snap(geometry, point, tolerance=np.nan)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_build_area_none():
    actual = pygeos.build_area(None)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
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
    actual = pygeos.build_area(geom)
    assert actual is not expected
    assert actual == expected


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
def test_make_valid_none():
    actual = pygeos.make_valid(None)
    assert actual is None


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (point, point),  # a valid geometry stays the same (but is copied)
        # an L shaped polygon without area is converted to a multilinestring
        (
            Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"),
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
    actual = pygeos.make_valid(geom)
    assert actual is not expected
    # normalize needed to handle variation in output across GEOS versions
    assert pygeos.normalize(actual) == expected


@pytest.mark.skipif(pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8")
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
    actual = pygeos.make_valid(geom)
    # normalize needed to handle variation in output across GEOS versions
    assert np.all(pygeos.normalize(actual) == pygeos.normalize(expected))


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
    actual = pygeos.normalize(geom)
    assert actual == expected


def test_offset_curve_empty():
    actual = pygeos.offset_curve(empty_line_string, 2.0)
    assert pygeos.is_empty(actual)


def test_offset_curve_distance_array():
    # check that kwargs are passed through
    result = pygeos.offset_curve([line_string, line_string], [-2.0, -3.0])
    assert result[0] == pygeos.offset_curve(line_string, -2.0)
    assert result[1] == pygeos.offset_curve(line_string, -3.0)


def test_offset_curve_kwargs():
    # check that kwargs are passed through
    result1 = pygeos.offset_curve(
        line_string, -2.0, quadsegs=2, join_style="mitre", mitre_limit=2.0
    )
    result2 = pygeos.offset_curve(line_string, -2.0)
    assert result1 != result2


def test_offset_curve_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        pygeos.offset_curve([line_string, line_string], 1, quadsegs=np.array([8, 9]))

    with pytest.raises(TypeError, match=msg):
        pygeos.offset_curve(
            [line_string, line_string], 1, join_style=["round", "bevel"]
        )

    with pytest.raises(TypeError, match=msg):
        pygeos.offset_curve([line_string, line_string], 1, mitre_limit=[5.0, 6.0])


def test_offset_curve_join_style():
    with pytest.raises(KeyError):
        pygeos.offset_curve(line_string, 1.0, join_style="nonsense")


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize(
    "geom,expected",
    [
        (
            pygeos.Geometry("LINESTRING (0 0, 1 2)"),
            pygeos.Geometry("LINESTRING (1 2, 0 0)"),
        ),
        (
            pygeos.Geometry("LINEARRING (0 0, 1 2, 1 3, 0 0)"),
            pygeos.Geometry("LINEARRING (0 0, 1 3, 1 2, 0 0)"),
        ),
        (
            pygeos.Geometry("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            pygeos.Geometry("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        ),
        (
            pygeos.Geometry(
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"
            ),
            pygeos.Geometry(
                "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))"
            ),
        ),
        (
            pygeos.Geometry("MULTILINESTRING ((0 0, 1 2), (3 3, 4 4))"),
            pygeos.Geometry("MULTILINESTRING ((1 2, 0 0), (4 4, 3 3))"),
        ),
        (
            pygeos.Geometry(
                "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"
            ),
            pygeos.Geometry(
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
    assert pygeos.equals(pygeos.reverse(geom), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
def test_reverse_none():
    assert pygeos.reverse(None) is None
    assert pygeos.reverse([None]).tolist() == [None]

    geometry = pygeos.Geometry("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    expected = pygeos.Geometry("POLYGON ((0 0,  0 1, 1 1, 1 0, 0 0))")
    result = pygeos.reverse([None, geometry])
    assert result[0] is None
    assert pygeos.equals(result[1], expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 7, 0), reason="GEOS < 3.7")
@pytest.mark.parametrize("geom", ["Not a geometry", 1])
def test_reverse_invalid_type(geom):
    with pytest.raises(TypeError, match="One of the arguments is of incorrect type"):
        pygeos.reverse(geom)


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
    geom, expected = pygeos.Geometry(geom), pygeos.Geometry(expected)
    actual = pygeos.clip_by_rect(geom, 10, 10, 20, 20)
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize(
    "geom, rect, expected",
    [
        # Polygon hole (CCW) fully on rectangle boundary"""
        (
            "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))",
            (10, 10, 20, 20),
            "GEOMETRYCOLLECTION EMPTY"
        ),
        # Polygon hole (CW) fully on rectangle boundary"""
        (
            "POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 10 20, 20 20, 20 10, 10 10))",
            (10, 10, 20, 20),
            "GEOMETRYCOLLECTION EMPTY"
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
        )
    ],
)
def test_clip_by_rect_polygon(geom, rect, expected):
    geom, expected = pygeos.Geometry(geom), pygeos.Geometry(expected)
    actual = pygeos.clip_by_rect(geom, *rect)
    assert pygeos.equals(actual, expected)


@pytest.mark.parametrize("geometry", all_types)
def test_clip_by_rect_array(geometry):
    actual = pygeos.clip_by_rect([geometry, geometry], 0.0, 0.0, 1.0, 1.0)
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)


def test_clip_by_rect_missing():
    actual = pygeos.clip_by_rect(None, 0, 0, 1, 1)
    assert actual is None


@pytest.mark.parametrize("geom", [empty, empty_line_string, empty_polygon])
def test_clip_by_rect_empty(geom):
    # TODO empty point
    actual = pygeos.clip_by_rect(geom, 0, 0, 1, 1)
    assert actual == Geometry("GEOMETRYCOLLECTION EMPTY")


def test_clip_by_rect_non_scalar_kwargs():
    msg = "only accepts scalar values"
    with pytest.raises(TypeError, match=msg):
        pygeos.clip_by_rect([line_string, line_string], 0, 0, 1, np.array([0, 1]))
