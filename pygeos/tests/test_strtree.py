import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import pygeos
from pygeos import box, UnsupportedGEOSOperation

from .common import (
    assert_decreases_refcount,
    assert_increases_refcount,
    empty,
    empty_line_string,
    empty_point,
    point,
)

# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture(scope="session")
def tree():
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


@pytest.fixture(scope="session")
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = pygeos.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield pygeos.STRtree(geoms)


@pytest.fixture(scope="session")
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = pygeos.buffer(
        pygeos.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quadsegs=32
    )
    yield pygeos.STRtree(geoms)


@pytest.mark.parametrize(
    "geometry,count, hits",
    [
        # Empty array produces empty tree
        ([], 0, 0),
        ([point], 1, 1),
        # None geometries are ignored when creating tree
        ([None], 0, 0),
        ([point, None], 1, 1),
        # empty geometries are ignored when creating tree
        ([empty, empty_point, empty_line_string], 0, 0),
        # only the valid geometry should have a hit
        ([empty, point, empty_point, empty_line_string], 1, 1),
    ],
)
def test_init(geometry, count, hits):
    tree = pygeos.STRtree(np.array(geometry))
    assert len(tree) == count
    assert tree.query(box(0, 0, 100, 100)).size == hits


def test_init_with_invalid_geometry():
    with pytest.raises(TypeError):
        pygeos.STRtree(np.array(["Not a geometry"], dtype=object))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(point):
        _ = pygeos.STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_decreases_refcount(point):
        del tree


def test_flush_geometries():
    arr = pygeos.points(np.arange(10), np.arange(10))
    tree = pygeos.STRtree(arr)
    # Dereference geometries
    arr[:] = None
    import gc

    gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_geometries_property():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    assert arr is tree.geometries


def test_query_invalid_geometry(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    assert tree.query(None).size == 0


@pytest.mark.parametrize("geometry", [empty, empty_point, empty_line_string])
def test_query_empty(tree, geometry):
    assert tree.query(geometry).size == 0


@pytest.mark.parametrize(
    "tree_geometry, geometry,expected",
    [
        ([point], box(0, 0, 10, 10), [0]),
        # None is ignored in the tree, but the index of the valid geometry should
        # be retained.
        ([None, point], box(0, 0, 10, 10), [1]),
        ([None, empty, point], box(0, 0, 10, 10), [2]),
    ],
)
def test_query(tree_geometry, geometry, expected):
    tree = pygeos.STRtree(np.array(tree_geometry))
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box contains points
        (box(0, 0, 1, 1), [0, 1]),
        # box contains points
        (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
        # envelope of buffer contains points
        (pygeos.buffer(pygeos.points(3, 3), 1), [2, 3, 4]),
        # envelope of points contains points
        (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (pygeos.points(0, 0), [0]),
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line
        (pygeos.points(0, 0.5), [0]),
        # point at shared vertex between 2 lines
        (pygeos.points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # envelope of buffer overlaps envelope of 2 lines
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [2, 3]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        (pygeos.multipoints([[5, 7], [7, 5]]), [4, 5, 6, 7]),
    ],
)
def test_query_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        (pygeos.points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (pygeos.points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box overlaps envelope of 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # envelope of buffer overlaps envelope of 3 polygons
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps 3 polygons
        (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6, 7]),
    ],
)
def test_query_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry), expected)


def test_query_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query(pygeos.points(1, 1), predicate="bad_predicate")


def test_query_unsupported_predicate(tree):
    # valid GEOS binary predicate, but not supported for query
    with pytest.raises(ValueError):
        tree.query(pygeos.points(1, 1), predicate="disjoint")


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("intersects", [0, 1, 2]),
        ("within", []),
        ("contains", [1]),
        ("overlaps", []),
        ("crosses", []),
        ("covers", [0, 1, 2]),
        ("covered_by", []),
        ("contains_properly", [1]),
    ],
)
def test_query_prepared_inputs(tree, predicate, expected):
    geom = box(0, 0, 2, 2)
    pygeos.prepare(geom)
    assert_array_equal(tree.query(geom, predicate=predicate), expected)


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(
            "intersects",
            marks=pytest.mark.xfail(reason="intersects does not raise exception"),
        ),
        pytest.param(
            "within",
            marks=pytest.mark.xfail(
                pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8"
            ),
        ),
        pytest.param(
            "contains",
            marks=pytest.mark.xfail(
                pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8"
            ),
        ),
        "overlaps",
        "crosses",
        "touches",
        pytest.param(
            "covers",
            marks=pytest.mark.xfail(
                pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8"
            ),
        ),
        pytest.param(
            "covered_by",
            marks=pytest.mark.xfail(
                pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8"
            ),
        ),
        pytest.param(
            "contains_properly",
            marks=pytest.mark.xfail(
                pygeos.geos_version < (3, 8, 0), reason="GEOS < 3.8"
            ),
        ),
    ],
)
def test_query_predicate_errors(tree, predicate):
    with pytest.raises(pygeos.GEOSException):
        tree.query(pygeos.linestrings([1, 1], [1, float("nan")]), predicate=predicate)


def test_query_tree_with_none():
    # valid GEOS binary predicate, but not supported for query
    tree = pygeos.STRtree(
        [pygeos.Geometry("POINT (0 0)"), None, pygeos.Geometry("POINT (2 2)")]
    )
    assert tree.query(pygeos.points(2, 2), predicate="intersects") == [2]


### predicate == 'intersects'


def test_query_with_prepared(tree):
    geom = box(0, 0, 1, 1)
    expected = tree.query(geom, predicate="intersects")

    pygeos.prepare(geom)
    assert_array_equal(expected, tree.query(geom, predicate="intersects"))


# TEMPORARY xfail: MultiPoint intersects with prepared geometries does not work
# properly on GEOS 3.5.x; it was fixed in 3.6+
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box contains points
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        (pygeos.buffer(pygeos.points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        pytest.param(
            pygeos.multipoints([[5, 5], [7, 7]]),
            [5, 7],
            marks=pytest.mark.xfail(pygeos.geos_version < (3, 6, 0), reason="GEOS 3.5"),
        ),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        pytest.param(
            pygeos.multipoints([[5, 7], [7, 7]]),
            [7],
            marks=pytest.mark.xfail(pygeos.geos_version < (3, 6, 0), reason="GEOS 3.5"),
        ),
    ],
)
def test_query_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (pygeos.points(0, 0), [0]),
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (pygeos.points(0, 0.5), []),
        # point at shared vertex between 2 lines
        (pygeos.points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [0, 1]),
        # buffer intersects 2 lines
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [2, 3]),
        # buffer intersects midpoint of line at tangent
        (pygeos.buffer(pygeos.points(2, 1), HALF_UNIT_DIAG), [1]),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [6, 7]),
    ],
)
def test_query_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (pygeos.points(0, 0.5), [0]),
        (pygeos.points(0.5, 0), [0]),
        # midpoint between two polygons intersects both
        (pygeos.points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (pygeos.points(1, 1), [1]),
        # box overlaps envelope of 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box intersects 3 polygons
        (box(0, 0, 1.5, 1.5), [0, 1, 2]),
        # buffer overlaps 3 polygons
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 3, 4, 5]),
        # envelope of points overlaps polygons, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint within polygon
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="intersects"), expected)


### predicate == 'within'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box not within points
        (box(3, 3, 6, 6), []),
        # envelope of buffer not within points
        (pygeos.buffer(pygeos.points(3, 3), 1), []),
        # multipoints intersect but are not within points in tree
        (pygeos.multipoints([[5, 5], [7, 7]]), []),
        # only one point of multipoint intersects, but multipoints are not
        # within any points in tree
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_within_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # endpoint not within first line
        (pygeos.points(0, 0), []),
        # point within first line
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (pygeos.points(0, 0.5), []),
        # point at shared vertex between 2 lines (but within neither)
        (pygeos.points(1, 1), []),
        # box not within line
        (box(0, 0, 1, 1), []),
        # buffer intersects 2 lines but not within either
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects, but both are not within line
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        (pygeos.multipoints([[6.5, 6.5], [7, 7]]), [6]),
    ],
)
def test_query_within_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="within"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (pygeos.points(0, 0.5), [0]),
        (pygeos.points(0.5, 0), [0]),
        # midpoint between two polygons intersects both
        (pygeos.points(0.5, 0.5), [0, 1]),
        # point intersects single polygon
        (pygeos.points(1, 1), [1]),
        # box overlaps envelope of 2 polygons but within neither
        (box(0, 0, 1, 1), []),
        # box within polygon
        (box(0, 0, 0.5, 0.5), [0]),
        # larger box intersects 3 polygons but within none
        (box(0, 0, 1.5, 1.5), []),
        # buffer intersects 3 polygons but only within one
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [3]),
        # larger buffer overlaps 6 polygons (touches midpoints) but within none
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), []),
        # envelope of points overlaps polygons, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint within polygon
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points in multipoint within polygon
        (pygeos.multipoints([[5.25, 5.5], [5.25, 5.0]]), [5]),
    ],
)
def test_query_within_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="within"), expected)


### predicate == 'contains'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box contains points (2 are at edges and not contained)
        (box(3, 3, 6, 6), [4, 5]),
        # envelope of buffer contains more points than within buffer
        # due to diagonal distance
        (pygeos.buffer(pygeos.points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_contains_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any lines (not valid relation)
        (pygeos.points(0, 0), []),
        # box contains first line (touches edge of 1 but does not contain it)
        (box(0, 0, 1, 1), [0]),
        # buffer intersects 2 lines but contains neither
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points intersect but do not contain any lines (not valid relation)
        (pygeos.multipoints([[5, 5], [6, 6]]), []),
    ],
)
def test_query_contains_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="contains"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any polygons (not valid relation)
        (pygeos.points(0, 0), []),
        # box overlaps envelope of 2 polygons but contains neither
        (box(0, 0, 1, 1), []),
        # larger box intersects 3 polygons but contains only one
        (box(0, 0, 2, 2), [1]),
        # buffer overlaps 3 polygons but contains none
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), []),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_contains_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="contains"), expected)


### predicate == 'overlaps'
# Overlaps only returns results where geometries are of same dimensions
# and do not completely contain each other.
# See: https://postgis.net/docs/ST_Overlaps.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect but do not overlap
        (pygeos.points(1, 1), []),
        # box overlaps points including those at edge but does not overlap
        # (completely contains all points)
        (box(3, 3, 6, 6), []),
        # envelope of buffer contains points, but does not overlap
        (pygeos.buffer(pygeos.points(3, 3), 1), []),
        # multipoints intersect but do not overlap (both completely contain each other)
        (pygeos.multipoints([[5, 5], [7, 7]]), []),
        # envelope of points contains points in tree, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects but does not overlap
        # the intersecting point from multipoint completely contains point in tree
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_overlaps_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects line but is completely contained by it
        (pygeos.points(0, 0), []),
        # box overlaps second line (contains first line)
        # but of different dimensions so does not overlap
        (box(0, 0, 1.5, 1.5), []),
        # buffer intersects 2 lines but of different dimensions so does not overlap
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points intersect but different dimensions
        (pygeos.multipoints([[5, 5], [6, 6]]), []),
    ],
)
def test_query_overlaps_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="overlaps"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not overlap any polygons (different dimensions)
        (pygeos.points(0, 0), []),
        # box overlaps 2 polygons
        (box(0, 0, 1, 1), [0, 1]),
        # larger box intersects 3 polygons and contains one
        (box(0, 0, 2, 2), [0, 2]),
        # buffer overlaps 3 polygons and contains 1
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 4]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1, 2, 4, 5]),
        # one of two points intersects but different dimensions
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_overlaps_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="overlaps"), expected)


### predicate == 'crosses'
# Only valid for certain geometry combinations
# See: https://postgis.net/docs/ST_Crosses.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points intersect but not valid relation
        (pygeos.points(1, 1), []),
        # all points of result from tree are in common with box
        (box(3, 3, 6, 6), []),
        # all points of result from tree are in common with buffer
        (pygeos.buffer(pygeos.points(3, 3), 1), []),
        # only one point of multipoint intersects but not valid relation
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
    ],
)
def test_query_crosses_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line but is completely in common with line
        (pygeos.points(0, 0), []),
        # box overlaps envelope of first 2 lines, contains first and crosses second
        (box(0, 0, 1.5, 1.5), [1]),
        # buffer intersects 2 lines
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [2, 3]),
        # line crosses line
        (pygeos.linestrings([(1, 0), (0, 1)]), [0]),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7], [7, 8]]), []),
    ],
)
def test_query_crosses_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="crosses"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon but not valid relation
        (pygeos.points(0, 0.5), []),
        # box overlaps 2 polygons but not valid relation
        (box(0, 0, 1.5, 1.5), []),
        # buffer overlaps 3 polygons but not valid relation
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), []),
        # only one point of multipoint within
        (pygeos.multipoints([[5, 7], [7, 7], [7, 8]]), [7]),
    ],
)
def test_query_crosses_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="crosses"), expected)


### predicate == 'touches'
# See: https://postgis.net/docs/ST_Touches.html
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect but not valid relation
        (pygeos.points(1, 1), []),
        # box contains points but touches only those at edges
        (box(3, 3, 6, 6), [3, 6]),
        # buffer completely contains point in tree
        (pygeos.buffer(pygeos.points(3, 3), 1), []),
        # buffer intersects 2 points but touches only one
        (pygeos.buffer(pygeos.points(0, 1), 1), [1]),
        # multipoints intersect but not valid relation
        (pygeos.multipoints([[5, 5], [7, 7]]), []),
    ],
)
def test_query_touches_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        (pygeos.points(0, 0), [0]),
        # point is within line
        (pygeos.points(0.5, 0.5), []),
        # point at shared vertex between 2 lines
        (pygeos.points(1, 1), [0, 1]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        (box(0, 0, 1, 1), [1]),
        # buffer intersects 2 lines but does not touch edges of either
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # buffer intersects midpoint of line at tangent but there is a little overlap
        # due to precision issues
        (pygeos.buffer(pygeos.points(2, 1), HALF_UNIT_DIAG + 1e-7), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects at vertex between lines
        (pygeos.multipoints([[5, 7], [7, 7], [7, 8]]), [6, 7]),
    ],
)
def test_query_touches_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="touches"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        (pygeos.points(0, 0.5), []),
        # point is at edge of first polygon
        (pygeos.points(HALF_UNIT_DIAG + EPS, 0), [0]),
        # box overlaps envelope of 2 polygons does not touch any at edge
        (box(0, 0, 1, 1), []),
        # box overlaps 2 polygons and touches edge of first
        (box(HALF_UNIT_DIAG + EPS, 0, 2, 2), [0]),
        # buffer overlaps 3 polygons but does not touch any at edge
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG + EPS), []),
        # only one point of multipoint within polygon but does not touch
        (pygeos.multipoints([[0, 0], [7, 7], [7, 8]]), []),
    ],
)
def test_query_touches_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="touches"), expected)


### predicate == 'covers'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect and thus no point is outside the other
        (pygeos.points(1, 1), [1]),
        # box covers any points that intersect or are within
        (box(3, 3, 6, 6), [3, 4, 5, 6]),
        # envelope of buffer covers more points than are covered by buffer
        # due to diagonal distance
        (pygeos.buffer(pygeos.points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect and thus no point is outside the other
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_covers_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="covers"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not cover any lines (not valid relation)
        (pygeos.points(0, 0), []),
        # box covers first line (intersects another does not contain it)
        (box(0, 0, 1.5, 1.5), [0]),
        # box completely covers 2 lines (touches edges of 2 others)
        (box(1, 1, 3, 3), [1, 2]),
        # buffer intersects 2 lines but does not completely cover either
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects a line, but does not completely cover it
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points intersect but do not cover any lines (not valid relation)
        (pygeos.multipoints([[5, 5], [6, 6]]), []),
    ],
)
def test_query_covers_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="covers"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not cover any polygons (not valid relation)
        (pygeos.points(0, 0), []),
        # box overlaps envelope of 2 polygons but does not completely cover either
        (box(0, 0, 1, 1), []),
        # larger box intersects 3 polygons but covers only one
        (box(0, 0, 2, 2), [1]),
        # buffer overlaps 3 polygons but does not completely cover any
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), []),
        # larger buffer overlaps 6 polygons (touches midpoints) but covers only one
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_covers_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="covers"), expected)


### predicate == 'covered_by'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # box not covered by points
        (box(3, 3, 6, 6), []),
        # envelope of buffer not covered by points
        (pygeos.buffer(pygeos.points(3, 3), 1), []),
        # multipoints intersect but are not covered by points in tree
        (pygeos.multipoints([[5, 5], [7, 7]]), []),
        # only one point of multipoint intersects, but multipoints are not
        # covered by any points in tree
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # envelope of points overlaps points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_covered_by_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="covered_by"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # endpoint is covered by first line
        (pygeos.points(0, 0), [0]),
        # point covered by first line
        (pygeos.points(0.5, 0.5), [0]),
        # point within envelope of first line but does not intersect
        (pygeos.points(0, 0.5), []),
        # point at shared vertex between 2 lines and is covered by both
        (pygeos.points(1, 1), [0, 1]),
        # line intersects 3 lines, but is covered by only one
        (pygeos.linestrings([[1, 1], [2, 2]]), [1]),
        # line intersects 2 lines, but is covered by neither
        (pygeos.linestrings([[1.5, 1.5], [2.5, 2.5]]), []),
        # box not covered by line (not valid geometric relation)
        (box(0, 0, 1, 1), []),
        # buffer intersects 2 lines but not within either (not valid geometric relation)
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
        # envelope of points overlaps lines but intersects none
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects, but both are not covered by line
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points are covered by a line
        (pygeos.multipoints([[6.5, 6.5], [7, 7]]), [6]),
    ],
)
def test_query_covered_by_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query(geometry, predicate="covered_by"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point covered by polygon
        (pygeos.points(0, 0.5), [0]),
        (pygeos.points(0.5, 0), [0]),
        (pygeos.points(1, 1), [1]),
        # midpoint between two polygons is covered by both
        (pygeos.points(0.5, 0.5), [0, 1]),
        # line intersects multiple polygons but is not covered by any
        (pygeos.linestrings([[0, 0], [2, 2]]), []),
        # line intersects multiple polygons but is covered by only one
        (pygeos.linestrings([[1.5, 1.5], [2.5, 2.5]]), [2]),
        # box overlaps envelope of 2 polygons but not covered by either
        (box(0, 0, 1, 1), []),
        # box covered by polygon
        (box(0, 0, 0.5, 0.5), [0]),
        # larger box intersects 3 polygons but not covered by any
        (box(0, 0, 1.5, 1.5), []),
        # buffer intersects 3 polygons but only within one
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [3]),
        # larger buffer overlaps 6 polygons (touches midpoints) but within none
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), []),
        # envelope of points overlaps polygons, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint within polygon
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        # both points in multipoint within polygon
        (pygeos.multipoints([[5.25, 5.5], [5.25, 5.0]]), [5]),
    ],
)
def test_query_covered_by_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="covered_by"), expected)


### predicate == 'contains_properly'
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        (pygeos.points(0.5, 0.5), []),
        # points intersect
        (pygeos.points(1, 1), [1]),
        # line contains every point that is not on its first or last coordinate
        # these are on the "exterior" of the line
        (pygeos.linestrings([[0, 0], [2, 2]]), [1]),
        # slightly longer line contains multiple points
        (pygeos.linestrings([[0.5, 0.5], [2.5, 2.5]]), [1, 2]),
        # line intersects and contains one point
        (pygeos.linestrings([[0, 2], [2, 0]]), [1]),
        # box contains points (2 are at edges and not contained)
        (box(3, 3, 6, 6), [4, 5]),
        # envelope of buffer contains more points than within buffer
        # due to diagonal distance
        (pygeos.buffer(pygeos.points(3, 3), 1), [3]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [2, 3, 4]),
        # multipoints intersect
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        (pygeos.multipoints([[5, 7], [7, 7]]), [7]),
    ],
)
def test_query_contains_properly_points(tree, geometry, expected):
    assert_array_equal(tree.query(geometry, predicate="contains_properly"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # None of the following conditions satisfy the relation for linestrings
        # because they have no interior:
        # "a contains b if no points of b lie in the exterior of a, and at least one
        # point of the interior of b lies in the interior of a"
        (pygeos.points(0, 0), []),
        (pygeos.linestrings([[0, 0], [1, 1]]), []),
        (pygeos.linestrings([[0, 0], [2, 2]]), []),
        (pygeos.linestrings([[0, 2], [2, 0]]), []),
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        (pygeos.multipoints([[5, 7], [7, 7]]), []),
        (pygeos.multipoints([[5, 5], [6, 6]]), []),
        (box(0, 0, 1, 1), []),
        (box(0, 0, 2, 2), []),
        (pygeos.buffer(pygeos.points(3, 3), 0.5), []),
    ],
)
def test_query_contains_properly_lines(line_tree, geometry, expected):
    assert_array_equal(
        line_tree.query(geometry, predicate="contains_properly"), expected
    )


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point does not contain any polygons (not valid relation)
        (pygeos.points(0, 0), []),
        # line intersects multiple polygons but does not contain any (not valid relation)
        (pygeos.linestrings([[0, 0], [2, 2]]), []),
        # box overlaps envelope of 2 polygons but contains neither
        (box(0, 0, 1, 1), []),
        # larger box intersects 3 polygons but contains only one
        (box(0, 0, 2, 2), [1]),
        # buffer overlaps 3 polygons but contains none
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), []),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [3]),
        # envelope of points overlaps polygons, but points do not intersect
        # (not valid relation)
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
    ],
)
def test_query_contains_properly_polygons(poly_tree, geometry, expected):
    assert_array_equal(
        poly_tree.query(geometry, predicate="contains_properly"), expected
    )


### predicate = 'dwithin'


@pytest.mark.skipif(pygeos.geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
def test_query_dwithin_geos_version(tree):
    with pytest.raises(UnsupportedGEOSOperation, match="requires GEOS >= 3.10"):
        tree.query(pygeos.points(0, 0), predicate="dwithin", distance=1)


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "distance,match",
    [
        (None, "distance parameter must be provided"),
        ("foo", "could not convert string to float"),
        ([1.0], "distance must be a scalar value"),
        ([None], "distance must be a scalar value"),
    ],
)
def test_query_dwithin_invalid_distance(tree, distance, match):
    with pytest.raises(ValueError, match=match):
        tree.query(pygeos.points(0, 0), predicate="dwithin", distance=distance)


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (None, 1.0, []),
        (pygeos.points(0.25, 0.25), 0, []),
        (pygeos.points(0.25, 0.25), -1, []),
        (pygeos.points(0.25, 0.25), np.nan, []),
        (pygeos.Geometry("POINT EMPTY"), 1, []),
        (pygeos.points(0.25, 0.25), 0.5, [0]),
        (pygeos.points(0.25, 0.25), 2.5, [0, 1, 2]),
        (pygeos.points(3, 3), 1.5, [2, 3, 4]),
        # 2 equidistant points in tree
        (pygeos.points(0.5, 0.5), 0.75, [0, 1]),
        # all points intersect box
        (box(0, 0, 3, 3), 0, [0, 1, 2, 3]),
        (box(0, 0, 3, 3), 0.25, [0, 1, 2, 3]),
        # intersecting and nearby points
        (box(1, 1, 2, 2), 1.5, [0, 1, 2, 3]),
        # # return nearest point in tree for each point in multipoint
        (pygeos.multipoints([[0.25, 0.25], [1.5, 1.5]]), 0.75, [0, 1, 2]),
        # 2 equidistant points per point in multipoint
        (
            pygeos.multipoints([[0.5, 0.5], [3.5, 3.5]]),
            0.75,
            [0, 1, 3, 4],
        ),
    ],
)
def test_query_dwithin_points(tree, geometry, distance, expected):
    assert_array_equal(
        tree.query(geometry, predicate="dwithin", distance=distance), expected
    )


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (None, 1.0, []),
        (pygeos.points(0.5, 0.5), 0, [0]),
        (pygeos.points(0.5, 0.5), 1.0, [0, 1]),
        (pygeos.points(2, 2), 0.5, [1, 2]),
        (box(0, 0, 1, 1), 0.5, [0, 1]),
        (box(0.5, 0.5, 1.5, 1.5), 0.5, [0, 1]),
        # multipoints at endpoints of 2 lines each
        (pygeos.multipoints([[5, 5], [7, 7]]), 0.5, [4, 5, 6, 7]),
        # multipoints are equidistant from 2 lines
        (pygeos.multipoints([[5, 7], [7, 5]]), 1.5, [5, 6]),
    ],
)
def test_query_dwithin_lines(line_tree, geometry, distance, expected):
    assert_array_equal(
        line_tree.query(geometry, predicate="dwithin", distance=distance), expected
    )


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (pygeos.points(0, 0), 0, [0]),
        (pygeos.points(0, 0), 0.5, [0]),
        (pygeos.points(0, 0), 1.5, [0, 1]),
        (pygeos.points(0.5, 0.5), 1, [0, 1]),
        (pygeos.points(0.5, 0.5), 0.5, [0, 1]),
        (box(0, 0, 1, 1), 0, [0, 1]),
        (box(0, 0, 1, 1), 2, [0, 1, 2]),
        (pygeos.multipoints([[5, 5], [7, 7]]), 0.5, [5, 7]),
        (
            pygeos.multipoints([[5, 5], [7, 7]]),
            2.5,
            [3, 4, 5, 6, 7, 8, 9],
        ),
    ],
)
def test_query_dwithin_polygons(poly_tree, geometry, distance, expected):
    assert_array_equal(
        poly_tree.query(geometry, predicate="dwithin", distance=distance), expected
    )


### Bulk query tests
@pytest.mark.parametrize(
    "tree_geometry,geometry,expected",
    [
        # Empty tree returns no results
        ([], [None], (2, 0)),
        ([], [point], (2, 0)),
        # None is ignored when constructing and querying the tree
        ([None], [None], (2, 0)),
        ([point], [None], (2, 0)),
        ([None], [point], (2, 0)),
        # Empty is included in the tree, but ignored when querying the tree
        ([empty], [empty], (2, 0)),
        ([empty], [point], (2, 0)),
        ([point, empty], [empty], (2, 0)),
        # Only the non-empty geometry gets hits
        ([point, empty], [point, empty], (2, 1)),
        (
            [point, empty, empty_point, empty_line_string],
            [point, empty, empty_point, empty_line_string],
            (2, 1),
        ),
    ],
)
def test_query_bulk(tree_geometry, geometry, expected):
    tree = pygeos.STRtree(np.array(tree_geometry))
    assert tree.query_bulk(np.array(geometry)).shape == expected


def test_query_bulk_wrong_dimensions(tree):
    with pytest.raises(TypeError, match="Array should be one dimensional"):
        tree.query_bulk([[pygeos.points(0.5, 0.5)]])


@pytest.mark.parametrize("geometry", [[], "foo", 1])
def test_query_bulk_wrong_type(tree, geometry):
    with pytest.raises(TypeError, match="Array should be of object dtype"):
        tree.query_bulk(geometry)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        ([pygeos.points(0.5, 0.5)], [[], []]),
        # points intersect
        ([pygeos.points(1, 1)], [[0], [1]]),
        # first and last points intersect
        (
            [pygeos.points(1, 1), pygeos.points(-1, -1), pygeos.points(2, 2)],
            [[0, 2], [1, 2]],
        ),
        # box contains points
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # bigger box contains more points
        ([box(5, 5, 15, 15)], [[0, 0, 0, 0, 0], [5, 6, 7, 8, 9]]),
        # first and last boxes contains points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(5, 5, 15, 15)],
            [[0, 0, 2, 2, 2, 2, 2], [0, 1, 5, 6, 7, 8, 9]],
        ),
        # envelope of buffer contains points
        ([pygeos.buffer(pygeos.points(3, 3), 1)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of points contains points
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
        # nulls and empty should be skipped
        ([None, empty, pygeos.points(1, 1)], [[2], [1]]),
    ],
)
def test_query_bulk_points(tree, geometry, expected):
    assert_array_equal(tree.query_bulk(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        ([pygeos.points(0, 0)], [[0], [0]]),
        ([pygeos.points(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line
        ([pygeos.points(0, 0.5)], [[0], [0]]),
        # point at shared vertex between 2 lines
        ([pygeos.points(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # envelope of buffer overlaps envelope of 2 lines
        ([pygeos.buffer(pygeos.points(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # envelope of points overlaps 5 lines (touches edge of 2 envelopes)
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[0, 0, 0, 0], [4, 5, 6, 7]]),
    ],
)
def test_query_bulk_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_bulk(geometry), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects edge of envelopes of 2 polygons
        ([pygeos.points(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        ([pygeos.points(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap envelope of 2 polyons
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # larger box overlaps envelope of 3 polygons
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # envelope of buffer overlaps envelope of 3 polygons
        ([pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # envelope of larger buffer overlaps envelope of 6 polygons
        (
            [pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps 3 polygons
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[0, 0, 0], [5, 6, 7]]),
    ],
)
def test_query_bulk_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query_bulk(geometry), expected)


def test_query_bulk_invalid_predicate(tree):
    with pytest.raises(ValueError):
        tree.query_bulk(pygeos.points(1, 1), predicate="bad_predicate")


### predicate == 'intersects'


def test_query_bulk_with_prepared(tree):
    geom = np.array([box(0, 0, 1, 1), box(3, 3, 5, 5)])
    expected = tree.query_bulk(geom, predicate="intersects")

    # test with array of partially prepared geometries
    pygeos.prepare(geom[0])
    assert_array_equal(expected, tree.query_bulk(geom, predicate="intersects"))

    # test with fully prepared geometries
    pygeos.prepare(geom)
    assert_array_equal(expected, tree.query_bulk(geom, predicate="intersects"))


# TEMPORARY xfail: MultiPoint intersects with prepared geometries does not work
# properly on GEOS 3.5.x; it was fixed in 3.6+
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # points do not intersect
        ([pygeos.points(0.5, 0.5)], [[], []]),
        # points intersect
        ([pygeos.points(1, 1)], [[0], [1]]),
        # box contains points
        ([box(3, 3, 6, 6)], [[0, 0, 0, 0], [3, 4, 5, 6]]),
        # first and last boxes contain points
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(3, 3, 6, 6)],
            [[0, 0, 2, 2, 2, 2], [0, 1, 3, 4, 5, 6]],
        ),
        # envelope of buffer contains more points than intersect buffer
        # due to diagonal distance
        ([pygeos.buffer(pygeos.points(3, 3), 1)], [[0], [3]]),
        # envelope of buffer with 1/2 distance between points should intersect
        # same points as envelope
        (
            [pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0], [2, 3, 4]],
        ),
        # multipoints intersect
        pytest.param(
            [pygeos.multipoints([[5, 5], [7, 7]])],
            [[0, 0], [5, 7]],
            marks=pytest.mark.xfail(reason="GEOS 3.5"),
        ),
        # envelope of points contains points, but points do not intersect
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        pytest.param(
            [pygeos.multipoints([[5, 7], [7, 7]])],
            [[0], [7]],
            marks=pytest.mark.xfail(reason="GEOS 3.5"),
        ),
    ],
)
def test_query_bulk_intersects_points(tree, geometry, expected):
    assert_array_equal(tree.query_bulk(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point intersects first line
        ([pygeos.points(0, 0)], [[0], [0]]),
        ([pygeos.points(0.5, 0.5)], [[0], [0]]),
        # point within envelope of first line but does not intersect
        ([pygeos.points(0, 0.5)], [[], []]),
        # point at shared vertex between 2 lines
        ([pygeos.points(1, 1)], [[0, 0], [0, 1]]),
        # box overlaps envelope of first 2 lines (touches edge of 1)
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap multiple lines each
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2, 2], [0, 1, 1, 2, 3]],
        ),
        # buffer intersects 2 lines
        ([pygeos.buffer(pygeos.points(3, 3), 0.5)], [[0, 0], [2, 3]]),
        # buffer intersects midpoint of line at tangent
        ([pygeos.buffer(pygeos.points(2, 1), HALF_UNIT_DIAG)], [[0], [1]]),
        # envelope of points overlaps lines but intersects none
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint intersects
        ([pygeos.multipoints([[5, 7], [7, 7]])], [[0, 0], [6, 7]]),
    ],
)
def test_query_bulk_intersects_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_bulk(geometry, predicate="intersects"), expected)


@pytest.mark.parametrize(
    "geometry,expected",
    [
        # point within first polygon
        ([pygeos.points(0, 0.5)], [[0], [0]]),
        ([pygeos.points(0.5, 0)], [[0], [0]]),
        # midpoint between two polygons intersects both
        ([pygeos.points(0.5, 0.5)], [[0, 0], [0, 1]]),
        # point intersects single polygon
        ([pygeos.points(1, 1)], [[0], [1]]),
        # box overlaps envelope of 2 polygons
        ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]),
        # first and last boxes overlap
        (
            [box(0, 0, 1, 1), box(100, 100, 110, 110), box(2, 2, 3, 3)],
            [[0, 0, 2, 2], [0, 1, 2, 3]],
        ),
        # larger box intersects 3 polygons
        ([box(0, 0, 1.5, 1.5)], [[0, 0, 0], [0, 1, 2]]),
        # buffer overlaps 3 polygons
        ([pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG)], [[0, 0, 0], [2, 3, 4]]),
        # larger buffer overlaps 6 polygons (touches midpoints)
        (
            [pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG)],
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]],
        ),
        # envelope of points overlaps polygons, but points do not intersect
        ([pygeos.multipoints([[5, 7], [7, 5]])], [[], []]),
        # only one point of multipoint within polygon
        ([pygeos.multipoints([[5, 7], [7, 7]])], [[0], [7]]),
    ],
)
def test_query_bulk_intersects_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query_bulk(geometry, predicate="intersects"), expected)


def test_query_bulk_predicate_errors(tree):
    with pytest.raises(pygeos.GEOSException):
        tree.query_bulk(
            [pygeos.linestrings([1, 1], [1, float("nan")])], predicate="touches"
        )


@pytest.mark.skipif(pygeos.geos_version >= (3, 10, 0), reason="GEOS >= 3.10")
def test_query_bulk_dwithin_geos_version(tree):
    with pytest.raises(UnsupportedGEOSOperation, match="requires GEOS >= 3.10"):
        tree.query_bulk(pygeos.points(0, 0), predicate="dwithin", distance=1)


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "distance,match",
    [
        (None, "distance parameter must be provided"),
        ("foo", "could not convert string to float"),
        (["foo"], "could not convert string to float"),
        ([0, 1], "Could not broadcast distance to match geometry"),
        ([[1.0]], "should be one dimensional"),
    ],
)
def test_query_bulk_dwithin_invalid_distance(tree, distance, match):
    with pytest.raises(ValueError, match=match):
        tree.query_bulk(pygeos.points(0, 0), predicate="dwithin", distance=distance)


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (pygeos.points(0.25, 0.25), 0, [[], []]),
        (pygeos.points(0.25, 0.25), -1, [[], []]),
        (pygeos.points(0.25, 0.25), np.nan, [[], []]),
        (pygeos.Geometry("POINT EMPTY"), 1, [[], []]),
        (pygeos.points(0.25, 0.25), 0.5, [[0], [0]]),
        (pygeos.points(0.25, 0.25), [0.5], [[0], [0]]),
        (pygeos.points(0.25, 0.25), 2.5, [[0, 0, 0], [0, 1, 2]]),
        (pygeos.points(3, 3), 1.5, [[0, 0, 0], [2, 3, 4]]),
        # 2 equidistant points in tree
        (pygeos.points(0.5, 0.5), 0.75, [[0, 0], [0, 1]]),
        (
            [None, pygeos.points(0.5, 0.5)],
            0.75,
            [
                [
                    1,
                    1,
                ],
                [0, 1],
            ],
        ),
        (
            [pygeos.points(0.5, 0.5), pygeos.points(0.25, 0.25)],
            0.75,
            [[0, 0, 1], [0, 1, 0]],
        ),
        (
            [pygeos.points(0, 0.2), pygeos.points(1.75, 1.75)],
            [0.25, 2],
            [[0, 1, 1, 1], [0, 1, 2, 3]],
        ),
        # all points intersect box
        (box(0, 0, 3, 3), 0, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        (box(0, 0, 3, 3), 0.25, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        # intersecting and nearby points
        (box(1, 1, 2, 2), 1.5, [[0, 0, 0, 0], [0, 1, 2, 3]]),
        # # return nearest point in tree for each point in multipoint
        (pygeos.multipoints([[0.25, 0.25], [1.5, 1.5]]), 0.75, [[0, 0, 0], [0, 1, 2]]),
        # 2 equidistant points per point in multipoint
        (
            pygeos.multipoints([[0.5, 0.5], [3.5, 3.5]]),
            0.75,
            [[0, 0, 0, 0], [0, 1, 3, 4]],
        ),
    ],
)
def test_query_bulk_dwithin_points(tree, geometry, distance, expected):
    assert_array_equal(
        tree.query_bulk(geometry, predicate="dwithin", distance=distance), expected
    )


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (pygeos.points(0.5, 0.5), 0, [[0], [0]]),
        (pygeos.points(0.5, 0.5), 1.0, [[0, 0], [0, 1]]),
        (pygeos.points(2, 2), 0.5, [[0, 0], [1, 2]]),
        (box(0, 0, 1, 1), 0.5, [[0, 0], [0, 1]]),
        (box(0.5, 0.5, 1.5, 1.5), 0.5, [[0, 0], [0, 1]]),
        # multipoints at endpoints of 2 lines each
        (pygeos.multipoints([[5, 5], [7, 7]]), 0.5, [[0, 0, 0, 0], [4, 5, 6, 7]]),
        # multipoints are equidistant from 2 lines
        (pygeos.multipoints([[5, 7], [7, 5]]), 1.5, [[0, 0], [5, 6]]),
    ],
)
def test_query_bulk_dwithin_lines(line_tree, geometry, distance, expected):
    assert_array_equal(
        line_tree.query_bulk(geometry, predicate="dwithin", distance=distance), expected
    )


@pytest.mark.skipif(pygeos.geos_version < (3, 10, 0), reason="GEOS < 3.10")
@pytest.mark.parametrize(
    "geometry,distance,expected",
    [
        (pygeos.points(0, 0), 0, [[0], [0]]),
        (pygeos.points(0, 0), 0.5, [[0], [0]]),
        (pygeos.points(0, 0), 1.5, [[0, 0], [0, 1]]),
        (pygeos.points(0.5, 0.5), 1, [[0, 0], [0, 1]]),
        (pygeos.points(0.5, 0.5), 0.5, [[0, 0], [0, 1]]),
        (box(0, 0, 1, 1), 0, [[0, 0], [0, 1]]),
        (box(0, 0, 1, 1), 2, [[0, 0, 0], [0, 1, 2]]),
        (pygeos.multipoints([[5, 5], [7, 7]]), 0.5, [[0, 0], [5, 7]]),
        (
            pygeos.multipoints([[5, 5], [7, 7]]),
            2.5,
            [[0, 0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8, 9]],
        ),
    ],
)
def test_query_bulk_dwithin_polygons(poly_tree, geometry, distance, expected):
    assert_array_equal(
        poly_tree.query_bulk(geometry, predicate="dwithin", distance=distance), expected
    )


### STRtree nearest


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_nearest_empty_tree():
    tree = pygeos.STRtree([])
    assert_array_equal(tree.nearest(point), [[], []])


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("geometry", ["I am not a geometry"])
def test_nearest_invalid_geom(tree, geometry):
    with pytest.raises(TypeError):
        tree.nearest(geometry)


# TODO: add into regular results
@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("geometry,expected", [(None, [[], []]), ([None], [[], []])])
def test_nearest_none(tree, geometry, expected):
    assert_array_equal(tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0.25, 0.25), [[0], [0]]),
        (pygeos.points(0.75, 0.75), [[0], [1]]),
        (pygeos.points(1, 1), [[0], [1]]),
        ([pygeos.points(1, 1), pygeos.points(0, 0)], [[0, 1], [1, 0]]),
        ([pygeos.points(1, 1), pygeos.points(0.25, 1)], [[0, 1], [1, 1]]),
        ([pygeos.points(-10, -10), pygeos.points(100, 100)], [[0, 1], [0, 9]]),
        (box(0.5, 0.5, 0.75, 0.75), [[0], [1]]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), HALF_UNIT_DIAG), [[0], [2]]),
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [[0], [3]]),
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [[0], [7]]),
        (pygeos.multipoints([[5, 7], [7, 5]]), [[0], [6]]),
        (None, [[], []]),
        ([None], [[], []]),
    ],
)
def test_nearest_points(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.xfail(reason="equidistant geometries may produce nondeterministic results")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # 2 equidistant points in tree
        (pygeos.points(0.5, 0.5), [0, 1]),
        # multiple points in box
        (box(0, 0, 3, 3), [0, 1, 2, 3]),
        # return nearest point in tree for each point in multipoint
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
    ],
)
def test_nearest_points_equidistant(tree, geometry, expected):
    result = tree.nearest(geometry)
    assert result[1] in expected


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0.5, 0.5), [[0], [0]]),
        (pygeos.points(1.5, 0.5), [[0], [0]]),
        (pygeos.box(0.5, 1.5, 1, 2), [[0], [1]]),
        (pygeos.linestrings([[0, 0.5], [1, 2.5]]), [[0], [0]]),
    ],
)
def test_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.nearest(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.xfail(reason="equidistant geometries may produce nondeterministic results")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # at junction between 2 lines
        (pygeos.points(2, 2), [1, 2]),
        # contains one line, intersects with another
        (box(0, 0, 1, 1), [0, 1]),
        # overlaps 2 lines
        (box(0.5, 0.5, 1.5, 1.5), [0, 1]),
        # box overlaps 2 lines and intersects endpoints of 2 more
        (box(3, 3, 5, 5), [2, 3, 4, 5]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), HALF_UNIT_DIAG), [1, 2]),
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3]),
        # multipoints at endpoints of 2 lines each
        (pygeos.multipoints([[5, 5], [7, 7]]), [4, 5, 6, 7]),
        # second point in multipoint at endpoints of 2 lines
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [6, 7]),
        # multipoints are equidistant from 2 lines
        (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6]),
    ],
)
def test_nearest_lines_equidistant(line_tree, geometry, expected):
    result = line_tree.nearest(geometry)
    assert result[1] in expected


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0, 0), [[0], [0]]),
        (pygeos.points(2, 2), [[0], [2]]),
        (pygeos.box(0, 5, 1, 6), [[0], [3]]),
        (pygeos.multipoints([[5, 7], [7, 5]]), [[0], [6]]),
    ],
)
def test_nearest_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.nearest(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.xfail(reason="equidistant geometries may produce nondeterministic results")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        # 2 polygons in tree overlap point
        (pygeos.points(0.5, 0.5), [0, 1]),
        # box overlaps multiple polygons
        (box(0, 0, 1, 1), [0, 1]),
        (box(0.5, 0.5, 1.5, 1.5), [0, 1, 2]),
        (box(3, 3, 5, 5), [3, 4, 5]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), HALF_UNIT_DIAG), [2, 3]),
        # completely overlaps one polygon, touches 2 others
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2, 3, 4]),
        # each point in multi point intersects a polygon in tree
        (pygeos.multipoints([[5, 5], [7, 7]]), [5, 7]),
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [5, 7]),
    ],
)
def test_nearest_polygons_equidistant(poly_tree, geometry, expected):
    result = poly_tree.nearest(geometry)
    assert result[1] in expected


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_nearest_all_empty_tree():
    tree = pygeos.STRtree([])
    assert_array_equal(tree.nearest_all(point), [[], []])


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize("geometry", ["I am not a geometry"])
def test_nearest_all_invalid_geom(tree, geometry):
    with pytest.raises(TypeError):
        tree.nearest_all(geometry)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,return_distance,expected",
    [(None, False, [[], []]), ([None], False, [[], []]), (None, True, ([[], []], []))],
)
def test_nearest_all_none(tree, geometry, return_distance, expected):
    if return_distance:
        index, distance = tree.nearest_all(geometry, return_distance=True)
        assert_array_equal(index, expected[0])
        assert_array_equal(distance, expected[1])

    else:
        assert_array_equal(tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected", [(empty, [[], []]), ([empty, point], [[1, 1], [2, 3]])]
)
def test_nearest_all_empty_geom(tree, geometry, expected):
    assert_array_equal(tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0.25, 0.25), [[0], [0]]),
        (pygeos.points(0.75, 0.75), [[0], [1]]),
        (pygeos.points(1, 1), [[0], [1]]),
        # 2 equidistant points in tree
        (pygeos.points(0.5, 0.5), [[0, 0], [0, 1]]),
        ([pygeos.points(1, 1), pygeos.points(0, 0)], [[0, 1], [1, 0]]),
        ([pygeos.points(1, 1), pygeos.points(0.25, 1)], [[0, 1], [1, 1]]),
        ([pygeos.points(-10, -10), pygeos.points(100, 100)], [[0, 1], [0, 9]]),
        (box(0.5, 0.5, 0.75, 0.75), [[0], [1]]),
        # multiple points in box
        (box(0, 0, 3, 3), [[0, 0, 0, 0], [0, 1, 2, 3]]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), 1), [[0, 0], [2, 3]]),
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [[0], [3]]),
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [[0], [7]]),
        (pygeos.multipoints([[5, 7], [7, 5]]), [[0], [6]]),
        # return nearest point in tree for each point in multipoint
        (pygeos.multipoints([[5, 5], [7, 7]]), [[0, 0], [5, 7]]),
        # 2 equidistant points per point in multipoint
        (pygeos.multipoints([[0.5, 0.5], [3.5, 3.5]]), [[0, 0, 0, 0], [0, 1, 3, 4]]),
    ],
)
def test_nearest_all_points(tree, geometry, expected):
    assert_array_equal(tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0.5, 0.5), [[0], [0]]),
        # at junction between 2 lines, will return both
        (pygeos.points(2, 2), [[0, 0], [1, 2]]),
        # contains one line, intersects with another
        (box(0, 0, 1, 1), [[0, 0], [0, 1]]),
        # overlaps 2 lines
        (box(0.5, 0.5, 1.5, 1.5), [[0, 0], [0, 1]]),
        # second box overlaps 2 lines and intersects endpoints of 2 more
        ([box(0, 0, 0.5, 0.5), box(3, 3, 5, 5)], [[0, 1, 1, 1, 1], [0, 2, 3, 4, 5]]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), 1), [[0, 0, 0], [1, 2, 3]]),
        (pygeos.buffer(pygeos.points(3, 3), 0.5), [[0, 0], [2, 3]]),
        # multipoints at endpoints of 2 lines each
        (pygeos.multipoints([[5, 5], [7, 7]]), [[0, 0, 0, 0], [4, 5, 6, 7]]),
        # second point in multipoint at endpoints of 2 lines
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [[0, 0], [6, 7]]),
        # multipoints are equidistant from 2 lines
        (pygeos.multipoints([[5, 7], [7, 5]]), [[0, 0], [5, 6]]),
    ],
)
def test_nearest_all_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [
        (pygeos.points(0, 0), [[0], [0]]),
        (pygeos.points(2, 2), [[0], [2]]),
        # 2 polygons in tree overlap point
        (pygeos.points(0.5, 0.5), [[0, 0], [0, 1]]),
        # box overlaps multiple polygons
        (box(0, 0, 1, 1), [[0, 0], [0, 1]]),
        (box(0.5, 0.5, 1.5, 1.5), [[0, 0, 0], [0, 1, 2]]),
        ([box(0, 0, 1, 1), box(3, 3, 5, 5)], [[0, 0, 1, 1, 1], [0, 1, 3, 4, 5]]),
        (pygeos.buffer(pygeos.points(2.5, 2.5), HALF_UNIT_DIAG), [[0, 0], [2, 3]]),
        # completely overlaps one polygon, touches 2 others
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [[0, 0, 0], [2, 3, 4]]),
        # each point in multi point intersects a polygon in tree
        (pygeos.multipoints([[5, 5], [7, 7]]), [[0, 0], [5, 7]]),
        (pygeos.multipoints([[5.5, 5], [7, 7]]), [[0, 0], [5, 7]]),
        (pygeos.multipoints([[5, 7], [7, 5]]), [[0], [6]]),
    ],
)
def test_nearest_all_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.nearest_all(geometry), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,max_distance,expected",
    [
        # using unset max_distance should return all nearest
        (pygeos.points(0.5, 0.5), None, [[0, 0], [0, 1]]),
        # using large max_distance should return all nearest
        (pygeos.points(0.5, 0.5), 10, [[0, 0], [0, 1]]),
        # using small max_distance should return no results
        (pygeos.points(0.5, 0.5), 0.1, [[], []]),
        # using small max_distance should only return results in that distance
        ([pygeos.points(0.5, 0.5), pygeos.points(0, 0)], 0.1, [[1], [0]]),
    ],
)
def test_nearest_all_max_distance(tree, geometry, max_distance, expected):
    assert_array_equal(tree.nearest_all(geometry, max_distance=max_distance), expected)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,max_distance",
    [(pygeos.points(0.5, 0.5), 0), (pygeos.points(0.5, 0.5), -1)],
)
def test_nearest_all_invalid_max_distance(tree, geometry, max_distance):
    with pytest.raises(ValueError, match="max_distance must be greater than 0"):
        tree.nearest_all(geometry, max_distance=max_distance)


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
def test_nearest_all_nonscalar_max_distance(tree):
    with pytest.raises(ValueError, match="parameter only accepts scalar values"):
        tree.nearest_all(pygeos.points(0.5, 0.5), max_distance=[1])


@pytest.mark.skipif(pygeos.geos_version < (3, 6, 0), reason="GEOS < 3.6")
@pytest.mark.parametrize(
    "geometry,expected",
    [(pygeos.points(0, 0), [0.0]), (pygeos.points(0.5, 0.5), [0.7071, 0.7071])],
)
def test_nearest_all_return_distance(tree, geometry, expected):
    assert_array_equal(
        np.round(tree.nearest_all(geometry, return_distance=True)[1], 4), expected
    )


def test_strtree_threaded_query():
    ## Create data
    polygons = pygeos.polygons(np.random.randn(1000, 3, 2))
    # needs to be big enough to trigger the segfault
    N = 100_000
    points = pygeos.points(4 * np.random.random(N) - 2, 4 * np.random.random(N) - 2)

    ## Slice parts of the arrays -> 4x4 => 16 combinations
    n = int(len(polygons) / 4)
    polygons_parts = [
        polygons[:n],
        polygons[n : 2 * n],
        polygons[2 * n : 3 * n],
        polygons[3 * n :],
    ]
    n = int(len(points) / 4)
    points_parts = [
        points[:n],
        points[n : 2 * n],
        points[2 * n : 3 * n],
        points[3 * n :],
    ]

    ## Creating the trees in advance
    trees = []
    for i in range(4):
        left = points_parts[i]
        tree = pygeos.STRtree(left)
        trees.append(tree)

    ## The function querying the trees in parallel

    def thread_func(idxs):
        i, j = idxs
        tree = trees[i]
        right = polygons_parts[j]
        return tree.query_bulk(right, predicate="contains")

    with ThreadPoolExecutor() as pool:
        list(pool.map(thread_func, itertools.product(range(4), range(4))))
