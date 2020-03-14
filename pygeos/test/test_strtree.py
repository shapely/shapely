import math
import pygeos
from pygeos import box
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from .common import point, empty, assert_increases_refcount, assert_decreases_refcount


# the distance between 2 points spaced at whole numbers along a diagonal
HALF_UNIT_DIAG = math.sqrt(2) / 2
EPS = 1e-9


@pytest.fixture
def tree():
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


@pytest.fixture
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = pygeos.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield pygeos.STRtree(geoms)


@pytest.fixture
def poly_tree():
    # create buffers so that midpoint between two buffers intersects
    # each buffer.  NOTE: add EPS to help mitigate rounding errors at midpoint.
    geoms = pygeos.buffer(
        pygeos.points(np.arange(10), np.arange(10)), HALF_UNIT_DIAG + EPS, quadsegs=32
    )
    yield pygeos.STRtree(geoms)


def test_init_with_none():
    tree = pygeos.STRtree(np.array([None]))
    assert tree.query(point).size == 0


def test_init_with_no_geometry():
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
    import gc; gc.collect()
    # Still it does not lead to a segfault
    tree.query(point)


def test_len():
    arr = np.array([point, None, point])
    tree = pygeos.STRtree(arr)
    assert len(tree) == 2


def test_geometries_property():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    assert arr is tree.geometries


def test_query_no_geom(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    assert tree.query(None).size == 0


def test_query_empty(tree):
    assert tree.query(empty).size == 0


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


def test_query_tree_with_none():
    # valid GEOS binary predicate, but not supported for query
    tree = pygeos.STRtree([
        pygeos.Geometry("POINT (0 0)"), None, pygeos.Geometry("POINT (2 2)")
    ])
    assert tree.query(pygeos.points(2, 2), predicate="intersects") == [2]


### predicate == 'intersects'

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
        pytest.param(pygeos.multipoints([[5, 5], [7, 7]]), [5, 7], marks=pytest.mark.xfail(pygeos.geos_version<(3, 6, 0), reason="GEOS 3.5")),
        # envelope of points contains points, but points do not intersect
        (pygeos.multipoints([[5, 7], [7, 5]]), []),
        # only one point of multipoint intersects
        pytest.param(pygeos.multipoints([[5, 7], [7, 7]]), [7], marks=pytest.mark.xfail(pygeos.geos_version<(3, 6, 0), reason="GEOS 3.5")),
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
        # point does not contain any polygs (not valid relation)
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
        (box(0, 0, 2, 2), [0,2]),
        # buffer overlaps 3 polygons and contains 1
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG), [2,4]),
        # larger buffer overlaps 6 polygons (touches midpoints) but contains one
        (pygeos.buffer(pygeos.points(3, 3), 3 * HALF_UNIT_DIAG), [1,2,4,5]),
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
        # buffer crosses line
        (pygeos.buffer(pygeos.points(2, 1), 1), [1]),
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
        (pygeos.buffer(pygeos.points(2, 1), HALF_UNIT_DIAG), []),
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
        (pygeos.points(HALF_UNIT_DIAG+EPS, 0), [0]),
        # box overlaps envelope of 2 polygons does not touch any at edge
        (box(0, 0, 1, 1), []),
        # box overlaps 2 polygons and touches edge of first
        (box(HALF_UNIT_DIAG+EPS, 0, 2, 2), [0]),
        # buffer overlaps 3 polygons but does not touch any at edge
        (pygeos.buffer(pygeos.points(3, 3), HALF_UNIT_DIAG+EPS), []),
        # only one point of multipoint within polygon but does not touch
        (pygeos.multipoints([[0, 0], [7,7], [7, 8]]), []),
    ],
)
def test_query_touches_polygons(poly_tree, geometry, expected):
    assert_array_equal(poly_tree.query(geometry, predicate="touches"), expected)
