import gc
import os
import pickle
import subprocess
import sys

import pytest

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Point, Polygon, box
from shapely.geos import geos_version
from shapely import strtree
from shapely.strtree import STRtree
from shapely import wkt

from .conftest import requires_geos_342


point = Point(2, 3)
empty = wkt.loads("GEOMETRYCOLLECTION EMPTY")


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,num_results",
    [(Point(2, 2).buffer(0.99), 1), (Point(2, 2).buffer(1.0), 3)],
)
def test_query(geoms, query_geom, num_results):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)
    results = tree.query(query_geom)
    assert len(results) == num_results


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,expected",
    [(Point(2, 2).buffer(0.99), [2]), (Point(2, 2).buffer(1.0), [1, 2, 3])],
)
def test_query_enumeration_idx(geoms, query_geom, expected):
    """Store enumeration idx"""
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, range(len(geoms)))
    results = tree.query_items(query_geom)
    assert sorted(results) == sorted(expected)


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(5)]])
@pytest.mark.parametrize("items", [None, list(range(1, 6)), list("abcde")])
@pytest.mark.parametrize(
    "query_geom,expected",
    [(Point(2, 2).buffer(0.99), [2]), (Point(2, 2).buffer(1.0), [1, 2, 3])],
)
def test_query_items(geoms, items, query_geom, expected):
    """Store enumeration idx"""
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, items)
    results = tree.query_items(query_geom)
    expected = [items[idx] for idx in expected] if items is not None else expected
    assert sorted(results) == sorted(expected)


@pytest.mark.parametrize(
    "tree_geometry, geometry,expected",
    [
        ([point], box(0, 0, 10, 10), [0]),
        # None/empty is ignored in the tree, but the index of the valid geometry
        # should be retained.
        ([None, point], box(0, 0, 10, 10), [1]),
        ([None, empty, point], box(0, 0, 10, 10), [2]),
    ],
)
def test_query_items_with_empty(tree_geometry, geometry, expected):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(tree_geometry)
    assert tree.query_items(geometry) == expected


@requires_geos_342
def test_insert_empty_geometry():
    """
    Passing nothing but empty geometries results in an empty strtree.
    The query segfaults if the empty geometry was actually inserted.
    """
    empty = Polygon()
    geoms = [empty]
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)
    query = Polygon([(0, 0), (1, 1), (2, 0), (0, 0)])
    results = tree.query(query)
    assert len(results) == 0


@requires_geos_342
def test_query_empty_geometry():
    """
    Empty geometries should be filtered out.
    The query segfaults if the empty geometry was actually inserted.
    """
    empty = Polygon()
    point = Point(1, 0.5)
    geoms = [empty, point]
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)
    query = Polygon([(0, 0), (1, 1), (2, 0), (0, 0)])
    results = tree.query(query)
    assert len(results) == 1
    assert results[0] == point


@requires_geos_342
def test_references():
    """Don't crash due to dangling references"""
    empty = Polygon()
    point = Point(1, 0.5)
    geoms = [empty, point]
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)

    empty = None
    point = None
    gc.collect()

    query = Polygon([(0, 0), (1, 1), (2, 0), (0, 0)])
    results = tree.query(query)
    assert len(results) == 1
    assert results[0] == Point(1, 0.5)


@requires_geos_342
def test_safe_delete():
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree([])

    _lgeos = strtree.lgeos
    strtree.lgeos = None

    del tree

    strtree.lgeos = _lgeos


@requires_geos_342
def test_pickle_persistence():
    """
    Don't crash trying to use unpickled GEOS handle.
    """
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree([Point(i, i).buffer(0.1) for i in range(3)], range(3))

    pickled_strtree = pickle.dumps(tree)
    unpickle_script_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unpickle-strtree.py")
    proc = subprocess.Popen(
        [sys.executable, str(unpickle_script_file_path)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(input=pickled_strtree)
    proc.wait()
    assert proc.returncode == 0


@pytest.mark.skipif(geos_version < (3, 6, 0), reason="GEOS 3.6.0 required")
@pytest.mark.parametrize(
    "geoms",
    [
        [
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
            Point(0, 0.5),
        ]
    ],
)
@pytest.mark.parametrize("query_geom", [Point(0, 0.4)])
def test_nearest_geom(geoms, query_geom):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms)
    result = tree.nearest(query_geom)
    assert result.geom_type == "Point"
    assert result.x == 0.0
    assert result.y == 0.5


@pytest.mark.skipif(geos_version < (3, 6, 0), reason="GEOS 3.6.0 required")
@pytest.mark.parametrize(
    "geoms",
    [
        [
            Point(0, 0.5),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ]
    ],
)
@pytest.mark.parametrize("items", [list(range(1, 4)), list("abc")])
@pytest.mark.parametrize("query_geom", [Point(0, 0.4)])
def test_nearest_item(geoms, items, query_geom):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, items)
    assert tree.nearest_item(query_geom) == items[0]


@pytest.mark.parametrize(["geoms", "items"], [([], None), ([], [])])
def test_nearest_empty(geoms, items):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, items)
    assert tree.nearest_item(None) is None


@pytest.mark.parametrize(["geoms", "items"], [([], None), ([], [])])
def test_nearest_items(geoms, items):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, items)
    assert tree.nearest_item(None) is None


@pytest.mark.skipif(geos_version < (3, 6, 0), reason="GEOS 3.6.0 required")
@pytest.mark.parametrize(
    "geoms",
    [
        [
            Point(0, 0.5),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        ]
    ],
)
@pytest.mark.parametrize("items", [list(range(1, 4)), list("abc")])
@pytest.mark.parametrize("query_geom", [Point(0, 0.5)])
def test_nearest_item_exclusive(geoms, items, query_geom):
    with pytest.warns(ShapelyDeprecationWarning):
        tree = STRtree(geoms, items)
    assert tree.nearest_item(query_geom, exclusive=True) != items[0]
