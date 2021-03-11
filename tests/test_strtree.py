import ctypes
import gc
import os
import pickle
import subprocess
import sys

import pytest

from shapely.geometry import Point, Polygon
from shapely import strtree
from shapely.strtree import STRtree, query_callback

from .conftest import requires_geos_342


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,num_results", [(Point(2, 2).buffer(0.99), 1), (Point(2, 2).buffer(1.0), 3)]
)
def test_query(geoms, query_geom, num_results):
    tree = STRtree(geoms)
    results = tree.query(query_geom)
    assert len(results) == num_results


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,num_results", [(Point(2, 2).buffer(0.99), 1), (Point(2, 2).buffer(1.0), 3)]
)
def test_query_cb(geoms, query_geom, num_results):
    tree = STRtree(geoms)

    results = []

    def callback(item, userdata):
        obj = ctypes.cast(item, ctypes.py_object).value
        results.append(obj)

    tree.query_cb(query_geom, callback=callback)

    assert len(results) == num_results


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize("values", [["Hi!" for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,num_results", [(Point(2, 2).buffer(0.99), 1), (Point(2, 2).buffer(1.0), 3)]
)
def test_query_cb_str(geoms, values, query_geom, num_results):
    tree = STRtree(zip(geoms, values))

    results = []

    @query_callback
    def callback(value, userdata):
        results.append(value)

    tree.query_cb(query_geom, callback=callback)

    assert list(results) == ["Hi!"] * num_results


@requires_geos_342
@pytest.mark.parametrize("geoms", [[Point(i, i) for i in range(10)]])
@pytest.mark.parametrize("values", [[{"a": 1, "b": 2} for i in range(10)]])
@pytest.mark.parametrize(
    "query_geom,num_results", [(Point(2, 2).buffer(0.99), 1), (Point(2, 2).buffer(1.0), 3)]
)
def test_query_cb_dict(geoms, values, query_geom, num_results):
    tree = STRtree(zip(geoms, values))

    results = []

    @query_callback
    def callback(value, userdata):
        results.append(value)

    tree.query_cb(query_geom, callback=callback)

    assert list(results) == [{"a": 1, "b": 2}] * num_results


@requires_geos_342
def test_insert_empty_geometry():
    """
    Passing nothing but empty geometries results in an empty strtree.
    The query segfaults if the empty geometry was actually inserted.
    """
    empty = Polygon()
    geoms = [empty]
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
    tree = STRtree([(Point(i, i).buffer(0.1), "Hi!") for i in range(3)])
    pickled_strtree = pickle.dumps(tree)
    unpickle_script_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unpickle-strtree.py")
    proc = subprocess.Popen(
        [sys.executable, str(unpickle_script_file_path)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(input=pickled_strtree)
    proc.wait()
    assert proc.returncode == 0
