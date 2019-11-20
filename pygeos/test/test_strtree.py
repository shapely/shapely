import pygeos
from pygeos import box
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from .common import point, empty, assert_increases_refcount, assert_decreases_refcount


@pytest.fixture
def tree():
    geoms = pygeos.points(np.arange(10), np.arange(10))
    yield pygeos.STRtree(geoms)


def test_init_with_none():
    tree = pygeos.STRtree(np.array([None]))
    assert tree.query(point).size == 0


def test_init_with_no_geometry():
    with pytest.raises(TypeError):
        pygeos.STRtree(np.array(["Not a geometry"], dtype=object))


def test_init_increases_refcount():
    arr = np.array([point])
    with assert_increases_refcount(arr):
        _ = pygeos.STRtree(arr)


def test_del_decreases_refcount():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    with assert_decreases_refcount(arr):
        del tree


def test_geometries_property():
    arr = np.array([point])
    tree = pygeos.STRtree(arr)
    assert arr is tree.geometries


def test_flush_geometries(tree):
    arr = pygeos.points(np.arange(10), np.arange(10))
    tree = pygeos.STRtree(arr)
    # Dereference geometries
    arr[:] = None
    # Still it does not lead to a segfault
    tree.query(point)


def test_query_no_geom(tree):
    with pytest.raises(TypeError):
        tree.query("I am not a geometry")


def test_query_none(tree):
    assert tree.query(None).size == 0


def test_query_empty(tree):
    assert tree.query(empty).size == 0


@pytest.mark.parametrize("envelope,expected", [
    (pygeos.points(1, 1), [1]),
    (box(0, 0, 1, 1), [0, 1]),
    (box(5, 5, 15, 15), [5, 6, 7, 8, 9]),
    (pygeos.multipoints([[5, 7], [7, 5]]), [5, 6, 7]),  # query by envelope
])
def test_query(tree, envelope, expected):
    assert_array_equal(tree.query(envelope), expected)
