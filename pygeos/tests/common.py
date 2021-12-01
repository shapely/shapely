import sys
from contextlib import contextmanager

import numpy as np
import pytest

import pygeos

point_polygon_testdata = (
    pygeos.points(np.arange(6), np.arange(6)),
    pygeos.box(2, 2, 4, 4),
)
point = pygeos.points(2, 3)
line_string = pygeos.linestrings([(0, 0), (1, 0), (1, 1)])
linear_ring = pygeos.linearrings([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
polygon = pygeos.polygons([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
multi_point = pygeos.multipoints([(0, 0), (1, 2)])
multi_line_string = pygeos.multilinestrings([[(0, 0), (1, 2)]])
multi_polygon = pygeos.multipolygons(
    [
        [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        [(2.1, 2.1), (2.2, 2.1), (2.2, 2.2), (2.1, 2.2), (2.1, 2.1)],
    ]
)
geometry_collection = pygeos.geometrycollections(
    [pygeos.points(51, -1), pygeos.linestrings([(52, -1), (49, 2)])]
)
point_z = pygeos.points(2, 3, 4)
line_string_z = pygeos.linestrings([(0, 0, 4), (1, 0, 4), (1, 1, 4)])
polygon_z = pygeos.polygons([(0, 0, 4), (2, 0, 4), (2, 2, 4), (0, 2, 4), (0, 0, 4)])
geometry_collection_z = pygeos.geometrycollections([point_z, line_string_z])
polygon_with_hole = pygeos.Geometry(
    "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"
)
empty_point = pygeos.Geometry("POINT EMPTY")
empty_point_z = pygeos.Geometry("POINT Z EMPTY")
empty_line_string = pygeos.Geometry("LINESTRING EMPTY")
empty_line_string_z = pygeos.Geometry("LINESTRING Z EMPTY")
empty_polygon = pygeos.Geometry("POLYGON EMPTY")
empty = pygeos.Geometry("GEOMETRYCOLLECTION EMPTY")
line_string_nan = pygeos.linestrings([(np.nan, np.nan), (np.nan, np.nan)])
multi_point_z = pygeos.multipoints([(0, 0, 4), (1, 2, 4)])
multi_line_string_z = pygeos.multilinestrings([[(0, 0, 4), (1, 2, 4)]])
multi_polygon_z = pygeos.multipolygons(
    [
        [(0, 0, 4), (1, 0, 4), (1, 1, 4), (0, 1, 4), (0, 0, 4)],
        [(2.1, 2.1, 4), (2.2, 2.1, 4), (2.2, 2.2, 4), (2.1, 2.2, 4), (2.1, 2.1, 4)],
    ]
)
polygon_with_hole_z = pygeos.Geometry(
    "POLYGON Z((0 0 4, 0 10 4, 10 10 4, 10 0 4, 0 0 4), (2 2 4, 2 4 4, 4 4 4, 4 2 4, 2 2 4))"
)

all_types = (
    point,
    line_string,
    linear_ring,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
    geometry_collection,
    empty,
)


@contextmanager
def assert_increases_refcount(obj):
    try:
        before = sys.getrefcount(obj)
    except AttributeError:  # happens on Pypy
        pytest.skip("sys.getrefcount is not available.")
    yield
    assert sys.getrefcount(obj) == before + 1


@contextmanager
def assert_decreases_refcount(obj):
    try:
        before = sys.getrefcount(obj)
    except AttributeError:  # happens on Pypy
        pytest.skip("sys.getrefcount is not available.")
    yield
    assert sys.getrefcount(obj) == before - 1


@contextmanager
def ignore_invalid(condition=True):
    if condition:
        with np.errstate(invalid="ignore"):
            yield
    else:
        yield
