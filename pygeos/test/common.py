import numpy as np
import pygeos
import sys
from contextlib import contextmanager

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
point_z = pygeos.points(1.0, 1.0, 1.0)
line_string_z = pygeos.linestrings([(0, 0, 0), (1, 0, 1), (1, 1, 2)])
polygon_z = pygeos.polygons([(0, 0, 0), (2, 0, 1), (2, 2, 2), (0, 2, 3), (0, 0, 0)])
geometry_collection_z = pygeos.geometrycollections([point_z, line_string_z])
polygon_with_hole = pygeos.Geometry(
    "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"
)
empty_point = pygeos.Geometry("POINT EMPTY")
empty_line_string = pygeos.Geometry("LINESTRING EMPTY")
empty_polygon = pygeos.Geometry("POLYGON EMPTY")
empty = pygeos.Geometry("GEOMETRYCOLLECTION EMPTY")
point_nan = pygeos.points(np.nan, np.nan)

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
    before = sys.getrefcount(obj)
    yield
    assert sys.getrefcount(obj) == before + 1


@contextmanager
def assert_decreases_refcount(obj):
    before = sys.getrefcount(obj)
    yield
    assert sys.getrefcount(obj) == before - 1
