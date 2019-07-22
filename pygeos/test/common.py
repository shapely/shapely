import numpy as np
import pygeos

point_polygon_testdata = (
    pygeos.points(np.arange(6), np.arange(6)),
    pygeos.box(2, 2, 4, 4),
)
point = pygeos.points(2, 2)
line_string = pygeos.linestrings([[0, 0], [1, 0], [1, 1]])
linear_ring = pygeos.linearrings(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0)))
polygon = pygeos.polygons(((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0), (0.0, 0.0)))
multi_point = pygeos.multipoints([[0.0, 0.0], [1.0, 2.0]])
multi_line_string = pygeos.multilinestrings([[[0.0, 0.0], [1.0, 2.0]]])
multi_polygon = pygeos.multipolygons(
    [
        ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        ((0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1)),
    ]
)
geometry_collection = pygeos.geometrycollections(
    [pygeos.points(51, -1), pygeos.linestrings([(52, -1), (49, 2)])]
)
point_z = pygeos.points(1.0, 1.0, 1.0)

all_types = (
    point,
    line_string,
    linear_ring,
    polygon,
    multi_point,
    multi_line_string,
    multi_polygon,
    geometry_collection,
    pygeos.Empty,
)
