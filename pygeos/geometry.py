from enum import IntEnum
import numpy as np
from . import ufuncs

__all__ = [
    "Geometry",
    "GeometryType",
    "get_type_id",
    "get_dimensions",
    "get_coordinate_dimensions",
    "get_num_coordinates",
    "normalize",
    "get_srid",
    "set_srid",
    "get_x",
    "get_y",
    "get_num_points",
    "get_start_point",
    "get_end_point",
    "get_point",
    "get_exterior_ring",
    "get_num_interior_rings",
    "get_interior_ring",
    "get_geometry",
]


Geometry = ufuncs.Geometry


class GeometryType(IntEnum):
    POINT = 0
    LINESTRING = 1
    LINEARRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7


# generic


def get_type_id(geometries):
    return ufuncs.get_type_id(geometries)


def get_dimensions(geometries):
    return ufuncs.get_dimensions(geometries)


def get_coordinate_dimensions(geometries):
    return ufuncs.get_coordinate_dimensions(geometries)


def get_num_coordinates(geometries):
    return ufuncs.get_num_coordinates(geometries)


def normalize(geometries):
    return ufuncs.normalize(geometries)


def get_srid(geometries):
    return ufuncs.get_srid(geometries)


def set_srid(geometries, srid):
    return ufuncs.set_srid(geometries, np.intc(srid))


# points


def get_x(point):
    return ufuncs.get_x(point)


def get_y(point):
    return ufuncs.get_y(point)


# linestrings


def get_num_points(linestring):
    return ufuncs.get_num_points(linestring)


def get_start_point(linestring):
    return ufuncs.get_start_point(linestring)


def get_end_point(linestring):
    return ufuncs.get_end_point(linestring)


def get_point(linestring, index):
    return ufuncs.get_point(linestring, np.intc(index))


# polygons


def get_exterior_ring(polygon):
    return ufuncs.get_exterior_ring(polygon)


def get_num_interior_rings(polygon):
    return ufuncs.get_num_interior_rings(polygon)


def get_interior_ring(polygon, index):
    return ufuncs.get_interior_ring(polygon, np.intc(index))


# collections


def get_geometry(geometries, index):
    return ufuncs.get_geometry(geometries, np.intc(index))
