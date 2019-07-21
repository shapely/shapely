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
    "get_num_geometries",
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


def get_type_id(geometry):
    return ufuncs.get_type_id(geometry)


def get_dimensions(geometry):
    return ufuncs.get_dimensions(geometry)


def get_coordinate_dimensions(geometry):
    return ufuncs.get_coordinate_dimensions(geometry)


def get_num_coordinates(geometry):
    return ufuncs.get_num_coordinates(geometry)


def normalize(geometry):
    return ufuncs.normalize(geometry)


def get_srid(geometry):
    return ufuncs.get_srid(geometry)


def set_srid(geometry, srid):
    return ufuncs.set_srid(geometry, np.intc(srid))


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


def get_num_geometries(collection):
    return ufuncs.get_num_geometries(collection)


def get_geometry(collection, index):
    return ufuncs.get_geometry(collection, np.intc(index))
