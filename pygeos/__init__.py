from . import geos_ufuncs as ufuncs

from functools import wraps
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import \
    Point, LineString, LinearRing, Polygon, MultiPoint, MultiLineString,\
    MultiPolygon, GeometryCollection

GEOM_CLASSES = [
    Point,
    LineString,
    LinearRing,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection
]

GEOM_DTYPE = np.dtype([("obj", "O"), ("_ptr", "intp")], align=True)


def _obj_to_ptr_elem(obj):
    # if obj is None  # TODO How to return NULL pointer?
    return obj.__geom__


_ptr_from_obj = np.vectorize(_obj_to_ptr_elem, otypes=[np.intp])


def _ptr_to_obj_elem(ptr, geom_type_id, has_z):
    # if ptr == NULL  # TODO How to check for a NULL pointer?
    ob = BaseGeometry()
    ob.__class__ = GEOM_CLASSES[geom_type_id]
    ob.__geom__ = ptr
    ob._ndim = 3 if has_z else 2
    ob._is_empty = False
    return ob


_ptr_to_obj = np.vectorize(_ptr_to_obj_elem, otypes=[np.object])


def garr_from_shapely(objs):
    """Construct a geometry-dtype array from an (iterable of) shapely geometry
    objects.
    """
    if isinstance(objs, BaseGeometry):
        objs = [objs]
    objs = np.asarray(objs, dtype=np.object)
    result = np.empty_like(objs, dtype=GEOM_DTYPE)
    result['obj'] = objs
    result['_ptr'] = _ptr_from_obj(objs)
    return result


def garr_finalize(arr):
    """Construct shapely objects for newly created GEOSGeometry pointers in a
    geometry-dtype array

    The created shapely objects take ownership of the GEOSGeometry.
    """
    mask = arr['obj'] == None
    arr_filt = arr[mask]
    objs = _ptr_to_obj(
        arr_filt['_ptr'],
        ufuncs.geom_type_id(arr_filt),
        ufuncs.has_z(arr_filt)
    )
    arr['obj'][mask] = objs
    return arr


def garr_from_pointers(arr):
    """Construct a geometry-dtype array from an array of GEOSGeometry pointers.

    The created shapely objects take ownership of the GEOSGeometry.
    """
    result = np.empty_like(arr, dtype=GEOM_DTYPE)
    result['_ptr'] = arr
    return garr_finalize(arr)


def _maybe_convert(x):
    if not (isinstance(x, np.ndarray) and x.dtype == GEOM_DTYPE):
        x = garr_from_shapely(x)
    return x


def geos_func_G_x(f):
    @wraps(f)
    def func(a, *args, **kwargs):
        return f(_maybe_convert(a), *args, **kwargs)
    return func


def geos_func_GG_x(f):
    @wraps(f)
    def func(a, b, *args, **kwargs):
        return f(_maybe_convert(a), _maybe_convert(b), *args, **kwargs)
    return func


def geos_func_G_G(f):
    @wraps(f)
    def func(a, *args, **kwargs):
        return garr_finalize(f(_maybe_convert(a), *args, **kwargs))
    return func


def geos_func_GG_G(f):
    @wraps(f)
    def func(a, b, *args, **kwargs):
        return garr_finalize(
            f(_maybe_convert(a), _maybe_convert(b), *args, **kwargs)
        )
    return func


interpolate = geos_func_G_G(ufuncs.interpolate)
interpolate_normalized = geos_func_G_G(ufuncs.interpolate_normalized)
buffer = geos_func_G_G(ufuncs.buffer)
clone = geos_func_G_G(ufuncs.clone)
envelope = geos_func_G_G(ufuncs.envelope)
intersection = geos_func_GG_G(ufuncs.intersection)
convex_hull = geos_func_G_G(ufuncs.convex_hull)
difference = geos_func_GG_G(ufuncs.difference)
symmetric_difference = geos_func_GG_G(ufuncs.symmetric_difference)
boundary = geos_func_G_G(ufuncs.boundary)
union = geos_func_GG_G(ufuncs.union)
unary_union = geos_func_G_G(ufuncs.unary_union)
point_on_surface = geos_func_G_G(ufuncs.point_on_surface)
get_centroid = geos_func_G_G(ufuncs.get_centroid)
line_merge = geos_func_G_G(ufuncs.line_merge)
simplify = geos_func_G_G(ufuncs.simplify)
topology_preserve_simplify = geos_func_G_G(ufuncs.topology_preserve_simplify)
extract_unique_points = geos_func_G_G(ufuncs.extract_unique_points)
disjoint = geos_func_GG_G(ufuncs.disjoint)
touches = geos_func_GG_x(ufuncs.touches)
intersects = geos_func_GG_x(ufuncs.intersects)
crosses = geos_func_GG_x(ufuncs.crosses)
within = geos_func_GG_x(ufuncs.within)
contains = geos_func_GG_x(ufuncs.contains)
overlaps = geos_func_GG_x(ufuncs.overlaps)
equals = geos_func_GG_x(ufuncs.equals)
covers = geos_func_GG_x(ufuncs.covers)
covered_by = geos_func_GG_x(ufuncs.covered_by)
is_empty = geos_func_G_x(ufuncs.is_empty)
is_simple = geos_func_G_x(ufuncs.is_simple)
is_ring = geos_func_G_x(ufuncs.is_ring)
has_z = geos_func_G_x(ufuncs.has_z)
is_closed = geos_func_G_x(ufuncs.is_closed)
is_valid = geos_func_G_x(ufuncs.is_valid)
geom_type_id = geos_func_G_x(ufuncs.geom_type_id)
get_srid = geos_func_G_x(ufuncs.get_srid)
get_num_geometries = geos_func_G_x(ufuncs.get_num_geometries)
get_num_interior_rings = geos_func_G_x(ufuncs.get_num_interior_rings)
get_num_points = geos_func_G_x(ufuncs.get_num_points)
get_x = geos_func_G_x(ufuncs.get_x)
get_y = geos_func_G_x(ufuncs.get_y)
get_interior_ring_n = geos_func_G_G(ufuncs.get_interior_ring_n)
get_exterior_ring = geos_func_G_G(ufuncs.get_exterior_ring)
get_num_coordinates = geos_func_G_x(ufuncs.get_num_coordinates)
get_dimensions = geos_func_G_x(ufuncs.get_dimensions)
get_coordinate_dimensions = geos_func_G_x(ufuncs.get_coordinate_dimensions)
get_point_n = geos_func_G_G(ufuncs.get_point_n)
get_start_point = geos_func_G_G(ufuncs.get_start_point)
get_end_point = geos_func_G_G(ufuncs.get_end_point)
area = geos_func_G_x(ufuncs.area)
length = geos_func_G_x(ufuncs.length)
distance = geos_func_GG_x(ufuncs.distance)
hausdorff_distance = geos_func_GG_x(ufuncs.hausdorff_distance)
get_length = geos_func_G_x(ufuncs.get_length)
