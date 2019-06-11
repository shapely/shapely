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


is_empty = geos_func_G_x(ufuncs.is_empty)
is_simple = geos_func_G_x(ufuncs.is_simple)
is_ring = geos_func_G_x(ufuncs.is_ring)
has_z = geos_func_G_x(ufuncs.has_z)
is_closed = geos_func_G_x(ufuncs.is_closed)
is_valid = geos_func_G_x(ufuncs.is_valid)
clone = geos_func_G_G(ufuncs.clone)
envelope = geos_func_G_G(ufuncs.envelope)
convex_hull = geos_func_G_G(ufuncs.convex_hull)
minimum_rotated_rectangle = geos_func_G_G(ufuncs.minimum_rotated_rectangle)
minimum_width = geos_func_G_G(ufuncs.minimum_width)
minimum_clearance_line = geos_func_G_G(ufuncs.minimum_clearance_line)
boundary = geos_func_G_G(ufuncs.boundary)
unary_union = geos_func_G_G(ufuncs.unary_union)
union_cascaded = geos_func_G_G(ufuncs.union_cascaded)
point_on_surface = geos_func_G_G(ufuncs.point_on_surface)
get_centroid = geos_func_G_G(ufuncs.get_centroid)
node = geos_func_G_G(ufuncs.node)
line_merge = geos_func_G_G(ufuncs.line_merge)
extract_unique_points = geos_func_G_G(ufuncs.extract_unique_points)
minimum_clearance = geos_func_G_G(ufuncs.minimum_clearance)
interpolate = geos_func_G_G(ufuncs.interpolate)
interpolate_normalized = geos_func_G_G(ufuncs.interpolate_normalized)
simplify = geos_func_G_G(ufuncs.simplify)
topology_preserve_simplify = geos_func_G_G(ufuncs.topology_preserve_simplify)
contains = geos_func_GG_x(ufuncs.contains)
covered_by = geos_func_GG_x(ufuncs.covered_by)
covers = geos_func_GG_x(ufuncs.covers)
crosses = geos_func_GG_x(ufuncs.crosses)
disjoint = geos_func_GG_x(ufuncs.disjoint)
equals = geos_func_GG_x(ufuncs.equals)
intersects = geos_func_GG_x(ufuncs.intersects)
overlaps = geos_func_GG_x(ufuncs.overlaps)
touches = geos_func_GG_x(ufuncs.touches)
within = geos_func_GG_x(ufuncs.within)
difference = geos_func_GG_G(ufuncs.difference)
symmetric_difference = geos_func_GG_G(ufuncs.symmetric_difference)
intersection = geos_func_GG_G(ufuncs.intersection)
shared_paths = geos_func_GG_G(ufuncs.shared_paths)
union = geos_func_GG_G(ufuncs.union)
area = geos_func_G_x(ufuncs.area)
length = geos_func_G_x(ufuncs.length)
distance = geos_func_GG_x(ufuncs.distance)
hausdorff_distance = geos_func_GG_x(ufuncs.hausdorff_distance)
geom_type_id = geos_func_G_x(ufuncs.geom_type_id)
