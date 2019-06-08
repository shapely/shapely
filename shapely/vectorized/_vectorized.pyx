import cython
cimport cpython.array

import numpy as np
cimport numpy as np

import shapely.prepared


include "../_geos.pxi"


# Define the predicate function, which returns True/False from functions such as GEOSPreparedContains_r
# and GEOSPreparedWithin_r.
ctypedef char (* predicate)(GEOSContextHandle_t, const GEOSPreparedGeometry *, const GEOSGeometry *) nogil


def contains(geometry, x, y):
    """
    Vectorized (element-wise) version of `contains` which checks whether
    multiple points are contained by a single geometry.

    Parameters
    ----------
    geometry : PreparedGeometry or subclass of BaseGeometry
        The geometry which is to be checked to see whether each point is
        contained within. The geometry will be "prepared" if it is not already
        a PreparedGeometry instance.
    x : array
        The x coordinates of the points to check. 
    y : array
        The y coordinates of the points to check.

    Returns
    -------
    Mask of points contained by the given `geometry`.

    """
    return _predicated_elementwise(geometry, x, y, GEOSPreparedContains_r)


def touches(geometry, x, y):
    """
    Vectorized (element-wise) version of `touches` which checks whether
    multiple points touch the exterior of a single geometry.

    Parameters
    ----------
    geometry : PreparedGeometry or subclass of BaseGeometry
        The geometry which is to be checked to see whether each point is
        contained within. The geometry will be "prepared" if it is not already
        a PreparedGeometry instance.
    x : array
        The x coordinates of the points to check. 
    y : array
        The y coordinates of the points to check.

    Returns
    -------
    Mask of points which touch the exterior of the given `geometry`.

    """
    return _predicated_elementwise(geometry, x, y, GEOSPreparedTouches_r)


cdef _predicated_elementwise(geometry, x, y, predicate fn):
    """
    Implements elementwise True/False testing, given a predicate function
    such as "contains" or "touches". x and y arrays can be of any type, order
    and dtype, and will be cast for appropriate calling to "_predicated_1d".

    """
    # Support coordinate sequences and other array like objects.
    x, y = np.asanyarray(x), np.asanyarray(y)
    if x.shape != y.shape:
        raise ValueError('X and Y shapes must be equivalent.')

    if x.dtype != np.float64:
        x = x.astype(np.float64)
    if y.dtype != np.float64:
        y = y.astype(np.float64)

    result = _predicated_1d(geometry, x.ravel(), y.ravel(), fn)
    return result.reshape(x.shape)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef _predicated_1d(geometry, np.double_t[:] x, np.double_t[:] y, predicate fn):
    
    cdef Py_ssize_t idx
    cdef unsigned int n = x.size
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.uint8)
    cdef GEOSContextHandle_t geos_handle
    cdef GEOSPreparedGeometry *geos_prepared_geom
    cdef GEOSCoordSequence *cs
    cdef GEOSGeometry *point

    # Prepare the geometry if it hasn't already been prepared.
    if not isinstance(geometry, shapely.prepared.PreparedGeometry):
        geometry = shapely.prepared.prep(geometry)

    geos_h = get_geos_context_handle()
    geos_geom = geos_from_prepared(geometry)

    with nogil:
        for idx in xrange(n):
            # Construct a coordinate sequence with our x, y values.
            cs = GEOSCoordSeq_create_r(geos_h, 1, 2)
            GEOSCoordSeq_setX_r(geos_h, cs, 0, x[idx])
            GEOSCoordSeq_setY_r(geos_h, cs, 0, y[idx])
            
            # Construct a point with this sequence.
            p = GEOSGeom_createPoint_r(geos_h, cs)
            
            # Put the result of whether the point is "contained" by the
            # prepared geometry into the result array. 
            result[idx] = <np.uint8_t> fn(geos_h, geos_geom, p)
            GEOSGeom_destroy_r(geos_h, p)

    return result.view(dtype=np.bool)


def array_transform(func, geom):
    """Applies `func` to all coordinates of `geom` and returns a new
    geometry of the same type from the transformed coordinates.

    `func` maps x, y, and optionally z to output xp, yp, zp. The input
    parameters may iterable types like lists or arrays or single values.
    The output shall be of the same type. Scalars in, scalars out.
    Lists in, lists out.

    For example, here is an identity function applicable to both types
    of input.

      def id_func(x, y, z=None):
          return tuple(filter(None, [x, y, z]))

      g2 = transform(id_func, g1)

    A partially applied transform function from pyproj satisfies the
    requirements for `func`.

      from functools import partial
      from pyproj import Transformer

      transformer = Transformer.from_crs("epsg:4326", "epsg:26913")
      g2 = transform(transformer.transform, g1)

    Lambda expressions such as the one in

      g2 = transform(lambda x, y, z=None: (x+1.0, y+1.0), g1)

    also satisfy the requirements for `func`.
    """
    if geom.is_empty:
        return geom

    if geom.type in ("Point", "LineString", "LinearRing", "Polygon"):

        def transform_line(line_geom):
            geom_array = np.asarray(geom)
            if geom_array.shape[-1] == 2:
                geom_array[:, 0], geom_array[:, 1] = func(
                    geom_array[:, 0], geom_array[:, 1]
                )
            else:
                geom_array[:, 0], geom_array[:, 1], geom_array[
                    :, 2
                ] = func(
                    geom_array[:, 0], geom_array[:, 1], geom_array[:, 2]
                )
            return type(line_geom)(geom_array)

        if geom.type == "Point":
            geom_array = np.asarray(geom)
            if geom_array.shape == (2,):
                geom_array[0], geom_array[1] = func(
                    geom_array[0], geom_array[1]
                )
            else:
                geom_array[0], geom_array[1], geom_array[2] = func(
                    geom_array[0], geom_array[1], geom_array[2]
                )
            return type(geom)(geom_array)
        elif geom.type in ("LineString", "LinearRing"):
            return transform_line(geom)
        elif geom.type == "Polygon":
            shell = transform_line(geom.exterior)
            holes = list(transform_line(ring) for ring in geom.interiors)
            return type(geom)(shell, holes)

    elif geom.type.startswith("Multi") or geom.type == "GeometryCollection":
        return type(geom)([array_transform(part) for part in geom.geoms])
    else:
        raise ValueError("Type %r not recognized" % geom.type)
