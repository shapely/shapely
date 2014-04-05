import cython
cimport cpython.array
import numpy as np
cimport numpy as np

from shapely.geos import lgeos
import shapely.prepared

include "../_geos.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def contains(geometry, np.double_t[:] x, np.double_t[:] y):
    """
    Vectorized (element-wise) version of `contains` for multiple points within
    a single geometry.

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
    Mask of points contained within the given `geometry`
    """
    cdef size_t idx
    cdef unsigned int n = x.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] result = np.empty(n, dtype=np.bool)

    cdef GEOSContextHandle_t geos_handle
    cdef GEOSPreparedGeometry *geos_prepared_geom
    cdef GEOSCoordSequence *cs
    cdef GEOSGeometry *point

    # Prepare the geometry if it hasn't already been prepared.
    if not isinstance(geometry, shapely.prepared.PreparedGeometry):
        geometry = shapely.prepared.prep(geometry)

    geos_h = get_geos_context_handle()
    geos_geom = geos_from_prepared(geometry)

    for idx in xrange(n):
        # Construct a coordinate sequence with our x, y values.
        cs = GEOSCoordSeq_create_r(geos_h, 1, 2)
        GEOSCoordSeq_setX_r(geos_h, cs, 0, x[idx])
        GEOSCoordSeq_setY_r(geos_h, cs, 0, y[idx])
        
        # Construct a point with this sequence.
        p = GEOSGeom_createPoint_r(geos_h, cs)
        
        # Put the result of whether the point is "contained" by the
        # prepared geometry into the result array. 
        result[idx] = GEOSPreparedContains_r(geos_h, geos_geom, p)
        GEOSGeom_destroy_r(geos_h, p)

    return result

