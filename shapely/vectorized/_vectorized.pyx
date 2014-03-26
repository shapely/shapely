import cython
import numpy as np
cimport numpy as np

from shapely.geometry import Point
from shapely.geos import lgeos
import shapely.prepared


ctypedef np.double_t float64

include "../_geos.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def contains(geometry not None,
             np.ndarray[float64, ndim=2] x,
             np.ndarray[float64, ndim=2] y):
    """
    Vectorized (element-wise) version of "contains" for multiple points within
    a single geometry.

    Parameters
    ----------
    geometry : PreparedGeometry or subclass of BaseGeometry
        The geometry which is to be checked to see whether each point is
        contained within. The geometry will be "prepared" if it is not already
        a PreparedGeometry instance.
    x : np.array
        The x coordinates of the points to check. The array's dtype must be
        np.float64 and it must be 2 dimensional.
    y : np.array
        The y coordinates of the points to check. The array's dtype must be
        np.float64 and it must be 2 dimensional.
    
    """
    # Note: This has not been written with maximal efficiency in mind - 
    # the GIL really could be released within this function's for loop.
    cdef int i, j, ni, nj
    ni, nj = x.shape[0], x.shape[1]
    result = np.empty([ni, nj], dtype=np.bool)

    # Prepare the geometry if it hasn't already been prepared.
    if not isinstance(geometry, shapely.prepared.PreparedGeometry):
        geometry = shapely.prepared.prep(geometry)

    geos_handle = <GEOSContextHandle_t> get_geos_context_handle()
    geos_prepared_geom = <GEOSPreparedGeometry *> geos_from_prepared(geometry)

    # N.B. The point and associated CS must be constructed for each point.
    # I'm not certain why the coordinate sequence can't be updated as we go,
    # but that results in incorrect results (the tests will fail).
    for j in range(nj):
        for i in range(ni):
            # Construct a coordinate sequence with our x, y values.
            cs = <GEOSCoordSequence *> GEOSCoordSeq_create_r(geos_handle, 1, 2)
            GEOSCoordSeq_setX_r(geos_handle, cs, 0, x[i, j])
            GEOSCoordSeq_setY_r(geos_handle, cs, 0, y[i, j])
            
            # Construct a point with this sequence.
            point = <GEOSGeometry *> GEOSGeom_createPoint_r(geos_handle, cs)
            
            # Put the result of whether the point is "contained" by the
            # prepared geometry into the result array. 
            result[i, j] = <char> GEOSPreparedContains_r(geos_handle, geos_prepared_geom, point)
            GEOSGeom_destroy_r(geos_handle, point)
    return result
