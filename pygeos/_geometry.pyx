from cpython cimport PyObject
cimport cython

import numpy as np
cimport numpy as np
import pygeos

from pygeos._geos cimport (
    GEOSGeometry,
    GEOSGeom_clone_r,
    GEOSGetGeometryN_r,
    get_geos_handle
)
from pygeos._pygeos_api cimport (
    import_pygeos_c_api,
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry
)

# initialize PyGEOS C API
import_pygeos_c_api()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_parts(object[:] array):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *part = NULL

    counts = pygeos.get_num_geometries(array)

    # None elements in array return -1 for count, so
    # they must be filtered out before calculating total count
    cdef Py_ssize_t count = counts[counts>0].sum()

    if count <= 0:
        # return immeidately if there are no geometries to return
        # count is negative when the only entries in array are None
        return (
            np.empty(shape=(0, ), dtype=np.object),
            np.empty(shape=(0, ), dtype=np.intp)
        )

    parts = np.empty(shape=(count, ), dtype=np.object)
    index = np.empty(shape=(count, ), dtype=np.intp)

    cdef int[:] counts_view = counts
    cdef object[:] parts_view = parts
    cdef np.intp_t[:] index_view = index

    with get_geos_handle() as geos_handle:
        for geom_idx in range(array.size):
            if counts_view[geom_idx] <= 0:
                # No parts to return, skip this item
                continue

            if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
                raise TypeError("One of the arguments is of incorrect type. "
                                "Please provide only Geometry objects.")

            if geom == NULL:
                continue

            for part_idx in range(counts_view[geom_idx]):
                index_view[idx] = geom_idx
                part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)

                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                parts_view[idx] = PyGEOS_CreateGeometry(part, geos_handle)

                idx += 1

    return parts, index
