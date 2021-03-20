# distutils: define_macros=GEOS_USE_ONLY_R_API

from cpython cimport PyObject
cimport cython
from cython cimport view

import numpy as np
cimport numpy as np
import pygeos

from pygeos._geos cimport (
    GEOSGeometry,
    GEOSGeom_clone_r,
    GEOSGetGeometryN_r,
    get_geos_handle,
    GEOSGeom_destroy_r,
    GEOSGeom_createCollection_r,
    GEOSGeomTypeId_r,
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
    cdef const GEOSGeometry *part = NULL

    counts = pygeos.get_num_geometries(array)
    cdef Py_ssize_t count = counts.sum()

    if count == 0:
        # return immediately if there are no geometries to return
        return (
            np.empty(shape=(0, ), dtype=np.object_),
            np.empty(shape=(0, ), dtype=np.intp)
        )

    parts = np.empty(shape=(count, ), dtype=np.object_)
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
                # cast part back to <GEOSGeometry> to discard const qualifier
                # pending issue #227
                parts_view[idx] = PyGEOS_CreateGeometry(<GEOSGeometry *>part, geos_handle)

                idx += 1

    return parts, index


@cython.boundscheck(False)
@cython.wraparound(False)
def collections_1d(object geometries, object indices, int geometry_type = 7):
    cdef Py_ssize_t geom_idx_1 = 0
    cdef Py_ssize_t coll_idx = 0
    cdef unsigned int coll_size = 0
    cdef Py_ssize_t coll_geom_idx = 0
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *coll = NULL
    cdef int expected_type = -1
    cdef int expected_type_alt = -1
    cdef int curr_type = -1

    if geometry_type == 4:  # MULTIPOINT
        expected_type = 0
    elif geometry_type == 5:  # MULTILINESTRING
        expected_type = 1
        expected_type_alt = 2
    elif geometry_type == 6:  # MULTIPOLYGON
        expected_type = 3
    elif geometry_type == 7:
        pass
    else:
        raise ValueError(f"Invalid geometry_type: {geometry_type}.")

    # Cast input arrays and define memoryviews for later usage
    geometries = np.asarray(geometries, dtype=np.object)
    if geometries.ndim != 1:
        raise TypeError("geometries is not a one-dimensional array.")

    indices = np.asarray(indices, dtype=np.int32)
    if indices.ndim != 1:
        raise TypeError("indices is not a one-dimensional array.")

    if geometries.shape[0] != indices.shape[0]:
        raise ValueError("geometries and indices do not have equal size.")

    if geometries.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=np.object_)

    if np.any(indices[1:] < indices[:indices.shape[0] - 1]):
        raise ValueError("The indices should be sorted.")  

    cdef object[:] geometries_view = geometries
    cdef int[:] indices_view = indices

    # get the geometry count per collection
    cdef int[:] collection_size = np.bincount(indices).astype(np.int32)

    # A temporary array for the geometries that will be given to CreateCollection.
    # Its size equals max(collection_size) to accomodate the largest collection.
    temp_geoms = np.empty(shape=(np.max(collection_size), ), dtype=np.intp)
    cdef np.intp_t[:] temp_geoms_view = temp_geoms

    # The final target array
    cdef Py_ssize_t n_colls = collection_size.shape[0]
    result = np.empty(shape=(n_colls, ), dtype=np.object_)
    cdef object[:] result_view = result

    with get_geos_handle() as geos_handle:
        for coll_idx in range(n_colls):
            coll_size = 0

            # fill the temporary array with geometries belonging to this collection
            for coll_geom_idx in range(collection_size[coll_idx]):
                if PyGEOS_GetGEOSGeometry(<PyObject *>geometries_view[geom_idx_1 + coll_geom_idx], &geom) == 0:
                    # deallocate previous temp geometries (preventing memory leaks)
                    for coll_geom_idx in range(coll_size):
                        GEOSGeom_destroy_r(geos_handle, <GEOSGeometry *>temp_geoms_view[coll_geom_idx])
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # ignore missing values
                if geom == NULL:
                    continue

                # Check geometry subtype for non-geometrycollections
                if geometry_type != 7:
                    curr_type = GEOSGeomTypeId_r(geos_handle, geom)
                    if curr_type != expected_type and curr_type != expected_type_alt:
                        # deallocate previous temp geometries (preventing memory leaks)
                        for coll_geom_idx in range(coll_size):
                            GEOSGeom_destroy_r(geos_handle, <GEOSGeometry *>temp_geoms_view[coll_geom_idx])
                        raise TypeError(
                            f"One of the arguments has unexpected geometry type {curr_type}."
                        )

                # assign to the temporary geometry array                   
                temp_geoms_view[coll_size] = <np.intp_t>GEOSGeom_clone_r(geos_handle, geom)
                coll_size += 1

            # create the collection
            coll = GEOSGeom_createCollection_r(
                geos_handle,
                geometry_type, 
                <GEOSGeometry**> &temp_geoms_view[0],
                coll_size
            )

            result_view[coll_idx] = PyGEOS_CreateGeometry(coll, geos_handle)

            geom_idx_1 += collection_size[coll_idx]

    return result
