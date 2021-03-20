"""
Provides a wrapper for GEOS types and functions.

Note: GEOS functions in Cython must be called using the get_geos_handle context manager.
Example:
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)
"""

cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t

    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil

    ctypedef struct GEOSGeometry

    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t handle,
                                           const GEOSGeometry* g,
                                           int n) nogil except NULL
    int GEOSGeomTypeId_r(GEOSContextHandle_t handle, GEOSGeometry* g) nogil except -1
    void GEOSGeom_destroy_r(GEOSContextHandle_t handle,
                                    GEOSGeometry* g) nogil
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t handle,
                                   const GEOSGeometry* g) nogil except NULL

    GEOSGeometry* GEOSGeom_createCollection_r(
        GEOSContextHandle_t handle,
        int type,
        GEOSGeometry** geoms,
        unsigned int ngeoms
    ) nogil except NULL


cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef GEOSContextHandle_t __enter__(self)
