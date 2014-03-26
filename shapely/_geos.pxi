# The beginnings of a Cython definition of GEOS. In the future much of this
# could be auto-generated.

ctypedef long ptr


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence
    ctypedef struct GEOSPreparedGeometry
    
    GEOSCoordSequence *GEOSCoordSeq_create_r(GEOSContextHandle_t, unsigned int, unsigned int)
    GEOSCoordSequence *GEOSGeom_getCoordSeq_r(GEOSContextHandle_t, GEOSGeometry *)

    int GEOSCoordSeq_getSize_r(GEOSContextHandle_t, GEOSCoordSequence *, int *)
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_getX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *)
    int GEOSCoordSeq_getY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *)
    int GEOSCoordSeq_getZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *)

    GEOSGeometry *GEOSGeom_createPoint_r(GEOSContextHandle_t, GEOSCoordSequence *)
    GEOSGeometry *GEOSGeom_createLineString_r(GEOSContextHandle_t, GEOSCoordSequence *)
    GEOSGeometry *GEOSGeom_createLinearRing_r(GEOSContextHandle_t, GEOSCoordSequence *)

    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry *)

    char GEOSPreparedContains_r(GEOSContextHandle_t, const GEOSPreparedGeometry *,
                                const GEOSGeometry *)


cdef GEOSContextHandle_t get_geos_context_handle():
    cdef ptr handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


cdef GEOSPreparedGeometry *geos_from_prepared(shapely_geom) except *:
    """Get the Prepared GEOS geometry pointer from the given shapely geometry."""
    cdef ptr geos_geom = shapely_geom._geom
    return <GEOSPreparedGeometry *>geos_geom