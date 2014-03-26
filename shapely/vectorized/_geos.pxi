ctypedef long ptr


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    
    ctypedef struct GEOSGeometry:
        pass
    
    ctypedef struct GEOSCoordSequence:
        pass
    
    ctypedef struct GEOSPreparedGeometry:
        pass

    char GEOSPreparedContains_r(GEOSContextHandle_t, const GEOSPreparedGeometry *,
                                const GEOSGeometry *)
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double)
    GEOSCoordSequence *GEOSGeom_getCoordSeq(GEOSGeometry *)
    GEOSGeometry *GEOSGeom_createPoint_r(GEOSContextHandle_t, GEOSCoordSequence *)
    GEOSCoordSequence *GEOSCoordSeq_create_r(GEOSContextHandle_t,
                                             unsigned int, unsigned int)
    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry*)


cdef GEOSContextHandle_t get_geos_context_handle():
    cdef ptr handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


cdef GEOSPreparedGeometry *geos_from_prepared(shapely_geom) except *:
    """Get the Prepared GEOS geometry pointer from the given shapely geometry."""
    cdef ptr geos_geom = shapely_geom._geom
    return <GEOSPreparedGeometry *>geos_geom