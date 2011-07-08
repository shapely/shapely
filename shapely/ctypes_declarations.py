# Prototyping of libgeos_c functions, now using a function written by
# `tartley`: http://trac.gispython.org/lab/ticket/189

import ctypes

class allocated_c_char_p(ctypes.c_char_p):
    pass

EXCEPTION_HANDLER_FUNCTYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

def prototype(lgeos, geosVersion):

    lgeos.initGEOS.argtypes = [EXCEPTION_HANDLER_FUNCTYPE, EXCEPTION_HANDLER_FUNCTYPE]
    lgeos.initGEOS.restype = None

    lgeos.finishGEOS.argtypes = []
    lgeos.finishGEOS.restype = None

    lgeos.GEOSversion.argtypes = []
    lgeos.GEOSversion.restype = ctypes.c_char_p

    lgeos.GEOSGeomFromWKT.restype = ctypes.c_void_p
    lgeos.GEOSGeomFromWKT.argtypes = [ctypes.c_char_p]

    lgeos.GEOSGeomToWKT.restype = allocated_c_char_p
    lgeos.GEOSGeomToWKT.argtypes = [ctypes.c_void_p]

    lgeos.GEOS_setWKBOutputDims.restype = ctypes.c_int
    lgeos.GEOS_setWKBOutputDims.argtypes = [ctypes.c_int]

    lgeos.GEOSGeomFromWKB_buf.restype = ctypes.c_void_p
    lgeos.GEOSGeomFromWKB_buf.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    lgeos.GEOSGeomToWKB_buf.restype = allocated_c_char_p
    lgeos.GEOSGeomToWKB_buf.argtypes = [ctypes.c_void_p , ctypes.POINTER(ctypes.c_size_t)]

    lgeos.GEOSCoordSeq_create.restype = ctypes.c_void_p
    lgeos.GEOSCoordSeq_create.argtypes = [ctypes.c_uint, ctypes.c_uint]

    lgeos.GEOSCoordSeq_clone.restype = ctypes.c_void_p
    lgeos.GEOSCoordSeq_clone.argtypes = [ctypes.c_void_p]

    lgeos.GEOSCoordSeq_destroy.restype = None
    lgeos.GEOSCoordSeq_destroy.argtypes = [ctypes.c_void_p]

    lgeos.GEOSCoordSeq_setX.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_setX.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double]

    lgeos.GEOSCoordSeq_setY.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_setY.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double]

    lgeos.GEOSCoordSeq_setZ.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_setZ.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double]

    lgeos.GEOSCoordSeq_setOrdinate.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_setOrdinate.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_double]

    lgeos.GEOSCoordSeq_getX.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_getX.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]

    lgeos.GEOSCoordSeq_getY.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_getY.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]

    lgeos.GEOSCoordSeq_getZ.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_getZ.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]

    lgeos.GEOSCoordSeq_getSize.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_getSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSCoordSeq_getDimensions.restype = ctypes.c_int
    lgeos.GEOSCoordSeq_getDimensions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSGeom_createPoint.restype = ctypes.c_void_p
    lgeos.GEOSGeom_createPoint.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_createLinearRing.restype = ctypes.c_void_p
    lgeos.GEOSGeom_createLinearRing.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_createLineString.restype = ctypes.c_void_p
    lgeos.GEOSGeom_createLineString.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_createPolygon.restype = ctypes.c_void_p
    lgeos.GEOSGeom_createPolygon.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

    lgeos.GEOSGeom_createCollection.restype = ctypes.c_void_p
    lgeos.GEOSGeom_createCollection.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]

    lgeos.GEOSGeom_clone.restype = ctypes.c_void_p
    lgeos.GEOSGeom_clone.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_destroy.restype = None
    lgeos.GEOSGeom_destroy.argtypes = [ctypes.c_void_p]

    lgeos.GEOSEnvelope.restype = ctypes.c_void_p
    lgeos.GEOSEnvelope.argtypes = [ctypes.c_void_p]

    lgeos.GEOSIntersection.restype = ctypes.c_void_p
    lgeos.GEOSIntersection.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSBuffer.restype = ctypes.c_void_p
    lgeos.GEOSBuffer.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]

    lgeos.GEOSSimplify.restype = ctypes.c_void_p
    lgeos.GEOSSimplify.argtypes = [ctypes.c_void_p, ctypes.c_double]

    lgeos.GEOSTopologyPreserveSimplify.restype = ctypes.c_void_p
    lgeos.GEOSTopologyPreserveSimplify.argtypes = [ctypes.c_void_p, ctypes.c_double]

    lgeos.GEOSConvexHull.restype = ctypes.c_void_p
    lgeos.GEOSConvexHull.argtypes = [ctypes.c_void_p]

    lgeos.GEOSDifference.restype = ctypes.c_void_p
    lgeos.GEOSDifference.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSSymDifference.restype = ctypes.c_void_p
    lgeos.GEOSSymDifference.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSBoundary.restype = ctypes.c_void_p
    lgeos.GEOSBoundary.argtypes = [ctypes.c_void_p]

    lgeos.GEOSUnion.restype = ctypes.c_void_p
    lgeos.GEOSUnion.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSPointOnSurface.restype = ctypes.c_void_p
    lgeos.GEOSPointOnSurface.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGetCentroid.restype = ctypes.c_void_p
    lgeos.GEOSGetCentroid.argtypes = [ctypes.c_void_p]

    lgeos.GEOSRelate.restype = allocated_c_char_p
    lgeos.GEOSRelate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSPolygonize.restype = ctypes.c_void_p
    lgeos.GEOSPolygonize.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    lgeos.GEOSLineMerge.restype = ctypes.c_void_p
    lgeos.GEOSLineMerge.argtypes = [ctypes.c_void_p]

    lgeos.GEOSRelatePattern.restype = ctypes.c_char
    lgeos.GEOSRelatePattern.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]

    lgeos.GEOSDisjoint.restype = ctypes.c_byte
    lgeos.GEOSDisjoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSTouches.restype = ctypes.c_byte
    lgeos.GEOSTouches.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSIntersects.restype = ctypes.c_byte
    lgeos.GEOSIntersects.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSCrosses.restype = ctypes.c_byte
    lgeos.GEOSCrosses.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSWithin.restype = ctypes.c_byte
    lgeos.GEOSWithin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSContains.restype = ctypes.c_byte
    lgeos.GEOSContains.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSOverlaps.restype = ctypes.c_byte
    lgeos.GEOSOverlaps.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSEquals.restype = ctypes.c_byte
    lgeos.GEOSEquals.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSEqualsExact.restype = ctypes.c_byte
    lgeos.GEOSEqualsExact.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]

    lgeos.GEOSisEmpty.restype = ctypes.c_byte
    lgeos.GEOSisEmpty.argtypes = [ctypes.c_void_p]

    lgeos.GEOSisValid.restype = ctypes.c_byte
    lgeos.GEOSisValid.argtypes = [ctypes.c_void_p]

    lgeos.GEOSisSimple.restype = ctypes.c_byte
    lgeos.GEOSisSimple.argtypes = [ctypes.c_void_p]

    lgeos.GEOSisRing.restype = ctypes.c_byte
    lgeos.GEOSisRing.argtypes = [ctypes.c_void_p]

    lgeos.GEOSHasZ.restype = ctypes.c_byte
    lgeos.GEOSHasZ.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeomType.restype = ctypes.c_char_p
    lgeos.GEOSGeomType.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeomTypeId.restype = ctypes.c_int
    lgeos.GEOSGeomTypeId.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGetSRID.restype = ctypes.c_int
    lgeos.GEOSGetSRID.argtypes = [ctypes.c_void_p]

    lgeos.GEOSSetSRID.restype = None
    lgeos.GEOSSetSRID.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lgeos.GEOSGetNumGeometries.restype = ctypes.c_int
    lgeos.GEOSGetNumGeometries.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGetGeometryN.restype = ctypes.c_void_p
    lgeos.GEOSGetGeometryN.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lgeos.GEOSGetNumInteriorRings.restype = ctypes.c_int
    lgeos.GEOSGetNumInteriorRings.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGetInteriorRingN.restype = ctypes.c_void_p
    lgeos.GEOSGetInteriorRingN.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lgeos.GEOSGetExteriorRing.restype = ctypes.c_void_p
    lgeos.GEOSGetExteriorRing.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGetNumCoordinates.restype = ctypes.c_int
    lgeos.GEOSGetNumCoordinates.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_getCoordSeq.restype = ctypes.c_void_p
    lgeos.GEOSGeom_getCoordSeq.argtypes = [ctypes.c_void_p]

    lgeos.GEOSGeom_getDimensions.restype = ctypes.c_int
    lgeos.GEOSGeom_getDimensions.argtypes = [ctypes.c_void_p]

    lgeos.GEOSArea.restype = ctypes.c_double
    lgeos.GEOSArea.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSLength.restype = ctypes.c_int
    lgeos.GEOSLength.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lgeos.GEOSDistance.restype = ctypes.c_int
    lgeos.GEOSDistance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    if geosVersion >= (1, 5, 0):

        if hasattr(lgeos, 'GEOSFree'):
            lgeos.GEOSFree.restype = None
            lgeos.GEOSFree.argtypes = [ctypes.c_void_p]

        # Prepared geometry, GEOS C API 1.5.0+
        lgeos.GEOSPrepare.restype = ctypes.c_void_p
        lgeos.GEOSPrepare.argtypes = [ctypes.c_void_p]

        lgeos.GEOSPreparedGeom_destroy.restype = None
        lgeos.GEOSPreparedGeom_destroy.argtypes = [ctypes.c_void_p]

        lgeos.GEOSPreparedIntersects.restype = ctypes.c_int
        lgeos.GEOSPreparedIntersects.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        lgeos.GEOSPreparedContains.restype = ctypes.c_int
        lgeos.GEOSPreparedContains.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        lgeos.GEOSPreparedContainsProperly.restype = ctypes.c_int
        lgeos.GEOSPreparedContainsProperly.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        lgeos.GEOSPreparedCovers.restype = ctypes.c_int
        lgeos.GEOSPreparedCovers.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        lgeos.GEOSisValidReason.restype = allocated_c_char_p
        lgeos.GEOSisValidReason.argtypes = [ctypes.c_void_p]

    # Other, GEOS C API 1.5.0+
    if geosVersion >= (1, 5, 0):
        lgeos.GEOSUnionCascaded.restype = ctypes.c_void_p
        lgeos.GEOSUnionCascaded.argtypes = [ctypes.c_void_p]

    # 1.6.0
    if geosVersion >= (1, 6, 0):
        # Linear referencing features aren't found in versions 1.5,
        # but not in all libs versioned 1.6.0 either!
        if hasattr(lgeos, 'GEOSProject'):
            lgeos.GEOSSingleSidedBuffer.restype = ctypes.c_void_p
            lgeos.GEOSSingleSidedBuffer.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]

            lgeos.GEOSProject.restype = ctypes.c_double
            lgeos.GEOSProject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            lgeos.GEOSProjectNormalized.restype = ctypes.c_double
            lgeos.GEOSProjectNormalized.argtypes = [ctypes.c_void_p, 
                                                    ctypes.c_void_p]

            lgeos.GEOSInterpolate.restype = ctypes.c_void_p
            lgeos.GEOSInterpolate.argtypes = [ctypes.c_void_p, 
                                              ctypes.c_double]

            lgeos.GEOSInterpolateNormalized.restype = ctypes.c_void_p
            lgeos.GEOSInterpolateNormalized.argtypes = [ctypes.c_void_p, 
                                                        ctypes.c_double]

