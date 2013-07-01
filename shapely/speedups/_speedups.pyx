# geos_linestring_from_py was transcribed from shapely.geometry.linestring
# geos_linearring_from_py was transcribed from shapely.geometry.polygon
# coordseq_ctypes was transcribed from shapely.coords.CoordinateSequence.ctypes
#
# Copyright (c) 2007, Sean C. Gillies
# Transcription to cython: Copyright (c) 2011, Oliver Tonnhofer

import ctypes
from shapely.geos import lgeos

cdef extern from "geos_c.h":
    ctypedef struct GEOSCoordSequence
    ctypedef struct GEOSGeometry
    cdef struct GEOSContextHandle_HS
    GEOSCoordSequence *GEOSCoordSeq_create_r(GEOSContextHandle_HS *,double, double)
    GEOSCoordSequence *GEOSGeom_getCoordSeq_r(GEOSContextHandle_HS *, GEOSGeometry *)
    int GEOSCoordSeq_getSize_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int *)
    int GEOSCoordSeq_setX_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_setY_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double)
    int GEOSCoordSeq_getX_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double *)
    int GEOSCoordSeq_getY_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double *)
    int GEOSCoordSeq_getZ_r(GEOSContextHandle_HS *, GEOSCoordSequence *, int, double *)
    GEOSGeometry *GEOSGeom_createLineString_r(GEOSContextHandle_HS *, GEOSCoordSequence *)
    GEOSGeometry *GEOSGeom_createLinearRing_r(GEOSContextHandle_HS *, GEOSCoordSequence *)
    void GEOSGeom_destroy_r(GEOSContextHandle_HS *, GEOSGeometry *)

cdef inline GEOSGeometry *cast_geom(unsigned long geom_addr):
    return <GEOSGeometry *>geom_addr

cdef inline GEOSContextHandle_HS *cast_handle(unsigned long handle_addr):
    return <GEOSContextHandle_HS *>handle_addr

cdef inline GEOSCoordSequence *cast_seq(unsigned long handle_addr):
    return <GEOSCoordSequence *>handle_addr

def destroy(geom):
    GEOSGeom_destroy_r(cast_handle(lgeos.geos_handle), cast_geom(geom))

def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    cdef double *cp
    cdef GEOSContextHandle_HS *handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef double dx, dy, dz
    cdef int i, n, m
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")
        try:
            n = array['shape'][1]
        except IndexError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], ctypes.Array):
            cp = <double *><unsigned long>ctypes.addressof(array['data'])
        else:
            cp = <double *><unsigned long>array['data'][0]

        # Create a coordinate sequence
        if update_geom is not None:
            cs = GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, <int>m, <int>n)

        # add to coordinate sequence
        for i in xrange(m):
            dx = cp[n*i]
            dy = cp[n*i+1]
            dz = 0
            if n == 3:
                dz = cp[n*i+2]
                
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)
    
    except AttributeError:
        # Fall back on list
        m = len(ob)
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")
        try:
            n = len(ob[0])
        except TypeError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Create a coordinate sequence
        if update_geom is not None:
            cs = GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, <int>m, <int>n)

        # add to coordinate sequence
        for i in xrange(m):
            coords = ob[i]
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                if len(coords) != 3:
                    raise ValueError("Inconsistent coordinate dimensionality")
                dz = coords[2]
            
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

    if update_geom is not None:
        return None
    else:
        return <unsigned long>GEOSGeom_createLineString_r(handle, cs), n


def geos_linearring_from_py(ob, update_geom=None, update_ndim=0):
    cdef double *cp
    cdef GEOSContextHandle_HS *handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef double dx, dy, dz
    cdef int i, n, m, M
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        n = array['shape'][1]
        if m < 3:
            raise ValueError(
                "A LinearRing must have at least 3 coordinate tuples")
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], ctypes.Array):
            cp = <double *><unsigned long>ctypes.addressof(array['data'])
        else:
            cp = <double *><unsigned long>array['data'][0]

        # Add closing coordinates to sequence?
        if cp[0] != cp[m*n-n] or cp[1] != cp[m*n-n+1]:
            M = m + 1
        else:
            M = m

        # Create a coordinate sequence
        if update_geom is not None:
            cs = GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, M, n)

        # add to coordinate sequence
        for i in xrange(m):
            dx = cp[n*i]
            dy = cp[n*i+1]
            dz = 0
            if n == 3:
                dz = cp[n*i+2]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

        # Add closing coordinates to sequence?
        if M > m:
            dx = cp[0]
            dy = cp[1]
            dz = 0
            if n == 3:
                dz = cp[2]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, M-1, dx)
            GEOSCoordSeq_setY_r(handle, cs, M-1, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, M-1, dz)
            
    except AttributeError:
        # Fall back on list
        m = len(ob)
        n = len(ob[0])
        if m < 3:
            raise ValueError(
                "A LinearRing must have at least 3 coordinate tuples")
        assert (n == 2 or n == 3)

        # Add closing coordinates if not provided
        if m == 3 or ob[0][0] != ob[-1][0] or ob[0][1] != ob[-1][1]:
            M = m + 1
        else:
            M = m

        # Create a coordinate sequence
        if update_geom is not None:
            cs = GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, M, n)
        
        # add to coordinate sequence
        for i in xrange(m):
            coords = ob[i]
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                dz = coords[2]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

        # Add closing coordinates to sequence?
        if M > m:
            coords = ob[0]
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                dz = coords[2]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, M-1, dx)
            GEOSCoordSeq_setY_r(handle, cs, M-1, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, M-1, dz)

    if update_geom is not None:
        return None
    else:
        return <unsigned long>GEOSGeom_createLinearRing_r(handle, cs), n


def coordseq_ctypes(self):
    cdef int i, n, m
    cdef double temp = 0
    cdef GEOSContextHandle_HS *handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef double *data_p
    self._update()
    n = self._ndim
    m = self.__len__()
    array_type = ctypes.c_double * (m * n)
    data = array_type()
    
    cs = cast_seq(self._cseq)
    data_p = <double *><unsigned long>ctypes.addressof(data)
    
    for i in xrange(m):
        GEOSCoordSeq_getX_r(handle, cs, i, &temp)
        data_p[n*i] = temp
        GEOSCoordSeq_getY_r(handle, cs, i, &temp)
        data_p[n*i+1] = temp
        if n == 3: # TODO: use hasz
            GEOSCoordSeq_getZ_r(handle, cs, i, &temp)
            data_p[n*i+2] = temp
    return data

