# geos_linestring_from_py was transcribed from shapely.geometry.linestring
# geos_linearring_from_py was transcribed from shapely.geometry.polygon
# coordseq_ctypes was transcribed from shapely.coords.CoordinateSequence.ctypes
#
# Copyright (c) 2007, Sean C. Gillies
# Transcription to cython: Copyright (c) 2011, Oliver Tonnhofer

import ctypes
from shapely.geos import lgeos
from shapely.geometry import Point, LineString, LinearRing

include "../_geos.pxi"
    

cdef inline GEOSGeometry *cast_geom(unsigned long geom_addr):
    return <GEOSGeometry *>geom_addr

cdef inline GEOSContextHandle_t cast_handle(unsigned long handle_addr):
    return <GEOSContextHandle_t>handle_addr

cdef inline GEOSCoordSequence *cast_seq(unsigned long handle_addr):
    return <GEOSCoordSequence *>handle_addr

def destroy(geom):
    GEOSGeom_destroy_r(cast_handle(lgeos.geos_handle), cast_geom(geom))

def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    cdef double *cp
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef GEOSGeometry *g
    cdef double dx, dy, dz
    cdef int i, n, m, sm, sn

    # If a LineString is passed in, just clone it and return
    # If a LinearRing is passed in, clone the coord seq and return a LineString
    if isinstance(ob, LineString):
        g = cast_geom(ob._geom)
        if GEOSHasZ_r(handle, g):
            n = 3
        else:
            n = 2

        if type(ob) == LineString:
            return <unsigned long>GEOSGeom_clone_r(handle, g), n
        else:
            cs = GEOSGeom_getCoordSeq_r(handle, g)
            cs = GEOSCoordSeq_clone_r(handle, cs)
            return <unsigned long>GEOSGeom_createLineString_r(handle, cs), n

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

        # Use strides to properly index into cp
        # ob[i, j] == cp[sm*i + sn*j]
        # Just to avoid a referenced before assignment warning.
        dx = 0
        if array.get('strides', None):
            sm = array['strides'][0]/sizeof(dx)
            sn = array['strides'][1]/sizeof(dx)
        else:
            sm = n
            sn = 1

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
            dx = cp[sm*i]
            dy = cp[sm*i+sn]
            dz = 0
            if n == 3:
                dz = cp[sm*i+2*sn]
                
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

    except AttributeError:
        # Fall back on list
        try:
            m = len(ob)
        except TypeError:  # Iterators, e.g. Python 3 zip
            ob = list(ob)
            m = len(ob)
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")

        def _coords(o):
            if isinstance(o, Point):
                return o.coords[0]
            else:
                return o

        try:
            n = len(_coords(ob[0]))
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
            coords = _coords(ob[i])
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
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSGeometry *g
    cdef GEOSCoordSequence *cs
    cdef double dx, dy, dz
    cdef int i, n, m, M, sm, sn

    # If a LinearRing is passed in, just clone it and return
    # If a LineString is passed in, clone the coord seq and return a LinearRing
    if isinstance(ob, LineString):
        g = cast_geom(ob._geom)
        if GEOSHasZ_r(handle, g):
            n = 3
        else:
            n = 2

        if type(ob) == LinearRing:
            return <unsigned long>GEOSGeom_clone_r(handle, g), n
        else:
            cs = GEOSGeom_getCoordSeq_r(handle, g)
            GEOSCoordSeq_getSize_r(handle, cs, &m)
            if GEOSisClosed_r(handle, g) and m >= 4:
                cs = GEOSCoordSeq_clone_r(handle, cs)
                return <unsigned long>GEOSGeom_createLinearRing_r(handle, cs), n

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

        # Use strides to properly index into cp
        # ob[i, j] == cp[sm*i + sn*j]
        dx = 0  # Just to avoid a referenced before assignment warning.
        if array.get('strides', None):
            sm = array['strides'][0]/sizeof(dx)
            sn = array['strides'][1]/sizeof(dx)
        else:
            sm = n
            sn = 1

        # Add closing coordinates to sequence?
        # Check whether the first set of coordinates matches the last.
        # If not, we'll have to close the ring later
        if (cp[0] != cp[sm*(m-1)] or cp[sn] != cp[sm*(m-1)+sn] or
            (n == 3 and cp[2*sn] != cp[sm*(m-1)+2*sn])):
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
            dx = cp[sm*i]
            dy = cp[sm*i+sn]
            dz = 0
            if n == 3:
                dz = cp[sm*i+2*sn]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

        # Add closing coordinates to sequence?
        if M > m:
            dx = cp[0]
            dy = cp[sn]
            dz = 0
            if n == 3:
                dz = cp[2*sn]
        
            # Because of a bug in the GEOS C API, 
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, M-1, dx)
            GEOSCoordSeq_setY_r(handle, cs, M-1, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, M-1, dz)
            
    except AttributeError:
        # Fall back on list
        try:
            m = len(ob)
        except TypeError:  # Iterators, e.g. Python 3 zip
            ob = list(ob)
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
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
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

def coordseq_iter(self):
    cdef int i
    cdef double dx
    cdef double dy
    cdef double dz
    cdef int has_z

    self._update()

    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cs = cast_seq(self._cseq)

    has_z = self._ndim == 3
    for i in range(self.__len__()):
        GEOSCoordSeq_getX_r(handle, cs, i, &dx)
        GEOSCoordSeq_getY_r(handle, cs, i, &dy)
        if has_z == 1:
            GEOSCoordSeq_getZ_r(handle, cs, i, &dz)
            yield (dx, dy, dz)
        else:
            yield (dx, dy)
