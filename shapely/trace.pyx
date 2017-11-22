# (C) British Crown Copyright 2011 - 2017, Met Office
#
# This file was part of cartopy.  It is now part of shapely
#
# Shapely is licensed under the BSD 3-clause "New" or "Revised" License

"""
This module pulls together _trace.cpp, proj.4, and GEOS.
"""

from libc.stdint cimport uintptr_t as ptr


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry:
        pass


import shapely.geometry as sgeom
from shapely.geos import lgeos


cdef extern from "proj_api.h":
    ctypedef void *projPJ


cdef extern from "_trace.h":
    cdef cppclass Interpolator:
        pass

    cdef cppclass SphericalInterpolator:
        SphericalInterpolator(projPJ src_proj, projPJ dest_proj)

    cdef cppclass CartesianInterpolator:
        CartesianInterpolator(projPJ src_proj, projPJ dest_proj)

    # XXX Rename? It handles LinearRings too.
    GEOSGeometry *_project_line_string(GEOSContextHandle_t handle,
                                       GEOSGeometry *g_line_string,
                                       Interpolator *interpolator,
                                       GEOSGeometry *g_domain,
                                       double threshold)


cdef GEOSContextHandle_t get_geos_context_handle():
    cdef ptr handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


cdef GEOSGeometry *geos_from_shapely(shapely_geom) except *:
    """Get the GEOS pointer from the given shapely geometry."""
    cdef ptr geos_geom = shapely_geom._geom
    return <GEOSGeometry *>geos_geom


cdef shapely_from_geos(GEOSGeometry *geom):
    """Turn the given GEOS geometry pointer into a shapely geometry."""
    # This is the "correct" way to do it...
    #   return geom_factory(<ptr>geom)
    # ... but it's quite slow, so we do it by hand.
    multi_line_string = sgeom.base.BaseGeometry()
    multi_line_string.__class__ = sgeom.MultiLineString
    multi_line_string.__geom__ = <ptr>geom
    multi_line_string.__parent__ = None
    multi_line_string._ndim = 2
    return multi_line_string
