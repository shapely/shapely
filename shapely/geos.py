"""
Proxies for the libgeos_c shared lib, GEOS-specific exceptions, and utilities
"""

import atexit
import logging
import os
import sys
import threading
import ctypes
from ctypes import cdll, CDLL, CFUNCTYPE, c_char_p, c_void_p, string_at
from ctypes.util import find_library

import ftools
from ctypes_declarations import prototype, EXCEPTION_HANDLER_FUNCTYPE



# Begin by creating a do-nothing handler and adding to this module's logger.
class NullHandler(logging.Handler):
    def emit(self, record):
        pass
LOG = logging.getLogger(__name__)
LOG.addHandler(NullHandler())

# Find and load the GEOS and C libraries
# If this ever gets any longer, we'll break it into separate modules

def load_dll(libname, fallbacks=None):
    lib = find_library(libname)
    if lib is not None:
        return CDLL(lib)
    else:
        if fallbacks is not None:
            for name in fallbacks:
                try:
                    return CDLL(name)
                except OSError:
                    # move on to the next fallback
                    pass
        # the end
        raise OSError(
            "Could not find library %s or load any of its variants %s" % (
                libname, fallbacks or []))
       
if sys.platform.startswith('linux'):
    _lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = load_dll('c').free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == 'darwin':
    alt_paths = [
            # The Framework build from Kyng Chaos:
            "/Library/Frameworks/GEOS.framework/Versions/Current/GEOS",
            # macports
            '/opt/local/lib/libgeos_c.dylib',
    ]
    _lgeos = load_dll('geos_c', fallbacks=alt_paths)
    free = load_dll('c').free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == 'win32':
    try:
        egg_dlls = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     r"..\DLLs"))
        wininst_dlls =  os.path.abspath(os.__file__ + "../../../DLLs")
        original_path = os.environ['PATH']
        os.environ['PATH'] = "%s;%s;%s" % (egg_dlls, wininst_dlls, original_path)
        _lgeos = CDLL("geos.dll")
    except (ImportError, WindowsError, OSError):
        raise
    def free(m):
        try:
            cdll.msvcrt.free(m)
        except WindowsError:
            # XXX: See http://trac.gispython.org/projects/PCL/ticket/149
            pass

elif sys.platform == 'sunos5':
    _lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = CDLL('libc.so.1').free
    free.argtypes = [c_void_p]
    free.restype = None

else: # other *nix systems
    _lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = load_dll('c', fallbacks=['libc.so.6']).free
    free.argtypes = [c_void_p]
    free.restype = None

def _geos_c_version():
    func = _lgeos.GEOSversion
    func.argtypes = []
    func.restype = c_char_p
    v = func().split('-')[2]
    return tuple(int(n) for n in v.split('.'))

geos_capi_version = geos_c_version = _geos_c_version()

# If we have the new interface, then record a baseline so that we know what
# additional functions are declared in ctypes_declarations.
if geos_c_version >= (1,5,0):
    start_set = set(_lgeos.__dict__)

# Apply prototypes for the libgeos_c functions
prototype(_lgeos, geos_c_version)

# If we have the new interface, automatically detect all function
# declarations, and declare their re-entrant counterpart.
if geos_c_version >= (1,5,0):
    end_set = set(_lgeos.__dict__)
    new_func_names = end_set - start_set
    
    for func_name in new_func_names:
        new_func_name = "%s_r" % func_name
        if hasattr(_lgeos, new_func_name):
            new_func = getattr(_lgeos, new_func_name)
            old_func = getattr(_lgeos, func_name)
            new_func.restype = old_func.restype
            if old_func.argtypes is None:
                # Handle functions that didn't take an argument before,
                # finishGEOS.
                new_func.argtypes = [c_void_p]
            else:
                new_func.argtypes = [c_void_p] + old_func.argtypes
            if old_func.errcheck is not None:
                new_func.errcheck = old_func.errcheck
    
    # Handle special case.
    _lgeos.initGEOS_r.restype = c_void_p
    _lgeos.initGEOS_r.argtypes = [EXCEPTION_HANDLER_FUNCTYPE, EXCEPTION_HANDLER_FUNCTYPE]
    _lgeos.finishGEOS_r.argtypes = [c_void_p]

# Exceptions

class ReadingError(Exception):
    pass

class DimensionError(Exception):
    pass

class TopologicalError(Exception):
    pass

class PredicateError(Exception):
    pass

def error_handler(fmt, list):
    LOG.error("%s", list)
error_h = EXCEPTION_HANDLER_FUNCTYPE(error_handler)

def notice_handler(fmt, list):
    LOG.warning("%s", list)
notice_h = EXCEPTION_HANDLER_FUNCTYPE(notice_handler)

def cleanup():
    if _lgeos is not None :
        _lgeos.finishGEOS()

atexit.register(cleanup)

# Errcheck functions

def errcheck_wkb(result, func, argtuple):
    if not result:
        return None
    size_ref = argtuple[-1]
    size = size_ref.contents
    retval = ctypes.string_at(result, size.value)[:]
    lgeos.GEOSFree(result)
    return retval

def errcheck_just_free(result, func, argtuple):
    retval = string_at(result)
    lgeos.GEOSFree(result)
    return retval

def errcheck_predicate(result, func, argtuple):
    if result == 2:
        raise PredicateError("Failed to evaluate %s" % repr(func))
    return result


class LGEOSBase(threading.local):
    """Proxy for the GEOS_C DLL/SO

    This is a base class. Do not instantiate.
    """
    methods = {}
    def __init__(self, dll):
        self._lgeos = dll
        self.geos_handle = None
        

class LGEOS14(LGEOSBase):    
    """Proxy for the GEOS_C DLL/SO API version 1.4
    """
    geos_capi_version = (1, 4, 0)
    def __init__(self, dll):
        super(LGEOS14, self).__init__(dll)
        self.geos_handle = self._lgeos.initGEOS(notice_h, error_h)
        keys = self._lgeos.__dict__.keys()
        for key in keys:
            setattr(self, key, getattr(self._lgeos, key))
        self.GEOSFree = self._lgeos.free
        self.GEOSGeomToWKB_buf.errcheck = errcheck_wkb
        self.GEOSGeomToWKT.errcheck = errcheck_just_free
        self.GEOSRelate.errcheck = errcheck_just_free
        for pred in ( self.GEOSDisjoint,
              self.GEOSTouches,
              self.GEOSIntersects,
              self.GEOSCrosses,
              self.GEOSWithin,
              self.GEOSContains,
              self.GEOSOverlaps,
              self.GEOSEquals,
              self.GEOSEqualsExact,
              self.GEOSisEmpty,
              self.GEOSisValid,
              self.GEOSisSimple,
              self.GEOSisRing,
              self.GEOSHasZ
              ):
            pred.errcheck = errcheck_predicate

        self.methods['area'] = self.GEOSArea
        self.methods['boundary'] = self.GEOSBoundary
        self.methods['buffer'] = self.GEOSBuffer
        self.methods['centroid'] = self.GEOSGetCentroid
        self.methods['representative_point'] = self.GEOSPointOnSurface
        self.methods['convex_hull'] = self.GEOSConvexHull
        self.methods['distance'] = self.GEOSDistance
        self.methods['envelope'] = self.GEOSEnvelope
        self.methods['length'] = self.GEOSLength
        self.methods['has_z'] = self.GEOSHasZ
        self.methods['is_empty'] = self.GEOSisEmpty
        self.methods['is_ring'] = self.GEOSisRing
        self.methods['is_simple'] = self.GEOSisSimple
        self.methods['is_valid'] = self.GEOSisValid
        self.methods['disjoint'] = self.GEOSDisjoint
        self.methods['touches'] = self.GEOSTouches
        self.methods['intersects'] = self.GEOSIntersects
        self.methods['crosses'] = self.GEOSCrosses
        self.methods['within'] = self.GEOSWithin
        self.methods['contains'] = self.GEOSContains
        self.methods['overlaps'] = self.GEOSOverlaps
        self.methods['equals'] = self.GEOSEquals
        self.methods['equals_exact'] = self.GEOSEqualsExact
        self.methods['relate'] = self.GEOSRelate
        self.methods['difference'] = self.GEOSDifference
        self.methods['symmetric_difference'] = self.GEOSSymDifference
        self.methods['union'] = self.GEOSUnion
        self.methods['intersection'] = self.GEOSIntersection
        self.methods['simplify'] = self.GEOSSimplify
        self.methods['topology_preserve_simplify'] = \
            self.GEOSTopologyPreserveSimplify


class LGEOS15(LGEOSBase):    
    """Proxy for the reentrant GEOS_C DLL/SO API version 1.5
    """
    geos_capi_version = (1, 5, 0)    
    def __init__(self, dll):
        super(LGEOS15, self).__init__(dll)
        self.geos_handle = self._lgeos.initGEOS_r(notice_h, error_h)
        keys = self._lgeos.__dict__.keys()
        for key in filter(lambda x: not x.endswith('_r'), keys):
            if key + '_r' in keys:
                reentr_func = getattr(self._lgeos, key + '_r')
                attr = ftools.partial(reentr_func, self.geos_handle)
                attr.__name__ = reentr_func.__name__
                setattr(self, key, attr)
            else:
                setattr(self, key, getattr(self._lgeos, key))
        if not hasattr(self, 'GEOSFree'):
            self.GEOSFree = self._lgeos.free
        self.GEOSGeomToWKB_buf.func.errcheck = errcheck_wkb
        self.GEOSGeomToWKT.func.errcheck = errcheck_just_free
        self.GEOSRelate.func.errcheck = errcheck_just_free
        for pred in ( self.GEOSDisjoint,
              self.GEOSTouches,
              self.GEOSIntersects,
              self.GEOSCrosses,
              self.GEOSWithin,
              self.GEOSContains,
              self.GEOSOverlaps,
              self.GEOSEquals,
              self.GEOSEqualsExact,
              self.GEOSisEmpty,
              self.GEOSisValid,
              self.GEOSisSimple,
              self.GEOSisRing,
              self.GEOSHasZ
              ):
            pred.func.errcheck = errcheck_predicate

        self.GEOSisValidReason.func.errcheck = errcheck_just_free
        
        self.methods['area'] = self.GEOSArea
        self.methods['boundary'] = self.GEOSBoundary
        self.methods['buffer'] = self.GEOSBuffer
        self.methods['centroid'] = self.GEOSGetCentroid
        self.methods['representative_point'] = self.GEOSPointOnSurface
        self.methods['convex_hull'] = self.GEOSConvexHull
        self.methods['distance'] = self.GEOSDistance
        self.methods['envelope'] = self.GEOSEnvelope
        self.methods['length'] = self.GEOSLength
        self.methods['has_z'] = self.GEOSHasZ
        self.methods['is_empty'] = self.GEOSisEmpty
        self.methods['is_ring'] = self.GEOSisRing
        self.methods['is_simple'] = self.GEOSisSimple
        self.methods['is_valid'] = self.GEOSisValid
        self.methods['disjoint'] = self.GEOSDisjoint
        self.methods['touches'] = self.GEOSTouches
        self.methods['intersects'] = self.GEOSIntersects
        self.methods['crosses'] = self.GEOSCrosses
        self.methods['within'] = self.GEOSWithin
        self.methods['contains'] = self.GEOSContains
        self.methods['overlaps'] = self.GEOSOverlaps
        self.methods['equals'] = self.GEOSEquals
        self.methods['equals_exact'] = self.GEOSEqualsExact
        self.methods['relate'] = self.GEOSRelate
        self.methods['difference'] = self.GEOSDifference
        self.methods['symmetric_difference'] = self.GEOSSymDifference
        self.methods['union'] = self.GEOSUnion
        self.methods['intersection'] = self.GEOSIntersection
        self.methods['prepared_intersects'] = self.GEOSPreparedIntersects
        self.methods['prepared_contains'] = self.GEOSPreparedContains
        self.methods['prepared_contains_properly'] = \
            self.GEOSPreparedContainsProperly
        self.methods['prepared_covers'] = self.GEOSPreparedCovers
        self.methods['simplify'] = self.GEOSSimplify
        self.methods['topology_preserve_simplify'] = \
            self.GEOSTopologyPreserveSimplify

class LGEOS16(LGEOS15):
    """Proxy for the reentrant GEOS_C DLL/SO API version 1.6
    """
    geos_capi_version = (1, 6, 0)
    def __init__(self, dll):
        super(LGEOS16, self).__init__(dll)


class LGEOS16LR(LGEOS16):    
    """Proxy for the reentrant GEOS_C DLL/SO API version 1.6 with linear
    referencing
    """
    geos_capi_version = geos_c_version
    def __init__(self, dll):
        super(LGEOS16LR, self).__init__(dll)

        self.methods['parallel_offset'] = self.GEOSSingleSidedBuffer
        self.methods['project'] = self.GEOSProject
        self.methods['project_normalized'] = self.GEOSProjectNormalized
        self.methods['interpolate'] = self.GEOSInterpolate
        self.methods['interpolate_normalized'] = \
            self.GEOSInterpolateNormalized


if geos_c_version >= (1, 6, 0):
    if hasattr(_lgeos, 'GEOSProject'):
        L = LGEOS16LR
    else:
        L = LGEOS16
elif geos_c_version >= (1, 5, 0):
        L = LGEOS15
else:
        L = LGEOS14

lgeos = L(_lgeos)

