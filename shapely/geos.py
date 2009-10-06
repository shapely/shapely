"""
Exports the libgeos_c shared lib, GEOS-specific exceptions, and utilities.
"""

import atexit
import os
import sys
from threading import local
import ctypes
from ctypes import cdll, CDLL, PyDLL, CFUNCTYPE, c_char_p, c_void_p
from ctypes.util import find_library

from ctypes_declarations import prototype


if sys.platform == 'win32':
    try:
        local_dlls = os.path.abspath(os.__file__ + "../../../DLLs")
        original_path = os.environ['PATH']
        os.environ['PATH'] = "%s;%s" % (local_dlls, original_path)
        _lgeos = CDLL("geos.dll")
    except (ImportError, WindowsError):
        raise
    def free(m):
        try:
            cdll.msvcrt.free(m)
        except WindowsError:
            # XXX: See http://trac.gispython.org/projects/PCL/ticket/149
            pass
elif sys.platform == 'darwin':
    lib = find_library('geos_c')
    if lib is None:
        ## try a few more locations
        lib_paths = [
            # The Framework build from Kyng Chaos:
            "/Library/Frameworks/GEOS.framework/Versions/Current/GEOS",
            # macports
            '/opt/local/lib/libgeos_c.dylib',
        ]
        for path in lib_paths:
            if os.path.exists(path):
                lib = path
                break
    if lib is None:
        raise ImportError, "Could not find geos_c library"
    _lgeos = CDLL(lib)
    free = CDLL(find_library('libc')).free
    free.argtypes = [c_void_p]
    free.restype = None
else:
    # Try the major versioned name first, falling back on the unversioned name.
    try:
        _lgeos = CDLL('libgeos_c.so.1')
    except (OSError, ImportError):
        _lgeos = CDLL('libgeos_c.so')
    except:
        raise
    free = CDLL('libc.so.6').free
    free.argtypes = [c_void_p]
    free.restype = None

def _geos_c_version():
    func = _lgeos.GEOSversion
    func.restype = c_char_p
    v = func().split('-')[2]
    return tuple(int(n) for n in v.split('.'))

geos_c_version = _geos_c_version()

class allocated_c_char_p(c_char_p):
    pass

# If we have the new interface, then record a baseline so that we know what
# additional functions are declared in ctypes_declarations.
if geos_c_version >= (1,5,0):
    start_set = set(_lgeos.__dict__)

# Prototype the libgeos_c functions using new code from `tarley` in
# http://trac.gispython.org/lab/ticket/189
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
    _lgeos.initGEOS_r.argtypes = [c_void_p, c_void_p]

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
    print "ERROR: '%s'" % (list)
    pass
error_h = CFUNCTYPE(None, c_char_p, c_char_p)(error_handler)

def notice_handler(fmt, list):
    print "NOTICE: '%s'" % (list)
    pass
notice_h = CFUNCTYPE(None, c_char_p, c_char_p)(notice_handler)

def cleanup():
    if _lgeos is not None:
        _lgeos.finishGEOS()
atexit.register(cleanup)

import time
import functools

class LGEOS(local):
    
    def __init__(self, dll):
        self._lgeos = dll

    def __getattr__(self, name):
        if 'geos_handle' == name:
            self.geos_handle = lgeos._lgeos.initGEOS_r(notice_h, error_h)
            return self.geos_handle
        if name == 'GEOSFree':
            attr = getattr(self._lgeos, name)
            return attr

        old_func = getattr(self._lgeos, name)
        if geos_c_version >= (1,5,0):
            real_func = getattr(self._lgeos, name + '_r')

            #attr = wrapper(old_func, real_func)
            attr = functools.partial(real_func, self.geos_handle)
            attr.__name__ = real_func.__name__
        else:
            attr = old_func

        # Store the function, or function wrapper in a thread specific attribute.
        setattr(self, name, attr)
        return attr
        
lgeos = LGEOS(_lgeos)

func = lgeos.GEOSGeomToWKB_buf
def errcheck_wkb(result, func, argtuple):
   if not result:
      return None
   size_ref = argtuple[2]
   size = size_ref._obj
   retval = ctypes.string_at(result, size.value)[:]
   lgeos.GEOSFree(result)
   return retval
func.func.errcheck = errcheck_wkb

def errcheck_just_free(result, func, argtuple):
   retval = result.value
   lgeos.GEOSFree(result)
   return retval

func = lgeos.GEOSGeomToWKT
func.func.errcheck = errcheck_just_free
func = lgeos.GEOSRelate
func.func.errcheck = errcheck_just_free

def errcheck_predicate(result, func, argtuple):
    if result == 2:
        raise PredicateError, "Failed to evaluate %s" % repr(func)
    return result

for pred in [lgeos.GEOSDisjoint, lgeos.GEOSTouches, lgeos.GEOSIntersects, lgeos.GEOSCrosses,
             lgeos.GEOSWithin, lgeos.GEOSContains, lgeos.GEOSOverlaps, lgeos.GEOSEquals,
             lgeos.GEOSEqualsExact]:
    pred.func.errcheck = errcheck_predicate

for pred in [lgeos.GEOSisEmpty, lgeos.GEOSisValid, lgeos.GEOSisSimple,
             lgeos.GEOSisRing, lgeos.GEOSHasZ]:
    pred.func.errcheck = errcheck_predicate
