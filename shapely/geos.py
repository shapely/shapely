"""
Exports the libgeos_c shared lib, GEOS-specific exceptions, and utilities.
"""

import atexit
import os
import sys
from threading import local
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
else:
    # Try the major versioned name first, falling back on the unversioned name.
    try:
        _lgeos = CDLL('libgeos_c.so.1')
    except (OSError, ImportError):
        _lgeos = CDLL('libgeos_c.so')
    except:
        raise
    free = CDLL('libc.so.6').free

def _geos_c_version():
    func = _lgeos.GEOSversion
    func.restype = c_char_p
    v = func().split('-')[2]
    return tuple(int(n) for n in v.split('.'))

geos_c_version = _geos_c_version()

# Prototype the libgeos_c functions using new code from `tarley` in
# http://trac.gispython.org/lab/ticket/189
prototype(_lgeos, geos_c_version)

class allocated_c_char_p(c_char_p):
    pass

# Exceptions

class ReadingError(Exception):
    pass

class DimensionError(Exception):
    pass

class TopologicalError(Exception):
    pass

class PredicateError(Exception):
    pass

thread_data = local()
thread_data.geos_handle = None

def error_handler(fmt, list):
    pass
error_h = CFUNCTYPE(None, c_char_p, c_char_p)(error_handler)

def notice_handler(fmt, list):
    pass
notice_h = CFUNCTYPE(None, c_char_p, c_char_p)(notice_handler)

if geos_c_version >= (1,5,0):
    def cleanup():
        if _lgeos is not None:
            _lgeos.finishGEOS_r(thread_data.geos_handle)
else:
    def cleanup():
        if _lgeos is not None:
            _lgeos.finishGEOS()
atexit.register(cleanup)


class LGEOS(object):
    
    def __init__(self, dll):
        self._lgeos = dll
        self._geos_c_version = geos_c_version
        # GEOS initialization
        if self._geos_c_version >= (1,5,0):
            self._lgeos.initGEOS_r.restype = c_void_p
            self._lgeos.initGEOS_r.argtypes = [c_void_p, c_void_p]
            thread_data.geos_handle = self._lgeos.initGEOS_r(notice_h, error_h)
        else:
            self._lgeos.initGEOS(notice_h, error_h)
    
    def __getattr__(self, name):
        func = getattr(self._lgeos, name)
        # Use GEOS reentrant API if possible
        if self._geos_c_version >= (1,5,0):
            ob = getattr(self._lgeos, name + '_r')
            class wrapper(object):
                __name__ = None
                errcheck = None
                restype = None
                def __init__(self):
                    self.func = ob
                    self.__name__ = ob.__name__
                    self.func.restype = func.restype
                    if func.argtypes is None: self.func.argtypes = None
                    else: self.func.argtypes = [c_void_p] + func.argtypes
                def __call__(self, *args):
                    if self.errcheck is not None:
                        self.func.errcheck = self.errcheck
                    if self.restype is not None:
                        self.func.restype = self.restype
                    return self.func(thread_data.geos_handle, *args)
            attr = wrapper()
        else:
            attr = func
        return attr

lgeos = LGEOS(_lgeos)
