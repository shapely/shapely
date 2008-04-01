"""
Exports the libgeos_c shared lib, GEOS-specific exceptions, and utilities.
"""

import atexit
from ctypes import cdll, CDLL, pydll, PyDLL, CFUNCTYPE, c_char_p
from ctypes.util import find_library
import os
import sys

import shapely

if sys.platform == 'win32':
    try:
        local_dlls = os.path.abspath(os.__file__ + "../../../DLLs")
        original_path = os.environ['PATH']
        os.environ['PATH'] = "%s;%s" % (local_dlls, original_path)
        lgeos = PyDLL("geos.dll")
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
        ]
        for path in lib_paths:
            if os.path.exists(path):
                lib = path
                break
            else:
                raise ImportError, "Could not find geos_c library"
    lgeos = PyDLL(lib)
    free = CDLL(find_library('libc')).free
else:
    # Try the major versioned name first, falling back on the unversioned name.
    try:
        lgeos = PyDLL('libgeos_c.so.1')
    except (OSError, ImportError):
        lgeos = PyDLL('libgeos_c.so')
    except:
        raise
    free = CDLL('libc.so.6').free


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


# GEOS error handlers, which currently do nothing.

def error_handler(fmt, list):
    pass
error_h = CFUNCTYPE(None, c_char_p, c_char_p)(error_handler)

def notice_handler(fmt, list):
    pass
notice_h = CFUNCTYPE(None, c_char_p, c_char_p)(notice_handler)

# Register a cleanup function

def cleanup():
    lgeos.finishGEOS()

atexit.register(cleanup)

lgeos.initGEOS(notice_h, error_h)


