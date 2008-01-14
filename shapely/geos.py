"""
Exports the libgeos_c shared lib, GEOS-specific exceptions, and utilities.
"""

import atexit
from ctypes import cdll, CDLL, CFUNCTYPE, c_char_p
from ctypes.util import find_library
import os
import sys

import shapely

if sys.platform == 'win32':
    try:
        geospath = os.path.abspath(
            shapely.__file__ + "../../../../../DLLs/geos_c.dll"
            )
        lgeos = CDLL(geospath)
    except (ImportError, WindowsError):
        raise
    free = cdll.msvcrt.free
elif sys.platform == 'darwin':
    lgeos = CDLL(find_library('geos_c'))
    free = CDLL(find_library('libc')).free
else:
    # Try the major versioned name first, falling back on the unversioned name.
    try:
        lgeos = CDLL('libgeos_c.so.1')
    except ImportError:
        lgeos = CDLL('libgeos_c.so')
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


