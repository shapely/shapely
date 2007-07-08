"""
Exports the libgeos_c shared lib, GEOS-specific exceptions, and utilities.
"""

import atexit
from ctypes import CDLL, CFUNCTYPE, c_char_p
import os, sys

# The GEOS shared lib

if os.name == 'nt':
    dll = 'libgeos_c-1.dll'
else:
    dll = 'libgeos_c.so'
lgeos = CDLL(dll)


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

# Init geos, and register a cleanup function

lgeos.initGEOS(notice_h, error_h)
atexit.register(lgeos.finishGEOS)


