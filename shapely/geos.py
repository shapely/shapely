
import atexit
from ctypes import CDLL, CFUNCTYPE, c_char_p
    #string_at, create_string_buffer, \
    #c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte, \
    #c_void_p, byref
import os, sys

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

class OperationError(Exception):
    pass

class PredicateError(Exception):
    pass


# Predicates

# A callable non-data descriptor
class BinaryPredicate(object):

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        self.source = obj
        return self

    def __call__(self, target):
        retval = self.fn(self.source._geom, target._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate equals()"
        return bool(retval)


# A data descriptor
class UnaryPredicate(object):

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        retval = self.fn(obj._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate equals()"
        return bool(retval)
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"


# Do-nothing handlers
def error_handler(fmt, list):
    pass

error_h = CFUNCTYPE(None, c_char_p, c_char_p)(error_handler)

def notice_handler(fmt, list):
    pass

notice_h = CFUNCTYPE(None, c_char_p, c_char_p)(notice_handler)

# Init geos, and register a cleanup function
lgeos.initGEOS(notice_h, error_h)
atexit.register(lgeos.finishGEOS)


