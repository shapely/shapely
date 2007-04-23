
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


