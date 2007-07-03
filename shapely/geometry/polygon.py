"""
"""

#from ctypes import string_at, create_string_buffer, \
#    c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte, \
#    c_void_p, byref

#from shapely.geos import lgeos, DimensionError
from shapely.geometry.base import BaseGeometry

class Polygon(BaseGeometry):

    """A polygon geometry.
    
    """

    _ctypes_data = None

    def __init__(self):
        """Initialize.
        """
        super(Polygon).__init__(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

