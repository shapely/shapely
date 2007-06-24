"""
"""

from ctypes import string_at, create_string_buffer, \
    c_char_p, c_double, c_float, c_int, c_uint, c_size_t, c_ubyte, \
    c_void_p, byref

from shapely.geos import lgeos, DimensionError
from shapely.geometry.base import BaseGeometry

class Polygon(BaseGeometry):

    """A polygon geometry.
    
    """

    _ctypes_data = None

    def __init__(self):
        """Initialize.
        """
        BaseGeometry.__init__(self)

    @property
    def array(self):
        """Return a GeoJSON coordinate array."""
        array = [self.x, self.y]
        if self._ndim == 3: # TODO: use hasz
            array.append(self.z)
        return array

    @property
    def ctypes(self):
        if not self._ctypes_data:
            if self._ndim == 3: # TODO: use hasz
                array = c_double * 3
                self._ctypes_data = array(self.x, self.y, self.z)
            else:
                array = c_double * 2
                self._ctypes_data = array(self.x, self.y, self.z)
        return self._ctypes_data

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        return {
            'version': 3,
            'shape': (self._ndim,),
            'typestr': '>f8',
            'data': self.ctypes
            }

    # Python feature protocol
    @property
    def type(self):
        return self.geometryType()

    @property
    def coordinates(self):
        return self.array

# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

