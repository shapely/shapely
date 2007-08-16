"""
Multiple linestring.
"""

from ctypes import byref, c_double, c_int, c_void_p, cast, POINTER, pointer

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, GeometrySequence
from shapely.geometry.linestring import LineString


class MultiLineString(BaseGeometry):

    """a multiple linestring geometry.
    """

    def __init__(self, coordinates=None):
        """Initialize.

        Parameters
        ----------
        
        coordinates : sequence or array
            This may be an object that satisfies the numpy array protocol,
            providing an M x 2 or M x 3 (with z) array, or it may be a sequence
            of x, y (,z) coordinate sequences.

        Example
        -------

        >>> geom = MultiLineString([[[0.0, 0.0], [1.0, 2.0]]])
        >>> geom = LineString(array([[[0.0, 0.0], [1.0, 2.0]]]))
        
        Each result in a collection containing one line string.
        """
        BaseGeometry.__init__(self)

        if coordinates is None:
            # allow creation of null lines, to support unpickling
            pass
        else:
            self._geom, self._ndim = self._geos_from_py(coordinates)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

