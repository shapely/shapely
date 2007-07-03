"""
"""

#from shapely.geos import lgeos, DimensionError
from shapely.geometry.base import BaseGeometry

class GeometryCollection(BaseGeometry):

    """A geometry collection.
    
    """

    _ctypes_data = None

    def __init__(self):
        """Initialize.
        """
        super(GeometryCollection).__init__(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

