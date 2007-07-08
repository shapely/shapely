"""
Geometry collections.
"""

from shapely.geometry.base import BaseGeometry

class GeometryCollection(BaseGeometry):

    """A geometry collection.
    """

    def __init__(self):
        BaseGeometry.__init__(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

