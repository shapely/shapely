"""
Multiple linestring.
"""

from shapely.geometry.base import BaseGeometry


class MultiLineString(BaseGeometry):

    """a multiple linestring geometry.
    """

    def __init__(self):
        BaseGeometry.__init__(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

