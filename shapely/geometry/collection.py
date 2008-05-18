"""
Geometry collections.
"""

from shapely.geometry.base import BaseMultiPartGeometry
from shapely.geometry.base import HeterogeneousGeometrySequence, exceptNull


class GeometryCollection(BaseMultiPartGeometry):

    """A geometry collection.
    """

    def __init__(self):
        BaseMultiPartGeometry.__init__(self)

    @property
    def __geo_interface__(self):
        geometries = []
        for geom in self.geoms:
            geometries.append(geom.__geo_interface__)
        return dict(type='GeometryCollection', geometries=geometries)

    @property
    @exceptNull
    def geoms(self):
        return HeterogeneousGeometrySequence(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

