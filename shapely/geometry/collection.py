"""Multi-part collections of geometries
"""

from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.base import HeterogeneousGeometrySequence


class GeometryCollection(BaseMultipartGeometry):

    """A heterogenous collection of geometries

    Attributes
    ----------
    geoms : sequence
        A sequence of Shapely geometry instances
    """

    def __init__(self):
        BaseMultipartGeometry.__init__(self)

    @property
    def __geo_interface__(self):
        geometries = []
        for geom in self.geoms:
            geometries.append(geom.__geo_interface__)
        return dict(type='GeometryCollection', geometries=geometries)

    @property
    def geoms(self):
        if self.is_empty:
            return []
        return HeterogeneousGeometrySequence(self)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

