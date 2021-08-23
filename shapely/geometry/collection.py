"""Multi-part collections of geometries
"""

from ctypes import c_void_p

from shapely.geometry.base import BaseGeometry
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.base import HeterogeneousGeometrySequence

import pygeos


class GeometryCollection(BaseMultipartGeometry):

    """A heterogenous collection of geometries

    Attributes
    ----------
    geoms : sequence
        A sequence of Shapely geometry instances
    """

    __slots__ = []

    def __new__(self, geoms=None):
        """
        Parameters
        ----------
        geoms : list
            A list of shapely geometry instances, which may be heterogenous.

        Example
        -------
        Create a GeometryCollection with a Point and a LineString

          >>> from shapely.geometry import LineString, Point
          >>> p = Point(51, -1)
          >>> l = LineString([(52, -1), (49, 2)])
          >>> gc = GeometryCollection([p, l])
        """
        if not geoms:
            # TODO better empty constructor
            return pygeos.from_wkt("GEOMETRYCOLLECTION EMPTY")
        if isinstance(geoms, BaseGeometry):
            # TODO(shapely-2.0) do we actually want to split Multi-part geometries?
            # this is needed for the split() tests
            if hasattr(geoms, "geoms"):
                geoms = geoms.geoms
            else:
                geoms = [geoms]

        return pygeos.geometrycollections(geoms)

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


pygeos.lib.registry[7] = GeometryCollection
