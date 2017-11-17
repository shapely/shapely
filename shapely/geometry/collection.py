"""Multi-part collections of geometries
"""

from ctypes import c_void_p

from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.base import HeterogeneousGeometrySequence
from shapely.geometry.base import geos_geom_from_py


class GeometryCollection(BaseMultipartGeometry):

    """A heterogenous collection of geometries

    Attributes
    ----------
    geoms : sequence
        A sequence of Shapely geometry instances
    """

    def __init__(self, geoms=None, colors=None):
        """
        Parameters
        ----------
        geoms : list
            A list of shapely geometry instances, which may be heterogenous.
        colors : list
            A list of html colors to render the geometries in Jupyter Notebook.

        Example
        -------
        Create a GeometryCollection with a Point and a LineString

          >>> p = Point(51, -1)
          >>> l = LineString([(52, -1), (49, 2)])
          >>> gc = GeometryCollection([p, l])
        """
        BaseMultipartGeometry.__init__(self)
        if not geoms:
            pass
        else:
            self._geom, self._ndim = geos_geometrycollection_from_py(geoms)
        self.colors = colors or []

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

    def svg(self, scale_factor=1., color=None):
        """Returns a group of SVG elements for the multipart geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        color : str, optional
            Hex string for stroke or fill color. Default is to use "#66cc99"
            if geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if not self.colors:
            return super(GeometryCollection, self).svg(scale_factor, color)
        return '<g>' + ''.join(p.svg(scale_factor, c) for p, c in zip(
            self, self.colors)) + '</g>'


def geos_geometrycollection_from_py(ob):
    """Creates a GEOS GeometryCollection from a list of geometries"""
    L = len(ob)
    N = 2
    subs = (c_void_p * L)()
    for l in range(L):
        assert(isinstance(ob[l], BaseGeometry))
        if ob[l].has_z:
            N = 3
        geom, n = geos_geom_from_py(ob[l])
        subs[l] = geom

    return (lgeos.GEOSGeom_createCollection(7, subs, L), N)


# Test runner
def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
