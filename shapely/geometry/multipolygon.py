"""Collections of polygons and related utilities
"""

from ctypes import c_void_p, cast
import warnings

from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry import polygon

import pygeos


__all__ = ['MultiPolygon']


class MultiPolygon(BaseMultipartGeometry):

    """A collection of one or more polygons

    If component polygons overlap the collection is `invalid` and some
    operations on it may fail.

    Attributes
    ----------
    geoms : sequence
        A sequence of `Polygon` instances
    """

    __slots__ = []

    def __new__(self, polygons=None):
        """
        Parameters
        ----------
        polygons : sequence
            A sequence of (shell, holes) tuples where shell is the sequence
            representation of a linear ring (see linearring.py) and holes is
            a sequence of such linear rings

        Example
        -------
        Construct a collection from a sequence of coordinate tuples

          >>> from shapely.geometry import Polygon
          >>> ob = MultiPolygon( [
          ...     (
          ...     ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
          ...     [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))]
          ...     )
          ... ] )
          >>> len(ob.geoms)
          1
          >>> type(ob.geoms[0]) == Polygon
          True
        """
        if not polygons:
            # allow creation of empty multipolygons, to support unpickling
            # TODO better empty constructor
            return pygeos.from_wkt("MULTIPOLYGON EMPTY")
        elif isinstance(polygons, MultiPolygon):
            return polygons

        polygons = getattr(polygons, 'geoms', polygons)
        polygons = [p for p in polygons
            if p and not (isinstance(p, polygon.Polygon) and p.is_empty)]

        L = len(polygons)

        # Bail immediately if we have no input points.
        if L == 0:
            return pygeos.from_wkt("MULTIPOLYGON EMPTY")

        # This function does not accept sequences of MultiPolygons: there is
        # no implicit flattening.
        if isinstance(polygons[0], MultiPolygon):
            raise ValueError("Sequences of multi-polygons are not valid arguments")

        subs = []
        for i in range(L):
            ob = polygons[i]
            if not isinstance(ob, polygon.Polygon):
                shell = ob[0]
                holes = ob[1]
                p = polygon.Polygon(shell, holes)
            else:
                p = polygon.Polygon(ob)
            subs.append(p)

        return pygeos.multipolygons(subs)

    def shape_factory(self, *args):
        return polygon.Polygon(*args)

    @property
    def __geo_interface__(self):
        allcoords = []
        for geom in self.geoms:
            coords = []
            coords.append(tuple(geom.exterior.coords))
            for hole in geom.interiors:
                coords.append(tuple(hole.coords))
            allcoords.append(tuple(coords))
        return {
            'type': 'MultiPolygon',
            'coordinates': allcoords
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns group of SVG path elements for the MultiPolygon geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Defaul value is 0.6
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
            ''.join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms) + \
            '</g>'


pygeos.lib.registry[6] = MultiPolygon
