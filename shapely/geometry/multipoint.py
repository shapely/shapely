"""Collections of points and related utilities
"""

from ctypes import byref, c_double, c_void_p, cast
import warnings

from shapely.errors import EmptyPartError, ShapelyDeprecationWarning
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry import point

import pygeos


__all__ = ['MultiPoint']


class MultiPoint(BaseMultipartGeometry):

    """A collection of one or more points

    A MultiPoint has zero area and zero length.

    Attributes
    ----------
    geoms : sequence
        A sequence of Points
    """

    __slots__ = []

    def __new__(self, points=None):
        """
        Parameters
        ----------
        points : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples or a
            sequence of objects that implement the numpy array interface,
            including instances of Point.

        Example
        -------
        Construct a 2 point collection

          >>> from shapely.geometry import Point
          >>> ob = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
          >>> len(ob.geoms)
          2
          >>> type(ob.geoms[0]) == Point
          True
        """
        if points is None:
            # allow creation of empty multipoints, to support unpickling
            # TODO better empty constructor
            return pygeos.from_wkt("MULTIPOINT EMPTY")
        elif isinstance(points, MultiPoint):
            return points

        m = len(points)
        subs = []
        for i in range(m):
            p = point.Point(points[i])
            if p.is_empty:
                raise EmptyPartError("Can't create MultiPoint with empty component")
            subs.append(p)

        if len(points) == 0:
            return pygeos.from_wkt("MULTIPOINT EMPTY")

        return pygeos.multipoints(subs)


    def shape_factory(self, *args):
        return point.Point(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPoint',
            'coordinates': tuple([g.coords[0] for g in self.geoms])
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns a group of SVG circle elements for the MultiPoint geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
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


pygeos.lib.registry[4] = MultiPoint
