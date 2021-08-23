"""Collections of linestrings and related utilities
"""

from ctypes import c_void_p, cast
import warnings

from shapely.errors import EmptyPartError
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry import linestring

import pygeos


__all__ = ['MultiLineString']


class MultiLineString(BaseMultipartGeometry):
    """
    A collection of one or more line strings

    A MultiLineString has non-zero length and zero area.

    Attributes
    ----------
    geoms : sequence
        A sequence of LineStrings
    """

    __slots__ = []

    def __new__(self, lines=None):
        """
        Parameters
        ----------
        lines : sequence
            A sequence of line-like coordinate sequences or objects that
            provide the numpy array interface, including instances of
            LineString.

        Example
        -------
        Construct a collection containing one line string.

          >>> lines = MultiLineString( [[[0.0, 0.0], [1.0, 2.0]]] )
        """
        if not lines:
            # allow creation of empty multilinestrings, to support unpickling
            # TODO better empty constructor
            return pygeos.from_wkt("MULTILINESTRING EMPTY")
        elif isinstance(lines, MultiLineString):
            return lines

        lines = getattr(lines, 'geoms', lines)
        m = len(lines)
        subs = []
        for i in range(m):
            l = linestring.LineString(lines[i])
            if l.is_empty:
                raise EmptyPartError("Can't create MultiLineString with empty component")
            subs.append(l)

        if len(lines) == 0:
            return pygeos.from_wkt("MULTILINESTRING EMPTY")

        return pygeos.multilinestrings(subs)

    def shape_factory(self, *args):
        return linestring.LineString(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': tuple(tuple(c for c in g.coords) for g in self.geoms)
            }

    def svg(self, scale_factor=1., stroke_color=None, opacity=None):
        """Returns a group of SVG polyline elements for the LineString geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Defaul value is 0.8
        """
        if self.is_empty:
            return '<g />'
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
            ''.join(p.svg(scale_factor, stroke_color, opacity) for p in self.geoms) + \
            '</g>'


pygeos.lib.registry[5] = MultiLineString
