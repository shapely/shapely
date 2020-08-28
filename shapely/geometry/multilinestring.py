"""Collections of linestrings and related utilities
"""

from ctypes import c_void_p, cast
import warnings

from shapely.errors import EmptyPartError
from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry, geos_geom_from_py
from shapely.geometry import linestring


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

    def __init__(self, lines=None):
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
        super().__init__()

        if not lines:
            # allow creation of empty multilinestrings, to support unpickling
            pass
        else:
            geom, n = geos_multilinestring_from_py(lines)
            self._set_geom(geom)
            self._ndim = n

    def shape_factory(self, *args):
        return linestring.LineString(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiLineString',
            'coordinates': tuple(tuple(c for c in g.coords) for g in self.geoms)
            }

    def svg(self, scale_factor=1., stroke_color=None):
        """Returns a group of SVG polyline elements for the LineString geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        stroke_color : str, optional
            Hex string for stroke color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if stroke_color is None:
            stroke_color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
            ''.join(p.svg(scale_factor, stroke_color) for p in self.geoms) + \
            '</g>'


def geos_multilinestring_from_py(ob):
    # ob must be either a MultiLineString, a sequence, or 
    # array of sequences or arrays
    
    if isinstance(ob, MultiLineString):
         return geos_geom_from_py(ob)

    obs = getattr(ob, 'geoms', ob)
    L = len(obs)
    assert L >= 1
    exemplar = obs[0]
    try:
        N = len(exemplar[0])
    except TypeError:
        N = exemplar._ndim
    if N not in (2, 3):
        raise ValueError("Invalid coordinate dimensionality")

    # Array of pointers to point geometries
    subs = (c_void_p * L)()
    
    # add to coordinate sequence
    for l in range(L):
        geom, ndims = linestring.geos_linestring_from_py(obs[l])

        if lgeos.GEOSisEmpty(geom):
            raise EmptyPartError("Can't create MultiLineString with empty component")

        subs[l] = cast(geom, c_void_p)
            
    return (lgeos.GEOSGeom_createCollection(5, subs, L), N)
