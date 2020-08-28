"""Collections of points and related utilities
"""

from ctypes import byref, c_double, c_void_p, cast
import warnings

from shapely.errors import EmptyPartError, ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry.base import (
    BaseMultipartGeometry, exceptNull, geos_geom_from_py)
from shapely.geometry import point


__all__ = ['MultiPoint']


class MultiPoint(BaseMultipartGeometry):

    """A collection of one or more points

    A MultiPoint has zero area and zero length.

    Attributes
    ----------
    geoms : sequence
        A sequence of Points
    """

    def __init__(self, points=None):
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

          >>> ob = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
          >>> len(ob.geoms)
          2
          >>> type(ob.geoms[0]) == Point
          True
        """
        super(MultiPoint, self).__init__()

        if points is None or len(points) == 0:
            # allow creation of empty multipoints, to support unpickling
            pass
        else:
            self._geom, self._ndim = geos_multipoint_from_py(points)

    def shape_factory(self, *args):
        return point.Point(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPoint',
            'coordinates': tuple([g.coords[0] for g in self.geoms])
            }

    def svg(self, scale_factor=1., fill_color=None):
        """Returns a group of SVG circle elements for the MultiPoint geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
            ''.join(p.svg(scale_factor, fill_color) for p in self) + \
            '</g>'


def geos_multipoint_from_py(ob):
    if isinstance(ob, MultiPoint):
        return geos_geom_from_py(ob)

    m = len(ob)
    try:
        n = len(ob[0])
    except TypeError:
        n = ob[0]._ndim
    assert n == 2 or n == 3

    # Array of pointers to point geometries
    subs = (c_void_p * m)()

    # add to coordinate sequence
    for i in range(m):
        coords = ob[i]
        geom, ndims = point.geos_point_from_py(coords)

        if lgeos.GEOSisEmpty(geom):
            raise EmptyPartError("Can't create MultiPoint with empty component")

        subs[i] = cast(geom, c_void_p)

    return lgeos.GEOSGeom_createCollection(4, subs, m), n
