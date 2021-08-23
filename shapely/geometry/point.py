"""Points and related utilities
"""

from ctypes import c_double
import warnings

import pygeos

from shapely.errors import DimensionError, ShapelyDeprecationWarning
from shapely.geometry.base import BaseGeometry


__all__ = ['Point']


class Point(BaseGeometry):
    """
    A zero dimensional feature

    A point has zero length and zero area.

    Attributes
    ----------
    x, y, z : float
        Coordinate values

    Example
    -------
      >>> p = Point(1.0, -1.0)
      >>> print(p)
      POINT (1 -1)
      >>> p.y
      -1.0
      >>> p.x
      1.0
    """

    __slots__ = []

    def __new__(self, *args):
        """
        Parameters
        ----------
        There are 2 cases:

        1) 1 parameter: this must satisfy the numpy array protocol.
        2) 2 or more parameters: x, y, z : float
            Easting, northing, and elevation.
        """
        if len(args) == 0:
            # empty geometry
            # TODO better constructor
            return pygeos.from_wkt("POINT EMPTY")
        elif len(args) > 3:
            raise TypeError(
                "Point() takes at most 3 arguments ({} given)".format(len(args))
            )
        elif len(args) == 1:
            coords = args[0]
            if isinstance(coords, Point):
                return coords

            # Accept either (x, y) or [(x, y)]
            if not hasattr(coords, '__getitem__'):  # generators
                coords = list(coords)

            if isinstance(coords[0], tuple):
                coords = coords[0]

            geom = pygeos.points(coords)
        else:
            # 2 or 3 args
            geom = pygeos.points(*args)

        if not isinstance(geom, Point):
            raise ValueError("Invalid values passed to Point constructor")
        return geom

    # Coordinate getters and setters

    @property
    def x(self):
        """Return x coordinate."""
        return pygeos.get_x(self)

    @property
    def y(self):
        """Return y coordinate."""
        return pygeos.get_y(self)

    @property
    def z(self):
        """Return z coordinate."""
        if not pygeos.has_z(self):
            raise DimensionError("This point has no z coordinate.")
        # return pygeos.get_z(self) -> get_z only supported for GEOS 3.7+
        return self.coords[0][2]

    @property
    def __geo_interface__(self):
        return {
            'type': 'Point',
            'coordinates': self.coords[0]
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
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
        if opacity is None:
            opacity = 0.6 
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="{4}" />'
            ).format(self, 3. * scale_factor, 1. * scale_factor, fill_color, opacity)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values

        Example:
          >>> x, y = Point(0, 0).xy
          >>> list(x)
          [0.0]
          >>> list(y)
          [0.0]
        """
        return self.coords.xy


pygeos.lib.registry[0] = Point
