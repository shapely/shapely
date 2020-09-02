"""Points and related utilities
"""

from ctypes import c_double
import warnings

from shapely.errors import DimensionError, ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, geos_geom_from_py


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
      POINT (1.0000000000000000 -1.0000000000000000)
      >>> p.y
      -1.0
      >>> p.x
      1.0
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        There are 2 cases:

        1) 1 parameter: this must satisfy the numpy array protocol.
        2) 2 or more parameters: x, y, z : float
            Easting, northing, and elevation.
        """
        BaseGeometry.__init__(self)
        if len(args) > 0:
            if len(args) == 1:
                self._geom, self._ndim = geos_point_from_py(args[0])
            elif len(args) > 3:
                raise TypeError(
                    "Point() takes at most 3 arguments ({} given)".format(len(args))
                )
            else:
                self._geom, self._ndim = geos_point_from_py(tuple(args))

    # Coordinate getters and setters

    @property
    def x(self):
        """Return x coordinate."""
        return self.coords[0][0]

    @property
    def y(self):
        """Return y coordinate."""
        return self.coords[0][1]

    @property
    def z(self):
        """Return z coordinate."""
        if self._ndim != 3:
            raise DimensionError("This point has no z coordinate.")
        return self.coords[0][2]

    @property
    def __geo_interface__(self):
        return {
            'type': 'Point',
            'coordinates': self.coords[0]
            }

    def svg(self, scale_factor=1., fill_color=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="0.6" />'
            ).format(self, 3. * scale_factor, 1. * scale_factor, fill_color)

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        try:
            xy = self.coords[0]
        except IndexError:
            return ()
        return (xy[0], xy[1], xy[0], xy[1])

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


def geos_point_from_py(ob, update_geom=None, update_ndim=0):
    """Create a GEOS geom from an object that is a Point, a coordinate sequence
    or that provides the array interface.

    Returns the GEOS geometry and the number of its dimensions.
    """
    if isinstance(ob, Point):
        return geos_geom_from_py(ob)

    # Accept either (x, y) or [(x, y)]
    if not hasattr(ob, '__getitem__'):  # generators
        ob = list(ob)

    if isinstance(ob[0], tuple):
        coords = ob[0]
    else:
        coords = ob
    n = len(coords)
    dx = c_double(coords[0])
    dy = c_double(coords[1])
    dz = None
    if n == 3:
        dz = c_double(coords[2])

    if update_geom:
        cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
        if n != update_ndim:
            raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: "
                "%d" % update_ndim)
    else:
        cs = lgeos.GEOSCoordSeq_create(1, n)

    # Because of a bug in the GEOS C API, always set X before Y
    lgeos.GEOSCoordSeq_setX(cs, 0, dx)
    lgeos.GEOSCoordSeq_setY(cs, 0, dy)
    if n == 3:
        lgeos.GEOSCoordSeq_setZ(cs, 0, dz)

    if update_geom:
        return None
    else:
        return lgeos.GEOSGeom_createPoint(cs), n


def update_point_from_py(geom, ob):
    geos_point_from_py(ob, geom._geom, geom._ndim)
