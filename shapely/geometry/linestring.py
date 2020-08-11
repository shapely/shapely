"""Line strings and related utilities
"""

from ctypes import c_double
import warnings

from shapely.errors import ShapelyDeprecationWarning
from shapely.geos import lgeos, TopologicalError
from shapely.geometry.base import (
    BaseGeometry, geom_factory, JOIN_STYLE, geos_geom_from_py
)
from shapely.geometry.point import Point

__all__ = ['LineString']


class LineString(BaseGeometry):
    """
    A one-dimensional figure comprising one or more line segments

    A LineString has non-zero length and zero area. It may approximate a curve
    and need not be straight. Unlike a LinearRing, a LineString is not closed.
    """

    def __init__(self, coordinates=None):
        """
        Parameters
        ----------
        coordinates : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples or
            an object that provides the numpy array interface, including
            another instance of LineString.

        Example
        -------
        Create a line with two segments

          >>> a = LineString([[0, 0], [1, 0], [1, 1]])
          >>> a.length
          2.0
        """
        BaseGeometry.__init__(self)
        if coordinates is not None:
            ret = geos_linestring_from_py(coordinates)
            if ret is not None:
                self._geom, self._ndim = ret

    @property
    def __geo_interface__(self):
        return {
            'type': 'LineString',
            'coordinates': tuple(self.coords)
            }

    def svg(self, scale_factor=1., stroke_color=None):
        """Returns SVG polyline element for the LineString geometry.

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
        pnt_format = " ".join(["{},{}".format(*c) for c in self.coords])
        return (
            '<polyline fill="none" stroke="{2}" stroke-width="{1}" '
            'points="{0}" opacity="0.8" />'
            ).format(pnt_format, 2. * scale_factor, stroke_color)

    @property
    def ctypes(self):
        if not self._ctypes_data:
            self._ctypes_data = self.coords.ctypes
        return self._ctypes_data

    def array_interface(self):
        """Provide the Numpy array protocol."""
        if self.is_empty:
            ai = {'version': 3, 'typestr': '<f8', 'shape': (0,), 'data': (c_double * 0)()}
        else:
            ai = self.coords.array_interface()
        return ai

    __array_interface__ = property(array_interface)

    # Coordinate access
    def _set_coords(self, coordinates):
        warnings.warn(
            "Setting the 'coords' to mutate a Geometry in place is deprecated,"
            " and will not be possible any more in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        self.empty()
        ret = geos_linestring_from_py(coordinates)
        if ret is not None:
            self._geom, self._ndim = ret

    coords = property(BaseGeometry._get_coords, _set_coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values

        Example:

          >>> x, y = LineString(((0, 0), (1, 1))).xy
          >>> list(x)
          [0.0, 1.0]
          >>> list(y)
          [0.0, 1.0]
        """
        return self.coords.xy

    def parallel_offset(
            self, distance, side='right',
            resolution=16, join_style=JOIN_STYLE.round, mitre_limit=5.0):

        """Returns a LineString or MultiLineString geometry at a distance from
        the object on its right or its left side.

        The side parameter may be 'left' or 'right' (default is 'right'). The
        resolution of the buffer around each vertex of the object increases by
        increasing the resolution keyword parameter or third positional
        parameter. Vertices of right hand offset lines will be ordered in
        reverse.

        The join style is for outside corners between line segments. Accepted
        values are JOIN_STYLE.round (1), JOIN_STYLE.mitre (2), and
        JOIN_STYLE.bevel (3).

        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled.
        """
        if mitre_limit == 0.0:
            raise ValueError(
                'Cannot compute offset from zero-length line segment')
        try:
            return geom_factory(self.impl['parallel_offset'](
                self, distance, resolution, join_style, mitre_limit, side))
        except OSError:
            raise TopologicalError()


def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    # If a LineString is passed in, clone it and return
    # If a LinearRing is passed in, clone the coord seq and return a
    # LineString.
    #
    # NB: access to coordinates using the array protocol has been moved
    # entirely to the speedups module.

    if isinstance(ob, LineString):
        if type(ob) == LineString:
            return geos_geom_from_py(ob)
        else:
            return geos_geom_from_py(ob, lgeos.GEOSGeom_createLineString)

    try:
        m = len(ob)
    except TypeError:  # generators
        ob = list(ob)
        m = len(ob)

    if m == 0:
        return None
    elif m == 1:
        raise ValueError("LineStrings must have at least 2 coordinate tuples")

    if m < 2:
        raise ValueError(
            "LineStrings must have at least 2 coordinate tuples")

    def _coords(o):
        if isinstance(o, Point):
            return o.coords[0]
        else:
            return o

    try:
        n = len(_coords(ob[0]))
    except TypeError:
        raise ValueError(
            "Input %s is the wrong shape for a LineString" % str(ob))
    assert n == 2 or n == 3

    # Create a coordinate sequence
    if update_geom is not None:
        cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
        if n != update_ndim:
            raise ValueError(
                "Wrong coordinate dimensions; this geometry has "
                "dimensions: %d" % update_ndim)
    else:
        cs = lgeos.GEOSCoordSeq_create(m, n)

    # add to coordinate sequence
    for i in range(m):
        coords = _coords(ob[i])
        # Because of a bug in the GEOS C API,
        # always set X before Y
        lgeos.GEOSCoordSeq_setX(cs, i, coords[0])
        lgeos.GEOSCoordSeq_setY(cs, i, coords[1])
        if n == 3:
            try:
                lgeos.GEOSCoordSeq_setZ(cs, i, coords[2])
            except IndexError:
                raise ValueError("Inconsistent coordinate dimensionality")

    if update_geom is not None:
        return None
    else:
        ptr = lgeos.GEOSGeom_createLineString(cs)
        if not ptr:
            raise ValueError("GEOSGeom_createLineString returned a NULL pointer")
        return ptr, n


def update_linestring_from_py(geom, ob):
    geos_linestring_from_py(ob, geom._geom, geom._ndim)
