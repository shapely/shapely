"""Support for various GEOS geometry operations
"""

import sys

if sys.version_info[0] < 3:
    from itertools import izip
else:
    izip = zip

from ctypes import byref, c_void_p, c_double

from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString, asMultiLineString, Point

__all__ = ['cascaded_union', 'linemerge', 'operator', 'polygonize',
           'polygonize_full', 'transform', 'unary_union', 'triangulate']


class CollectionOperator(object):

    def shapeup(self, ob):
        if isinstance(ob, BaseGeometry):
            return ob
        else:
            try:
                return asShape(ob)
            except ValueError:
                return asLineString(ob)

    def polygonize(self, lines):
        """Creates polygons from a source of lines

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.
        """
        source = getattr(lines, 'geoms', None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(l) for l in source]
        geom_array_type = c_void_p * len(obs)
        geom_array = geom_array_type()
        for i, line in enumerate(obs):
            geom_array[i] = line._geom
        product = lgeos.GEOSPolygonize(byref(geom_array), len(obs))
        collection = geom_factory(product)
        for g in collection.geoms:
            clone = lgeos.GEOSGeom_clone(g._geom)
            g = geom_factory(clone)
            g._owned = False
            yield g

    def polygonize_full(self, lines):
        """Creates polygons from a source of lines, returning the polygons
        and leftover geometries.

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.

        Returns a tuple of objects: (polygons, dangles, cut edges, invalid ring
        lines). Each are a geometry collection.

        Dangles are edges which have one or both ends which are not incident on
        another edge endpoint. Cut edges are connected at both ends but do not
        form part of polygon. Invalid ring lines form rings which are invalid
        (bowties, etc).
        """
        source = getattr(lines, 'geoms', None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(l) for l in source]
        L = len(obs)
        subs = (c_void_p * L)()
        for i, g in enumerate(obs):
            subs[i] = g._geom
        collection = lgeos.GEOSGeom_createCollection(5, subs, L)
        dangles = c_void_p()
        cuts = c_void_p()
        invalids = c_void_p()
        product = lgeos.GEOSPolygonize_full(
            collection, byref(dangles), byref(cuts), byref(invalids))
        return (
            geom_factory(product),
            geom_factory(dangles),
            geom_factory(cuts),
            geom_factory(invalids)
            )

    def linemerge(self, lines):
        """Merges all connected lines from a source

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.  Returns a
        LineString or MultiLineString when lines are not contiguous.
        """
        source = None
        if hasattr(lines, 'type') and lines.type == 'MultiLineString':
            source = lines
        elif hasattr(lines, '__iter__'):
            try:
                source = asMultiLineString([ls.coords for ls in lines])
            except AttributeError:
                source = asMultiLineString(lines)
        if source is None:
            raise ValueError("Cannot linemerge %s" % lines)
        result = lgeos.GEOSLineMerge(source._geom)
        return geom_factory(result)

    def cascaded_union(self, geoms):
        """Returns the union of a sequence of geometries

        This is the most efficient method of dissolving many polygons.
        """
        try:
            L = len(geoms)
        except TypeError:
            geoms = [geoms]
            L = 1
        subs = (c_void_p * L)()
        for i, g in enumerate(geoms):
            subs[i] = g._geom
        collection = lgeos.GEOSGeom_createCollection(6, subs, L)
        return geom_factory(lgeos.methods['cascaded_union'](collection))

    def unary_union(self, geoms):
        """Returns the union of a sequence of geometries

        This method replaces :meth:`cascaded_union` as the
        prefered method for dissolving many polygons.

        """
        try:
            L = len(geoms)
        except TypeError:
            geoms = [geoms]
            L = 1
        subs = (c_void_p * L)()
        for i, g in enumerate(geoms):
            subs[i] = g._geom
        collection = lgeos.GEOSGeom_createCollection(6, subs, L)
        return geom_factory(lgeos.methods['unary_union'](collection))

operator = CollectionOperator()
polygonize = operator.polygonize
polygonize_full = operator.polygonize_full
linemerge = operator.linemerge
cascaded_union = operator.cascaded_union
unary_union = operator.unary_union


def triangulate(geom, tolerance=0.0, edges=False):
    """Creates the Delaunay triangulation and returns a list of geometries

    The source may be any geometry type. All vertices of the geometry will be
    used as the points of the triangulation.

    From the GEOS documentation:
    tolerance is the snapping tolerance used to improve the robustness of
    the triangulation computation. A tolerance of 0.0 specifies that no
    snapping will take place.

    If edges is False, a list of Polygons (triangles) will be returned.
    Otherwise the list of LineString edges is returned.

    """
    func = lgeos.methods['delaunay_triangulation']
    gc = geom_factory(func(geom._geom, tolerance, int(edges)))
    return [g for g in gc.geoms]

class ValidateOp(object):
    def __call__(self, this):
        return lgeos.GEOSisValidReason(this._geom)

validate = ValidateOp()


def transform(func, geom):
    """Applies `func` to all coordinates of `geom` and returns a new
    geometry of the same type from the transformed coordinates.

    `func` maps x, y, and optionally z to output xp, yp, zp. The input
    parameters may iterable types like lists or arrays or single values.
    The output shall be of the same type. Scalars in, scalars out.
    Lists in, lists out.

    For example, here is an identity function applicable to both types
    of input.

      def id_func(x, y, z=None):
          return tuple(filter(None, [x, y, z]))

      g2 = transform(id_func, g1)

    A partially applied transform function from pyproj satisfies the
    requirements for `func`.

      from functools import partial
      import pyproj

      project = partial(
          pyproj.transform,
          pyproj.Proj(init='epsg:4326'),
          pyproj.Proj(init='epsg:26913'))

      g2 = transform(project, g1)

    Lambda expressions such as the one in

      g2 = transform(lambda x, y, z=None: (x+1.0, y+1.0), g1)

    also satisfy the requirements for `func`.
    """
    if geom.is_empty:
        return geom
    if geom.type in ('Point', 'LineString', 'LinearRing', 'Polygon'):

        # First we try to apply func to x, y, z sequences. When func is
        # optimized for sequences, this is the fastest, though zipping
        # the results up to go back into the geometry constructors adds
        # extra cost.
        try:
            if geom.type in ('Point', 'LineString', 'LinearRing'):
                return type(geom)(zip(*func(*izip(*geom.coords))))
            elif geom.type == 'Polygon':
                shell = type(geom.exterior)(
                    zip(*func(*izip(*geom.exterior.coords))))
                holes = list(type(ring)(zip(*func(*izip(*ring.coords))))
                             for ring in geom.interiors)
                return type(geom)(shell, holes)

        # A func that assumes x, y, z are single values will likely raise a
        # TypeError, in which case we'll try again.
        except TypeError:
            if geom.type in ('Point', 'LineString', 'LinearRing'):
                return type(geom)([func(*c) for c in geom.coords])
            elif geom.type == 'Polygon':
                shell = type(geom.exterior)(
                    [func(*c) for c in geom.exterior.coords])
                holes = list(type(ring)([func(*c) for c in ring.coords])
                             for ring in geom.interiors)
                return type(geom)(shell, holes)

    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        return type(geom)([transform(func, part) for part in geom.geoms])
    else:
        raise ValueError('Type %r not recognized' % geom.type)


def nearest_points(g1, g2):
    """Returns the calculated nearest points in the input geometries
    
    The points are returned in the same order as the input geometries.
    """
    seq = lgeos.methods['nearest_points'](g1._geom, g2._geom)
    if seq is None:
        if g1.is_empty:
            raise ValueError('The first input geometry is empty')
        else:
            raise ValueError('The second input geometry is empty')
    x1 = c_double()
    y1 = c_double()
    x2 = c_double()
    y2 = c_double()
    lgeos.GEOSCoordSeq_getX(seq, 0, byref(x1))
    lgeos.GEOSCoordSeq_getY(seq, 0, byref(y1))
    lgeos.GEOSCoordSeq_getX(seq, 1, byref(x2))
    lgeos.GEOSCoordSeq_getY(seq, 1, byref(y2))
    p1 = Point(x1.value, y1.value)
    p2 = Point(x2.value, y2.value)
    return (p1, p2)
