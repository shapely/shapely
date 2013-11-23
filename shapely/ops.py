"""Support for various GEOS geometry operations
"""

import sys

if sys.version_info[0] < 3:
    from itertools import izip
else:
    izip = zip

from ctypes import byref, c_void_p

from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString, asMultiLineString

__all__ = ['cascaded_union', 'linemerge', 'operator', 'polygonize',
           'polygonize_full', 'transform', 'unary_union']


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
        L = len(geoms)
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
        L = len(geoms)
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
          pyproj.Proj(init='espg:4326'),
          pyproj.Proj(init='epsg:26913'))

      g2 = transform(project, g1)

    Lambda expressions such as the one in

      g2 = transform(lambda x, y, z=None: (x+1.0, y+1.0), g1)

    also satisfy the requirements for `func`.
    """
    if geom.is_empty:
        return geom
    if geom.type in ('Point', 'LineString', 'Polygon'):

        # First we try to apply func to x, y, z sequences. When func is
        # optimized for sequences, this is the fastest, though zipping
        # the results up to go back into the geometry constructors adds
        # extra cost.
        try:
            if geom.type in ('Point', 'LineString'):
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
            if geom.type in ('Point', 'LineString'):
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
