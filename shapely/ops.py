"""Support for various GEOS geometry operations
"""

from ctypes import byref, c_void_p

from shapely.geos import lgeos
from shapely.geometry.base import geom_factory, BaseGeometry
from shapely.geometry import asShape, asLineString, asMultiLineString

__all__= ['operator', 'polygonize', 'linemerge', 'cascaded_union']


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
        return geom_factory(lgeos.GEOSUnionCascaded(collection))


operator = CollectionOperator()
polygonize = operator.polygonize
linemerge = operator.linemerge
cascaded_union = operator.cascaded_union

class ValidateOp(object):
    def __call__(self, this):
        return lgeos.GEOSisValidReason(this._geom)

validate = ValidateOp()

