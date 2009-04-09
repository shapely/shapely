"""
Support for GEOS topological operations.
"""

from shapely.geos import TopologicalError


class OpWrapper(object):
    
    def __init__(self, fn, context, factory):
        self.fn = fn
        self.context = context
        self.factory = factory
        
    def __call__(self, other):
        context = self.context
        if other._geom is None:
            raise ValueError, "Null geometry supports no operations"
        product = self.fn(context._geom, other._geom)
        if not product:
            # Check validity of geometries
            if not context.is_valid:
                raise TopologicalError, \
                "The operation '%s' produced a null geometry. Likely cause is invalidity of the geometry %s" % (self.fn.__name__, repr(context))
            elif not other.is_valid:
                raise TopologicalError, \
                "The operation '%s' produced a null geometry. Likely cause is invalidity of the 'other' geometry %s" % (self.fn.__name__, repr(other))
        return self.factory(product)
        
                    
class BinaryTopologicalOp(object):
    
    """A non-data descriptor that returns a callable.
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """
    
    fn = None
    factory = None
    
    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory
    
    def __get__(self, obj, objtype=None):
        if not obj._geom:
            raise ValueError, "Null geometry supports no operations"
        return OpWrapper(self.fn, obj, self.factory)


class UnaryTopologicalOp(object):
    
    """A data descriptor.
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """
    
    fn = None
    factory = None
    
    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory
    
    def __get__(self, obj, objtype=None):
        if obj._geom is None:
            raise ValueError, "Null geometry supports no operations"
        return self.factory(self.fn(obj._geom), obj)
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"

