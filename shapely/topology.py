"""
Support for GEOS topological operations.
"""

from shapely.geos import TopologicalError


class BinaryTopologicalOp(object):

    """A callable non-data descriptor.

    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    fn = None
    context = None
    factory = None

    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory

    def __get__(self, obj, objtype=None):
        self.context = obj
        return self

    def __call__(self, other):
        if self.context._geom is None or other._geom is None:
            raise ValueError, "Null geometry supports no operations"
        product = self.fn(self.context._geom, other._geom)
        if not product:
            # Check validity of geometries
            if not self.context.is_valid:
                raise TopologicalError, \
                "The operation '%s' produced a null geometry. Likely cause is invalidity of the geometry %s" % (self.fn.__name__, repr(self.context))
            elif not other.is_valid:
                raise TopologicalError, \
                "The operation '%s' produced a null geometry. Likely cause is invalidity of the 'other' geometry %s" % (self.fn.__name__, repr(other))

        return self.factory(product, self.context)


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

