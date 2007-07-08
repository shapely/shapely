"""
Support for GEOS topological operations.
"""

from shapely.geos import TopologicalError


class BinaryTopologicalOp(object):

    """A callable non-data descriptor.

    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory

    def __get__(self, obj, objtype=None):
        self.source = obj
        return self

    def __call__(self, target):
        return self.factory(self.fn(self.source._geom, target._geom))


class UnaryTopologicalOp(object):

    """A data descriptor.
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory

    def __get__(self, obj, objtype=None):
        return self.factory(self.fn(obj._geom))
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"

