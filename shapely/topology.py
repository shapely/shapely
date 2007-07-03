from shapely.geos import TopologicalError


# A callable non-data descriptor
class BinaryTopologicalOp(object):

    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory

    def __get__(self, obj, objtype=None):
        self.source = obj
        return self

    def __call__(self, target):
        return self.factory(self.fn(self.source._geom, target._geom))


# A data descriptor
class UnaryTopologicalOp(object):

    def __init__(self, fn, factory):
        self.fn = fn
        self.factory = factory

    def __get__(self, obj, objtype=None):
        return self.factory(self.fn(obj._geom))
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"



