"""
Support for GEOS spatial predicates.
"""

from shapely.geos import PredicateError

# Predicates

class BinaryPredicate(object):
    
    """A callable non-data descriptor.
    """
    
    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        self.source = obj
        return self

    def __call__(self, target):
        retval = self.fn(self.source._geom, target._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate %s" % repr(self.fn)
        return bool(retval)


# A data descriptor
class UnaryPredicate(object):

    """A data descriptor.
    """

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        retval = self.fn(obj._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate %s" % repr(self.fn)
        return bool(retval)
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"



