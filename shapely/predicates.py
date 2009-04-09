"""
Support for GEOS spatial predicates.
"""

from shapely.geos import PredicateError


class OpWrapper(object):
    
    def __init__(self, fn, context):
        self.fn = fn
        self.context = context
        
    def __call__(self, other):
        if not other._geom:
            raise ValueError, "Null geometry can not be operated upon"
        return bool(self.fn(self.context._geom, other._geom))


class BinaryPredicate(object):
    
    """A callable non-data descriptor.
    """
   
    fn = None
    context = None

    def __init__(self, fn):
        self.fn = fn
        def errcheck(result, func, argtuple):
            if result == 2:
                raise PredicateError, "Failed to evaluate %s" % repr(self.fn)
            return result
        self.fn.errcheck = errcheck

    def __get__(self, obj, objtype=None):
        if not obj._geom:
            raise ValueError, "Null geometry supports no operations"
        return OpWrapper(self.fn, obj)


# A data descriptor
class UnaryPredicate(object):

    """A data descriptor.
    """

    fn = None

    def __init__(self, fn):
        self.fn = fn
        def errcheck(result, func, argtuple):
            if result == 2:
                raise PredicateError, "Failed to evaluate %s" % repr(self.fn)
            return result
        self.fn.errcheck = errcheck

    def __get__(self, obj, objtype=None):
        if not obj._geom:
            raise ValueError, "Null geometry supports no operations"
        return bool(self.fn(obj._geom))
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"

