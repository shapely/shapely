from shapely.geos import PredicateError

# Predicates

# A callable non-data descriptor
class BinaryPredicate(object):

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        self.source = obj
        return self

    def __call__(self, target):
        retval = self.fn(self.source._geom, target._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate equals()"
        return bool(retval)


# A data descriptor
class UnaryPredicate(object):

    def __init__(self, fn):
        self.fn = fn

    def __get__(self, obj, objtype=None):
        retval = self.fn(obj._geom)
        if retval == 2:
            raise PredicateError, "Failed to evaluate equals()"
        return bool(retval)
    
    def __set__(self, obj, value=None):
        raise AttributeError, "Attribute is read-only"



