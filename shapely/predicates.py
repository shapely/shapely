"""
Support for GEOS spatial predicates
"""

from shapely.geos import PredicateError, lgeos


class Validating(object):
    def _validate(self, ob):
        try:
            assert ob is not None
            assert ob._geom is not None
        except AssertionError:
            raise ValueError("Null geometry supports no operations")


class Delegated(Validating):
    fn = None
    factory = None
    context = None
    def __init__(self, name, context, factory=None):
        self.fn = lgeos.methods[name]
        self.factory = factory
        self.context = context


class PredicateProperty(Delegated):
    """A predicate property of a geometry
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self):
        self._validate(self.context)
        return bool(self.fn(self.context._geom))


class RelateOp(Delegated):
    """A predicate property of a geometry
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, other):
        self._validate(self.context)
        self._validate(other)
        return self.fn(self.context._geom, other._geom)


class BinaryPredicateOp(Delegated):
    """A predicate property of a geometry
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, other, *args):
        self._validate(self.context)
        self._validate(other)
        newargs = [self.context._geom, other._geom]
        if len(args) > 0:
            newargs = newargs + list(args)
        return bool(self.fn(*newargs))

