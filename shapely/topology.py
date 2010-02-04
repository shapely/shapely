"""
Support for GEOS topological operations
"""

from ctypes import byref, c_double
from shapely.geos import TopologicalError, lgeos


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


class RealProperty(Delegated):
    
    """A real-valued property of a geometry
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self):
        self._validate(self.context)
        d = c_double()
        retval = self.fn(self.context._geom, byref(d))
        return d.value


class DistanceOp(Delegated):
    
    """A real-valued function of two geometries
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, other):
        self._validate(self.context)
        self._validate(other)
        d = c_double()
        retval = self.fn(self.context._geom, other._geom, byref(d))
        return d.value


class TopologicalProperty(Delegated):
    
    """A topological property of a geometry
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self):
        self._validate(self.context)
        return self.factory(self.fn(self.context._geom), self.context)


class UnaryTopologicalOp(Delegated):

    """A unary topological operation
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, *args):
        self._validate(self.context)
        return self.factory(self.fn(self.context._geom, *args), self.context)


class BinaryTopologicalOp(Delegated):

    """A binary topological operation
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, other):
        self._validate(self.context)
        self._validate(other)
        product = self.fn(self.context._geom, other._geom)
        if product is None:
            # Check validity of geometries
            if not self.context.is_valid:
                raise TopologicalError(
                    "The operation '%s' produced a null geometry. Likely cause is invalidity of the geometry %s" % (self.fn.__name__, repr(self.context)))
            elif not other.is_valid:
                raise TopologicalError(
                    "The operation '%s' produced a null geometry. Likely cause is invalidity of the 'other' geometry %s" % (self.fn.__name__, repr(other)))
            else:
                raise TopologicalError(
                    "This operation produced a null geometry. Reason: unknown")
        return self.factory(product)

