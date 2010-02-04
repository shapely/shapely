"""
Intermediaries supporting GEOS topological operations

These methods all take Shapely geometries and other Python objects and delegate
to GEOS functions via ctypes.

These methods return ctypes objects that should be recast by the caller.
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

class Delegating(Validating):
    def __init__(self, name):
        self.fn = lgeos.methods[name]

class BinaryRealProperty(Delegating):
    def __call__(self, this, other):
        self._validate(this)
        self._validate(other)
        d = c_double()
        retval = self.fn(this._geom, other._geom, byref(d))
        return d.value

class UnaryRealProperty(Delegating):
    def __call__(self, this):
        self._validate(this)
        d = c_double()
        retval = self.fn(this._geom, byref(d))
        return d.value

class BinaryTopologicalOp(Delegating):
    def __call__(self, this, other, *args):
        self._validate(this)
        self._validate(other)
        product = self.fn(this._geom, other._geom, *args)
        if product is None:
            if not this.is_valid:
                raise TopologicalError(
                    "The operation '%s' produced a null geometry. Likely cause is invalidity of the geometry %s" % (self.fn.__name__, repr(this)))
            elif not other.is_valid:
                raise TopologicalError(
                    "The operation '%s' produced a null geometry. Likely cause is invalidity of the 'other' geometry %s" % (self.fn.__name__, repr(other)))
            else:
                raise TopologicalError(
                    "This operation produced a null geometry. Reason: unknown")
        return product

class UnaryTopologicalOp(Delegating):
    def __call__(self, this, *args):
        self._validate(this)
        return self.fn(this._geom, *args)

