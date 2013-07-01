"""
Support for GEOS spatial predicates
"""

from shapely.topology import Delegating

class BinaryPredicate(Delegating):
    def __call__(self, this, other, *args):
        self._validate(this)
        self._validate(other, stop_prepared=True)
        return self.fn(this._geom, other._geom, *args)

class RelateOp(Delegating):
    def __call__(self, this, other):
        self._validate(this)
        self._validate(other, stop_prepared=True)
        return self.fn(this._geom, other._geom)

class UnaryPredicate(Delegating):
    def __call__(self, this):
        self._validate(this)
        return self.fn(this._geom)

