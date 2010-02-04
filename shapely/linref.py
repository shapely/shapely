"""Linear referencing
"""

from shapely.topology import Delegated


class LinearRefBase(Delegated):
    def _validate_line(self, ob):
        super(LinearRefBase, self)._validate(ob)
        try:
            assert ob.geom_type in ['LineString', 'MultiLineString']
        except AssertionError:
            raise TypeError("Only linear types support this operation")


class ProjectOp(LinearRefBase):
    """A real-valued function of two geometries
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, other):
        self._validate_line(self.context)
        self._validate(other)
        return self.fn(self.context._geom, other._geom)


class InterpolateOp(LinearRefBase):
    """Function of one geometry and a real value
    
    Wraps a GEOS function. The factory is a callable which wraps results in
    the appropriate shapely geometry class.
    """

    def __call__(self, distance):
        self._validate_line(self.context)
        return self.factory(self.fn(self.context._geom, distance))


