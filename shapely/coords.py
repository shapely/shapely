"""Coordinate sequence utilities
"""

import sys
from array import array
from ctypes import byref, c_double, c_uint
import warnings

from shapely.errors import ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.topology import Validating


class CoordinateSequence(object):
    """
    Iterative access to coordinate tuples from the parent geometry's coordinate
    sequence.

    Example:

      >>> from shapely.wkt import loads
      >>> g = loads('POINT (0.0 0.0)')
      >>> list(g.coords)
      [(0.0, 0.0)]

    """

    def __init__(self, coords):
        self._coords = coords

    def __len__(self):
        return self._coords.shape[0]

    def __iter__(self):
        for i in range(self.__len__()):
            yield tuple(self._coords[i].tolist())

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return tuple(self._coords[i].tolist())
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(tuple(self._coords[i].tolist()))
            return res
        else:
            raise TypeError("key must be an index or slice")

    def __array__(self, dtype=None):
        return self._coords

    @property
    def xy(self):
        """X and Y arrays"""
        m = self.__len__()
        x = array('d')
        y = array('d')
        for i in range(m):
            xy = self._coords[i].tolist()
            x.append(xy[0])
            y.append(xy[1])
        return x, y


class BoundsOp(Validating):

    def __init__(self, *args):
        pass

    def __call__(self, this):
        self._validate(this)
        env = this.envelope
        if env.geom_type == 'Point':
            return env.bounds
        cs = lgeos.GEOSGeom_getCoordSeq(env.exterior._geom)
        cs_len = c_uint(0)
        lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
        minx = 1.e+20
        maxx = -1e+20
        miny = 1.e+20
        maxy = -1e+20
        temp = c_double()
        for i in range(cs_len.value):
            lgeos.GEOSCoordSeq_getX(cs, i, byref(temp))
            x = temp.value
            if x < minx: minx = x
            if x > maxx: maxx = x
            lgeos.GEOSCoordSeq_getY(cs, i, byref(temp))
            y = temp.value
            if y < miny: miny = y
            if y > maxy: maxy = y
        return (minx, miny, maxx, maxy)
