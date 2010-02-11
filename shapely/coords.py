"""Coordinate sequence utilities
"""

from array import array
from ctypes import string_at, byref, c_char_p, c_double, c_void_p
from ctypes import c_int, c_size_t, c_uint
import sys

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

    # Attributes
    # ----------
    # _cseq : c_void_p
    #     Ctypes pointer to GEOS coordinate sequence
    # _ndim : int
    #     Number of dimensions (2 or 3, generally)
    # __p__ : object
    #     Parent (Shapely) geometry    
    _cseq = None
    _ndim = None
    __p__ = None

    def __init__(self, parent):
        self.__p__ = parent

    def _update(self):
        self._ndim = self.__p__._ndim
        self._cseq = lgeos.GEOSGeom_getCoordSeq(self.__p__._geom)
    
    def __len__(self):
        self._update()
        cs_len = c_uint(0)
        lgeos.GEOSCoordSeq_getSize(self._cseq, byref(cs_len))
        return cs_len.value

    def __iter__(self):
        self._update()
        dx = c_double()
        dy = c_double()
        dz = c_double()
        for i in range(self.__len__()):
            lgeos.GEOSCoordSeq_getX(self._cseq, i, byref(dx))
            lgeos.GEOSCoordSeq_getY(self._cseq, i, byref(dy))
            if self._ndim == 3: # TODO: use hasz
                lgeos.GEOSCoordSeq_getZ(self._cseq, i, byref(dz))
                yield (dx.value, dy.value, dz.value)
            else:
                yield (dx.value, dy.value)

    def __getitem__(self, i):
        self._update()
        M = self.__len__()
        if i + M < 0 or i >= M:
            raise IndexError("index out of range")
        if i < 0:
            ii = M + i
        else:
            ii = i
        dx = c_double()
        dy = c_double()
        dz = c_double()
        lgeos.GEOSCoordSeq_getX(self._cseq, ii, byref(dx))
        lgeos.GEOSCoordSeq_getY(self._cseq, ii, byref(dy))
        if self._ndim == 3: # TODO: use hasz
            lgeos.GEOSCoordSeq_getZ(self._cseq, ii, byref(dz))
            return (dx.value, dy.value, dz.value)
        else:
            return (dx.value, dy.value)

    @property
    def ctypes(self):
        self._update()
        n = self._ndim
        m = self.__len__()
        array_type = c_double * (m * n)
        data = array_type()
        temp = c_double()
        for i in xrange(m):
            lgeos.GEOSCoordSeq_getX(self._cseq, i, byref(temp))
            data[n*i] = temp.value
            lgeos.GEOSCoordSeq_getY(self._cseq, i, byref(temp))
            data[n*i+1] = temp.value
            if n == 3: # TODO: use hasz
                lgeos.GEOSCoordSeq_getZ(self._cseq, i, byref(temp))
                data[n*i+2] = temp.value
        return data

    def array_interface(self):
        """Provide the Numpy array protocol."""
        if sys.byteorder == 'little':
            typestr = '<f8'
        elif sys.byteorder == 'big':
            typestr = '>f8'
        else:
            raise ValueError(
                "Unsupported byteorder: neither little nor big-endian")
        ai = {
            'version': 3,
            'typestr': typestr,
            'data': self.ctypes,
            }
        ai.update({'shape': (len(self), self._ndim)})
        return ai
    
    __array_interface__ = property(array_interface)
    
    @property
    def xy(self):
        """X and Y arrays"""
        self._update()
        m = self.__len__()
        x = array('d')
        y = array('d')
        temp = c_double()
        for i in xrange(m):
            lgeos.GEOSCoordSeq_getX(self._cseq, i, byref(temp))
            x.append(temp.value)
            lgeos.GEOSCoordSeq_getY(self._cseq, i, byref(temp))
            y.append(temp.value)
        return x, y
            

class BoundsOp(Validating):

    def __init__(self, *args):
        pass

    def __call__(self, this):
        self._validate(this)
        env = this.envelope
        cs = lgeos.GEOSGeom_getCoordSeq(env.exterior._geom)
        cs_len = c_uint(0)
        lgeos.GEOSCoordSeq_getSize(cs, byref(cs_len))
        minx = 1.e+20
        maxx = -1e+20
        miny = 1.e+20
        maxy = -1e+20
        temp = c_double()
        for i in xrange(cs_len.value):
            lgeos.GEOSCoordSeq_getX(cs, i, byref(temp))
            x = temp.value
            if x < minx: minx = x
            if x > maxx: maxx = x
            lgeos.GEOSCoordSeq_getY(cs, i, byref(temp))
            y = temp.value
            if y < miny: miny = y
            if y > maxy: maxy = y
        return (minx, miny, maxx, maxy)

