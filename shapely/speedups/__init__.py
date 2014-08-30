import warnings
import sys

from shapely.geometry import linestring, polygon
from shapely import coords

try:
    from shapely.speedups import _speedups
    available = True
    import_error_msg = None
except ImportError:
    import sys
    available = False
    # TODO: This does not appear to do anything useful
    import_error_msg = sys.exc_info()[1]

if sys.version_info < (3, 0):
    from types import MethodType

__all__ = ['available', 'enable', 'disable']
_orig = {}

def enable():
    if not available:
        warnings.warn("shapely.speedups not available", RuntimeWarning)
        return
    
    if _orig:
        return
    
    _orig['CoordinateSequence.ctypes'] = coords.CoordinateSequence.ctypes
    coords.CoordinateSequence.ctypes = property(_speedups.coordseq_ctypes)
    
    _orig['CoordinateSequence.__iter__'] = coords.CoordinateSequence.__iter__
    if sys.version_info < (3, 0):
        coords.CoordinateSequence.__iter__ = MethodType(_speedups.coordseq_iter, None, coords.CoordinateSequence)
    else:
        coords.CoordinateSequence.__iter__ = _speedups.coordseq_iter

    _orig['geos_linestring_from_py'] = linestring.geos_linestring_from_py
    linestring.geos_linestring_from_py = _speedups.geos_linestring_from_py

    _orig['geos_linearring_from_py']  = polygon.geos_linearring_from_py
    polygon.geos_linearring_from_py = _speedups.geos_linearring_from_py

def disable():
    if not _orig:
        return

    coords.CoordinateSequence.ctypes = _orig['CoordinateSequence.ctypes']
    coords.CoordinateSequence.__iter__ = _orig['CoordinateSequence.__iter__']
    linestring.geos_linestring_from_py = _orig['geos_linestring_from_py']
    polygon.geos_linearring_from_py = _orig['geos_linearring_from_py']
    _orig.clear()
