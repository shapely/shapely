import warnings

from shapely.geometry import linestring, polygon
from shapely import coords

try:
    from shapely.speedups import _speedups
    available = True
    import_error_msg = None
except ImportError:
    import sys
    available = False
    import_error_msg = tuple(sys.exc_info()[1])

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
    
    _orig['geos_linestring_from_py'] = linestring.geos_linestring_from_py
    linestring.geos_linestring_from_py = _speedups.geos_linestring_from_py

    _orig['geos_linearring_from_py']  = polygon.geos_linearring_from_py
    polygon.geos_linearring_from_py = _speedups.geos_linearring_from_py

def disable():
    if not _orig:
        return

    coords.CoordinateSequence.ctypes = _orig['CoordinateSequence.ctypes']
    linestring.geos_linestring_from_py = _orig['geos_linestring_from_py']
    polygon.geos_linearring_from_py = _orig['geos_linearring_from_py']
    _orig.clear()