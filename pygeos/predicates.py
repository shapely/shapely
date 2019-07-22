import warnings

from . import ufuncs
from .geometry import Geometry

__all__ = [
    "is_closed",
    "is_empty",
    "is_ring",
    "is_simple",
    "is_valid",
    "crosses",
    "contains",
    "covered_by",
    "covers",
    "disjoint",
    "equals",
    "equals_exact",
    "intersects",
    "overlaps",
    "touches",
    "within",
]


def is_closed(geometry, **kwargs):
    """Returns True if a linestring's first and last points are equal.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    `ST_IsClosed <https://postgis.net/docs/ST_IsClosed.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_closed(Geometry("LINESTRING (0 0, 1 1)"))
    False
    >>> is_closed(Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)"))
    True
    >>> is_closed(Geometry("POINT (0 0)"))
    False
    """
    return ufuncs.is_closed(geometry, **kwargs)


def is_empty(geometry, **kwargs):
    """Returns True if a geometry is an empty point, polygon, etc.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    `ST_IsEmpty <https://postgis.net/docs/ST_IsEmpty.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_empty(Geometry("POINT EMPTY"))
    True
    >>> is_empty(Geometry("POINT (0 0)"))
    False
    """
    return ufuncs.is_empty(geometry, **kwargs)


def is_ring(geometry, **kwargs):
    """Returns True if a linestring is closed and simple.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for non-linestrings.

    See also
    --------
    is_closed
    is_simple
    `ST_IsRing <https://postgis.net/docs/ST_IsRing.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_ring(Geometry("POINT (0 0)"))
    False
    >>> geom = Geometry("LINESTRING(0 0, 1 1)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (False, True, False)
    >>> geom = Geometry("LINESTRING(0 0, 0 1, 1 1, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, True, True)
    >>> geom = Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)")
    >>> is_closed(geom), is_simple(geom), is_ring(geom)
    (True, False, False)
    """
    return ufuncs.is_ring(geometry, **kwargs)


def is_simple(geometry, **kwargs):
    """Returns True if a Geometry has no anomalous geometric points, such as
    self-intersections or self tangency.

    Parameters
    ----------
    geometry : Geometry or array_like
        This function will return False for geometrycollections.

    See also
    --------
    `ST_IsSimple <https://postgis.net/docs/ST_IsSimple.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_simple(Geometry("POLYGON((1 2, 3 4, 5 6, 1 2))"))
    True
    >>> is_simple(Geometry("LINESTRING(0 0, 1 1, 0 1, 1 0, 0 0)"))
    False
    """
    return ufuncs.is_simple(geometry, **kwargs)


def is_valid(geometry, **kwargs):
    """Returns True if a geometry is well formed.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    is_valid_reason
    `ST_IsValid <https://postgis.net/docs/ST_IsValid.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_valid(Geometry("LINESTRING(0 0, 1 1)"))
    True
    >>> is_valid(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    False
    """
    # GEOS is valid will emit warnings for invalid geometries. Suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ufuncs.is_valid(geometry, **kwargs)
    return result


def is_valid_reason(geometry, **kwargs):
    """Returns a string stating if a geometry is valid and if not, why.

    Parameters
    ----------
    geometry : Geometry or array_like
        Any geometry type is accepted.

    See also
    --------
    is_valid
    `ST_IsValidReason <https://postgis.net/docs/ST_IsValidReason.html>`_

    Notes
    -----
    Keyword arguments (``**kwargs``) are passed into the underlying ufunc. To
    use methods such as ``.at``, import the underlying ufunc from
    ``pygeos.ufuncs``. See the
    `NumPy docs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.

    Examples
    --------
    >>> is_valid_reason(Geometry("LINESTRING(0 0, 1 1)"))
    'Valid Geometry'
    >>> is_valid_reason(Geometry("POLYGON((0 0, 1 1, 1 2, 1 1, 0 0))"))
    'Self-intersection[0 0]'
    """
    return ufuncs.is_valid_reason(geometry, **kwargs)


def crosses(a, b):
    return ufuncs.crosses(a, b)


def contains(a, b):
    return ufuncs.contains(a, b)


def covered_by(a, b):
    return ufuncs.covered_by(a, b)


def covers(a, b):
    return ufuncs.covers(a, b)


def disjoint(a, b):
    return ufuncs.disjoint(a, b)


def equals(a, b):
    return ufuncs.equals(a, b)


def equals_exact(a, b, tolerance):
    return ufuncs.equals_exact(a, b, tolerance)


def intersects(a, b):
    return ufuncs.intersects(a, b)


def overlaps(a, b):
    return ufuncs.overlaps(a, b)


def touches(a, b):
    return ufuncs.touches(a, b)


def within(a, b):
    return ufuncs.within(a, b)
