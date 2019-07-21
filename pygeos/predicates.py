from . import ufuncs


__all__ = [
    "has_z",
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

from .ufuncs import has_z  # NOQA
from .ufuncs import is_valid  # NOQA
from .ufuncs import is_closed  # NOQA
from .ufuncs import is_empty  # NOQA
from .ufuncs import is_ring  # NOQA
from .ufuncs import is_simple  # NOQA


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
