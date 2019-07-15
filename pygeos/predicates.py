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


def has_z(geometries):
    return ufuncs.has_z(geometries)


def is_closed(geometries):
    return ufuncs.is_closed(geometries)


def is_empty(geometries):
    return ufuncs.is_empty(geometries)


def is_ring(geometries):
    return ufuncs.is_ring(geometries)


def is_simple(geometries):
    return ufuncs.is_simple(geometries)


def is_valid(geometries):
    return ufuncs.is_valid(geometries)


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
