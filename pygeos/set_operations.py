from . import ufuncs

__all__ = ["difference", "intersection", "symmetric_difference", "union"]


def difference(a, b=None, axis=0):
    if b is None:
        return ufuncs.difference.reduce(a, axis=axis)
    else:
        return ufuncs.difference(a, b)


def intersection(a, b=None, axis=0):
    if b is None:
        return ufuncs.intersection.reduce(a, axis=axis)
    else:
        return ufuncs.intersection(a, b)


def symmetric_difference(a, b=None, axis=0):
    if b is None:
        return ufuncs.symmetric_difference.reduce(a, axis=axis)
    else:
        return ufuncs.symmetric_difference(a, b)


def union(a, b=None, axis=0):
    if b is None:
        a = ufuncs.create_collection(a, axis=axis)
        return ufuncs.unary_union(a)
    else:
        return ufuncs.union(a, b)
