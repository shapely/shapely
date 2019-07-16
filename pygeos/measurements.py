from . import ufuncs

__all__ = ["area", "distance", "length", "hausdorff_distance"]


def area(geometries):
    return ufuncs.area(geometries)


def distance(a, b):
    return ufuncs.distance(a, b)


def length(geometries):
    return ufuncs.length(geometries)


def hausdorff_distance(a, b, densify=None):
    if densify is None:
        return ufuncs.hausdorff_distance(a, b)
    else:
        return ufuncs.haussdorf_distance_densify(a, b, densify)
