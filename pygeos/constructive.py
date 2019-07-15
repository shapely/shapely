from enum import IntEnum
import numpy as np
from . import ufuncs


__all__ = [
    "BufferCapStyles",
    "BufferJoinStyles",
    "boundary",
    "buffer",
    "centroid",
    "clone",
    "convex_hull",
    "envelope",
    "extract_unique_points",
    "point_on_surface",
    "simplify",
    "snap",
]


class BufferCapStyles(IntEnum):
    ROUND = 1
    FLAT = 2
    SQUARE = 3


class BufferJoinStyles(IntEnum):
    ROUND = 1
    MITRE = 2
    BEVEL = 3


def boundary(geometries):
    return ufuncs.boundary(geometries)


def buffer(
    geometries,
    width,
    quadsegs=8,
    cap_style="round",
    join_style="round",
    mitre_limit=5.0,
    single_sided=False,
):
    """
    Computes the buffer of a geometry, for both positive and negative
    buffer distances.

    In GIS, the positive (or negative) buffer of a geometry is defined
    as the Minkowski sum (or difference) of the geometry with a circle
    with radius equal to the absolute value of the buffer distance. In
    the CAD/CAM world buffers are known as offset curves. In
    morphological analysis the operation of positive and negative
    buffering is referred to as erosion and dilation.

    The buffer operation always returns a polygonal result. The negative
    or zero-distance buffer of lines and points is always an empty Polygon.

    Since true buffer curves may contain circular arcs, computed buffer
    polygons can only be approximations to the true geometry. The user
    can control the accuracy of the curve approximation by specifying
    the number of linear segments with which to approximate a curve.

    Parameters
    ----------
    geometries : Geometry
    width : float
    quadsegs : int
    cap_style : {'round', 'flat', 'square'}
    join_style : {'round', 'mitre', 'bevel'}
    mitre_limit : float
    single_sided : bool
    """
    if isinstance(cap_style, str):
        cap_style = BufferCapStyles[cap_style.upper()].value
    if isinstance(join_style, str):
        join_style = BufferJoinStyles[join_style.upper()].value
    if not np.isscalar(quadsegs):
        raise TypeError("quadsegs only accepts scalar values")
    if not np.isscalar(cap_style):
        raise TypeError("cap_style only accepts scalar values")
    if not np.isscalar(join_style):
        raise TypeError("join_style only accepts scalar values")
    if not np.isscalar(mitre_limit):
        raise TypeError("mitre_limit only accepts scalar values")
    if not np.isscalar(single_sided):
        raise TypeError("single_sided only accepts scalar values")
    return ufuncs.buffer(
        geometries,
        width,
        np.intc(quadsegs),
        np.intc(cap_style),
        np.intc(join_style),
        mitre_limit,
        np.bool(single_sided),
    )


def centroid(geometries):
    return ufuncs.centroid(geometries)


def clone(geometries):
    return ufuncs.clone(geometries)


def convex_hull(geometries):
    return ufuncs.convex_hull(geometries)


def envelope(geometries):
    return ufuncs.envelope(geometries)


def extract_unique_points(geometries):
    return ufuncs.extract_unique_points(geometries)


def point_on_surface(geometries):
    return ufuncs.point_on_surface(geometries)


def simplify(geometries, tolerance, preserve_topology=False):
    if preserve_topology:
        return ufuncs.simplify_preserve_topology(geometries, tolerance)
    else:
        return ufuncs.simplify(geometries, tolerance)


def snap(geometries, reference, tolerance):
    return ufuncs.snap(geometries, reference, tolerance)
