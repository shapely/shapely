from enum import IntEnum
import numpy as np
from . import ufuncs
from functools import wraps


__all__ = [
    "BufferCapStyles",
    "BufferJoinStyles",
    "boundary",
    "buffer",
    "centroid",
    "clone",
    "convex_hull",
    "envelope",
    "point_on_surface",
    "simplify",
    "simplify_preserve_topology",
]


class BufferCapStyles(IntEnum):
    ROUND = 1
    FLAT = 2
    SQUARE = 3


class BufferJoinStyles(IntEnum):
    ROUND = 1
    MITRE = 2
    BEVEL = 3


@wraps(ufuncs.boundary)
def boundary(geoms):
    return ufuncs.boundary(geoms)


@wraps(ufuncs.buffer)
def buffer(
    geoms,
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
    geoms : Geometry
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
        geoms,
        width,
        np.intc(quadsegs),
        np.intc(cap_style),
        np.intc(join_style),
        mitre_limit,
        np.bool(single_sided),
    )


@wraps(ufuncs.centroid)
def centroid(geoms):
    return ufuncs.centroid(geoms)


@wraps(ufuncs.clone)
def clone(geoms):
    return ufuncs.clone(geoms)


@wraps(ufuncs.convex_hull)
def convex_hull(geoms):
    return ufuncs.convex_hull(geoms)


@wraps(ufuncs.envelope)
def envelope(geoms):
    return ufuncs.envelope(geoms)


@wraps(ufuncs.point_on_surface)
def point_on_surface(geoms):
    return ufuncs.point_on_surface(geoms)


@wraps(ufuncs.simplify)
def simplify(geoms, tolerance, preserve_topology=False):
    if preserve_topology:
        return simplify_preserve_toplogy(geoms, tolerance)
    return ufuncs.simplify(geoms, tolerance)


@wraps(ufuncs.simplify_preserve_topology)
def simplify_preserve_topology(geoms, tolerance):
    return ufuncs.simplify_preserve_topology(geoms, tolerance)
