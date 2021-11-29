import numpy as np

from . import Geometry  # NOQA
from . import lib
from .decorators import multithreading_enabled, requires_geos

__all__ = [
    "area",
    "distance",
    "bounds",
    "total_bounds",
    "length",
    "hausdorff_distance",
    "frechet_distance",
    "minimum_clearance",
    "minimum_bounding_radius",
]


@multithreading_enabled
def area(geometry, **kwargs):
    """Computes the area of a (multi)polygon.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> area(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    100.0
    >>> area(Geometry("MULTIPOLYGON (((0 0, 0 10, 10 10, 0 0)), ((0 0, 0 10, 10 10, 0 0)))"))
    100.0
    >>> area(Geometry("POLYGON EMPTY"))
    0.0
    >>> area(None)
    nan
    """
    return lib.area(geometry, **kwargs)


@multithreading_enabled
def distance(a, b, **kwargs):
    """Computes the Cartesian distance between two geometries.

    Parameters
    ----------
    a, b : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> point = Geometry("POINT (0 0)")
    >>> distance(Geometry("POINT (10 0)"), point)
    10.0
    >>> distance(Geometry("LINESTRING (1 1, 1 -1)"), point)
    1.0
    >>> distance(Geometry("POLYGON ((3 0, 5 0, 5 5, 3 5, 3 0))"), point)
    3.0
    >>> distance(Geometry("POINT EMPTY"), point)
    nan
    >>> distance(None, point)
    nan
    """
    return lib.distance(a, b, **kwargs)


@multithreading_enabled
def bounds(geometry, **kwargs):
    """Computes the bounds (extent) of a geometry.

    For each geometry these 4 numbers are returned: min x, min y, max x, max y.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> bounds(Geometry("POINT (2 3)")).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> bounds(Geometry("LINESTRING (0 0, 0 2, 3 2)")).tolist()
    [0.0, 0.0, 3.0, 2.0]
    >>> bounds(Geometry("POLYGON EMPTY")).tolist()
    [nan, nan, nan, nan]
    >>> bounds(None).tolist()
    [nan, nan, nan, nan]
    """
    # We need to provide the `out` argument here for compatibility with
    # numpy < 1.16. See https://github.com/numpy/numpy/issues/14949
    geometry_arr = np.asarray(geometry, dtype=np.object_)
    out = np.empty(geometry_arr.shape + (4,), dtype="float64")
    return lib.bounds(geometry_arr, out=out, **kwargs)


def total_bounds(geometry, **kwargs):
    """Computes the total bounds (extent) of the geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    numpy ndarray of [xmin, ymin, xmax, ymax]


    >>> total_bounds(Geometry("POINT (2 3)")).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> total_bounds([Geometry("POINT (2 3)"), Geometry("POINT (4 5)")]).tolist()
    [2.0, 3.0, 4.0, 5.0]
    >>> total_bounds([Geometry("LINESTRING (0 1, 0 2, 3 2)"),Geometry("LINESTRING (4 4, 4 6, 6 7)")]).tolist()
    [0.0, 1.0, 6.0, 7.0]
    >>> total_bounds(Geometry("POLYGON EMPTY")).tolist()
    [nan, nan, nan, nan]
    >>> total_bounds([Geometry("POLYGON EMPTY"), Geometry("POINT (2 3)")]).tolist()
    [2.0, 3.0, 2.0, 3.0]
    >>> total_bounds(None).tolist()
    [nan, nan, nan, nan]
    """
    b = bounds(geometry, **kwargs)
    if b.ndim == 1:
        return b

    return np.array(
        [
            np.nanmin(b[..., 0]),
            np.nanmin(b[..., 1]),
            np.nanmax(b[..., 2]),
            np.nanmax(b[..., 3]),
        ]
    )


@multithreading_enabled
def length(geometry, **kwargs):
    """Computes the length of a (multi)linestring or polygon perimeter.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> length(Geometry("LINESTRING (0 0, 0 2, 3 2)"))
    5.0
    >>> length(Geometry("MULTILINESTRING ((0 0, 1 0), (0 0, 1 0))"))
    2.0
    >>> length(Geometry("POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))"))
    40.0
    >>> length(Geometry("LINESTRING EMPTY"))
    0.0
    >>> length(None)
    nan
    """
    return lib.length(geometry, **kwargs)


@multithreading_enabled
def hausdorff_distance(a, b, densify=None, **kwargs):
    """Compute the discrete Hausdorff distance between two geometries.

    The Hausdorff distance is a measure of similarity: it is the greatest
    distance between any point in A and the closest point in B. The discrete
    distance is an approximation of this metric: only vertices are considered.
    The parameter 'densify' makes this approximation less coarse by splitting
    the line segments between vertices before computing the distance.

    Parameters
    ----------
    a, b : Geometry or array_like
    densify : float or array_like, optional
        The value of densify is required to be between 0 and 1.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING (130 0, 0 0, 0 150)")
    >>> line_2 = Geometry("LINESTRING (10 10, 10 150, 130 10)")
    >>> hausdorff_distance(line_1, line_2)  # doctest: +ELLIPSIS
    14.14...
    >>> hausdorff_distance(line_1, line_2, densify=0.5)
    70.0
    >>> hausdorff_distance(line_1, Geometry("LINESTRING EMPTY"))
    nan
    >>> hausdorff_distance(line_1, None)
    nan
    """
    if densify is None:
        return lib.hausdorff_distance(a, b, **kwargs)
    else:
        return lib.hausdorff_distance_densify(a, b, densify, **kwargs)


@requires_geos("3.7.0")
@multithreading_enabled
def frechet_distance(a, b, densify=None, **kwargs):
    """Compute the discrete Fréchet distance between two geometries.

    The Fréchet distance is a measure of similarity: it is the greatest
    distance between any point in A and the closest point in B. The discrete
    distance is an approximation of this metric: only vertices are considered.
    The parameter 'densify' makes this approximation less coarse by splitting
    the line segments between vertices before computing the distance.

    Fréchet distance sweep continuously along their respective curves
    and the direction of curves is significant. This makes it a better measure
    of similarity than Hausdorff distance for curve or surface matching.

    Parameters
    ----------
    a, b : Geometry or array_like
    densify : float or array_like, optional
        The value of densify is required to be between 0 and 1.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> line_1 = Geometry("LINESTRING (0 0, 100 0)")
    >>> line_2 = Geometry("LINESTRING (0 0, 50 50, 100 0)")
    >>> frechet_distance(line_1, line_2)  # doctest: +ELLIPSIS
    70.71...
    >>> frechet_distance(line_1, line_2, densify=0.5)
    50.0
    >>> frechet_distance(line_1, Geometry("LINESTRING EMPTY"))
    nan
    >>> frechet_distance(line_1, None)
    nan
    """
    if densify is None:
        return lib.frechet_distance(a, b, **kwargs)
    return lib.frechet_distance_densify(a, b, densify, **kwargs)


@requires_geos("3.6.0")
@multithreading_enabled
def minimum_clearance(geometry, **kwargs):
    """Computes the Minimum Clearance distance.

    A geometry's "minimum clearance" is the smallest distance by which
    a vertex of the geometry could be moved to produce an invalid geometry.

    If no minimum clearance exists for a geometry (for example, a single
    point, or an empty geometry), infinity is returned.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> minimum_clearance(Geometry("POLYGON((0 0, 0 10, 5 6, 10 10, 10 0, 5 4, 0 0))"))
    2.0
    >>> minimum_clearance(Geometry("POLYGON EMPTY"))
    inf
    >>> minimum_clearance(None)
    nan
    """
    return lib.minimum_clearance(geometry, **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def minimum_bounding_radius(geometry, **kwargs):
    """Computes the radius of the minimum bounding circle that encloses an input geometry.

    Parameters
    ----------
    geometry : Geometry or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.


    Examples
    --------
    >>> minimum_bounding_radius(Geometry("POLYGON ((0 5, 5 10, 10 5, 5 0, 0 5))"))
    5.0
    >>> minimum_bounding_radius(Geometry("LINESTRING (1 1, 1 10)"))
    4.5
    >>> minimum_bounding_radius(Geometry("MULTIPOINT (2 2, 4 2)"))
    1.0
    >>> minimum_bounding_radius(Geometry("POINT (0 1)"))
    0.0
    >>> minimum_bounding_radius(Geometry("GEOMETRYCOLLECTION EMPTY"))
    0.0

    See also
    --------
    minimum_bounding_circle
    """
    return lib.minimum_bounding_radius(geometry, **kwargs)
