"""Shapely CGA algorithms."""

import numpy as np

import shapely
from shapely import is_ccw


def signed_area(ring):
    """Return the signed area enclosed by a ring in linear time.

    Algorithm used: https://web.archive.org/web/20080209143651/http://cgafaq.info:80/wiki/Polygon_Area
    """
    coords = np.array(ring.coords)[:, :2]
    xs, ys = np.vstack([coords, coords[1]]).T
    return np.sum(xs[1:-1] * (ys[2:] - ys[:-2])) / 2.0


def reverse_conditioned(geometry, condition=True, **kwargs):
    """Return a copy of Geometry with order of coordinates potentially reversed.

    Returns a copy of a Geometry with the order of coordinates reversed
    (`condition=True`) or returnes the geometry as is (`condition=False`).

    If a Geometry is a polygon with interior rings, the interior rings are also
    reversed.

    Points are unchanged. None is returned where Geometry is None.

    Parameters
    ----------
    geometry : Geometry or array_like
    condition: bool or array_like
        If True, return reversed geometry,
        Otherwise, return geometry as is.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See Also
    --------
    is_ccw : Checks if a Geometry is clockwise.
    reverse : Returns a copy of a Geometry with the order of coordinates reversed.

    Examples
    --------
    >>> from shapely import LineString, Polygon, reverse
    >>> ls = LineString([(0, 0), (1, 2)])
    >>> reverse_conditioned(ls)
    <LINESTRING (1 2, 0 0)>
    >>> reverse_conditioned(ls, False)
    <LINESTRING (0 0, 1 2)>
    >>> list(reverse_conditioned(ls, [True, False]))
    [<LINESTRING (1 2, 0 0)>, <LINESTRING (0 0, 1 2)>]
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> list(reverse_conditioned([polygon, ls, polygon], [True, True, False]))
    [<POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>, <LINESTRING (1 2, 0 0)>, \
<POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))>]
    >>> reverse_conditioned(None) is None
    True
    """
    if np.all(condition):
        return shapely.reverse(geometry, **kwargs)
    elif np.any(condition):
        geometry = np.array(geometry)
        if geometry.size == 1:
            # broadcast condition
            geometry = np.repeat(geometry, len(condition))
        geometries_to_reverse = geometry[condition]
        geometries_to_reverse = shapely.reverse(geometries_to_reverse, **kwargs)
        geometry[condition] = geometries_to_reverse
    return geometry


def vectorize_geom(func, dtype=shapely.Geometry, dtype_arg: int = 0, nout: int = 1):
    """Decorate a function to vectorize geometry input.

    Decorator that vectorize a function that gets a ``dtype`` or array_like as
    its first parameter
    """

    def vectorize_wrapper(*args, **kwargs):
        if isinstance(args[dtype_arg], dtype):
            # eliminate the vectorization overhead if not needed
            return func(*args, **kwargs)
        else:
            nin = len(args) + len(kwargs)
            return np.frompyfunc(func, nin, nout)(*args, **kwargs)

    return vectorize_wrapper


@vectorize_geom
def force_ccw(geometry, ccw: bool = True):
    """Return a properly oriented copy of the given geometry.

    Forces (Multi)Polygons to use a counter-clockwise orientation for their
    exterior ring, and a clockwise orientation for their interior rings (if ccw=True).
    Forces LinearRings to use a counter-clockwise/clockwise orientation
    (according to ccw).
    Also processes geometries inside a GeometryCollection in the same way.
    Other geometries are returned unchanged.

    Parameters
    ----------
    geometry : Geometry or array_like
        The original geometry. Either a LinearRing, Polygon, MultiPolygon,
        or GeometryCollection.
    ccw : bool or array_like, default True
        If True, force counter-clockwise of outer rings, and clockwise inner-rings.
        Otherwise, force clockwise of outer rings, and counter-clockwise inner-rings.

    Returns
    -------
    Geometry or array_like
    """
    if geometry.geom_type in ["MultiPolygon", "GeometryCollection"]:
        return geometry.__class__(list(force_ccw(geometry.geoms, ccw)))
    elif geometry.geom_type in ["LinearRing"]:
        return reverse_conditioned(geometry, is_ccw(geometry) != ccw)
    elif geometry.geom_type == "Polygon":
        rings = np.array([geometry.exterior, *geometry.interiors])
        reverse_condition = is_ccw(rings)
        reverse_condition[0] = not reverse_condition[0]
        if not ccw:
            reverse_condition = np.logical_not(reverse_condition)
        if np.any(reverse_condition):
            rings = reverse_conditioned(rings, reverse_condition)
            return geometry.__class__(rings[0], rings[1:])
    return geometry


def orient(geom, sign=1.0):
    """Return a properly oriented copy of the given geometry.

    Parameters
    ----------
    geom : Geometry or array_like
        The original geometry. Either a Polygon, MultiPolygon, or GeometryCollection.
    sign : float or array_like, default 1.
        The sign of the result's signed area.
        A non-negative sign means that the coordinates of the geometry's exterior
        rings will be oriented counter-clockwise.

    Returns
    -------
    Geometry or array_like

    Refer to `shapely.force_ccw` for full documentation.
    """
    return force_ccw(geom, np.array(sign) >= 0.0)
