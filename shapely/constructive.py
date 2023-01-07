from typing import Optional, TYPE_CHECKING, Union

import numpy as np

from . import lib
from ._enum import ParamEnum
from .decorators import multithreading_enabled, requires_geos

__all__ = [
    "BufferCapStyle",
    "BufferJoinStyle",
    "boundary",
    "buffer",
    "offset_curve",
    "centroid",
    "clip_by_rect",
    "concave_hull",
    "convex_hull",
    "delaunay_triangles",
    "segmentize",
    "envelope",
    "extract_unique_points",
    "build_area",
    "make_valid",
    "normalize",
    "node",
    "point_on_surface",
    "polygonize",
    "polygonize_full",
    "remove_repeated_points",
    "reverse",
    "simplify",
    "snap",
    "voronoi_polygons",
    "oriented_envelope",
    "minimum_rotated_rectangle",
    "minimum_bounding_circle",
]

from shapely.shapely_typing import (
    GeometryArrayLike,
    GeometryArrayNLike,
    MaybeArrayN,
    MaybeArrayNLike,
    MaybeGeometryArrayN,
    MaybeGeometryArrayNLike,
    Tuple4,
)

if TYPE_CHECKING:
    from shapely import (
        EmptyGeometry,
        GeometryCollection,
        LineString,
        MultiLineString,
        MultiPoint,
        Point,
        Polygon,
    )
    from shapely.geometry.base import BaseGeometry


class BufferCapStyle(ParamEnum):
    round = 1
    flat = 2
    square = 3


class BufferJoinStyle(ParamEnum):
    round = 1
    mitre = 2
    bevel = 3


@multithreading_enabled
def boundary(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Returns the topological boundary of a geometry
    (lower dimension geometry that bounds the object).

    The boundary of a polygon is a line.
    the boundary of a line is a collection of points.
    The boundary of a point is an empty (null) collection.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
        This function will return None for geometrycollections.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, LinearRing, LineString, \
MultiLineString, MultiPoint, Point, Polygon
    >>> boundary(Point(0, 0))
    <GEOMETRYCOLLECTION EMPTY>
    >>> boundary(LineString([(0, 0), (1, 1), (1, 2)]))
    <MULTIPOINT (0 0, 1 2)>
    >>> boundary(LinearRing([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <MULTIPOINT EMPTY>
    >>> boundary(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)>
    >>> boundary(MultiPoint([(0, 0), (1, 2)]))
    <GEOMETRYCOLLECTION EMPTY>
    >>> boundary(MultiLineString([[(0, 0), (1, 1)], [(0, 1), (1, 0)]]))
    <MULTIPOINT (0 0, 0 1, 1 0, 1 1)>
    >>> boundary(GeometryCollection([Point(0, 0)])) is None
    True
    """
    return lib.boundary(geometry, **kwargs)


@multithreading_enabled
def buffer(
    geometry: MaybeGeometryArrayNLike,
    distance: MaybeArrayNLike[float],
    quad_segs: int = 8,
    cap_style: Union[BufferCapStyle, str] = "round",
    join_style: Union[BufferJoinStyle, str] = "round",
    mitre_limit: float = 5.0,
    single_sided: bool = False,
    **kwargs
) -> MaybeGeometryArrayN:
    """Computes the buffer of a geometry for positive and negative buffer distance,
    A geometry that represents all points within a distance of this geometry.

    A positive distance produces a dilation, a negative distance an erosion.
    A very small or zero distance may sometimes be used to "tidy" a polygon.

    The buffer of a geometry is defined as the Minkowski sum (or difference,
    for negative distance) of the geometry with a circle with radius equal
    to the absolute value of the buffer distance.

    The buffer operation always returns a polygonal result. The negative
    or zero-distance buffer of lines and points is always empty.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    distance : MaybeArrayNLike[float]
        Specifies the circle radius (distance to buffer around the object)
        in the Minkowski sum (or difference).
    quad_segs : int, default 8
        Specifies the number of linear segments in a quarter circle in the
        approximation of circular arcs.
    cap_style : shapely.BufferCapStyle or {'round', 'square', 'flat'}, default 'round'
        Specifies the shape of buffered line endings.
        'round' results in circular line endings (see ``quad_segs``).
        Both 'square' and 'flat' result in rectangular line endings,
        only 'flat' will end at the original vertex,
        while 'square' involves adding the buffer width.
    join_style : shapely.BufferJoinStyle or {'round', 'mitre', 'bevel'}, default 'round'
        Specifies the shape of buffered line midpoints.
        'round' results in rounded shapes.
        'bevel' results in a beveled edge that touches the original vertex.
        'mitre' results in a single vertex that is beveled depending on the
        ``mitre_limit`` parameter.
    mitre_limit : float, default 5.0
        Crops of 'mitre'-style joins if the point is displaced from the
        buffered vertex by more than this limit.
        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled.
    single_sided : bool, default False
        Only buffer at one side of the geometry.
        The side used is determined by the sign of the buffer distance:
            a positive distance indicates the left-hand side
            a negative distance indicates the right-hand side
        The single-sided buffer of point geometries is the same as the regular buffer.
        The End Cap Style for single-sided buffers is always ignored, and forced to
        the equivalent of CAP_FLAT.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Notes
    -----
    The return value is a strictly two-dimensional geometry. All
    Z coordinates of the original geometry will be ignored.

    Examples
    --------
    >>> from shapely import LineString, Point, Polygon, BufferCapStyle, BufferJoinStyle

    Ocatagon, 128-gon, triangle approximations of a unit radius circle:
    >>> p = Point(0, 0)
    >>> buffer(p, 1).area  # doctest: +ELLIPSIS
    3.121445152258...
    >>> buffer(p, 1, quad_segs=128).area  # doctest: +ELLIPSIS
    3.141513801144...
    >>> buffer(p, 1, quad_segs=3).area
    3.0

    >>> list(buffer(p, 1.0, cap_style="square").exterior.coords)
    [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
    >>> buffer(p, 1.0, cap_style="square").area
    4.0

    >>> p = Point(10, 10)
    >>> buffer(p, 2, quad_segs=1)
    <POLYGON ((12 10, 10 8, 8 10, 10 12, 12 10))>
    >>> buffer(p, 2, quad_segs=2)  # doctest: +ELLIPSIS
    <POLYGON ((12 10, 11.414 8.586, 10 8, 8.586 8.586, 8 10, 8.5...>
    >>> buffer(p, -2, quad_segs=1)
    <POLYGON EMPTY>

    >>> line = LineString([(10, 10), (20, 10)])
    >>> buffer(line, 2, cap_style="square")
    <POLYGON ((20 12, 22 12, 22 8, 10 8, 8 8, 8 12, 20 12))>
    >>> buffer(line, 2, cap_style="flat")
    <POLYGON ((20 12, 20 8, 10 8, 10 12, 20 12))>
    >>> buffer(line, 2, single_sided=True, cap_style="flat")
    <POLYGON ((20 10, 10 10, 10 12, 20 12, 20 10))>

    >>> line2 = LineString([(10, 10), (20, 10), (20, 20)])
    >>> buffer(line2, 2, cap_style="flat", join_style="bevel")
    <POLYGON ((18 12, 18 20, 22 20, 22 10, 20 8, 10 8, 10 12, 18 12))>
    >>> buffer(line2, 2, cap_style="flat", join_style="mitre")
    <POLYGON ((18 12, 18 20, 22 20, 22 8, 10 8, 10 12, 18 12))>
    >>> buffer(line2, 2, cap_style="flat", join_style="mitre", mitre_limit=1)
    <POLYGON ((18 12, 18 20, 22 20, 22 9.172, 20.828 8, 10 8, 10 12, 18 12))>

    >>> square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> buffer(square, 2, join_style="mitre")
    <POLYGON ((-2 -2, -2 12, 12 12, 12 -2, -2 -2))>
    >>> buffer(square, -2, join_style="mitre")
    <POLYGON ((2 2, 2 8, 8 8, 8 2, 2 2))>
    >>> buffer(square, -5, join_style="mitre")
    <POLYGON EMPTY>
    >>> buffer(line, float("nan")) is None
    True
    """
    if isinstance(cap_style, str):
        cap_style = BufferCapStyle.get_value(cap_style)
    if isinstance(join_style, str):
        join_style = BufferJoinStyle.get_value(join_style)
    if not np.isscalar(quad_segs):
        raise TypeError("quad_segs only accepts scalar values")
    if not np.isscalar(cap_style):
        raise TypeError("cap_style only accepts scalar values")
    if not np.isscalar(join_style):
        raise TypeError("join_style only accepts scalar values")
    if not np.isscalar(mitre_limit):
        raise TypeError("mitre_limit only accepts scalar values")
    if not np.isscalar(single_sided):
        raise TypeError("single_sided only accepts scalar values")
    return lib.buffer(
        geometry,
        distance,
        np.intc(quad_segs),
        np.intc(cap_style),
        np.intc(join_style),
        mitre_limit,
        np.bool_(single_sided),
        **kwargs
    )


@multithreading_enabled
def offset_curve(
    geometry: MaybeGeometryArrayNLike,
    distance: MaybeArrayNLike[float],
    quad_segs: int = 8,
    join_style: Union[BufferJoinStyle, str] = "round",
    mitre_limit: float = 5.0,
    **kwargs
) -> MaybeArrayNLike[Union["LineString", "MultiLineString"]]:
    """Returns a (Multi)LineString at a distance from the object
    on its right or its left side.

    The side is determined by the sign of the `distance` parameter.
    For positive distance the offset will be on the left side of the input
    line. For a negative distance it will be on the right side. In general,
    this function tries to preserve the direction of the input.

    Note: the behaviour regarding orientation of the resulting line depends
    on the GEOS version. With GEOS < 3.11, the line retains the same
    direction for a left offset (positive distance) or has opposite direction
    for a right offset (negative distance), and this behaviour was documented
    as such in previous Shapely versions. Starting with GEOS 3.11, the
    function tries to preserve the orientation of the original line.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    distance : MaybeArrayNLike[float]
        Specifies the offset distance from the input geometry.
        Negative for right side offset, positive for left side offset.
    quad_segs : int, default 8
        Specifies the number of linear segments in a quarter circle in the
        approximation of circular arcs.
        The resolution of the buffer around each vertex of the object increases
        by increasing the `quad_segs` parameter.
    join_style : shapely.BufferJoinStyle or {'round', 'mitre', 'bevel'}, default 'round'
        Specifies the shape of outside corners between line segments.
        'round' results in rounded shapes.
        'bevel' results in a beveled edge that touches the original vertex.
        'mitre' results in a single vertex that is beveled depending on the
        ``mitre_limit`` parameter.
    mitre_limit : float, default 5.0
        Crops of 'mitre'-style joins if the point is displaced from the
        buffered vertex by more than this limit.
        The mitre ratio limit is used for very sharp corners. It is the ratio
        of the distance from the corner to the end of the mitred offset corner.
        When two line segments meet at a sharp angle, a miter join will extend
        far beyond the original geometry. To prevent unreasonable geometry, the
        mitre limit allows controlling the maximum length of the join corner.
        Corners with a ratio which exceed the limit will be beveled.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString
    >>> line = LineString([(0, 0), (0, 2)])
    >>> offset_curve(line, 2)
    <LINESTRING (-2 0, -2 2)>
    >>> offset_curve(line, -2)
    <LINESTRING (2 0, 2 2)>
    """
    if isinstance(join_style, str):
        join_style = BufferJoinStyle.get_value(join_style)
    if not np.isscalar(quad_segs):
        raise TypeError("quad_segs only accepts scalar values")
    if not np.isscalar(join_style):
        raise TypeError("join_style only accepts scalar values")
    if not np.isscalar(mitre_limit):
        raise TypeError("mitre_limit only accepts scalar values")
    return lib.offset_curve(
        geometry,
        distance,
        np.intc(quad_segs),
        np.intc(join_style),
        np.double(mitre_limit),
        **kwargs
    )


@multithreading_enabled
def centroid(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeArrayN["Point"]:
    """Computes the geometric center (center-of-mass) of a geometry.

    For multipoints this is computed as the mean of the input coordinates.
    For multilinestrings the centroid is weighted by the length of each
    line segment. For multipolygons the centroid is weighted by the area of
    each polygon.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, Polygon
    >>> centroid(Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]))
    <POINT (5 5)>
    >>> centroid(LineString([(0, 0), (2, 2), (10, 10)]))
    <POINT (5 5)>
    >>> centroid(MultiPoint([(0, 0), (10, 10)]))
    <POINT (5 5)>
    >>> centroid(Polygon())
    <POINT EMPTY>
    """
    return lib.centroid(geometry, **kwargs)


@multithreading_enabled
def clip_by_rect(
    geometry: MaybeGeometryArrayNLike,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    **kwargs
) -> MaybeGeometryArrayN:
    """
    Returns the portion of a geometry within a rectangle.

    The geometry is clipped in a fast but possibly dirty way. The output is
    not guaranteed to be valid. No exceptions will be raised for topological
    errors.

    Note: empty geometries or geometries that do not overlap with the
    specified bounds will result in GEOMETRYCOLLECTION EMPTY.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
        The geometries to be clipped
    xmin : float
        Minimum x value of the rectangle
    ymin : float
        Minimum y value of the rectangle
    xmax : float
        Maximum x value of the rectangle
    ymax : float
        Maximum y value of the rectangle
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (10, 10)])
    >>> clip_by_rect(line, 0., 0., 1., 1.)
    <LINESTRING (0 0, 1 1)>
    >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> clip_by_rect(polygon, 0., 0., 1., 1.)
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    """
    if not all(np.isscalar(val) for val in [xmin, ymin, xmax, ymax]):
        raise TypeError("xmin/ymin/xmax/ymax only accepts scalar values")
    return lib.clip_by_rect(
        geometry,
        np.double(xmin),
        np.double(ymin),
        np.double(xmax),
        np.double(ymax),
        **kwargs
    )


@requires_geos("3.11.0")
@multithreading_enabled
def concave_hull(
    geometry: MaybeGeometryArrayNLike,
    ratio: float = 0.0,
    allow_holes: bool = False,
    **kwargs
) -> MaybeGeometryArrayN:
    """Computes a concave geometry that encloses an input geometry.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    ratio : float, default 0.0
        Number in the range [0, 1]. Higher numbers will include fewer vertices
        in the hull.
    allow_holes : bool, default False
        If set to True, the concave hull may have holes.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import MultiPoint, Polygon
    >>> concave_hull(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]), ratio=0.1)
    <POLYGON ((0 0, 0 3, 1 1, 3 3, 3 0, 0 0))>
    >>> concave_hull(MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)]), ratio=1.0)
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0))>
    >>> concave_hull(Polygon())
    <POLYGON EMPTY>
    """
    if not np.isscalar(ratio):
        raise TypeError("ratio must be scalar")
    if not np.isscalar(allow_holes):
        raise TypeError("allow_holes must be scalar")
    return lib.concave_hull(geometry, np.double(ratio), np.bool_(allow_holes), **kwargs)


@multithreading_enabled
def convex_hull(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Computes the minimum convex geometry that encloses an input geometry.
    Imagine an elastic band stretched around the geometry:
    that's a convex hull, more or less.

    The convex hull of a three member multipoint, for example, is a triangular polygon.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import MultiPoint, Polygon
    >>> convex_hull(MultiPoint([(0, 0), (10, 0), (10, 10)]))
    <POLYGON ((0 0, 10 10, 10 0, 0 0))>
    >>> convex_hull(Polygon())
    <GEOMETRYCOLLECTION EMPTY>
    """
    return lib.convex_hull(geometry, **kwargs)


@multithreading_enabled
def delaunay_triangles(
    geometry: MaybeGeometryArrayNLike,
    tolerance: MaybeArrayNLike[float] = 0.0,
    only_edges: MaybeArrayNLike[bool] = False,
    **kwargs
) -> MaybeArrayN["GeometryCollection"]:
    """Creates the Delaunay triangulation and returns a list of geometries

    The source may be any geometry type. All vertices of the geometry will be
    used as the points of the triangulation.

    From the GEOS documentation:
    tolerance is the snapping tolerance used to improve the robustness of
    the triangulation computation. A tolerance of 0.0 specifies that no
    snapping will take place.

    If edges is False, a list of Polygons (triangles) will be returned.
    Otherwise, the list of LineString edges is returned.
    """

    """Computes a Delaunay triangulation around the vertices of an input
    geometry.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see only_edges). Returns a None if an input geometry
    contains less than 3 vertices.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
        snapping tolerance used to improve the robustness of the triangulation
        computation. A tolerance of 0.0 specifies that no snapping will take place.
    only_edges : bool or array_like, default False
        If False, the triangulation will return a collection of polygons (triangles).
        Otherwise, a collection of LineString edges.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Polygon
    >>> points = MultiPoint([(50, 30), (60, 30), (100, 100)])
    >>> delaunay_triangles(points)
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(points, only_edges=True)
    <MULTILINESTRING ((50 30, 100 100), (50 30, 60 30), ...>
    >>> delaunay_triangles(MultiPoint([(50, 30), (51, 30), (60, 30), (100, 100)]), \
tolerance=2)
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(Polygon([(50, 30), (60, 30), (100, 100), (50, 30)]))
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(LineString([(50, 30), (60, 30), (100, 100)]))
    <GEOMETRYCOLLECTION (POLYGON ((50 30, 60 30, 100 100, 50 30)))>
    >>> delaunay_triangles(GeometryCollection([]))
    <GEOMETRYCOLLECTION EMPTY>
    """
    return lib.delaunay_triangles(geometry, tolerance, only_edges, **kwargs)


@multithreading_enabled
def envelope(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Computes the minimum bounding box that encloses an input geometry.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point
    >>> envelope(LineString([(0, 0), (10, 10)]))
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>
    >>> envelope(MultiPoint([(0, 0), (10, 10)]))
    <POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))>
    >>> envelope(Point(0, 0))
    <POINT (0 0)>
    >>> envelope(GeometryCollection([]))
    <POINT EMPTY>
    """
    return lib.envelope(geometry, **kwargs)


@multithreading_enabled
def extract_unique_points(
    geometry: MaybeGeometryArrayNLike, **kwargs
) -> MaybeArrayN["MultiPoint"]:
    """Returns all distinct vertices of an input geometry as a multipoint.

    Note that only 2 dimensions of the vertices are considered when testing
    for equality.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, Point, Polygon
    >>> extract_unique_points(Point(0, 0))
    <MULTIPOINT (0 0)>
    >>> extract_unique_points(LineString([(0, 0), (1, 1), (1, 1)]))
    <MULTIPOINT (0 0, 1 1)>
    >>> extract_unique_points(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <MULTIPOINT (0 0, 1 0, 1 1, 0 1)>
    >>> extract_unique_points(MultiPoint([(0, 0), (1, 1), (0, 0)]))
    <MULTIPOINT (0 0, 1 1)>
    >>> extract_unique_points(LineString())
    <MULTIPOINT EMPTY>
    """
    return lib.extract_unique_points(geometry, **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def build_area(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Creates an areal geometry formed by the constituent linework of given geometry.

    Equivalent of the PostGIS ST_BuildArea() function.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, Polygon
    >>> polygon1 = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    >>> polygon2 = Polygon([(1, 1), (1, 2), (2, 2), (1, 1)])
    >>> build_area(GeometryCollection([polygon1, polygon2]))
    <POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 2 2, 1 2, 1 1))>
    """
    return lib.build_area(geometry, **kwargs)


@requires_geos("3.8.0")
@multithreading_enabled
def make_valid(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Returns repaired geometries according to the GEOS MakeValid algorithm.

    If the input geometry is already valid, then it will be returned.

    If the geometry must be split into multiple parts of the same type to be made valid,
    then a multi-part geometry will be returned.

    If the geometry must be split into multiple parts of different types to be made valid,
    then a GeometryCollection will be returned.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
        Shapely geometry object(s) which should be made valid. If the object is already valid,
        it will be returned as-is.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    MaybeGeometryArrayN
        The input geometry object(s), made valid according to the GEOS MakeValid algorithm.

    Examples
    --------
    >>> from shapely import is_valid, Polygon
    >>> polygon = Polygon([(0, 0), (1, 1), (1, 2), (1, 1), (0, 0)])
    >>> is_valid(polygon)
    False
    >>> make_valid(polygon)
    <MULTILINESTRING ((0 0, 1 1), (1 1, 1 2))>
    """
    return lib.make_valid(geometry, **kwargs)


@multithreading_enabled
def normalize(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Converts Geometry to normal form (or canonical form).

    This method orders the coordinates, rings of a polygon and parts of
    multi geometries consistently. Typically useful for testing purposes
    (for example in combination with ``equals_exact``).

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import MultiLineString
    >>> line = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    >>> normalize(line)
    <MULTILINESTRING ((2 2, 3 3), (0 0, 1 1))>
    """
    return lib.normalize(geometry, **kwargs)


@multithreading_enabled
def point_on_surface(
    geometry: MaybeGeometryArrayNLike, **kwargs
) -> MaybeArrayN["Point"]:
    """Returns a point that intersects (guaranteed to be within) an input geometry, cheaply.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, Polygon
    >>> point_on_surface(Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]))
    <POINT (5 5)>
    >>> point_on_surface(LineString([(0, 0), (2, 2), (10, 10)]))
    <POINT (2 2)>
    >>> point_on_surface(MultiPoint([(0, 0), (10, 10)]))
    <POINT (0 0)>
    >>> point_on_surface(Polygon())
    <POINT EMPTY>
    """
    return lib.point_on_surface(geometry, **kwargs)


@multithreading_enabled
def node(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeArrayN["MultiLineString"]:
    """
    Returns the fully noded version of the linear input as MultiLineString.

    Given a linear input geometry, this function returns a new MultiLineString
    in which no lines cross each other but only touch at and points. To
    obtain this, all intersections between segments are computed and added
    to the segments, and duplicate segments are removed.

    Non-linear input (points) will result in an empty MultiLineString.

    This function can for example be used to create a fully-noded linework
    suitable to passed as input to ``polygonize``.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, Point
    >>> line = LineString([(0, 0), (1,1), (0, 1), (1, 0)])
    >>> node(line)
    <MULTILINESTRING ((0 0, 0.5 0.5), (0.5 0.5, 1 1, 0 1, 0.5 0.5), (0.5 0.5, 1 0))>
    >>> node(Point(1, 1))
    <MULTILINESTRING EMPTY>
    """
    return lib.node(geometry, **kwargs)


def polygonize(
    geometries: GeometryArrayNLike, **kwargs
) -> MaybeArrayN["GeometryCollection"]:
    """Creates polygons formed from the linework of a set of Geometries.

    Polygonizes an array of Geometries that contain linework which
    represents the edges of a planar graph. Any type of Geometry may be
    provided as input; only the constituent lines and rings will be used to
    create the output polygons.

    Lines or rings that when combined do not completely close a polygon
    will result in an empty GeometryCollection.  Duplicate segments are
    ignored.

    This function returns the polygons within a GeometryCollection.
    Individual Polygons can be obtained using ``get_geometry`` to get
    a single polygon or ``get_parts`` to get an array of polygons.
    MultiPolygons can be constructed from the output using
    ``shapely.multipolygons(shapely.get_parts(shapely.polygonize(geometries)))``.

    Parameters
    ----------
    geometries : GeometryArrayLike
        An array of geometries.
    axis : int
        Axis along which the geometries are polygonized.
        The default is to perform a reduction over the last dimension
        of the input array. A 1D array results in a scalar geometry.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    GeometryCollection or array of GeometryCollections

    See Also
    --------
    get_parts, get_geometry
    polygonize_full
    node

    Examples
    --------
    >>> from shapely import LineString
    >>> lines = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(0, 0), (0, 1)]),
    ...     LineString([(0, 1), (1, 1)])
    ... ]
    >>> polygonize(lines)
    <GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))>
    """
    return lib.polygonize(geometries, **kwargs)


def polygonize_full(
    geometries: GeometryArrayLike, **kwargs
) -> Tuple4["GeometryCollection"]:
    """Creates polygons formed from the linework of a set of Geometries and
    return all extra leftover geometries as well.

    Polygonizes an array of Geometries that contain linework which
    represents the edges of a planar graph. Any type of Geometry may be
    provided as input; only the constituent lines and rings will be used to
    create the output polygons.

    This function performs the same polygonization as ``polygonize`` but does
    not only return the polygonal result but all extra outputs as well. The
    return value consists of 4 elements:

    * The polygonal valid output.
    * **Cut edges**: edges connected on both ends but not part of polygonal output.
    * **dangles**: edges with one or both ends that are not incident on another edge endpoint.
    * **invalid rings**: polygons formed but which are not valid (bowties, etc).

    This function returns a tuple of 4 GeometryCollections.
    Individual geometries can be obtained using ``get_geometry`` to get
    a single geometry or ``get_parts`` to get an array of geometries.

    Parameters
    ----------
    geometries : GeometryArrayLike
        An array of geometries.
    axis : int
        Axis along which the geometries are polygonized.
        The default is to perform a reduction over the last dimension
        of the input array. A 1D array results in a scalar geometry.
        Note: axis keyword is passed to the ufunc via **kwargs (NumPy>=1.15).
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    (polygons, cuts, dangles, invalid)
        tuple of 4 GeometryCollections or arrays of GeometryCollections

    See Also
    --------
    polygonize

    Examples
    --------
    >>> from shapely import LineString
    >>> lines = [
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(0, 0), (0, 1), (1, 1)]),
    ...     LineString([(0, 1), (1, 1)])
    ... ]
    >>> polygonize_full(lines)  # doctest: +NORMALIZE_WHITESPACE
    (<GEOMETRYCOLLECTION (POLYGON ((1 1, 0 0, 0 1, 1 1)))>,
     <GEOMETRYCOLLECTION EMPTY>,
     <GEOMETRYCOLLECTION (LINESTRING (0 1, 1 1))>,
     <GEOMETRYCOLLECTION EMPTY>)
    """
    return lib.polygonize_full(geometries, **kwargs)


@requires_geos("3.11.0")
@multithreading_enabled
def remove_repeated_points(
    geometry: MaybeGeometryArrayNLike, tolerance: MaybeArrayNLike[float] = 0.0, **kwargs
) -> MaybeGeometryArrayN:
    """Returns a copy of a Geometry with repeated points removed.

    From the start of the coordinate sequence, each next point within the
    tolerance is removed.

    Removing repeated points with a non-zero tolerance may result in an invalid
    geometry being returned.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    tolerance : float or array_like, default=0.0
        Use 0.0 to remove only exactly repeated points.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> remove_repeated_points(LineString([(0,0), (0,0), (1,0)]), tolerance=0)
    <LINESTRING (0 0, 1 0)>
    >>> remove_repeated_points(Polygon([(0, 0), (0, .5), (0, 1), (.5, 1), (0,0)]), tolerance=.5)
    <POLYGON ((0 0, 0 1, 0 0, 0 0))>
    """
    return lib.remove_repeated_points(geometry, tolerance, **kwargs)


@requires_geos("3.7.0")
@multithreading_enabled
def reverse(geometry: MaybeGeometryArrayNLike, **kwargs) -> MaybeGeometryArrayN:
    """Returns a copy of a Geometry with the order of coordinates reversed.

    If a Geometry is a polygon with interior rings, the interior rings are also
    reversed.

    Points are unchanged. None is returned where Geometry is None.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    See also
    --------
    is_ccw : Checks if a Geometry is clockwise.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> reverse(LineString([(0, 0), (1, 2)]))
    <LINESTRING (1 2, 0 0)>
    >>> reverse(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    <POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))>
    >>> reverse(None) is None
    True
    """

    return lib.reverse(geometry, **kwargs)


@requires_geos("3.10.0")
@multithreading_enabled
def segmentize(
    geometry: MaybeGeometryArrayNLike,
    max_segment_length: MaybeArrayNLike[float],
    **kwargs
) -> MaybeGeometryArrayN:
    """Adds vertices to line segments based on maximum segment length.

    Additional vertices will be added to every line segment in an input geometry
    so that segments are no longer than the provided maximum segment length. New
    vertices will evenly subdivide each segment.

    Only linear components of input geometries are densified; other geometries
    are returned unmodified.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    max_segment_length : float or array_like
        Additional vertices will be added so that all line segments are no
        longer than this value.  Must be greater than 0.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (0, 10)])
    >>> segmentize(line, max_segment_length=5)
    <LINESTRING (0 0, 0 5, 0 10)>
    >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    >>> segmentize(polygon, max_segment_length=5)
    <POLYGON ((0 0, 5 0, 10 0, 10 5, 10 10, 5 10, 0 10, 0 5, 0 0))>
    >>> segmentize(None, max_segment_length=5) is None
    True
    """
    return lib.segmentize(geometry, max_segment_length, **kwargs)


@multithreading_enabled
def simplify(
    geometry: MaybeGeometryArrayNLike,
    tolerance: MaybeArrayNLike[float],
    preserve_topology: bool = True,
    **kwargs
) -> MaybeGeometryArrayN:
    """Returns a simplified version of an input geometry using the
    Douglas-Peucker algorithm.

    Coordinates of the simplified geometry will be no more than the
    tolerance distance from the original. Unless the topology preserving
    option is used, the algorithm may produce self-intersecting or
    otherwise invalid geometries.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    tolerance : float or array_like
        The maximum allowed geometry displacement. The higher this value, the
        smaller the number of vertices in the resulting geometry.
    preserve_topology : bool, default True
        By default (True), the operation will avoid creating invalid
        geometries (checking for collapses, ring-intersections, etc), but
        this is computationally more expensive.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import LineString, Polygon
    >>> line = LineString([(0, 0), (1, 10), (0, 20)])
    >>> simplify(line, tolerance=0.9)
    <LINESTRING (0 0, 1 10, 0 20)>
    >>> simplify(line, tolerance=1)
    <LINESTRING (0 0, 0 20)>
    >>> polygon_with_hole = Polygon(
    ...     [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    ...     holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]
    ... )
    >>> simplify(polygon_with_hole, tolerance=4, preserve_topology=True)
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 2 4, 4 4, 4 2...>
    >>> simplify(polygon_with_hole, tolerance=4, preserve_topology=False)
    <POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))>
    """
    if preserve_topology:
        return lib.simplify_preserve_topology(geometry, tolerance, **kwargs)
    else:
        return lib.simplify(geometry, tolerance, **kwargs)


@multithreading_enabled
def snap(
    geometry: MaybeGeometryArrayNLike,
    reference: MaybeGeometryArrayNLike,
    tolerance: MaybeArrayNLike[float],
    **kwargs
) -> MaybeGeometryArrayN:
    """Snaps an input geometry to reference geometry's vertices.

    Vertices of the first geometry are snapped to vertices of the second.
    geometry. The input geometries are not modified.
    The result geometry is the input geometry with the vertices snapped.
    If no snapping occurs then the input geometry is returned unchanged.
    The tolerance is used to control where snapping is performed.

    Snapping one geometry to another can improve robustness for overlay
    operations by eliminating nearly-coincident edges
    (which cause problems during noding and intersection calculation).

    Too much snapping can result in invalid topology being created,
    so the number and location of snapped vertices is decided using
    heuristics to determine when it is safe to snap.
    This can result in some potential snaps being omitted, however.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    reference : MaybeGeometryArrayNLike
    tolerance : float or array_like
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import snap, distance, LineString, Point, Polygon, MultiPolygon
    >>> point = Point(0.5, 2.5)
    >>> target_point = Point(0, 2)
    >>> snap(point, target_point, tolerance=1)
    <POINT (0 2)>
    >>> snap(point, target_point, tolerance=0.49)
    <POINT (0.5 2.5)>
    >>> polygon = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    >>> snap(polygon, Point(8, 10), tolerance=5)
    <POLYGON ((0 0, 0 10, 8 10, 10 0, 0 0))>
    >>> snap(polygon, LineString([(8, 10), (8, 0)]), tolerance=5)
    <POLYGON ((0 0, 0 10, 8 10, 8 0, 0 0))>
    >>> # example from: https://postgis.net/docs/ST_Snap.html
    >>> poly = MultiPolygon([
    ...     ([(26, 125), (26, 200), (126, 200), (126, 125), (26, 125)],
    ...      [[(51, 150), (101, 150), (76, 175), (51, 150)]]),
    ...     ([(151, 100), (151, 200), (176, 175), (151, 100)], [])
    ... ])
    >>> line = LineString([(5, 107), (54, 84), (101, 100)])
    >>> snap(poly, line, distance(poly, line) * 1.01)
    <MULTIPOLYGON (((26 125, 26 200, 126 200, 126 125, 101 100, 26 125), (51 150...>
    >>> snap(poly, line, distance(poly, line) * 1.25)
    <MULTIPOLYGON (((5 107, 26 200, 126 200, 126 125, 101 100, 54 84, 5 107), (5...>
    >>> snap(line, poly, distance(poly, line) * 1.01)
    <LINESTRING (5 107, 26 125, 54 84, 101 100)>
    >>> snap(line, poly, distance(poly, line) * 1.25)
    <LINESTRING (26 125, 54 84, 101 100)>
    """
    return lib.snap(geometry, reference, tolerance, **kwargs)


@multithreading_enabled
def voronoi_polygons(
    geometry: MaybeGeometryArrayNLike,
    tolerance: MaybeArrayNLike[float] = 0.0,
    extend_to: Optional[MaybeArrayNLike["BaseGeometry"]] = None,
    only_edges: MaybeArrayNLike[bool] = False,
    **kwargs
) -> MaybeGeometryArrayN:
    """Computes a Voronoi diagram [1] from the vertices of an input geometries.

    The output is a geometrycollection containing polygons (default)
    or linestrings (see only_edges). Returns empty if an input geometry
    contains less than 2 vertices or if the provided extent has zero area.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
        The input geometries whose vertices will be used to calculate
        the final diagram.
    tolerance : float or array_like, default 0.0
        Snap input vertices together if their distance is less than this value.
        Snapping tolerance is used to improve the robustness of the computation.
        A tolerance of 0.0 specifies that no snapping will take place.
    extend_to : MaybeGeometryArrayNLike, optional
        If None, the clipping envelope will be automatically determined.
        Otherwise, the diagram will be extended to cover the envelope of this
        geometry (unless this envelope is smaller than the input geometry).
    only_edges : bool or array_like, default False
        If False, the triangulation will return a collection of polygons.
        Otherwise, a collection of LineString edges.
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    GEOS documentation can be found at [2]

    Returns
    -------
    GeometryCollection
        geometries representing the Voronoi regions.

    Notes
    -----
    The tolerance `argument` can be finicky and is known to cause the
    algorithm to fail in several cases. If you're using `tolerance`
    and getting a failure, try removing it. The test cases in
    tests/test_voronoi_diagram.py show more details.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Voronoi_diagram
    [2] https://geos.osgeo.org/doxygen/geos__c_8h_source.html  (line 730)

    Examples
    --------
    >>> from shapely import LineString, MultiPoint, normalize, Point
    >>> points = MultiPoint([(2, 2), (4, 2)])
    >>> normalize(voronoi_polygons(points))
    <GEOMETRYCOLLECTION (POLYGON ((3 0, 3 4, 6 4, 6 0, 3 0)), POLYGON ((0 0, 0 4...>
    >>> voronoi_polygons(points, only_edges=True)
    <LINESTRING (3 4, 3 0)>
    >>> voronoi_polygons(MultiPoint([(2, 2), (4, 2), (4.2, 2)]), 0.5, only_edges=True)
    <LINESTRING (3 4.2, 3 -0.2)>
    >>> voronoi_polygons(points, extend_to=LineString([(0, 0), (10, 10)]), only_edges=True)
    <LINESTRING (3 10, 3 0)>
    >>> voronoi_polygons(LineString([(2, 2), (4, 2)]), only_edges=True)
    <LINESTRING (3 4, 3 0)>
    >>> voronoi_polygons(Point(2, 2))
    <GEOMETRYCOLLECTION EMPTY>
    """
    return lib.voronoi_polygons(geometry, tolerance, extend_to, only_edges, **kwargs)


@requires_geos("3.6.0")
@multithreading_enabled
def oriented_envelope(
    geometry: MaybeGeometryArrayNLike, **kwargs
) -> MaybeGeometryArrayN:
    """Computes the oriented envelope (minimum rotated rectangle)
    that encloses an input geometry.

    Unlike envelope this rectangle is not constrained to be parallel to the
    coordinate axes. If the convex hull of the object is a degenerate (line
    or point) this degenerate is returned.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> oriented_envelope(MultiPoint([(0, 0), (10, 0), (10, 10)]))
    <POLYGON ((0 0, 5 -5, 15 5, 10 10, 0 0))>
    >>> oriented_envelope(LineString([(1, 1), (5, 1), (10, 10)]))
    <POLYGON ((1 1, 3 -1, 12 8, 10 10, 1 1))>
    >>> oriented_envelope(Polygon([(1, 1), (15, 1), (5, 10), (1, 1)]))
    <POLYGON ((15 1, 15 10, 1 10, 1 1, 15 1))>
    >>> oriented_envelope(LineString([(1, 1), (10, 1)]))
    <LINESTRING (1 1, 10 1)>
    >>> oriented_envelope(Point(2, 2))
    <POINT (2 2)>
    >>> oriented_envelope(GeometryCollection([]))
    <POLYGON EMPTY>
    """
    return lib.oriented_envelope(geometry, **kwargs)


minimum_rotated_rectangle = oriented_envelope


@requires_geos("3.8.0")
@multithreading_enabled
def minimum_bounding_circle(
    geometry: MaybeGeometryArrayNLike, **kwargs
) -> MaybeArrayNLike[Union["Polygon", "Point", "EmptyGeometry"]]:
    """Computes the minimum bounding circle that encloses an input geometry.

    Parameters
    ----------
    geometry : MaybeGeometryArrayNLike
    **kwargs
        For other keyword-only arguments, see the
        `NumPy ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Examples
    --------
    >>> from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
    >>> minimum_bounding_circle(Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]))
    <POLYGON ((12.071 5, 11.935 3.621, 11.533 2.294, 10.879 1.07...>
    >>> minimum_bounding_circle(LineString([(1, 1), (10, 10)]))
    <POLYGON ((11.864 5.5, 11.742 4.258, 11.38 3.065, 10.791 1.9...>
    >>> minimum_bounding_circle(MultiPoint([(2, 2), (4, 2)]))
    <POLYGON ((4 2, 3.981 1.805, 3.924 1.617, 3.831 1.444, 3.707...>
    >>> minimum_bounding_circle(Point(0, 1))
    <POINT (0 1)>
    >>> minimum_bounding_circle(GeometryCollection([]))
    <POLYGON EMPTY>

    See also
    --------
    minimum_bounding_radius
    """
    return lib.minimum_bounding_circle(geometry, **kwargs)
