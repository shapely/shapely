import numpy as np

from shapely import Geometry, GeometryType, lib
from shapely._enum import ParamEnum
from shapely._geometry import get_parts
from shapely.decorators import multithreading_enabled, requires_geos

__all__ = [
    "coverage_clean",
    "coverage_invalid_edges",
    "coverage_is_valid",
    "coverage_simplify",
]


class CoverageCleanMergeStrategy(ParamEnum):
    """Enumeration of coverage clean merge strategies.

    Determines which neighboring polygon to merge overlapping areas into.

    Attributes
    ----------
    longest_border : int
        Polygon with longest common border is chosen.
    max_area : int
        Polygon with largest area is chosen.
    min_area : int
        Polygon with smallest area is chosen.
    min_index : int
        Polygon with smallest input index is chosen.

    """

    longest_border = 0
    max_area = 1
    min_area = 2
    min_index = 3


@requires_geos("3.12.0")
@multithreading_enabled
def coverage_is_valid(geometry, gap_width=0.0, **kwargs):
    """Verify if a coverage is valid.

    The coverage is represented by an array of polygonal geometries with
    exactly matching edges and no overlap.

    A valid coverage may contain holes (regions of no coverage). However,
    sometimes it might be desirable to detect narrow gaps as invalidities in
    the coverage. The `gap_width` parameter allows to specify the maximum
    width of gaps to detect. When gaps are detected, this function will
    return False and the `coverage_invalid_edges` function can be used to
    find the edges of those gaps.

    Geometries that are not Polygon or MultiPolygon are ignored.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : array_like
        Array of geometries to verify.
    gap_width : float, default 0.0
        The maximum width of gaps to detect.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    bool

    See Also
    --------
    coverage_invalid_edges, coverage_simplify

    """
    geometries = np.asarray(geometry)
    # we always consider the full array as a single coverage -> ravel the input
    # to pass a 1D array
    return lib.coverage_is_valid(geometries.ravel(order="K"), gap_width, **kwargs)


@requires_geos("3.12.0")
@multithreading_enabled
def coverage_invalid_edges(geometry, gap_width=0.0, **kwargs):
    """Verify if a coverage is valid and return invalid edges.

    This functions returns linear indicators showing the location of invalid
    edges (if any) in each polygon in the input array.

    The coverage is represented by an array of polygonal geometries with
    exactly matching edges and no overlap.

    A valid coverage may contain holes (regions of no coverage). However,
    sometimes it might be desirable to detect narrow gaps as invalidities in
    the coverage. The `gap_width` parameter allows to specify the maximum
    width of gaps to detect. When gaps are detected, the `coverage_is_valid`
    function will return False and this function can be used to find the
    edges of those gaps.

    Geometries that are not Polygon or MultiPolygon are ignored.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : array_like
        Array of geometries to verify.
    gap_width : float, default 0.0
        The maximum width of gaps to detect.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    Returns
    -------
    numpy.ndarray | shapely.Geometry

    See Also
    --------
    coverage_is_valid, coverage_simplify

    """
    geometries = np.asarray(geometry)
    # we always consider the full array as a single coverage -> ravel the input
    # to pass a 1D array
    return lib.coverage_invalid_edges(geometries.ravel(order="K"), gap_width, **kwargs)


@requires_geos("3.12.0")
@multithreading_enabled
def coverage_simplify(geometry, tolerance, *, simplify_boundary=True):
    """Return a simplified version of an input geometry using coverage simplification.

    Assumes that the geometry forms a polygonal coverage. Under this assumption, the
    function simplifies the edges using the Visvalingam-Whyatt algorithm, while
    preserving a valid coverage. In the most simplified case, polygons are reduced to
    triangles.

    A collection of valid polygons is considered a coverage if the polygons are:

    * **Non-overlapping** - polygons do not overlap (their interiors do not intersect)
    * **Edge-Matched** - vertices along shared edges are identical

    The function allows simplification of all edges including the outer boundaries of
    the coverage or simplification of only the inner (shared) edges.

    If there are other geometry types than Polygons or MultiPolygons present,
    the function will raise an error.

    If the geometry is polygonal but does not form a valid coverage due to overlaps,
    it will be simplified but it may result in invalid topology.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    geometry : Geometry or array_like
    tolerance : float or array_like
        The degree of simplification roughly equal to the square root of the area
        of triangles that will be removed.
    simplify_boundary : bool, optional
        By default (True), simplifies both internal edges of the coverage as well
        as its boundary. If set to False, only simplifies internal edges.

    Returns
    -------
    numpy.ndarray | shapely.Geometry

    See Also
    --------
    coverage_is_valid, coverage_invalid_edges

    Examples
    --------
    >>> import shapely
    >>> from shapely import Polygon
    >>> poly = Polygon([(0, 0), (20, 0), (20, 10), (10, 5), (0, 10), (0, 0)])
    >>> shapely.coverage_simplify(poly, tolerance=2)
    <POLYGON ((0 0, 20 0, 20 10, 10 5, 0 10, 0 0))>
    """
    scalar = False
    if isinstance(geometry, Geometry):
        scalar = True

    geometries = np.asarray(geometry)
    shape = geometries.shape
    geometries = geometries.ravel()

    # create_collection acts on the inner axis
    collections = lib.create_collection(
        geometries, np.intc(GeometryType.GEOMETRYCOLLECTION)
    )

    simplified = lib.coverage_simplify(collections, tolerance, simplify_boundary)
    parts = get_parts(simplified).reshape(shape)
    if scalar:
        return parts.item()
    return parts


@requires_geos("3.14.0")
@multithreading_enabled
def coverage_clean(
    geometry,
    *,
    gap_width=0.0,
    snapping_distance=-1,
    merge_strategy="longest_border",
):
    """Return a cleaned version of an input geometry.

    Assumes that the geometry forms a polygonal coverage. Ensures that no polygons
    overlap, small gaps are removed, and all shared edges are exactly identical.

    A collection of valid polygons is considered a coverage if the polygons are:

    * **Non-overlapping** - polygons do not overlap (their interiors do not intersect)
    * **Edge-Matched** - vertices along shared edges are identical

    If there are other geometry types than Polygons or MultiPolygons present,
    the function will raise an error.

    .. versionadded:: 2.2.0

    Parameters
    ----------
    geometry : Geometry or array_like
    gap_width : float, default 0.0
        Gaps smaller than this value are merged with the adjacent polygon with
        longest shared shared border. Set to 0.0 to (default) for no removal
        of gaps.
    snapping_distance : float
        Determines the node snapping step when nearby vertices are snapped
        together. Set to -1 by default, which automatically finds a small snapping
        distance based on the extend of the input. Set to 0.0 to disable snapping.
    merge_strategy : {'longest_border', 'max_area', 'min_area', 'min_index'}, default 'longest_border'
        Determines how overlapping areas are handled by choosing which polygons to merge
        them into. CoverageCleanMergeStrategy.longest_border ('longest_border') chooses
        polygon with longest common border. CoverageCleanMergeStrategy.max_area
        ('max_area') chooses polygon with largest area.
        CoverageCleanMergeStrategy.min_area ('min_area') chooses polygon with smallest
        area. CoverageCleanMergeStartegy.min_index ('min_index') chooses the first
        encountered polygon.

    Returns
    -------
    numpy.ndarray

    See Also
    --------
    coverage_is_valid, coverage_invalid_edges, coverage_simplify

    Examples
    --------
    >>> import shapely  # doctest: +SKIP
    >>> from shapely import box  # doctest: +SKIP
    >>> polygons = [box(0, 0, 1, 1), box(0.9, 0, 2, 1)]  # doctest: +SKIP
    >>> shapely.coverage_clean(polygons)  # doctest: +SKIP
    array([<POLYGON ((0 0, 0 1, 0.9 1, 1 1, 1 0, 0.9 0, 0 0))>,
       <POLYGON ((1 1, 2 1, 2 0, 1 0, 1 1))>], dtype=object)
    """  # noqa: E501
    if isinstance(merge_strategy, str):
        merge_strategy = CoverageCleanMergeStrategy.get_value(merge_strategy)

    geometries = np.asarray(geometry)
    # we always consider the full array as a single coverage -> ravel the input
    # to pass a 1D array
    return lib.coverage_clean(
        geometries.ravel(order="K"),
        gap_width,
        snapping_distance,
        np.intc(merge_strategy),
    )
