from typing import Any, Iterable, Sequence, Union

import numpy as np

from . import lib
from .decorators import requires_geos, UnsupportedGEOSOperation
from .enum import ParamEnum
from .geometry.base import BaseGeometry

__all__ = ["STRtree"]


class BinaryPredicate(ParamEnum):
    """The enumeration of GEOS binary predicates types"""

    intersects = 1
    within = 2
    contains = 3
    overlaps = 4
    crosses = 5
    touches = 6
    covers = 7
    covered_by = 8
    contains_properly = 9


# DEPRECATED: to be removed on a future release
VALID_PREDICATES = {e.name for e in BinaryPredicate}


class STRtree:
    """
    A query-only R-tree spatial index created using the
    Sort-Tile-Recursive (STR) [1]_ algorithm.

    For two-dimensional spatial data. The tree is constructed directly
    at initialization. The tree is immutable and query-only, meaning that
    once created nodes cannot be added or removed.

    An index is initialized from a sequence of geometry objects and
    optionally a sequence of items. If items are not provided, the
    indices of the geometry sequence will be used instead.

    Stored items and corresponding geometry objects can be spatially
    queried using another geometric object.

    Parameters
    ----------
    geoms : sequence
        A sequence of geometry objects.
    items : sequence, optional
        A sequence of objects which typically serve as identifiers in an
        application. This sequence must have the same length as geoms.
    leafsize : int, default 10
        the maximum number of child nodes that a node can have

    Attributes
    ----------
    node_capacity : int
        The maximum number of items per node. Default: 10.

    Examples
    --------
    Creating an index of polygons:

    >>> from shapely.strtree import STRtree
    >>> from shapely.geometry import Polygon
    >>>
    >>> polys = [Polygon(((0, 0), (1, 0), (1, 1))),
    ...          Polygon(((0, 1), (0, 0), (1, 0))),
    ...          Polygon(((100, 100), (101, 100), (101, 101)))]
    >>> tree = STRtree(polys)
    >>> query_geom = Polygon(((-1, -1), (2, 0), (2, 2), (-1, 2)))
    >>> result = tree.query(query_geom)
    >>> polys[0] in result
    True
    >>> polys[1] in result
    True
    >>> polys[2] in result
    False

    >>> import shapely
    >>> tree = shapely.STRtree(shapely.points(np.arange(10), np.arange(10)))
    >>> # Query geometries that overlap envelope of input geometries:
    >>> tree.query(shapely.box(2, 2, 4, 4)).tolist()
    [2, 3, 4]
    >>> # Query geometries that are contained by input geometry:
    >>> tree.query(shapely.box(2, 2, 4, 4), predicate='contains').tolist()
    [3]
    >>> # Query geometries that overlap envelopes of ``geoms``
    >>> tree.query_bulk([shapely.box(2, 2, 4, 4), shapely.box(5, 5, 6, 6)]).tolist()
    [[0, 0, 0, 1, 1], [2, 3, 4, 5, 6]]
    >>> tree.nearest( shapely.points(1,1), shapely.points(3,5)]).tolist()  # doctest: +SKIP
    [[0, 1], [1, 4]]

    References
    ----------
    .. [1] Leutenegger, Scott T.; Edgington, Jeffrey M.; Lopez, Mario A.
       (February 1997). "STR: A Simple and Efficient Algorithm for
       R-Tree Packing".
       https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf

    """

    def __init__(
        self,
        geoms: Iterable[BaseGeometry],
        items: Iterable[Any] = None,
        leafsize: int = 10,
    ):
        # Keep references to geoms
        self.geometries = np.asarray(geoms, dtype=np.object_)

        # initialize GEOS STRtree
        self._tree = lib.STRtree(self.geometries, leafsize)

        # handle items
        self._has_custom_items = items is not None
        if self._has_custom_items:
            items = np.asarray(items)
        else:
            # should never be accessed
            items = None
        self._items = items

    def __len__(self):
        return self._tree.count

    def __reduce__(self):
        if self._has_custom_items:
            return (STRtree, (self.geometries, self._items))
        else:
            return (STRtree, (self.geometries,))

    def query_items(
        self, geom: BaseGeometry, predicate=None, distance=None
    ) -> Sequence[Any]:
        """
        Return the index of all geometries in the tree with extents that
        intersect the envelope of the input geometry.

        Items are integers serving as identifiers for an application.

        If predicate is provided, a prepared version of the input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        The 'dwithin' predicate requires GEOS >= 3.10.

        If geometry is None, an empty array is returned.

        Parameters
        ----------
        geom : Geometry
            The envelope of the geometry is taken automatically for
            querying the tree.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses',\
'touches', 'covers', 'covered_by', 'contains_properly', 'dwithin'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.
        distance : number, optional
            Distance around the geometry within which to query the tree for the
            'dwithin' predicate.  Required if predicate='dwithin'.

        Returns
        -------
        ndarray
            Indexes of geometries in tree
        Note
        ----
        A geometry object's "envelope" is its minimum xy bounding
        rectangle.

        Examples
        --------
        >>> import shapely
        >>> tree = shapely.STRtree(shapely.points(np.arange(10), np.arange(10)))
        >>> tree.query(shapely.box(1,1, 3,3)).tolist()
        [1, 2, 3]
        >>> # Query geometries that are contained by input geometry
        >>> tree.query(shapely.box(2, 2, 4, 4), predicate='contains').tolist()
        [3]
        >>> # Query geometries within 1 unit distance of input geometry
        >>> tree.query(shapely.points(0.5, 0.5), predicate='dwithin', distance=1.0).tolist()  # doctest: +SKIP
        [0, 1]

        A buffer around a point can be used to control the extent
        of the query.

        >>> from shapely.strtree import STRtree
        >>> from shapely.geometry import Point
        >>> points = [Point(i, i) for i in range(10)]
        >>> tree = STRtree(points)
        >>> query_geom = Point(2,2).buffer(0.99)
        >>> [o.wkt for o in tree.query(query_geom)]
        ['POINT (2 2)']
        >>> query_geom = Point(2, 2).buffer(1.0)
        >>> [o.wkt for o in tree.query(query_geom)]
        ['POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)']

        A subsequent search through the returned subset using the
        desired binary predicate (eg. intersects, crosses, contains,
        overlaps) may be necessary to further filter the results
        according to their specific spatial relationships.

        >>> [o.wkt for o in tree.query(query_geom) if o.intersects(query_geom)]
        ['POINT (2 2)']

        """
        if geom is None:
            result = np.array([], dtype=np.intp)

        elif predicate is None:
            result = self._tree.query(geom, 0)

        elif predicate == "dwithin":
            if lib.geos_version < (3, 10, 0):
                raise UnsupportedGEOSOperation(
                    "dwithin predicate requires GEOS >= 3.10"
                )
            if distance is None:
                raise ValueError(
                    "distance parameter must be provided for dwithin predicate"
                )
            if not np.isscalar(distance):
                raise ValueError("distance must be a scalar value")

            geometry = np.array([geom])
            distance = np.array([distance], dtype="float64")
            result = self._tree.dwithin(geometry, distance)[1]

        else:
            predicate = BinaryPredicate.get_value(predicate)
            result = self._tree.query(geom, predicate)

        if self._has_custom_items:
            return self._items[result]
        else:
            return result

    def query_geoms(self, geom: BaseGeometry) -> Sequence[BaseGeometry]:
        """Query for nodes which intersect the geom's envelope to get
        geometries corresponding to the items stored in the nodes.

        Parameters
        ----------
        geom : geometry object
            The query geometry.

        Returns
        -------
        An array of geometry objects.

        """
        result = self._tree.query(geom, 0)
        return self.geometries[result]

    def query(self, geom: BaseGeometry) -> Sequence[BaseGeometry]:
        """Query for nodes which intersect the geom's envelope to get
        geometries corresponding to the items stored in the nodes.

        This method is an alias for query_geoms. It may be removed in
        version 2.0.

        Parameters
        ----------
        geom : geometry object
            The query geometry.

        Returns
        -------
        An array of geometry objects.

        """
        return self.query_geoms(geom)

    def query_bulk(self, geometry, predicate=None, distance=None):
        """Returns all combinations of each input geometry and geometries in the tree
        where the envelope of each input geometry intersects with the envelope of a
        tree geometry.

        If predicate is provided, a prepared version of each input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        The 'dwithin' predicate requires GEOS >= 3.10.

        This returns an array with shape (2,n) where the subarrays correspond
        to the indexes of the input geometries and indexes of the tree geometries
        associated with each.  To generate an array of pairs of input geometry
        index and tree geometry index, simply transpose the results.

        In the context of a spatial join, input geometries are the "left" geometries
        that determine the order of the results, and tree geometries are "right" geometries
        that are joined against the left geometries.  This effectively performs
        an inner join, where only those combinations of geometries that can be joined
        based on envelope overlap or optional predicate are returned.

        Any geometry that is None or empty in the input geometries is omitted from
        the output.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.  The envelope of each geometry
            is automatically calculated for querying the tree.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses',\
'touches', 'covers', 'covered_by', 'contains_properly', 'dwithin'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.
        distance : number or array_like, optional
            Distances around each input geometry within which to query the tree
            for the 'dwithin' predicate.  If array_like, shape must be
            broadcastable to shape of geometry.  Required if predicate='dwithin'.

        Returns
        -------
        ndarray with shape (2, n)
            The first subarray contains input geometry indexes.
            The second subarray contains tree geometry indexes.

        Examples
        --------
        >>> import shapely
        >>> tree = shapely.STRtree(shapely.points(np.arange(10), np.arange(10)))
        >>> tree.query_bulk([shapely.box(2, 2, 4, 4), shapely.box(5, 5, 6, 6)]).tolist()
        [[0, 0, 0, 1, 1], [2, 3, 4, 5, 6]]
        >>> # Query for geometries that contain tree geometries
        >>> tree.query_bulk([shapely.box(2, 2, 4, 4), shapely.box(5, 5, 6, 6)], predicate='contains').tolist()
        [[0], [3]]
        >>> # To get an array of pairs of index of input geometry, index of tree geometry,
        >>> # transpose the output:
        >>> tree.query_bulk([shapely.box(2, 2, 4, 4), shapely.box(5, 5, 6, 6)]).T.tolist()
        [[0, 2], [0, 3], [0, 4], [1, 5], [1, 6]]
        >>> # Query for tree geometries within 1 unit distance of input geometries
        >>> tree.query_bulk([shapely.points(0.5, 0.5)], predicate='dwithin', distance=1.0).tolist()  # doctest: +SKIP
        [[0, 0], [0, 1]]
        """

        geometry = np.asarray(geometry)
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)

        if predicate is None:
            return self._tree.query_bulk(geometry, 0)

        # Requires GEOS >= 3.10
        elif predicate == "dwithin":
            if lib.geos_version < (3, 10, 0):
                raise UnsupportedGEOSOperation(
                    "dwithin predicate requires GEOS >= 3.10"
                )
            if distance is None:
                raise ValueError(
                    "distance parameter must be provided for dwithin predicate"
                )
            distance = np.asarray(distance, dtype="float64")
            if distance.ndim > 1:
                raise ValueError("Distance array should be one dimensional")

            try:
                distance = np.broadcast_to(distance, geometry.shape)
            except ValueError:
                raise ValueError("Could not broadcast distance to match geometry")

            return self._tree.dwithin(geometry, distance)

        predicate = BinaryPredicate.get_value(predicate)
        return self._tree.query_bulk(geometry, predicate)

    @requires_geos("3.6.0")
    def nearest_bulk(self, geometry, exclusive: bool = False):
        """Returns the index of the nearest item in the tree for each input
        geometry.

        If there are multiple equidistant or intersected geometries in the tree,
        only a single result is returned for each input geometry, based on the
        order that tree geometries are visited; this order may be
        nondeterministic.

        Any geometry that is None or empty in the input geometries is omitted
        from the output.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.

        Returns
        -------
        ndarray with shape (2, n)
            The first subarray contains input geometry indexes.
            The second subarray contains tree geometry indexes.

        See also
        --------
        nearest_all: returns all equidistant geometries and optional distances

        Examples
        --------
        >>> import shapely
        >>> tree = shapely.STRtree(shapely.points(np.arange(10), np.arange(10)))
        >>> tree.nearest(shapely.points(1,1)).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> tree.nearest( shapely.box(1,1,3,3)]).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> points = shapely.points(0.5,0.5)
        >>> tree.nearest([None, shapely.points(10,10)]).tolist()  # doctest: +SKIP
        [[1], [9]]
        """
        # TODO(shapely-2.0)
        if exclusive:
            raise NotImplementedError(
                "The `exclusive` keyword is not yet implemented for Shapely 2.0"
            )

        geometry = np.asarray(geometry, dtype=object)
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)

        return self._tree.nearest(geometry)

    def _nearest_idx(self, geom: BaseGeometry, exclusive: bool = False) -> int:
        indices = self.nearest_bulk(geom, exclusive=exclusive)
        # nearest returns ndarray with shape (2, 1) -> index in input
        # geometries and index into tree geometries
        return indices[1, 0]

    @requires_geos("3.6.0")
    def nearest_item(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[Any, None]:
        """Query the tree for the node nearest to geom and get the item
        stored in the node.

        Items are integers serving as identifiers for an application.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the item corresponding to the given geom
            from results or not.  Default: False.

        Returns
        -------
        Stored item or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        Examples
        --------
        >>> from shapely.strtree import STRtree
        >>> from shapely.geometry import Point
        >>> tree = STRtree([Point(i, i) for i in range(10)])
        >>> tree.nearest(Point(2.2, 2.2)).wkt
        'POINT (2 2)'

        Will only return one object:

        >>> tree = STRtree ([Point(0, 0), Point(0, 0)])
        >>> tree.nearest(Point(0, 0)).wkt
        'POINT (0 0)'

        """
        if self._tree.count == 0:
            return None

        result = self._nearest_idx(geom, exclusive)
        if self._has_custom_items:
            return self._items[result]
        else:
            return result

    @requires_geos("3.6.0")
    def nearest_geom(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[BaseGeometry, None]:
        """Query the tree for the node nearest to geom and get the
        geometry corresponding to the item stored in the node.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the given geom from results or not.
            Default: False.

        Returns
        -------
        BaseGeometry or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        """
        if self._tree.count == 0:
            return None

        result = self._nearest_idx(geom, exclusive)
        return self.geometries[result]

    @requires_geos("3.6.0")
    def nearest(
        self, geom: BaseGeometry, exclusive: bool = False
    ) -> Union[BaseGeometry, None]:
        """Query the tree for the node nearest to geom and get the
        geometry corresponding to the item stored in the node.

        This method is an alias for nearest_geom. It may be removed in
        version 2.0.

        Parameters
        ----------
        geom : geometry object
            The query geometry.
        exclusive : bool, optional
            Whether to exclude the given geom from results or not.
            Default: False.

        Returns
        -------
        BaseGeometry or None.

        None is returned if this index is empty. This may change in
        version 2.0.

        """
        return self.nearest_geom(geom, exclusive=exclusive)

    @requires_geos("3.6.0")
    def nearest_all(self, geometry, max_distance=None, return_distance=False):
        """Returns the index of the nearest item(s) in the tree for each input
        geometry.

        If there are multiple equidistant or intersected geometries in tree, all
        are returned.  Tree indexes are returned in the order they are visited
        for each input geometry and may not be in ascending index order; no meaningful
        order is implied.

        The max_distance used to search for nearest items in the tree may have a
        significant impact on performance by reducing the number of input geometries
        that are evaluated for nearest items in the tree.  Only those input geometries
        with at least one tree item within +/- max_distance beyond their envelope will
        be evaluated.

        The distance, if returned, will be 0 for any intersected geometries in the tree.

        Any geometry that is None or empty in the input geometries is omitted from
        the output.

        Parameters
        ----------
        geometry : Geometry or array_like
            Input geometries to query the tree.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0.
        return_distance : bool, default False
            If True, will return distances in addition to indexes.

        Returns
        -------
        indices or tuple of (indices, distances)
            indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        See also
        --------
        nearest: returns singular nearest geometry for each input

        Examples
        --------
        >>> import shapely
        >>> tree = shapely.STRtree(shapely.points(np.arange(10), np.arange(10)))
        >>> tree.nearest_all(shapely.points(1,1)).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> tree.nearest_all( shapely.box(1,1,3,3)]).tolist()  # doctest: +SKIP
        [[0, 0, 0], [1, 2, 3]]
        >>> points = shapely.points(0.5,0.5)
        >>> index, distance = tree.nearest_all(points, return_distance=True)  # doctest: +SKIP
        >>> index.tolist()  # doctest: +SKIP
        [[0, 0], [0, 1]]
        >>> distance.round(4).tolist()  # doctest: +SKIP
        [0.7071, 0.7071]
        >>> tree.nearest_all(None).tolist()  # doctest: +SKIP
        [[], []]
        """

        geometry = np.asarray(geometry, dtype=object)
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)

        if max_distance is not None:
            if not np.isscalar(max_distance):
                raise ValueError("max_distance parameter only accepts scalar values")

            if max_distance <= 0:
                raise ValueError("max_distance must be greater than 0")

        # a distance of 0 means no max_distance is used
        max_distance = max_distance or 0

        if return_distance:
            return self._tree.nearest_all(geometry, max_distance)

        return self._tree.nearest_all(geometry, max_distance)[0]
