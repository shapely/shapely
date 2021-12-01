import numpy as np

from . import lib
from .decorators import requires_geos, UnsupportedGEOSOperation
from .enum import ParamEnum

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
    """A query-only R-tree created using the Sort-Tile-Recursive (STR)
    algorithm.

    For two-dimensional spatial data. The tree is constructed directly
    at initialization.

    Parameters
    ----------
    geometries : array_like
    leafsize : int, default 10
        the maximum number of child nodes that a node can have

    Examples
    --------
    >>> import pygeos
    >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
    >>> # Query geometries that overlap envelope of input geometries:
    >>> tree.query(pygeos.box(2, 2, 4, 4)).tolist()
    [2, 3, 4]
    >>> # Query geometries that are contained by input geometry:
    >>> tree.query(pygeos.box(2, 2, 4, 4), predicate='contains').tolist()
    [3]
    >>> # Query geometries that overlap envelopes of ``geoms``
    >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)]).tolist()
    [[0, 0, 0, 1, 1], [2, 3, 4, 5, 6]]
    >>> tree.nearest([pygeos.points(1,1), pygeos.points(3,5)]).tolist()  # doctest: +SKIP
    [[0, 1], [1, 4]]
    """

    def __init__(self, geometries, leafsize=10):
        self.geometries = np.asarray(geometries, dtype=np.object_)
        self._tree = lib.STRtree(self.geometries, leafsize)

    def __len__(self):
        return self._tree.count

    def query(self, geometry, predicate=None, distance=None):
        """Return the index of all geometries in the tree with extents that
        intersect the envelope of the input geometry.

        If predicate is provided, a prepared version of the input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        The 'dwithin' predicate requires GEOS >= 3.10.

        If geometry is None, an empty array is returned.

        Parameters
        ----------
        geometry : Geometry
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

        Examples
        --------
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.query(pygeos.box(1,1, 3,3)).tolist()
        [1, 2, 3]
        >>> # Query geometries that are contained by input geometry
        >>> tree.query(pygeos.box(2, 2, 4, 4), predicate='contains').tolist()
        [3]
        >>> # Query geometries within 1 unit distance of input geometry
        >>> tree.query(pygeos.points(0.5, 0.5), predicate='dwithin', distance=1.0).tolist()  # doctest: +SKIP
        [0, 1]
        """

        if geometry is None:
            return np.array([], dtype=np.intp)

        if predicate is None:
            return self._tree.query(geometry, 0)

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

            geometry = np.array([geometry])
            distance = np.array([distance], dtype="float64")
            return self._tree.dwithin(geometry, distance)[1]

        predicate = BinaryPredicate.get_value(predicate)
        return self._tree.query(geometry, predicate)

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
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)]).tolist()
        [[0, 0, 0, 1, 1], [2, 3, 4, 5, 6]]
        >>> # Query for geometries that contain tree geometries
        >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)], predicate='contains').tolist()
        [[0], [3]]
        >>> # To get an array of pairs of index of input geometry, index of tree geometry,
        >>> # transpose the output:
        >>> tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)]).T.tolist()
        [[0, 2], [0, 3], [0, 4], [1, 5], [1, 6]]
        >>> # Query for tree geometries within 1 unit distance of input geometries
        >>> tree.query_bulk([pygeos.points(0.5, 0.5)], predicate='dwithin', distance=1.0).tolist()  # doctest: +SKIP
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
    def nearest(self, geometry):
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
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.nearest(pygeos.points(1,1)).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> tree.nearest([pygeos.box(1,1,3,3)]).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> points = pygeos.points(0.5,0.5)
        >>> tree.nearest([None, pygeos.points(10,10)]).tolist()  # doctest: +SKIP
        [[1], [9]]
        """

        geometry = np.asarray(geometry, dtype=object)
        if geometry.ndim == 0:
            geometry = np.expand_dims(geometry, 0)

        return self._tree.nearest(geometry)

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
        >>> import pygeos
        >>> tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
        >>> tree.nearest_all(pygeos.points(1,1)).tolist()  # doctest: +SKIP
        [[0], [1]]
        >>> tree.nearest_all([pygeos.box(1,1,3,3)]).tolist()  # doctest: +SKIP
        [[0, 0, 0], [1, 2, 3]]
        >>> points = pygeos.points(0.5,0.5)
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
