"""
strtree
=======

Index geometry objects for efficient lookup of nearby or
nearest neighbors. Home of the `STRtree` class which is
an interface to the query-only GEOS R-tree packed using
the Sort-Tile-Recursive algorithm [1]_.

.. autoclass:: STRtree
    :members:

References
----------
  .. [1]  Leutenegger, Scott & Lopez, Mario & Edgington, Jeffrey. (1997).
     "STR: A Simple and Efficient Algorithm for R-Tree Packing." Proc.
     VLDB Conf. 497-506. 10.1109/ICDE.1997.582015.
     https://www.cs.odu.edu/~mln/ltrs-pdfs/icase-1997-14.pdf
"""

from shapely.geos import lgeos
import ctypes

class STRtree:
    """
    An STRtree is a spatial index; specifically, an R-tree created
    using the Sort-Tile-Recursive algorithm.

    Pass a list of geometry objects to the `STRtree` constructor to
    create a spatial index. References to these indexed objects are
    kept and stored in the R-tree. You can query them with another
    geometric object.

    The `STRtree` is query-only, meaning that once created
    you cannot add or remove geometries.

    *New in version 1.4.0*.

    Parameters
    ----------
    geoms : sequence of geometry objects
        geometry objects to be indexed

    Examples
    --------

    Creating an index of pologons:

    >>> from shapely.strtree import STRtree
    >>> from shapely.geometry import Polygon, Point
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

    Behavior if an `STRtree` is created empty:

    >>> tree = STRtree([])
    >>> tree.query(Point(0, 0))
    []
    >>> print(tree.nearest(Point(0, 0)))
    None
    """

    def __init__(self, geoms):
        # filter empty geometries out of the input
        geoms = [geom for geom in geoms if not geom.is_empty]
        self._n_geoms = len(geoms)

        self._init_tree_handle(geoms)

        # Keep references to geoms.
        self._geoms = list(geoms)

    def _init_tree_handle(self, geoms):
        # GEOS STRtree capacity has to be > 1
        self._tree_handle = lgeos.GEOSSTRtree_create(max(2, len(geoms)))
        for geom in geoms:
            lgeos.GEOSSTRtree_insert(self._tree_handle, geom._geom, ctypes.py_object(geom))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tree_handle"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_tree_handle(self._geoms)

    def __del__(self):
        if self._tree_handle is not None:
            try:
                lgeos.GEOSSTRtree_destroy(self._tree_handle)
            except AttributeError:
                pass  # lgeos might be empty on shutdown.

            self._tree_handle = None

    def query(self, geom):
        """
        Search the index for geometry objects whose extents
        intersect the extent of the given object.

        Parameters
        ----------
        geom : geometry object
            The query geometry

        Returns
        -------
        list of geometry objects
            All the geometry objects in the index whose extents
            intersect the extent of `geom`.

        Note
        ----
        A geometry object's "extent" is its the minimum xy bounding
        rectangle.

        Examples
        --------

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

        To get the original indices of the returned objects, create an
        auxiliary dictionary. But use the geometry *ids* as keys since
        the shapely geometry objects themselves are not hashable.

        >>> index_by_id = dict((id(pt), i) for i, pt in enumerate(points))
        >>> [(index_by_id[id(pt)], pt.wkt) for pt in tree.query(Point(2,2).buffer(1.0))]
        [(1, 'POINT (1 1)'), (2, 'POINT (2 2)'), (3, 'POINT (3 3)')]
        """
        if self._n_geoms == 0:
            return []

        result = []

        def callback(item, userdata):
            geom = ctypes.cast(item, ctypes.py_object).value
            result.append(geom)

        lgeos.GEOSSTRtree_query(self._tree_handle, geom._geom, lgeos.GEOSQueryCallback(callback), None)

        return result

    def nearest(self, geom):
        """
        Get the nearest object in the index to a geometry object.

        Parameters
        ----------
        geom : geometry object
            The query geometry

        Returns
        -------
        geometry object
            The nearest geometry object in the index to `geom`.

            Will always only return *one* object even if several
            in the index are the minimum distance away.

            `None` if the index is empty.

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
        if self._n_geoms == 0:
            return None

        envelope = geom.envelope

        def callback(item1, item2, distance, userdata):
            try:
                geom1 = ctypes.cast(item1, ctypes.py_object).value
                geom2 = ctypes.cast(item2, ctypes.py_object).value
                dist = ctypes.cast(distance, ctypes.POINTER(ctypes.c_double))
                lgeos.GEOSDistance(geom1._geom, geom2._geom, dist)
                return 1
            except:
                return 0

        item = lgeos.GEOSSTRtree_nearest_generic(self._tree_handle, ctypes.py_object(geom), envelope._geom, \
            lgeos.GEOSDistanceCallback(callback), None)
        result = ctypes.cast(item, ctypes.py_object).value

        return result
