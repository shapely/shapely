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

import ctypes
from functools import wraps
import logging
from warnings import warn

from shapely.errors import ShapelyDeprecationWarning
from shapely.geos import lgeos

log = logging.getLogger(__name__)


def nearest_callback(func):
    @wraps(func)
    def wrapper(arg1, arg2, arg3, arg4):
        value = ctypes.cast(arg1, ctypes.py_object).value
        geom = ctypes.cast(arg2, ctypes.py_object).value
        dist = ctypes.cast(arg3, ctypes.POINTER(ctypes.c_double))
        try:
            dist.contents.value = func(value, geom)
            return 1
        except Exception:
            log.exception("Caught exception")
            return 0
    return wrapper


def query_callback(func):
    @wraps(func)
    def wrapper(arg1, arg2):
        value = ctypes.cast(arg1, ctypes.py_object).value
        func(value)
    return wrapper


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

    def __init__(self, initdata=None):
        """Create a new STRtree.

        Parameters
        ----------
        initdata : iterable
            An iterable sequence of single geometry objects or (geom,
            value) tuples.

        Notes
        -----
        Items from initdata will be stored in two lists.

        """
        warn(
            "STRtree will be completely changed in 2.0.0. The exact API is not yet decided, but will be documented before 1.8.0",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        self._initdata = None
        self._tree = None
        if initdata:
            self._initdata = list(self._iteritems(initdata))
            self._init_tree(self._initdata)

    def _iteritems(self, initdata):
        for obj in initdata:
            if not isinstance(obj, tuple):
                geom = obj
                value = obj
            else:
                value, geom = obj
            if not geom.is_empty:
                yield value, geom

    def _init_tree(self, initdata):
        if initdata:
            node_capacity = 10
            self._tree = lgeos.GEOSSTRtree_create(node_capacity)
            for value, geom in self._iteritems(initdata):
                lgeos.GEOSSTRtree_insert(
                    self._tree, geom._geom, ctypes.py_object(value)
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tree"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_tree(self._initdata)

    def __del__(self):
        if self._tree is not None:
            try:
                lgeos.GEOSSTRtree_destroy(self._tree)
            except AttributeError:  # lgeos might be empty on shutdown.
                pass
            self._tree = None

    def query(self, geom):
        """
        Search the tree for nodes which intersect geom's envelope

        Parameters
        ----------
        geom : geometry object
            The query geometry

        Returns
        -------
        list
            A list of the values stored at the found nodes. The list
            will be empty if the tree is empty.

        Note
        ----
        A geometry object's envelope is its the minimum xy bounding
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

        """
        result = []

        @query_callback
        def callback(value):
            result.append(value)

        self.query_cb(geom, callback)
        return result

    def query_cb(self, geom, callback):
        """Searches the tree for nodes and applies a callback function.

        Parameters
        ----------
        geom : Geometry
            The query geometry
        callback : callable
            This is a function which takes a node value as argument and
            which is decorated by shapely.strtree.query_callback. See
            STRtree.query() for an example.

        Returns
        -------
        None

        """
        if self._tree is None or not self._initdata:
            return

        lgeos.GEOSSTRtree_query(
            self._tree, geom._geom, lgeos.GEOSQueryCallback(callback), None
        )

    def nearest(self, geom):
        """Finds the tree node nearest to a given geometry object.

        Parameters
        ----------
        geom : geometry object
            The query geometry

        Returns
        -------
        object or None
            The value of the tree node nearest to geom or None if the
            index is empty.

        Examples
        --------
        >>> from shapely.strtree import STRtree
        >>> from shapely.geometry import Point
        >>> tree = STRtree([Point(i, i) for i in range(10)])
        >>> tree.nearest(Point(2.2, 2.2)).wkt
        'POINT (2 2)'

        Will only return one object:

        >>> tree = STRtree([Point(0, 0), Point(0, 0)])
        >>> tree.nearest(Point(0, 0)).wkt
        'POINT (0 0)'

        """
        if self._tree is None or not self._initdata:
            return None

        # In a future version of shapely, geometries will be hashable
        # and we won't need to reindex like this.
        geoms = {id(v): g for v, g in self._initdata}

        @nearest_callback
        def callback(value, geom):
            value_geom = geoms[id(value)]
            return geom.distance(value_geom)

        envelope = geom.envelope

        item = lgeos.GEOSSTRtree_nearest_generic(
            self._tree,
            ctypes.py_object(geom),
            envelope._geom,
            lgeos.GEOSDistanceCallback(callback),
            None,
        )

        return ctypes.cast(item, ctypes.py_object).value
