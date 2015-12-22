from shapely.geos import lgeos
import ctypes

class STRtree:
    """
    STRtree is an R-tree that is created using the Sort-Tile-Recursive
    algorithm. STRtree takes a sequence of geometry objects as initialization
    parameter. After initialization the query method can be used to make a
    spatial query over those objects.

    >>> from shapely.geometry import Polygon
    >>> polys = [ Polygon(((0, 0), (1, 0), (1, 1))), Polygon(((0, 1), (0, 0), (1, 0))), Polygon(((100, 100), (101, 100), (101, 101))) ]
    >>> s = STRtree(polys)
    >>> query_geom = Polygon(((-1, -1), (2, 0), (2, 2), (-1, 2)))
    >>> result = s.query(query_geom)
    >>> polys[0] in result
    True
    >>> polys[1] in result
    True
    >>> polys[2] in result
    False
    >>> # Test empty tree
    >>> s = STRtree([])
    >>> s.query(query_geom)
    []
    >>> # Test tree with one object
    >>> s = STRtree([polys[0]])
    >>> result = s.query(query_geom)
    >>> polys[0] in result
    True
    """

    def __init__(self, geoms):
        # filter empty geometries out of the input
        geoms = [geom for geom in geoms if not geom.is_empty]
        self._n_geoms = len(geoms)
        # GEOS STRtree capacity has to be > 1
        self._tree_handle = lgeos.GEOSSTRtree_create(max(2, len(geoms)))
        for geom in geoms:
            lgeos.GEOSSTRtree_insert(self._tree_handle, geom._geom, ctypes.py_object(geom))

    def __del__(self):
        lgeos.GEOSSTRtree_destroy(self._tree_handle)

    def query(self, geom):
        if self._n_geoms == 0:
            return []

        result = []

        def callback(item, userdata):
            geom = ctypes.cast(item, ctypes.py_object).value
            result.append(geom)

        lgeos.GEOSSTRtree_query(self._tree_handle, geom._geom, lgeos.GEOSQueryCallback(callback), None)

        return result

if __name__ == "__main__":
    import doctest
    doctest.testmod()
