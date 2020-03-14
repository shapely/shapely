from enum import IntEnum
import numpy as np
from pygeos import lib


__all__ = ["STRtree"]


class BinaryPredicate(IntEnum):
    """The enumeration of GEOS binary predicates types"""

    intersects = 1
    within = 2
    contains = 3
    overlaps = 4
    crosses = 5
    touches = 6


VALID_PREDICATES = {e.name for e in BinaryPredicate}


class STRtree:
    """A query-only R-tree created using the Sort-Tile-Recursive (STR)
    algorithm.

    For two-dimensional spatial data. The actual tree will be constructed at the first
    query.

    Parameters
    ----------
    geometries : array_like
    leafsize : int
        the maximum number of child nodes that a node can have

    Examples
    --------
    >>> import pygeos
    >>> geoms = pygeos.points(np.arange(10), np.arange(10))
    >>> tree = pygeos.STRtree(geoms)
    >>> geom = pygeos.box(2, 2, 4, 4)
    >>> # Query geometries that overlap envelope of `geom`:
    >>> tree.query(geom).tolist()
    [2, 3, 4]
    >>> # Query geometries that overlap envelope of `geom`
    >>> # and are contained by `geom`:
    >>> tree.query(geom, predicate='contains').tolist()
    [3]
    """

    def __init__(self, geometries, leafsize=5):
        self.geometries = np.asarray(geometries, dtype=np.object)
        self._tree = lib.STRtree(self.geometries, leafsize)

    def __len__(self):
        return self._tree.count

    def query(self, geometry, predicate=None):
        """Return all items whose extent intersect the envelope of the input
        geometry.

        If predicate is provided, a prepared version of the input geometry
        is tested using the predicate function against each item whose
        extent intersects the envelope of the input geometry:
        predicate(geometry, tree_geometry).

        If geometry is None, an empty array is returned.

        Parameters
        ----------
        geometry : Geometry
            The envelope of the geometry is taken automatically for
            querying the tree.
        predicate : {None, 'intersects', 'within', 'contains', 'overlaps', 'crosses', 'touches'}, optional
            The predicate to use for testing geometries from the tree
            that are within the input geometry's envelope.
        """

        if geometry is None:
            return np.array([], dtype=np.intp)

        if predicate is None:
            predicate = 0

        else:
            if not predicate in VALID_PREDICATES:
                raise ValueError(
                    "Predicate {} is not valid; must be one of {}".format(
                        predicate, ", ".join(VALID_PREDICATES)
                    )
                )

            predicate = BinaryPredicate[predicate].value

        return self._tree.query(geometry, predicate)
