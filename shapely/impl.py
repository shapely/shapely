"""Implementation of the intermediary layer between Shapely and GEOS
"""

from shapely.coords import BoundsOp
from shapely.geos import lgeos
from shapely.linref import ProjectOp, InterpolateOp
from shapely.predicates import BinaryPredicate, UnaryPredicate
from shapely.topology import BinaryRealProperty, BinaryTopologicalOp
from shapely.topology import UnaryRealProperty, UnaryTopologicalOp


# These methods return ctypes ints, floats, and c_void_p

DefaultImplementation = {
    'area': UnaryRealProperty('area'),
    'distance': BinaryRealProperty('distance'),
    'length': UnaryRealProperty('length'),
    #
    'boundary': UnaryTopologicalOp('boundary'),
    'bounds': BoundsOp(),
    'centroid': UnaryTopologicalOp('centroid'),
    'envelope': UnaryTopologicalOp('envelope'),
    'convex_hull': UnaryTopologicalOp('convex_hull'),
    'buffer': UnaryTopologicalOp('buffer'),
    'simplify': UnaryTopologicalOp('simplify'),
    'topology_preserve_simplify': UnaryTopologicalOp('topology_preserve_simplify'),
    #
    'difference': BinaryTopologicalOp('difference'),
    'intersection': BinaryTopologicalOp('intersection'),
    'symmetric_difference': BinaryTopologicalOp('symmetric_difference'),
    'union': BinaryTopologicalOp('union'),
    #
    'has_z': UnaryPredicate('has_z'),
    'is_empty': UnaryPredicate('is_empty'),
    'is_ring': UnaryPredicate('is_ring'),
    'is_simple': UnaryPredicate('is_simple'),
    'is_valid': UnaryPredicate('is_valid'),
    #
    'relate': BinaryPredicate('relate'),
    'contains': BinaryPredicate('contains'),
    'crosses': BinaryPredicate('crosses'),
    'disjoint': BinaryPredicate('disjoint'),
    'equals': BinaryPredicate('equals'),
    'intersects': BinaryPredicate('intersects'),
    'overlaps': BinaryPredicate('overlaps'),
    'touches': BinaryPredicate('touches'),
    'within': BinaryPredicate('within'),
    'equals_exact': BinaryPredicate('equals_exact'),
    #
    #
    'project_normalized': ProjectOp('project_normalized'),
    'project': ProjectOp('project'),
    'interpolate_normalized': InterpolateOp('interpolate_normalized'),
    'interpolate': InterpolateOp('interpolate'),
    #
    'prepared_intersects': BinaryPredicate('prepared_intersects'),
    'prepared_contains': BinaryPredicate('prepared_contains'),
    'prepared_contains_properly': BinaryPredicate('prepared_contains_properly'),
    'prepared_covers': BinaryPredicate('prepared_covers'),
}

