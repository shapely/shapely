"""Implementation of the intermediary layer between Shapely and GEOS
"""

from shapely.coords import BoundsOp
from shapely.geos import lgeos
from shapely.linref import ProjectOp, InterpolateOp
from shapely.predicates import BinaryPredicate, UnaryPredicate
from shapely.topology import BinaryRealProperty, BinaryTopologicalOp
from shapely.topology import UnaryRealProperty, UnaryTopologicalOp


# Map geometry methods to their GEOS delegates

IMPL14 = {
    'area': (UnaryRealProperty, 'area'),
    'distance': (BinaryRealProperty, 'distance'),
    'length': (UnaryRealProperty, 'length'),
    #
    'boundary': (UnaryTopologicalOp, 'boundary'),
    'bounds': (BoundsOp, None),
    'centroid': (UnaryTopologicalOp, 'centroid'),
    'envelope': (UnaryTopologicalOp, 'envelope'),
    'convex_hull': (UnaryTopologicalOp, 'convex_hull'),
    'buffer': (UnaryTopologicalOp, 'buffer'),
    #
    'difference': (BinaryTopologicalOp, 'difference'),
    'intersection': (BinaryTopologicalOp, 'intersection'),
    'symmetric_difference': (BinaryTopologicalOp, 'symmetric_difference'),
    'union': (BinaryTopologicalOp, 'union'),
    #
    'has_z': (UnaryPredicate, 'has_z'),
    'is_empty': (UnaryPredicate, 'is_empty'),
    'is_ring': (UnaryPredicate, 'is_ring'),
    'is_simple': (UnaryPredicate, 'is_simple'),
    'is_valid': (UnaryPredicate, 'is_valid'),
    #
    'relate': (BinaryPredicate, 'relate'),
    'contains': (BinaryPredicate, 'contains'),
    'crosses': (BinaryPredicate, 'crosses'),
    'disjoint': (BinaryPredicate, 'disjoint'),
    'equals': (BinaryPredicate, 'equals'),
    'intersects': (BinaryPredicate, 'intersects'),
    'overlaps': (BinaryPredicate, 'overlaps'),
    'touches': (BinaryPredicate, 'touches'),
    'within': (BinaryPredicate, 'within'),
    'equals_exact': (BinaryPredicate, 'equals_exact'),
    }

IMPL15 = {
    'simplify': (UnaryTopologicalOp, 'simplify'),
    'topology_preserve_simplify': 
        (UnaryTopologicalOp, 'topology_preserve_simplify'),
    'prepared_intersects': (BinaryPredicate, 'prepared_intersects'),
    'prepared_contains': (BinaryPredicate, 'prepared_contains'),
    'prepared_contains_properly': 
        (BinaryPredicate, 'prepared_contains_properly'),
    'prepared_covers': (BinaryPredicate, 'prepared_covers'),
    }

IMPL16 = {
    }

IMPL16LR = {
    'project_normalized': (ProjectOp, 'project_normalized'),
    'project': (ProjectOp, 'project'),
    'interpolate_normalized': (InterpolateOp, 'interpolate_normalized'),
    'interpolate': (InterpolateOp, 'interpolate'),
    }

def build(defs):
    return [(k, v[0](v[1])) for k, v in defs.items()]

imp = dict(build(IMPL14))
if lgeos.geos_capi_version >= (1, 5, 0):
    imp.update(build(IMPL15))
if lgeos.geos_capi_version >= (1, 6, 0):
    imp.update(build(IMPL16))
    if 'project' in lgeos.methods:
        imp.update(build(IMPL16LR))

DefaultImplementation = imp
