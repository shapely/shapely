GEOS_POINT = 0
GEOS_LINESTRING = 1
GEOS_LINEARRING = 2
GEOS_POLYGON = 3
GEOS_MULTIPOINT = 4
GEOS_MULTILINESTRING = 5
GEOS_MULTIPOLYGON = 6
GEOS_GEOMETRYCOLLECTION = 7


from .ufuncs import GEOSGeometry  # NOQA
from .ufuncs import GEOSException  # NOQA
from .ufuncs import is_empty  # NOQA
from .ufuncs import is_simple  # NOQA
from .ufuncs import is_ring  # NOQA
from .ufuncs import has_z  # NOQA
from .ufuncs import is_closed  # NOQA
from .ufuncs import is_valid  # NOQA
from .ufuncs import disjoint  # NOQA
from .ufuncs import touches  # NOQA
from .ufuncs import intersects  # NOQA
from .ufuncs import crosses  # NOQA
from .ufuncs import within  # NOQA
from .ufuncs import contains  # NOQA
from .ufuncs import overlaps  # NOQA
from .ufuncs import equals  # NOQA
from .ufuncs import covers  # NOQA
from .ufuncs import covered_by  # NOQA
from .ufuncs import clone  # NOQA
from .ufuncs import envelope  # NOQA
from .ufuncs import convex_hull  # NOQA
from .ufuncs import boundary  # NOQA
from .ufuncs import unary_union  # NOQA
from .ufuncs import point_on_surface  # NOQA
from .ufuncs import get_centroid  # NOQA
from .ufuncs import line_merge  # NOQA
from .ufuncs import extract_unique_points  # NOQA
from .ufuncs import get_start_point  # NOQA
from .ufuncs import get_end_point  # NOQA
from .ufuncs import get_exterior_ring  # NOQA
from .ufuncs import normalize  # NOQA
from .ufuncs import get_interior_ring_n  # NOQA
from .ufuncs import get_point_n  # NOQA
from .ufuncs import get_geometry_n  # NOQA
from .ufuncs import set_srid  # NOQA
from .ufuncs import interpolate  # NOQA
from .ufuncs import interpolate_normalized  # NOQA
from .ufuncs import simplify  # NOQA
from .ufuncs import topology_preserve_simplify  # NOQA
from .ufuncs import intersection  # NOQA
from .ufuncs import difference  # NOQA
from .ufuncs import symmetric_difference  # NOQA
from .ufuncs import union  # NOQA
from .ufuncs import shared_paths  # NOQA
from .ufuncs import get_x  # NOQA
from .ufuncs import get_y  # NOQA
from .ufuncs import area  # NOQA
from .ufuncs import length  # NOQA
from .ufuncs import get_length  # NOQA
from .ufuncs import geom_type_id  # NOQA
from .ufuncs import get_dimensions  # NOQA
from .ufuncs import get_coordinate_dimensions  # NOQA
from .ufuncs import get_srid  # NOQA
from .ufuncs import get_num_geometries  # NOQA
from .ufuncs import get_num_interior_rings  # NOQA
from .ufuncs import get_num_points  # NOQA
from .ufuncs import get_num_coordinates  # NOQA
from .ufuncs import distance  # NOQA
from .ufuncs import hausdorff_distance  # NOQA
from .ufuncs import project  # NOQA
from .ufuncs import project_normalized  # NOQA
from .ufuncs import buffer  # NOQA
from .ufuncs import snap  # NOQA
from .ufuncs import equals_exact  # NOQA
from .construction import *  # NOQA
