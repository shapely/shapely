from shapely.lib import GEOSException  # noqa: F401, I001
from shapely import _version
from shapely._geometry import *  # noqa: F403
from shapely.constructive import *  # noqa: F403
from shapely.coordinates import *  # noqa: F403
from shapely.creation import *  # noqa: F403
from shapely.errors import setup_signal_checks

# Submodule always needs to be imported to ensure Geometry subclasses are registered
from shapely.geometry import (
    GeometryCollection,  # noqa: F401
    LinearRing,  # noqa: F401
    LineString,  # noqa: F401
    MultiLineString,  # noqa: F401
    MultiPoint,  # noqa: F401
    MultiPolygon,  # noqa: F401
    Point,  # noqa: F401
    Polygon,  # noqa: F401
)
from shapely.io import *  # noqa: F403
from shapely.lib import (
    Geometry,  # noqa: F401
    geos_capi_version,  # noqa: F401
    geos_capi_version_string,  # noqa: F401
    geos_version,  # noqa: F401
    geos_version_string,  # noqa: F401
)
from shapely.linear import *  # noqa: F403
from shapely.measurement import *  # noqa: F403
from shapely.predicates import *  # noqa: F403
from shapely.set_operations import *  # noqa: F403
from shapely.strtree import *  # noqa: F403

__version__ = _version.get_versions()["version"]

setup_signal_checks()
