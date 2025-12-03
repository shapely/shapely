"""Manipulation and analysis of geometric objects in the Cartesian plane."""

from shapely.lib import GEOSException
from shapely.lib import Geometry
from shapely.lib import geos_version, geos_version_string
from shapely.lib import geos_capi_version, geos_capi_version_string
from shapely.errors import setup_signal_checks
from shapely._geometry import *
from shapely.creation import *
from shapely.constructive import *
from shapely.predicates import *
from shapely.measurement import *
from shapely.set_operations import *
from shapely.linear import *
from shapely.coordinates import *
from shapely.strtree import *
from shapely.io import *
from shapely._coverage import *

# Submodule always needs to be imported to ensure Geometry subclasses are registered
from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
    LinearRing,
)

try:
    from shapely._version import __git_version__, __version__
except ImportError:
    __git_version__ = ""
    __version__ = "0+unknown"


setup_signal_checks()
