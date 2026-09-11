"""Manipulation and analysis of geometric objects in the Cartesian plane."""

import os
import platform

if platform.system() == "Windows":
    try:
        from shapely.lib import GEOSException
    except ImportError:  # DLL load failed while importing lib...
        # This is possibly a local build, which requires an
        # environment variable GEOS_LIBDIR to be specified at runtime
        # e.g.: set GEOS_LIBDIR=C:\OSGeo4W\bin
        if "GEOS_LIBDIR" in os.environ:
            os.add_dll_directory(os.environ["GEOS_LIBDIR"])
        else:
            raise

del os, platform

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

from shapely import _version

__version__ = _version.get_versions()["version"]

setup_signal_checks()
