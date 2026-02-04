"""Manipulation and analysis of geometric objects in the Cartesian plane."""


def _set_geos_libdir():
    b"""Might be required for editable builds to avoid ImportError.

    E.g. add -Csetup-args="-Dgeos_libdir=C:\\OSGeo4W\bin"
    """
    import os
    import sys

    if not sys.platform.startswith("win"):
        return

    import importlib.resources

    pth = importlib.resources.files("shapely").joinpath("geos_libdir.txt")
    try:
        geos_libdir = pth.read_text().strip()
    except FileNotFoundError:
        return
    os.add_dll_directory(geos_libdir)


_set_geos_libdir()
del _set_geos_libdir

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

# TODO: re-enable dynamic version
__version__ = "2.2.0.dev0"
__git_version__ = ""


setup_signal_checks()
