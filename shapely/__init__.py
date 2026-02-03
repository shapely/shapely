"""Manipulation and analysis of geometric objects in the Cartesian plane."""


def _set_geos_libdir():
    """Might be required for editable builds to avoid ImportError.

    E.g. add -Csetup-args="-Dgeos_libdir=/path/to/geos-install/lib64"
    """
    import importlib.resources
    import os
    import sys

    pth = importlib.resources.files("shapely").joinpath("geos_libdir.txt")
    try:
        geos_libdir = pth.read_text().strip()
    except FileNotFoundError:
        return
    if sys.platform.startswith("win"):
        os.add_dll_directory(geos_libdir)
    elif sys.platform.startswith("linux"):
        lib_pth = geos_libdir
        if "LD_LIBRARY_PATH" in os.environ:
            lib_pth += os.pathsep + os.environ["LD_LIBRARY_PATH"]
        os.environ["LD_LIBRARY_PATH"] = lib_pth


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
