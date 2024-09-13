"""Proxies for libgeos, GEOS-specific exceptions, and utilities."""

import warnings

import shapely

warnings.warn(
    "The 'shapely.geos' module is deprecated, and will be removed in a future version. "
    "All attributes of 'shapely.geos' are now available directly from the top-level "
    "'shapely' namespace.",
    DeprecationWarning,
    stacklevel=2,
)

geos_version_string = shapely.geos_capi_version_string
geos_version = shapely.geos_version
geos_capi_version = shapely.geos_capi_version
