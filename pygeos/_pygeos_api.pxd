"""
Provides a wrapper for the pygeos.lib C API for use in Cython.
Internally, the pygeos C extension uses a PyCapsule to provide run-time access
to function pointers within the C API.

To use these functions, you must first call the following function in each Cython module:
`import_pygeos_c_api()`

This uses a macro to dynamically load the functions from pointers in the PyCapsule.
Each C function in pygeos.lib exposed in the C API must be specially-wrapped to enable
this capability.

Segfaults will occur if the C API is not imported properly.
"""

cimport numpy as np
from cpython.ref cimport PyObject

from pygeos._geos cimport GEOSContextHandle_t, GEOSGeometry


cdef extern from "c_api.h":
    # pygeos.lib C API loader; returns -1 on error
    # MUST be called before calling other C API functions
    int import_pygeos_c_api() except -1

    # C functions provided by the pygeos.lib C API
    # Note: GeometryObjects are always managed as Python objects
    # in Cython to avoid memory leaks, not PyObject* (even though
    # they are declared that way in the header file).
    object PyGEOS_CreateGeometry(GEOSGeometry *ptr, GEOSContextHandle_t ctx)
    char PyGEOS_GetGEOSGeometry(PyObject *obj, GEOSGeometry **out) nogil
