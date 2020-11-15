/************************************************************************
 * PyGEOS C API
 *
 * This file wraps internal PyGEOS C extension functions for use in other
 * extensions.  These are specifically wrapped to enable dynamic loading
 * after Python initialization (see c_api.h and lib.c).
 *
 ***********************************************************************/
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#define PyGEOS_API_Module

#include "c_api.h"
#include "geos.h"
#include "pygeom.h"


extern PyObject* PyGEOS_CreateGeometry(GEOSGeometry *ptr, GEOSContextHandle_t ctx) {
    return GeometryObject_FromGEOS(ptr, ctx);
}

extern char PyGEOS_GetGEOSGeometry(PyObject *obj, GEOSGeometry **out) {
    return get_geom((GeometryObject*)obj, out);
}
