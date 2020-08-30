#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "geos.h"

/* This initializes a globally accessible GEOSException object */
PyObject *geos_exception[1] = {NULL};

int init_geos(PyObject *m)
{
    geos_exception[0] = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
    PyModule_AddObject(m, "GEOSException", geos_exception[0]);
    return 0;
}


/* Checks whether the geometry is a multipoint with an empty point in it
 *
 * According to https://github.com/libgeos/geos/issues/305, this check is not
 * necessary for GEOS 3.7.3, 3.8.2, or 3.9. When these versions are out, we 
 * should add version conditionals and test.
 * 
 * The return value is one of:
 * - PGERR_SUCCESS 
 * - PGERR_MULTIPOINT_WITH_POINT_EMPTY
 * - PGERR_GEOS_EXCEPTION
 */
char check_to_wkt_compatible(GEOSContextHandle_t ctx, GEOSGeometry *geom) {    
    int n, i;
    char geom_type, is_empty;
    const GEOSGeometry *sub_geom;

    geom_type = GEOSGeomTypeId_r(ctx, geom);
    if (geom_type == -1) { return PGERR_GEOS_EXCEPTION; }
    if (geom_type != GEOS_MULTIPOINT) { return PGERR_SUCCESS; }
    
    n = GEOSGetNumGeometries_r(ctx, geom);
    if (n == -1) { return PGERR_GEOS_EXCEPTION; }
    for(i = 0; i < n; i++) {
        sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
        if (sub_geom == NULL) { return PGERR_GEOS_EXCEPTION; }
        is_empty = GEOSisEmpty_r(ctx, sub_geom);
        // GEOS returns 2 on exception:
        if (is_empty == 2) { return PGERR_GEOS_EXCEPTION; }
        if (is_empty) { return PGERR_MULTIPOINT_WITH_POINT_EMPTY; }
    }
    return PGERR_SUCCESS;
}


/* Define GEOS error handlers. See GEOS_INIT / GEOS_FINISH macros in geos.h*/
void geos_error_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}

void geos_notice_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}
