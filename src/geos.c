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

/* Define GEOS error handlers. See GEOS_INIT / GEOS_FINISH macros in geos.h*/
void geos_error_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}

void geos_notice_handler(const char *message, void *userdata) {
    snprintf(userdata, 1024, "%s", message);
}
