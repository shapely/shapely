#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "geos.h"

/* This initializes a global GEOS Context */
void *geos_context[1] = {NULL};

static void HandleGEOSError(const char *message, void *userdata) {
    PyErr_SetString(userdata, message);
}

static void HandleGEOSNotice(const char *message, void *userdata) {
    PyErr_WarnEx(PyExc_Warning, message, 1);
}

int init_geos(PyObject *m)
{
    void *context_handle = GEOS_init_r();
    PyObject* GEOSException = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
    PyModule_AddObject(m, "GEOSException", GEOSException);
    GEOSContext_setErrorMessageHandler_r(context_handle, HandleGEOSError, GEOSException);
    GEOSContext_setNoticeMessageHandler_r(context_handle, HandleGEOSNotice, NULL);
    geos_context[0] = context_handle;  /* for global access */
    return 0;
}
