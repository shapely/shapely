#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL pygeos_UFUNC_API
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

#include "geos.h"
#include "pygeom.h"
#include "coords.h"
#include "ufuncs.h"


/* This tells Python what methods this module has. */
static PyMethodDef GeosModule[] = {
    {"count_coordinates", PyCountCoords, METH_VARARGS, "Counts the total amount of coordinates in a array with geometry objects"},
    {"get_coordinates", PyGetCoords, METH_VARARGS, "Gets the coordinates as an (N, 2) shaped ndarray of floats"},
    {"set_coordinates", PySetCoords, METH_VARARGS, "Sets coordinates to a geometry array"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "lib",
    NULL,
    -1,
    GeosModule,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_lib(void)
{
    PyObject *m, *d;

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    if (init_geos(m) < 0) {
        return NULL;
    };

    if (init_geom_type(m) < 0) {
        return NULL;
    };

    d = PyModule_GetDict(m);

    import_array();
    import_umath();

    /* export the version as a python string */
    PyModule_AddObject(m, "geos_version", PyUnicode_FromString(GEOS_VERSION));

    if (init_ufuncs(m, d) < 0) {
        return NULL;
    };

    return m;
}
