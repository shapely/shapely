#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD;
    void * __geom__;
} GeometryObject;

static PyObject *tp_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds) =
{

};

static PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.geometry.Geometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef geometrymodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "geometry",
    .m_doc = "Module that contains the geometry type.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_geometry(void)
{
    PyObject *m;
    if (PyType_Ready(&GeometryType) < 0)
        return NULL;

    m = PyModule_Create(&geometrymodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GeometryType);
    PyModule_AddObject(m, "Geometry", (PyObject *) &GeometryType);
    return m;
}
