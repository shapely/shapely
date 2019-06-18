#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <geos_c.h>

typedef struct {
    PyObject_HEAD;
    void *ptr;
    char geom_type_id;
    char has_z;
} GeometryObject;

static PyObject *GeometryObject_new(PyTypeObject *type, PyObject *args,
                                    PyObject *kwds)
{
    void *context_handle, *ptr;
    GeometryObject *self;
    long arg;
    int geos_result;
    self = (GeometryObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        context_handle = GEOS_init_r();
        if (!PyArg_ParseTuple(args, "l", &arg)) {
            goto fail;
        }
        self->ptr = (void *) arg;
        geos_result = GEOSGeomTypeId_r(context_handle, self->ptr);
        if ((geos_result < 0) | (geos_result > 255)) {
            goto fail;
        }
        self->geom_type_id = geos_result;
        geos_result = GEOSHasZ_r(context_handle, self->ptr);
        if ((geos_result < 0) | (geos_result > 1)) {
            goto fail;
        }
        self->has_z = geos_result;
        GEOS_finish_r(context_handle);
    }
    return (PyObject *) self;
    fail:
        PyErr_Format(PyExc_RuntimeError, "Geometry initialization failed");
        GEOS_finish_r(context_handle);
        Py_DECREF(self);
        return NULL;
}

static void GeometryObject_dealloc(GeometryObject *self)
{
    void *context_handle;
    if (self->ptr != NULL) {
        context_handle = GEOS_init_r();
        GEOSGeom_destroy_r(context_handle, self->ptr);
        GEOS_finish_r(context_handle);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.geometry.Geometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeometryObject_new,
    .tp_dealloc = (destructor) GeometryObject_dealloc,
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
