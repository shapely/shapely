#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#include "pygeom.h"
#include "geos.h"

/* Initializes a new geometry object */
PyObject *GeometryObject_FromGEOS(PyTypeObject *type, GEOSGeometry *ptr)
{
    if (ptr == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    GeometryObject *self = (GeometryObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    } else {
        self->ptr = ptr;
        return (PyObject *) self;
    }
}

static void GeometryObject_dealloc(GeometryObject *self)
{
    void *context_handle = geos_context[0];
    if (self->ptr != NULL) {
        GEOSGeom_destroy_r(context_handle, self->ptr);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef GeometryObject_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(GeometryObject, ptr), READONLY, "pointer to GEOSGeometry"},
    {NULL}  /* Sentinel */
};

static PyObject *GeometryObject_ToWKT(GeometryObject *obj, char *format)
{
    void *context_handle = geos_context[0];
    char *wkt;
    PyObject *result;
    if (obj->ptr == NULL) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    GEOSWKTWriter *writer = GEOSWKTWriter_create_r(context_handle);
    if (writer == NULL) {
        return NULL;
    }

    char trim = 1;
    int precision = 3;
    int dimension = 3;
    int use_old_3d = 0;
    GEOSWKTWriter_setRoundingPrecision_r(context_handle, writer, precision);
    GEOSWKTWriter_setTrim_r(context_handle, writer, trim);
    GEOSWKTWriter_setOutputDimension_r(context_handle, writer, dimension);
    GEOSWKTWriter_setOld3D_r(context_handle, writer, use_old_3d);
    wkt = GEOSWKTWriter_write_r(context_handle, writer, obj->ptr);
    result = PyUnicode_FromFormat(format, wkt);
    GEOSFree_r(context_handle, wkt);
    GEOSWKTWriter_destroy_r(context_handle, writer);
    return result;
}

static PyObject *GeometryObject_repr(GeometryObject *self)
{
    return GeometryObject_ToWKT(self, "<pygeos.Geometry %s>");
}

static PyObject *GeometryObject_str(GeometryObject *self)
{
    return GeometryObject_ToWKT(self, "%s");
}

static PyObject *GeometryObject_FromWKT(PyTypeObject *type, PyObject *value)
{
    void *context_handle = geos_context[0];
    PyObject *result = NULL;
    char *wkt;
    GEOSGeometry *geom;
    GEOSWKTReader *reader;

    /* Cast the PyObject str to char* */
    if (PyUnicode_Check(value)) {
        wkt = PyUnicode_AsUTF8(value);
        if (wkt == NULL) { return NULL; }
    } else {
        PyErr_Format(PyExc_TypeError, "Expected bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }

    reader = GEOSWKTReader_create_r(context_handle);
    if (reader == NULL) {
        return NULL;
    }
    geom = GEOSWKTReader_read_r(context_handle, reader, wkt);
    GEOSWKTReader_destroy_r(context_handle, reader);
    if (geom == NULL) {
        return NULL;
    }
    result = GeometryObject_FromGEOS(type, geom);
    if (result == NULL) {
        GEOSGeom_destroy_r(context_handle, geom);
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");
    }
    return result;
}

static PyObject *GeometryObject_new(PyTypeObject *type, PyObject *args,
                                    PyObject *kwds)
{
    PyObject *value;

    if (!PyArg_ParseTuple(args, "O", &value)) {
        return NULL;
    }
    else if (PyUnicode_Check(value)) {
        return GeometryObject_FromWKT(type, value);
    }
    else {
        PyErr_Format(PyExc_TypeError, "Expected string, got %s", value->ob_type->tp_name);
        return NULL;
    }
}

static PyMethodDef GeometryObject_methods[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.lib.GEOSGeometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeometryObject_new,
    .tp_dealloc = (destructor) GeometryObject_dealloc,
    .tp_members = GeometryObject_members,
    .tp_methods = GeometryObject_methods,
    .tp_repr = (reprfunc) GeometryObject_repr,
    .tp_str = (reprfunc) GeometryObject_str,
};


/* Get a GEOSGeometry pointer from a GeometryObject, or NULL if the input is
Py_None. Returns 0 on error, 1 on success. */
char get_geom(GeometryObject *obj, GEOSGeometry **out) {
    if (!PyObject_IsInstance((PyObject *) obj, (PyObject *) &GeometryType)) {
        if ((PyObject *) obj == Py_None) {
            *out = NULL;
            return 1;
        } else {
            PyErr_Format(PyExc_TypeError, "One of the arguments is of incorrect type. Please provide only Geometry objects.");
            return 0;
        }
    } else {
        *out = obj->ptr;
        return 1;
    }
}

int
init_geom_type(PyObject *m)
{
    if (PyType_Ready(&GeometryType) < 0) {
        return -1;
    }

    Py_INCREF(&GeometryType);
    PyModule_AddObject(m, "Geometry", (PyObject *) &GeometryType);
    return 0;
}
