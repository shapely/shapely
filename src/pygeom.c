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


static PyObject *to_wkt(GeometryObject *obj, char *format, char trim,
                        int precision, int dimension, int use_old_3d)
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


static PyObject *GeometryObject_ToWKT(GeometryObject *self, PyObject *args, PyObject *kw)
{
    char trim = 1;
    int precision = 6;
    int dimension = 3;
    int use_old_3d = 0;
    static char *kwlist[] = {"precision", "trim", "dimension", "use_old_3d", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|ibib", kwlist,
                                     &precision, &trim, &dimension, &use_old_3d))
    {
        return NULL;
    }
    return to_wkt(self, "%s", trim, precision, dimension, use_old_3d);
}

static PyObject *GeometryObject_repr(GeometryObject *self)
{
    if (self->ptr == NULL) {
        return PyUnicode_FromString("<pygeos.NaG>");
    } else {
        return to_wkt(self, "<pygeos.Geometry %s>", 1, 3, 3, 0);
    }
}

static PyObject *GeometryObject_ToWKB(GeometryObject *self, PyObject *args, PyObject *kw)
{
    void *context_handle = geos_context[0];
    unsigned char *wkb;
    size_t size;
    PyObject *result;
    int dimension = 3;
    int byte_order = 1;
    char include_srid = 0;
    char hex = 0;
    static char *kwlist[] = {"dimension", "byte_order", "include_srid", "hex", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|ibbb", kwlist,
                                     &dimension, &byte_order, &include_srid, &hex))
    {
        return NULL;
    }
    if (self->ptr == NULL) {
         Py_INCREF(Py_None);
         return Py_None;
    }
    GEOSWKBWriter *writer = GEOSWKBWriter_create_r(context_handle);
    if (writer == NULL) {
        return NULL;
    }
    GEOSWKBWriter_setOutputDimension_r(context_handle, writer, dimension);
    GEOSWKBWriter_setByteOrder_r(context_handle, writer, byte_order);
    GEOSWKBWriter_setIncludeSRID_r(context_handle, writer, include_srid);
    if (hex) {
        wkb = GEOSWKBWriter_writeHEX_r(context_handle, writer, self->ptr, &size);
    } else {
        wkb = GEOSWKBWriter_write_r(context_handle, writer, self->ptr, &size);
    }
    result = PyBytes_FromStringAndSize((char *) wkb, size);
    GEOSFree_r(context_handle, wkb);
    GEOSWKBWriter_destroy_r(context_handle, writer);
    return result;
}

static PyObject *GeometryObject_FromWKT(PyTypeObject *type, PyObject *value)
{
    void *context_handle = geos_context[0];
    PyObject *result = NULL;
    char *wkt;
    GEOSGeometry *geom;
    GEOSWKTReader *reader;

    /* Cast the PyObject (bytes or str) to char* */
    if (PyBytes_Check(value)) {
        wkt = PyBytes_AsString(value);
        if (wkt == NULL) { return NULL; }
    }
    else if (PyUnicode_Check(value)) {
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

static PyObject *GeometryObject_FromWKB(PyTypeObject *type, PyObject *value)
{
    void *context_handle = geos_context[0];
    PyObject *result = NULL;
    GEOSGeometry *geom;
    GEOSWKBReader *reader;
    char *wkb;
    Py_ssize_t size;
    char is_hex;

    /* Cast the PyObject (only bytes) to char* */
    if (!PyBytes_Check(value)) {
        PyErr_Format(PyExc_TypeError, "Expected bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }
    size = PyBytes_Size(value);
    wkb = PyBytes_AsString(value);
    if (wkb == NULL) {
        return NULL;
    }

    /* Check if this is a HEX WKB */
    if (size != 0) {
        is_hex = ((wkb[0] == 48) | (wkb[0] == 49));
    } else {
        is_hex = 0;
    }

    /* Create the reader and read the WKB */
    reader = GEOSWKBReader_create_r(context_handle);
    if (reader == NULL) {
        return NULL;
    }
    if (is_hex) {
        geom = GEOSWKBReader_readHEX_r(context_handle, reader, (unsigned char *) wkb, size);
    } else {
        geom = GEOSWKBReader_read_r(context_handle, reader, (unsigned char *) wkb, size);
    }
    GEOSWKBReader_destroy_r(context_handle, reader);
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

    if (PyBytes_Check(value)) {
        return GeometryObject_FromWKB(type, value);
    }
    else if (PyUnicode_Check(value)) {
        return GeometryObject_FromWKT(type, value);
    }
    else {
        PyErr_Format(PyExc_TypeError, "Expected string or bytes, found %s", value->ob_type->tp_name);
        return NULL;
    }
}

static PyMethodDef GeometryObject_methods[] = {
    {"to_wkt", (PyCFunction) GeometryObject_ToWKT, METH_VARARGS | METH_KEYWORDS,
     "Write the geometry to Well-Known Text (WKT) format"
    },
    {"to_wkb", (PyCFunction) GeometryObject_ToWKB, METH_VARARGS | METH_KEYWORDS,
     "Write the geometry to Well-Known Binary (WKB) format"
    },
    {"from_wkt", (PyCFunction) GeometryObject_FromWKT, METH_CLASS | METH_O,
     "Read the geometry from Well-Known Text (WKT) format"
    },
    {"from_wkb", (PyCFunction) GeometryObject_FromWKB, METH_CLASS | METH_O,
     "Read the geometry from Well-Known Binary (WKB) format"
    },
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
