#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <math.h>
#include <geos_c.h>
#include <structmember.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include "fast_loop_macros.h"


#define RAISE_ILLEGAL_GEOS if (!PyErr_Occurred()) {PyErr_Format(PyExc_RuntimeError, "Uncaught GEOS exception");}
#define RAISE_NO_MALLOC PyErr_Format(PyExc_MemoryError, "Could not allocate memory")
#define CREATE_COORDSEQ(SIZE, NDIM)\
    void *coord_seq = GEOSCoordSeq_create_r(context_handle, SIZE, NDIM);\
    if (coord_seq == NULL) {\
        return;\
    }

#define SET_COORD(N, DIM)\
    if (!GEOSCoordSeq_setOrdinate_r(context_handle, coord_seq, N, DIM, coord)) {\
        GEOSCoordSeq_destroy_r(context_handle, coord_seq);\
        RAISE_ILLEGAL_GEOS;\
        return;\
    }

#define GEOM_ISNAN_OR_NONE(GEOM) (npy_isnan(PyFloat_AS_DOUBLE((PyObject *) GEOM)) | ((PyObject *) GEOM == Py_None))

#define CHECK_GEOM(GEOM)\
    if (!PyObject_IsInstance((PyObject *) GEOM, (PyObject *) &GeometryType)) {\
        if (GEOM_ISNAN_OR_NONE(GEOM)) {\
            PyErr_Format(PyExc_ValueError, "NaN and None cannot be handled by this function.");\
        } else { \
            PyErr_Format(PyExc_TypeError, "One of the arguments is of incorrect type. Please provide only Geometry objects.");\
        }\
        return;\
    }\
    if (GEOM->ptr == NULL) {\
        PyErr_Format(PyExc_ValueError, "A geometry object is empty");\
        return;\
    }


#define INPUT_Y\
    GeometryObject *in1 = *(GeometryObject **)ip1;\
    CHECK_GEOM(in1)

#define INPUT_YY\
    GeometryObject *in1 = *(GeometryObject **)ip1;\
    GeometryObject *in2 = *(GeometryObject **)ip2;\
    CHECK_GEOM(in1);\
    CHECK_GEOM(in2)

#define OUTPUT_b\
    if (! ((ret == 0) || (ret == 1))) {\
        RAISE_ILLEGAL_GEOS;\
        return;\
    }\
    *(npy_bool *)op1 = ret

#define OUTPUT_Y\
    if (ret_ptr == NULL) {\
        RAISE_ILLEGAL_GEOS;\
        return;\
    }\
    PyObject *ret = GeometryObject_new_from_ptr(&GeometryType, ret_ptr);\
    if (ret == NULL) {\
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");\
        return;\
    }\
    PyObject **out = (PyObject **)op1;\
    Py_XDECREF(*out);\
    *out = ret

#define OUTPUT_Y_NAN\
    PyObject **out = (PyObject **)op1;\
    Py_XDECREF(*out);\
    *out = PyFloat_FromDouble(NPY_NAN)

/* This tells Python what methods this module has. */
static PyMethodDef GeosModule[] = {
    {NULL, NULL, 0, NULL},
    {NULL, NULL, 0, NULL}
};

/* This initializes a global GEOS Context */
static void *geos_context[1] = {NULL};

static void HandleGEOSError(const char *message, void *userdata) {
    PyErr_SetString(userdata, message);
}

static void HandleGEOSNotice(const char *message, void *userdata) {
    PyErr_WarnEx(PyExc_Warning, message, 1);
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs",
    NULL,
    -1,
    GeosModule,
    NULL,
    NULL,
    NULL,
    NULL
};

typedef struct {
    PyObject_HEAD
    void *ptr;
} GeometryObject;


static PyObject *GeometryObject_new_from_ptr(PyTypeObject *type, GEOSGeometry *ptr)
{
    GeometryObject *self = (GeometryObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ptr = ptr;
    }
    return (PyObject *) self;
}

static void GeometryObject_dealloc(GeometryObject *self)
{
    void *context_handle;
    if (self->ptr != NULL) {
        context_handle = geos_context[0];
        GEOSGeom_destroy_r(context_handle, self->ptr);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef GeometryObject_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(GeometryObject, ptr), READONLY, "pointer to GEOSGeometry"},
    {NULL}  /* Sentinel */
};


static PyObject *GeometryObject_ToWKT(GeometryObject *self, PyObject *args, PyObject *kw)
{
    void *context_handle = geos_context[0];
    char *wkt;
    PyObject *result;
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
    if (self->ptr == NULL) {
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
    wkt = GEOSWKTWriter_write_r(context_handle, writer, self->ptr);
    result = PyUnicode_FromString(wkt);
    GEOSFree_r(context_handle, wkt);
    GEOSWKTWriter_destroy_r(context_handle, writer);
    return result;
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
    result = GeometryObject_new_from_ptr(type, geom);
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
    result = GeometryObject_new_from_ptr(type, geom);
    if (result == NULL) {
        GEOSGeom_destroy_r(context_handle, geom);
        PyErr_Format(PyExc_RuntimeError, "Could not instantiate a new Geometry object");
    }
    return result;
}


static PyObject *GeometryObject_new(PyTypeObject *type, PyObject *args,
                                    PyObject *kwds)
{
    void *context_handle = geos_context[0];
    GEOSGeometry *arg;
    GEOSGeometry *ptr;
    PyObject *self;
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
    ptr = GEOSGeom_clone_r(context_handle, arg);
    if (ptr == NULL) {
        RAISE_ILLEGAL_GEOS;
        return NULL;
    }
    self = GeometryObject_new_from_ptr(type, ptr);
    return (PyObject *) self;
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

static PyTypeObject GeometryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.ufuncs.GEOSGeometry",
    .tp_doc = "Geometry type",
    .tp_basicsize = sizeof(GeometryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = GeometryObject_new,
    .tp_dealloc = (destructor) GeometryObject_dealloc,
    .tp_members = GeometryObject_members,
    .tp_methods = GeometryObject_methods,
};

/* Define the geom -> bool functions (Y_b) */
static void *is_empty_data[1] = {GEOSisEmpty_r};
static void *is_simple_data[1] = {GEOSisSimple_r};
static void *is_ring_data[1] = {GEOSisRing_r};
static void *has_z_data[1] = {GEOSHasZ_r};
static void *is_closed_data[1] = {GEOSisClosed_r};
static void *is_valid_data[1] = {GEOSisValid_r};
typedef char FuncGEOS_Y_b(void *context, void *a);
static char Y_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};
static void Y_b_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_b *func = (FuncGEOS_Y_b *)data;
    void *context_handle = geos_context[0];
    npy_bool nanvalue = 0;
    if (data == is_empty_data[0]) { nanvalue = 1; }

    UNARY_LOOP {
        if GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) {
            *(npy_bool *) op1 = nanvalue;
            continue;
        }
        INPUT_Y;
        npy_bool ret = func(context_handle, in1->ptr);
        OUTPUT_b;
    }
}
static PyUFuncGenericFunction Y_b_funcs[1] = {&Y_b_func};


/* Define the geom, geom -> bool functions (YY_b) */
static void *disjoint_data[1] = {GEOSDisjoint_r};
static void *touches_data[1] = {GEOSTouches_r};
static void *intersects_data[1] = {GEOSIntersects_r};
static void *crosses_data[1] = {GEOSCrosses_r};
static void *within_data[1] = {GEOSWithin_r};
static void *contains_data[1] = {GEOSContains_r};
static void *overlaps_data[1] = {GEOSOverlaps_r};
static void *equals_data[1] = {GEOSEquals_r};
static void *covers_data[1] = {GEOSCovers_r};
static void *covered_by_data[1] = {GEOSCoveredBy_r};
typedef char FuncGEOS_YY_b(void *context, void *a, void *b);
static char YY_b_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_BOOL};
static void YY_b_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_b *func = (FuncGEOS_YY_b *)data;
    void *context_handle = geos_context[0];
    npy_bool nanvalue = 0;
    if (data == disjoint_data[0]) { nanvalue = 1; }

    BINARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            *(npy_bool *)op1 = nanvalue;
            continue;
        }
        INPUT_YY;
        npy_bool ret = func(context_handle, in1->ptr, in2->ptr);
        OUTPUT_b;
    }
}
static PyUFuncGenericFunction YY_b_funcs[1] = {&YY_b_func};

/* Define the geom -> geom functions (Y_Y) */
static void *clone_data[1] = {GEOSGeom_clone_r};
static void *envelope_data[1] = {GEOSEnvelope_r};
static void *convex_hull_data[1] = {GEOSConvexHull_r};
static void *boundary_data[1] = {GEOSBoundary_r};
static void *unary_union_data[1] = {GEOSUnaryUnion_r};
static void *point_on_surface_data[1] = {GEOSPointOnSurface_r};
static void *centroid_data[1] = {GEOSGetCentroid_r};
static void *line_merge_data[1] = {GEOSLineMerge_r};
static void *extract_unique_points_data[1] = {GEOSGeom_extractUniquePoints_r};
static void *get_start_point_data[1] = {GEOSGeomGetStartPoint_r};
static void *get_end_point_data[1] = {GEOSGeomGetEndPoint_r};
static void *get_exterior_ring_data[1] = {GEOSGetExteriorRing_r};
/* the normalize funcion acts inplace */
static void *GEOSNormalize_r_with_clone(void *context, void *geom) {
    void *ret = GEOSGeom_clone_r(context, geom);
    if (ret == NULL) {
        return NULL;
    }
    if (GEOSNormalize_r(context, geom) == -1) {
        return NULL;
    }
    return ret;
}
static void *normalize_data[1] = {GEOSNormalize_r_with_clone};
/* a linear-ring to polygon conversion function */
static void *GEOSLinearRingToPolygon(void *context, void *geom) {
    void *shell = GEOSGeom_clone_r(context, geom);
    if (shell == NULL) {
        return NULL;
    }
    return GEOSGeom_createPolygon_r(context, shell, NULL, 0);
}
static void *polygons_without_holes_data[1] = {GEOSLinearRingToPolygon};
typedef void *FuncGEOS_Y_Y(void *context, void *a);
static char Y_Y_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};
static void Y_Y_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_Y *func = (FuncGEOS_Y_Y *)data;
    void *context_handle = geos_context[0];

    UNARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_Y;
        GEOSGeometry *ret_ptr = func(context_handle, in1->ptr);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Y_Y_funcs[1] = {&Y_Y_func};

/* Define the geom, double -> geom functions (Yd_Y) */
static void *interpolate_data[1] = {GEOSInterpolate_r};
static void *interpolate_normalized_data[1] = {GEOSInterpolateNormalized_r};
static void *simplify_data[1] = {GEOSSimplify_r};
static void *simplify_preserve_topology_data[1] = {GEOSTopologyPreserveSimplify_r};
typedef void *FuncGEOS_Yd_Y(void *context, void *a, double b);
static char Yd_Y_dtypes[3] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void Yd_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_Yd_Y *func = (FuncGEOS_Yd_Y *)data;
    void *context_handle = geos_context[0];

    BINARY_LOOP {
        double in2 = *(double *)ip2;
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | npy_isnan(in2)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_Y;
        GEOSGeometry *ret_ptr = func(context_handle, in1->ptr, in2);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Yd_Y_funcs[1] = {&Yd_Y_func};

/* Define the geom, int -> geom functions (Yi_Y) */
static void *get_interior_ring_data[1] = {GEOSGetInteriorRingN_r};
static void *get_point_data[1] = {GEOSGeomGetPointN_r};
static void *get_geometry_data[1] = {GEOSGetGeometryN_r};
/* the set srid funcion acts inplace */
static void *GEOSSetSRID_r_with_clone(void *context, void *geom, int srid) {
    void *ret = GEOSGeom_clone_r(context, geom);
    if (ret == NULL) {
        return NULL;
    }
    GEOSSetSRID_r(context, ret, srid);
    return ret;
}
static void *set_srid_data[1] = {GEOSSetSRID_r_with_clone};
typedef void *FuncGEOS_Yi_Y(void *context, void *a, int b);
static char Yi_Y_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void Yi_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_Yi_Y *func = (FuncGEOS_Yi_Y *)data;
    void *context_handle = geos_context[0];

    BINARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_Y;
        int in2 = *(int *) ip2;
        GEOSGeometry *ret_ptr = func(context_handle, in1->ptr, in2);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction Yi_Y_funcs[1] = {&Yi_Y_func};

/* Define the geom, geom -> geom functions (YY_Y) */
static void *intersection_data[1] = {GEOSIntersection_r};
static void *difference_data[1] = {GEOSDifference_r};
static void *symmetric_difference_data[1] = {GEOSSymDifference_r};
static void *union_data[1] = {GEOSUnion_r};
static void *shared_paths_data[1] = {GEOSSharedPaths_r};
typedef void *FuncGEOS_YY_Y(void *context, void *a, void *b);
static char YY_Y_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void YY_Y_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_Y *func = (FuncGEOS_YY_Y *)data;
    void *context_handle = geos_context[0];

    BINARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_YY;
        GEOSGeometry *ret_ptr = func(context_handle, in1->ptr, in2->ptr);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction YY_Y_funcs[1] = {&YY_Y_func};

/* Define the geom -> double functions (Y_d) */
static void *get_x_data[1] = {GEOSGeomGetX_r};
static void *get_y_data[1] = {GEOSGeomGetY_r};
static void *area_data[1] = {GEOSArea_r};
static void *length_data[1] = {GEOSLength_r};
typedef int FuncGEOS_Y_d(void *context, void *a, double *b);
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};
static void Y_d_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_d *func = (FuncGEOS_Y_d *)data;
    void *context_handle = geos_context[0];

    UNARY_LOOP {
        if GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) {
            *(npy_double *) op1 = NPY_NAN;
            continue;
        }
        INPUT_Y;
        if (func(context_handle, in1->ptr, (npy_double *) op1) == 0) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
    }
}
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};

/* Define the geom -> unsigned byte functions (Y_B) */
static void *get_type_id_data[1] = {GEOSGeomTypeId_r};
static void *get_dimensions_data[1] = {GEOSGeom_getDimensions_r};
static void *get_coordinate_dimensions_data[1] = {GEOSGeom_getCoordinateDimension_r};
typedef int FuncGEOS_Y_B(void *context, void *a);
static char Y_B_dtypes[2] = {NPY_OBJECT, NPY_UBYTE};
static void Y_B_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_B *func = (FuncGEOS_Y_B *)data;
    void *context_handle = geos_context[0];
    int ret;

    UNARY_LOOP {
        INPUT_Y;
        ret = func(context_handle, in1->ptr);
        if ((ret < 0) | (ret > NPY_MAX_UBYTE)) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
        *(npy_ubyte *)op1 = ret;
    }
}
static PyUFuncGenericFunction Y_B_funcs[1] = {&Y_B_func};

/* Define the geom -> int functions (Y_i) */
static void *get_srid_data[1] = {GEOSGetSRID_r};
static void *get_num_geometries_data[1] = {GEOSGetNumGeometries_r};
static void *get_num_interior_rings_data[1] = {GEOSGetNumInteriorRings_r};
static void *get_num_points_data[1] = {GEOSGeomGetNumPoints_r};
static void *get_num_coordinates_data[1] = {GEOSGetNumCoordinates_r};
typedef int FuncGEOS_Y_i(void *context, void *a);
static char Y_i_dtypes[2] = {NPY_OBJECT, NPY_INT};
static void Y_i_func(char **args, npy_intp *dimensions,
                     npy_intp *steps, void *data)
{
    FuncGEOS_Y_i *func = (FuncGEOS_Y_i *)data;
    void *context_handle = geos_context[0];
    int ret;

    UNARY_LOOP {
        INPUT_Y;
        ret = func(context_handle, in1->ptr);
        if ((ret < 0) | (ret > NPY_MAX_INT)) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
        *(npy_int *)op1 = ret;
    }
}
static PyUFuncGenericFunction Y_i_funcs[1] = {&Y_i_func};

/* Define the geom, geom -> double functions (YY_d) */
static void *distance_data[1] = {GEOSDistance_r};
static void *hausdorff_distance_data[1] = {GEOSHausdorffDistance_r};
typedef int FuncGEOS_YY_d(void *context, void *a,  void *b, double *c);
static char YY_d_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE};
static void YY_d_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    FuncGEOS_YY_d *func = (FuncGEOS_YY_d *)data;
    void *context_handle = geos_context[0];

    BINARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            *(npy_double *) op1 = NPY_NAN;
            continue;
        }
        INPUT_YY;
        if (func(context_handle, in1->ptr, in2->ptr, (double *) op1) == 0) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
    }
}
static PyUFuncGenericFunction YY_d_funcs[1] = {&YY_d_func};

/* Define the geom, geom -> double functions that have different GEOS call signature (YY_d_2) */
static void *project_data[1] = {GEOSProject_r};
static void *project_normalized_data[1] = {GEOSProjectNormalized_r};
typedef double FuncGEOS_YY_d_2(void *context, void *a, void *b);
static char YY_d_2_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE};
static void YY_d_2_func(char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    FuncGEOS_YY_d_2 *func = (FuncGEOS_YY_d_2 *)data;
    void *context_handle = geos_context[0];
    double ret;

    BINARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            *(npy_double *) op1 = NPY_NAN;
            continue;
        }
        INPUT_YY;
        ret = func(context_handle, in1->ptr, in2->ptr);
        if (ret == -1.0) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
        *(npy_double *) op1 = ret;
    }
}
static PyUFuncGenericFunction YY_d_2_funcs[1] = {&YY_d_2_func};

/* Define functions with unique call signatures */
static void *null_data[1] = {NULL};
static void buffer_inner(void *context_handle, GEOSBufferParams *params, void *ip1, void *ip2, void *op1, char *status) {
    *status = 0;
    double in2 = *(double *) ip2;
    if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | npy_isnan(in2)) {
        OUTPUT_Y_NAN;
    } else {
        INPUT_Y;
        GEOSGeometry *ret_ptr = GEOSBufferWithParams_r(context_handle, in1->ptr, params, in2);
        OUTPUT_Y;
    }
    *status = 1;
}

static char buffer_dtypes[8] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT, NPY_INT, NPY_INT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void buffer_func(char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *ip4 = args[3], *ip5 = args[4], *ip6 = args[5], *ip7 = args[6], *op1 = args[7];
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], is4 = steps[3], is5 = steps[4], is6 = steps[5], is7 = steps[6], os1 = steps[7];
    npy_intp n = dimensions[0];
    npy_intp i;
    char status;

    if ((is3 != 0) | (is4 != 0) | (is5 != 0) | (is6 != 0) | (is7 != 0)) {
        PyErr_Format(PyExc_ValueError, "Buffer function called with non-scalar parameters");
        return;
    }

    GEOSBufferParams *params = GEOSBufferParams_create_r(context_handle);
    if (params == 0) {
        RAISE_ILLEGAL_GEOS;
        return;
    }
    if (!GEOSBufferParams_setQuadrantSegments_r(context_handle, params, *(int *) ip3)) {
        goto fail;
    }
    if (!GEOSBufferParams_setEndCapStyle_r(context_handle, params, *(int *) ip4)) {
        goto fail;
    }
    if (!GEOSBufferParams_setJoinStyle_r(context_handle, params, *(int *) ip5)) {
        goto fail;
    }
    if (!GEOSBufferParams_setMitreLimit_r(context_handle, params, *(double *) ip6)) {
        goto fail;
    }
    if (!GEOSBufferParams_setSingleSided_r(context_handle, params, *(npy_bool *) ip7)) {
        goto fail;
    }

    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {
        buffer_inner(context_handle, params, ip1, ip2, op1, &status);
        if (!status) {
            goto fail;
        }
    }

    GEOSBufferParams_destroy_r(context_handle, params);
    return;

    fail:
        RAISE_ILLEGAL_GEOS;
        GEOSBufferParams_destroy_r(context_handle, params);
}
static PyUFuncGenericFunction buffer_funcs[1] = {&buffer_func};


static char snap_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};
static void snap_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];

    TERNARY_LOOP {
        INPUT_YY;
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            OUTPUT_Y_NAN;
            continue;
        }
        double in3 = *(double *) ip3;
        GEOSGeometry *ret_ptr = GEOSSnap_r(context_handle, in1->ptr, in2->ptr, in3);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction snap_funcs[1] = {&snap_func};

static char equals_exact_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL};
static void equals_exact_func(char **args, npy_intp *dimensions,
                      npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];

    TERNARY_LOOP {
        INPUT_YY;
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2)) {
            *(npy_bool *)op1 = 0;
            continue;
        }
        double in3 = *(double *) ip3;
        npy_bool ret = GEOSEqualsExact_r(context_handle, in1->ptr, in2->ptr, in3);
        OUTPUT_b;
    }
}
static PyUFuncGenericFunction equals_exact_funcs[1] = {&equals_exact_func};


static char haussdorf_distance_densify_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE};
static void haussdorf_distance_densify_func(char **args, npy_intp *dimensions,
                                            npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];

    TERNARY_LOOP {
        INPUT_YY;
        double in3 = *(double *) ip3;
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1) | GEOM_ISNAN_OR_NONE(*(PyObject **)ip2) | npy_isnan(in3)) {
            *(npy_double *) op1 = NPY_NAN;
            continue;
        }
        if (GEOSHausdorffDistanceDensify_r(context_handle, in1->ptr, in2->ptr, in3, (double *) op1) == 0) {
            RAISE_ILLEGAL_GEOS;
            return;
        }
    }
}
static PyUFuncGenericFunction haussdorf_distance_densify_funcs[1] = {&haussdorf_distance_densify_func};


static char delaunay_triangles_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void delaunay_triangles_func(char **args, npy_intp *dimensions,
                                    npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];

    TERNARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_Y;
        double in2 = *(double *) ip2;
        if (npy_isnan(in2)) {
            in2 = 0.0;
        }
        npy_bool in3 = *(npy_bool *) ip3;

        GEOSGeometry *ret_ptr = GEOSDelaunayTriangulation_r(context_handle, in1->ptr, in2, (int) in3);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction delaunay_triangles_funcs[1] = {&delaunay_triangles_func};


static char voronoi_polygons_dtypes[5] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};
static void voronoi_polygons_func(char **args, npy_intp *dimensions,
                                  npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    void *envelope_ptr;

    QUATERNARY_LOOP {
        if (GEOM_ISNAN_OR_NONE(*(PyObject **)ip1)) {
            OUTPUT_Y_NAN;
            continue;
        }
        INPUT_Y;
        GeometryObject *in2 = *(PyObject **)ip2;
        if (GEOM_ISNAN_OR_NONE(in2)) {
            envelope_ptr = NULL;
        } else {
            CHECK_GEOM(in2);
            envelope_ptr = in2->ptr;
        }
        double in3 = *(double *) ip3;
        if (npy_isnan(in3)) {
            in3 = 0.0;
        }
        npy_bool in4 = *(npy_bool *) ip4;
        GEOSGeometry *ret_ptr = GEOSVoronoiDiagram_r(context_handle, in1->ptr, envelope_ptr, in3, (int) in4);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction voronoi_polygons_funcs[1] = {&voronoi_polygons_func};

/* define double -> geometry construction functions */

static char points_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void points_func(char **args, npy_intp *dimensions,
                        npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    SINGLE_COREDIM_LOOP_OUTER {
        CREATE_COORDSEQ(1, n_c1);
        SINGLE_COREDIM_LOOP_INNER {
            double coord = *(double *) cp1;
            SET_COORD(0, i_c1);
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createPoint_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction points_funcs[1] = {&points_func};


static char linestrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linestrings_func(char **args, npy_intp *dimensions,
                              npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    DOUBLE_COREDIM_LOOP_OUTER {
        CREATE_COORDSEQ(n_c1, n_c2);
        DOUBLE_COREDIM_LOOP_INNER_1 {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) cp2;
                SET_COORD(i_c1, i_c2);
            }
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createLineString_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction linestrings_funcs[1] = {&linestrings_func};


static char linearrings_dtypes[2] = {NPY_DOUBLE, NPY_OBJECT};
static void linearrings_func(char **args, npy_intp *dimensions,
                             npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    DOUBLE_COREDIM_LOOP_OUTER {
        /* check if first and last coords are equal; duplicate if necessary */
        char ring_closure = 0;
        DOUBLE_COREDIM_LOOP_INNER_2 {
            double first_coord = *(double *) (ip1 + i_c2 * cs2);
            double last_coord = *(double *) (ip1 + (n_c1 - 1) * cs1 + i_c2 * cs2);
            if (first_coord != last_coord) {
                ring_closure = 1;
                break;
            }
        }
        /* fill the coordinate sequence */
        CREATE_COORDSEQ(n_c1 + ring_closure, n_c2);
        DOUBLE_COREDIM_LOOP_INNER_1 {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) cp2;
                SET_COORD(i_c1, i_c2);
            }
        }
        /* add the closing coordinate if necessary */
        if (ring_closure) {
            DOUBLE_COREDIM_LOOP_INNER_2 {
                double coord = *(double *) (ip1 + i_c2 * cs2);
                SET_COORD(n_c1, i_c2);
            }
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createLinearRing_r(context_handle, coord_seq);
        OUTPUT_Y;
    }
}
static PyUFuncGenericFunction linearrings_funcs[1] = {&linearrings_func};


static char polygons_with_holes_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};
static void polygons_with_holes_func(char **args, npy_intp *dimensions,
                                     npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    void *shell;
    int n_holes;
    GEOSGeometry **holes = malloc(sizeof(void *) * dimensions[1]);
    if (holes == NULL) {
        RAISE_NO_MALLOC;
        goto finish;
    }

    BINARY_SINGLE_COREDIM_LOOP_OUTER {
        GeometryObject *g = *(GeometryObject **)ip1;
        CHECK_GEOM(g);
        shell = GEOSGeom_clone_r(context_handle, g->ptr);
        if (shell == NULL) {
            goto finish;
        }
        n_holes = 0;
        cp1 = ip2;
        BINARY_SINGLE_COREDIM_LOOP_INNER {
            GeometryObject *g = *(GeometryObject **)cp1;
            if GEOM_ISNAN_OR_NONE(g) {
                continue;
            }
            CHECK_GEOM(g);
            GEOSGeometry *hole = GEOSGeom_clone_r(context_handle, g->ptr);
            if (hole == NULL) {
                goto finish;
            }
            holes[i_c1] = hole;
            n_holes++;
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createPolygon_r(context_handle, shell, holes, n_holes);
        OUTPUT_Y;
    }

    finish:
        if (holes != NULL) { free(holes); }
}
static PyUFuncGenericFunction polygons_with_holes_funcs[1] = {&polygons_with_holes_func};


static char create_collection_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};
static void create_collection_func(char **args, npy_intp *dimensions,
                                   npy_intp *steps, void *data)
{
    void *context_handle = geos_context[0];
    int n_geoms;
    GEOSGeometry **geoms = malloc(sizeof(void *) * dimensions[1]);
    if (geoms == NULL) {
        RAISE_NO_MALLOC;
        goto finish;
    }
    int type;

    BINARY_SINGLE_COREDIM_LOOP_OUTER {
        type = *(int *) ip2;
        n_geoms = 0;
        cp1 = ip1;
        BINARY_SINGLE_COREDIM_LOOP_INNER {
            GeometryObject *g = *(GeometryObject **)cp1;
            if GEOM_ISNAN_OR_NONE(g) {
                continue;
            }
            CHECK_GEOM(g);
            geoms[n_geoms] = GEOSGeom_clone_r(context_handle, g->ptr);
            if (geoms[n_geoms] == NULL) {
                goto finish;
            }
            n_geoms++;
        }
        GEOSGeometry *ret_ptr = GEOSGeom_createCollection_r(context_handle, type, geoms, n_geoms);
        OUTPUT_Y;
    }

    finish:
        if (geoms != NULL) { free(geoms); }
}
static PyUFuncGenericFunction create_collection_funcs[1] = {&create_collection_func};

/*
TODO polygonizer functions
TODO prepared geometry predicate functions
TODO relate functions
TODO G -> char function GEOSisValidReason_r
*/


#define DEFINE_Y_b(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_b_funcs, NAME ##_data, Y_b_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_b(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_b_funcs, NAME ##_data, YY_b_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_Y_funcs, NAME ##_data, Y_Y_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Yi_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Yi_Y_funcs, NAME ##_data, Yi_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Yd_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Yd_Y_funcs, NAME ##_data, Yd_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_Y(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_Y_funcs, NAME ##_data, YY_Y_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_d(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, NAME ##_data, Y_d_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_B(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_B_funcs, NAME ##_data, Y_B_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_Y_i(NAME)\
    ufunc = PyUFunc_FromFuncAndData(Y_i_funcs, NAME ##_data, Y_i_dtypes, 1, 1, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_d(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_d_funcs, NAME ##_data, YY_d_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_YY_d_2(NAME)\
    ufunc = PyUFunc_FromFuncAndData(YY_d_2_funcs, NAME ##_data, YY_d_2_dtypes, 1, 2, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_CUSTOM(NAME, N_IN)\
    ufunc = PyUFunc_FromFuncAndData(NAME ##_funcs, null_data, NAME ##_dtypes, 1, N_IN, 1, PyUFunc_None, # NAME, "", 0);\
    PyDict_SetItemString(d, # NAME, ufunc)

#define DEFINE_GENERALIZED(NAME, N_IN, SIGNATURE)\
    ufunc = PyUFunc_FromFuncAndDataAndSignature(NAME ##_funcs, null_data, NAME ##_dtypes, 1, N_IN, 1, PyUFunc_None, # NAME, "", 0, SIGNATURE);\
    PyDict_SetItemString(d, # NAME, ufunc)


PyMODINIT_FUNC PyInit_ufuncs(void)
{
    PyObject *m, *d, *ufunc;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    if (PyType_Ready(&GeometryType) < 0)
        return NULL;

    Py_INCREF(&GeometryType);
    PyModule_AddObject(m, "Geometry", (PyObject *) &GeometryType);

    d = PyModule_GetDict(m);

    import_array();
    import_umath();

    void *context_handle = GEOS_init_r();
    PyObject* GEOSException = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
    PyModule_AddObject(m, "GEOSException", GEOSException);
    GEOSContext_setErrorMessageHandler_r(context_handle, HandleGEOSError, GEOSException);
    GEOSContext_setNoticeMessageHandler_r(context_handle, HandleGEOSNotice, NULL);
    geos_context[0] = context_handle;  /* for global access */

    DEFINE_Y_b (is_empty);
    DEFINE_Y_b (is_simple);
    DEFINE_Y_b (is_ring);
    DEFINE_Y_b (has_z);
    DEFINE_Y_b (is_closed);
    DEFINE_Y_b (is_valid);

    DEFINE_YY_b (disjoint);
    DEFINE_YY_b (touches);
    DEFINE_YY_b (intersects);
    DEFINE_YY_b (crosses);
    DEFINE_YY_b (within);
    DEFINE_YY_b (contains);
    DEFINE_YY_b (overlaps);
    DEFINE_YY_b (equals);
    DEFINE_YY_b (covers);
    DEFINE_YY_b (covered_by);

    DEFINE_Y_Y (clone);
    DEFINE_Y_Y (envelope);
    DEFINE_Y_Y (convex_hull);
    DEFINE_Y_Y (boundary);
    DEFINE_Y_Y (unary_union);
    DEFINE_Y_Y (point_on_surface);
    DEFINE_Y_Y (centroid);
    DEFINE_Y_Y (line_merge);
    DEFINE_Y_Y (extract_unique_points);
    DEFINE_Y_Y (get_start_point);
    DEFINE_Y_Y (get_end_point);
    DEFINE_Y_Y (get_exterior_ring);
    DEFINE_Y_Y (normalize);

    DEFINE_Yi_Y (get_interior_ring);
    DEFINE_Yi_Y (get_point);
    DEFINE_Yi_Y (get_geometry);
    DEFINE_Yi_Y (set_srid);

    DEFINE_Yd_Y (interpolate);
    DEFINE_Yd_Y (interpolate_normalized);
    DEFINE_Yd_Y (simplify);
    DEFINE_Yd_Y (simplify_preserve_topology);

    DEFINE_YY_Y (intersection);
    DEFINE_YY_Y (difference);
    DEFINE_YY_Y (symmetric_difference);
    DEFINE_YY_Y (union);
    DEFINE_YY_Y (shared_paths);

    DEFINE_Y_d (get_x);
    DEFINE_Y_d (get_y);
    DEFINE_Y_d (area);
    DEFINE_Y_d (length);

    DEFINE_Y_B (get_type_id);
    DEFINE_Y_B (get_dimensions);
    DEFINE_Y_B (get_coordinate_dimensions);

    DEFINE_Y_i (get_srid);
    DEFINE_Y_i (get_num_geometries);
    DEFINE_Y_i (get_num_interior_rings);
    DEFINE_Y_i (get_num_points);
    DEFINE_Y_i (get_num_coordinates);

    DEFINE_YY_d (distance);
    DEFINE_YY_d (hausdorff_distance);

    DEFINE_YY_d_2 (project);
    DEFINE_YY_d_2 (project_normalized);

    DEFINE_CUSTOM (buffer, 7);
    DEFINE_CUSTOM (snap, 3);
    DEFINE_CUSTOM (equals_exact, 3);
    DEFINE_CUSTOM (haussdorf_distance_densify, 3);
    DEFINE_CUSTOM (delaunay_triangles, 3);
    DEFINE_CUSTOM (voronoi_polygons, 4);
    DEFINE_GENERALIZED(points, 1, "(d)->()");
    DEFINE_GENERALIZED(linestrings, 1, "(i, d)->()");
    DEFINE_GENERALIZED(linearrings, 1, "(i, d)->()");
    DEFINE_Y_Y (polygons_without_holes);
    DEFINE_GENERALIZED(polygons_with_holes, 2, "(),(i)->()");
    DEFINE_GENERALIZED(create_collection, 2, "(i),()->()");

    Py_DECREF(ufunc);
    return m;
}
