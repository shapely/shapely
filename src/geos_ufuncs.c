#include <Python.h>
#include <math.h>
#include <geos_c.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/* This tells Python what methods this module has. */
static PyMethodDef GeosModule[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "geos_ufuncs",
    NULL,
    -1,
    GeosModule,
    NULL,
    NULL,
    NULL,
    NULL
};

/* This defines the pygeos.GEOM_DTYPE in terms of a C struct. */
typedef struct GeomArrayValue {
   PyObject *obj;
   void *_ptr;
} GeomArrayValue;


/* This defines the ufunc for vector -> bool functions from GEOS. */
typedef char FuncGEOS_G_b(void *context, void *a);
static void PyUFuncGEOS_G_b(char **args, npy_intp *dimensions,
                               npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    void *context_handle;
    GeomArrayValue *a;
    npy_bool b;

    FuncGEOS_G_b *f = (FuncGEOS_G_b *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in;

        if (a->_ptr == NULL) {
            goto fail_hasnull;
        } else {
            b = f(context_handle, a->_ptr);
            if (b == 2) {
                goto fail;
            }
        }

        *out = (npy_bool) b;

        in += in_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail_hasnull:
        PyErr_Format(PyExc_ValueError, "Cannot apply unary predicate to NULL geometry");
        GEOS_finish_r(context_handle);
        return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}


/* This defines the ufunc for vector, vector -> bool functions from GEOS. */
typedef char FuncGEOS_GG_b(void *context, void *a, void *b);
static void PyUFuncGEOS_GG_b(char **args, npy_intp *dimensions,
                                npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    void *context_handle;
    GeomArrayValue *a, *b;
    npy_bool c;


    FuncGEOS_GG_b *f = (FuncGEOS_GG_b *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in1;
        b = (GeomArrayValue *)in2;
        if ((a->_ptr == NULL) || (b->_ptr == NULL)) {
            goto fail_hasnull;
        } else {
            c = f(context_handle, a->_ptr, b->_ptr);
            if (c == 2) {
                goto fail;
            }
        }

        *out = (npy_bool) c;

        in1 += in1_step;
        in2 += in2_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail_hasnull:
        PyErr_Format(PyExc_ValueError, "Cannot apply binary predicate to NULL geometries");
        GEOS_finish_r(context_handle);
        return -1;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}


/* This defines the ufunc for vector -> vector functions from GEOS. */
typedef void *FuncGEOS_G_G(void *context, void *a);
static void PyUFuncGEOS_G_G(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    void *context_handle;
    GeomArrayValue *a, *b;

    FuncGEOS_G_G *f = (FuncGEOS_G_G *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in;
        b = (GeomArrayValue *)out;
        b->obj = Py_None;

        if (a->_ptr == NULL) {
            b->_ptr = NULL;
        } else {
            b->_ptr = f(context_handle, a->_ptr);
            if (b->_ptr == NULL) {
                goto fail;
            }
        }

        in += in_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}


/* This defines the ufunc for vector, double -> vector functions from GEOS. */
typedef void *FuncGEOS_Gd_G(void *context, void *a, double *b);
static void PyUFuncGEOS_Gd_G(char **args, npy_intp *dimensions,
                             npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    void *context_handle;
    GeomArrayValue *a, *c;

    FuncGEOS_Gd_G *f = (FuncGEOS_Gd_G *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in1;
        c = (GeomArrayValue *)out;
        c->obj = Py_None;

        if (a->_ptr == NULL) {
            c->_ptr = NULL;
        } else {
            c->_ptr = f(context_handle, a->_ptr, *in2);
            if (c->_ptr == NULL) {
                goto fail;
            }
        }

        in1 += in1_step;
        in2 += in2_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}

/* This defines the ufunc for vector, vector -> vector functions from GEOS. */
typedef void *FuncGEOS_GG_G(void *context, void *a, void *b);
static void PyUFuncGEOS_GG_G(char **args, npy_intp *dimensions,
                             npy_intp* steps, void* data)
{

    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    void *context_handle;
    GeomArrayValue *a, *b, *c;

    FuncGEOS_GG_G *f = (FuncGEOS_GG_G *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in1;
        b = (GeomArrayValue *)in2;
        c = (GeomArrayValue *)out;
        c->obj = Py_None;

        if ((a->_ptr == NULL) || (b->_ptr == NULL)) {
            c->_ptr = NULL;
        } else {
            c->_ptr = f(context_handle, a->_ptr, b->_ptr);
            if (c->_ptr == NULL) {
                goto fail;
            }
        }

        in1 += in1_step;
        in2 += in2_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}



/* This defines the ufunc for vector -> double functions from GEOS. */
typedef int FuncGEOS_G_d(void *context, void *a, double *b);
static void PyUFuncGEOS_G_d(char **args, npy_intp *dimensions,
                           npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    void *context_handle;
    GeomArrayValue *a;

    FuncGEOS_G_d *f = (FuncGEOS_G_d *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in;

        if (a->_ptr == NULL) {
            *out = NPY_NAN;
        } else {
            if (f(context_handle, a->_ptr, out) == 0) {
                goto fail;
            }
        }

        in += in_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}


/* This defines the ufunc for vector, vector -> double functions from GEOS. */
typedef int FuncGEOS_GG_d(void *context, void *a, void *b, double *c);
static void PyUFuncGEOS_GG_d(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    void *context_handle;
    GeomArrayValue *a, *b;

    FuncGEOS_GG_d *f = (FuncGEOS_GG_d *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in1;
        b = (GeomArrayValue *)in2;

        if ((a->_ptr == NULL) || (b->_ptr == NULL)) {
            *out = NPY_NAN;
        } else {
            if (f(context_handle, a->_ptr, b->_ptr, out) == 0) {
                goto fail;
            }
        }

        in1 += in1_step;
        in2 += in2_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}

/* This defines the ufunc for vector -> uint8 functions from GEOS. */
typedef int FuncGEOS_G_i(void *context, void *a);
static void PyUFuncGEOS_G_u1(char **args, npy_intp *dimensions,
                             npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    void *context_handle;
    GeomArrayValue *a;
    int b;

    FuncGEOS_G_i *f = (FuncGEOS_G_i *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in;

        if (a->_ptr == NULL) {
            *out = 255;
        } else {
            b = f(context_handle, a->_ptr);
            if ((b < 0) | (b > UINT8_MAX)) {
                goto fail;
            } else {
                *out = (npy_uint8) b;
            }
        }

        in += in_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}


/* This defines the ufunc for vector -> uint32 functions from GEOS. */
static void PyUFuncGEOS_G_u4(char **args, npy_intp *dimensions,
                             npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    void *context_handle;
    GeomArrayValue *a;
    int b;

    FuncGEOS_G_i *f = (FuncGEOS_G_i *)data;
    context_handle = GEOS_init_r();

    for (i = 0; i < n; i++) {
        a = (GeomArrayValue *)in;

        if (a->_ptr == NULL) {
            *out = UINT32_MAX;
        } else {
            b = f(context_handle, a->_ptr);
            if ((b < 0) | (b > UINT32_MAX)) {
                goto fail;
            } else {
                *out = (npy_uint32) b;
            }
        }

        in += in_step;
        out += out_step;

    }

    GEOS_finish_r(context_handle);

    return;

    fail:
        PyErr_Format(PyExc_RuntimeError, "GEOS Operation failed");
        GEOS_finish_r(context_handle);
        return;
}

static void RegisterPyUFuncGEOS_G_b(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[2];
    dtypes[0] = geom_dtype;
    dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_G_b, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_GG_b(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[3];
    dtypes[0] = geom_dtype;
    dtypes[1] = geom_dtype;
    dtypes[2] = PyArray_DescrFromType(NPY_BOOL);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_GG_b, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_G_G(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[2];
    dtypes[0] = geom_dtype;
    dtypes[1] = geom_dtype;

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_G_G, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_Gd_G(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[3];
    dtypes[0] = geom_dtype;
    dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);
    dtypes[2] = geom_dtype;

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_Gd_G, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_GG_G(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[3];
    dtypes[0] = geom_dtype;
    dtypes[1] = geom_dtype;
    dtypes[2] = geom_dtype;

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_GG_G, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};


static void RegisterPyUFuncGEOS_G_d(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[2];
    dtypes[0] = geom_dtype;
    dtypes[1] = PyArray_DescrFromType(NPY_DOUBLE);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_G_d, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_GG_d(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 2, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[3];
    dtypes[0] = geom_dtype;
    dtypes[1] = geom_dtype;
    dtypes[2] = PyArray_DescrFromType(NPY_DOUBLE);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_GG_d, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_G_u1(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[2];
    dtypes[0] = geom_dtype;
    dtypes[1] = PyArray_DescrFromType(NPY_UINT8);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_G_u1, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

static void RegisterPyUFuncGEOS_G_u4(
        char *name, void *geos_func, PyArray_Descr *geom_dtype, PyObject *d) {
    PyObject *pyufunc;
    pyufunc = PyUFunc_FromFuncAndData(
        NULL, NULL, NULL, 0, 1, 1, PyUFunc_None, name, "", 0);

    PyArray_Descr *dtypes[2];
    dtypes[0] = geom_dtype;
    dtypes[1] = PyArray_DescrFromType(NPY_UINT32);

    PyUFunc_RegisterLoopForDescr(
        pyufunc, geom_dtype, &PyUFuncGEOS_G_u4, dtypes, geos_func
    );
    PyDict_SetItemString(d, name, pyufunc);
    Py_DECREF(pyufunc);
};

PyMODINIT_FUNC PyInit_geos_ufuncs(void)
{
    PyObject *m, *d, *dtype_dict;
    PyArray_Descr *dt;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    d = PyModule_GetDict(m);
    import_array();
    import_umath();

    /* Define the geometry structured dtype (see pygeos.GEOM_DTYPE) */

    dtype_dict = Py_BuildValue(
        "[(s, s), (s, s)]",
        "obj", "O", "_ptr", "intp"
    );
    PyArray_DescrConverter(dtype_dict, &dt);

    /* Register the GEOS functions. List is based on CAPI 3.3.0 */
    /* TODO GG -> d function GEOSProject_r */
    RegisterPyUFuncGEOS_Gd_G("interpolate", GEOSInterpolate_r, dt, d);
    /* TODO GG -> d function GEOSProjectNormalized_r */
    RegisterPyUFuncGEOS_Gd_G("interpolate_normalized", GEOSInterpolateNormalized_r, dt, d);
    /* TODO buffer functions */
    /* TODO possibly implement some creation functions */
    RegisterPyUFuncGEOS_G_G("clone", GEOSGeom_clone_r, dt, d);
    /* TODO G -> void function GEOSGeom_destroy_r */
    RegisterPyUFuncGEOS_G_G("envelope", GEOSEnvelope_r, dt, d);
    RegisterPyUFuncGEOS_GG_G("intersection", GEOSIntersection_r, dt, d);
    RegisterPyUFuncGEOS_G_G("convex_hull", GEOSConvexHull_r, dt, d);
    RegisterPyUFuncGEOS_GG_G("difference", GEOSDifference_r, dt, d);
    RegisterPyUFuncGEOS_GG_G("symmetric_difference", GEOSSymDifference_r, dt, d);
    RegisterPyUFuncGEOS_G_G("boundary", GEOSBoundary_r, dt, d);
    RegisterPyUFuncGEOS_GG_G("union", GEOSUnion_r, dt, d);
    RegisterPyUFuncGEOS_G_G("unary_union", GEOSUnaryUnion_r, dt, d);
    RegisterPyUFuncGEOS_G_G("point_on_surface", GEOSPointOnSurface_r, dt, d);
    RegisterPyUFuncGEOS_G_G("get_centroid", GEOSGetCentroid_r, dt, d);
    /* TODO polygonizer functions */
    RegisterPyUFuncGEOS_G_G("line_merge", GEOSLineMerge_r, dt, d);
    RegisterPyUFuncGEOS_Gd_G("simplify", GEOSSimplify_r, dt, d);
    RegisterPyUFuncGEOS_Gd_G("topology_preserve_simplify", GEOSTopologyPreserveSimplify_r, dt, d);
    RegisterPyUFuncGEOS_G_G("extract_unique_points", GEOSGeom_extractUniquePoints_r, dt, d);
    RegisterPyUFuncGEOS_GG_G("shared_paths", GEOSSharedPaths_r, dt, d);
    /* TODO GGd -> G function GEOSSnap_r */
    RegisterPyUFuncGEOS_GG_b("disjoint", GEOSDisjoint_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("touches", GEOSTouches_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("intersects", GEOSIntersects_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("crosses", GEOSCrosses_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("within", GEOSWithin_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("contains", GEOSContains_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("overlaps", GEOSOverlaps_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("equals", GEOSEquals_r, dt, d);
    /* TODO GGd -> b function GEOSEqualsExact_r */
    RegisterPyUFuncGEOS_GG_b("covers", GEOSCovers_r, dt, d);
    RegisterPyUFuncGEOS_GG_b("covered_by", GEOSCoveredBy_r, dt, d);
    /* TODO prepared geometry predicate functions */
    RegisterPyUFuncGEOS_G_b("is_empty", GEOSisEmpty_r, dt, d);
    RegisterPyUFuncGEOS_G_b("is_simple", GEOSisSimple_r, dt, d);
    RegisterPyUFuncGEOS_G_b("is_ring", GEOSisRing_r, dt, d);
    RegisterPyUFuncGEOS_G_b("has_z", GEOSHasZ_r, dt, d);
    RegisterPyUFuncGEOS_G_b("is_closed", GEOSisClosed_r, dt, d);
    /* TODO relate functions */
    RegisterPyUFuncGEOS_G_b("is_valid", GEOSisValid_r, dt, d);
    /* TODO G -> char function GEOSisValidReason_r */
    RegisterPyUFuncGEOS_G_u1("geom_type_id", GEOSGeomTypeId_r, dt, d);
    RegisterPyUFuncGEOS_G_u4("get_srid", GEOSGetSRID_r, dt, d);
    /* TODO Gi -> void function GEOSSetSRID_r */
    RegisterPyUFuncGEOS_G_u4("get_num_geometries", GEOSGetNumGeometries_r, dt, d);
    /* TODO G -> void function GEOSNormalize_r */
    RegisterPyUFuncGEOS_G_u4("get_num_interior_rings", GEOSGetNumInteriorRings_r, dt, d);
    RegisterPyUFuncGEOS_G_u4("get_num_points", GEOSGeomGetNumPoints_r, dt, d);
    RegisterPyUFuncGEOS_G_d("get_x", GEOSGeomGetX_r, dt, d);
    RegisterPyUFuncGEOS_G_d("get_y", GEOSGeomGetY_r, dt, d);
    /* TODO Gi -> G function GEOSGetInteriorRingN_r */
    RegisterPyUFuncGEOS_G_G("get_exterior_ring", GEOSGetExteriorRing_r, dt, d);
    RegisterPyUFuncGEOS_G_u4("get_num_coordinates", GEOSGetNumCoordinates_r, dt, d);
    RegisterPyUFuncGEOS_G_u1("get_dimensions", GEOSGeom_getDimensions_r, dt, d);
    RegisterPyUFuncGEOS_G_u1("get_coordinate_dimensions", GEOSGeom_getCoordinateDimension_r, dt, d);
    /* TODO Gi -> i function GEOSGeomGetPointN_r */
    RegisterPyUFuncGEOS_G_G("get_start_point", GEOSGeomGetStartPoint_r, dt, d);
    RegisterPyUFuncGEOS_G_G("get_end_point", GEOSGeomGetEndPoint_r, dt, d);
    RegisterPyUFuncGEOS_G_d("area", GEOSArea_r, dt, d);
    RegisterPyUFuncGEOS_G_d("length", GEOSLength_r, dt, d);
    RegisterPyUFuncGEOS_GG_d("distance", GEOSDistance_r, dt, d);
    RegisterPyUFuncGEOS_GG_d("hausdorff_distance", GEOSHausdorffDistance_r, dt, d);
    /* TODO GGd -> d function GEOSHausdorffDistanceDensify_r */
    RegisterPyUFuncGEOS_G_d("get_length", GEOSGeomGetLength_r, dt, d);

    Py_DECREF(dtype_dict);
    Py_DECREF(dt);
    return m;
}
