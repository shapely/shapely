#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "strtree.h"
#include "geos.h"
#include "pygeom.h"
#include "kvec.h"

/* GEOS function that takes a prepared geometry and a regular geometry
 * and returns bool value */

typedef char FuncGEOS_YpY_b(void *context, const GEOSPreparedGeometry *a,
                            const GEOSGeometry *b);


/* Copy values from arr to a new numpy integer array.
 * The order of values from arr is inverted, because arr is created by pushing
 * values onto the end. */

static PyArrayObject *copy_kvec_to_npy(npy_intp_vec *arr)
{
    npy_intp i;
    npy_intp size = kv_size(*arr);
    npy_intp *ptr;

    npy_intp dims[1] = {size};
    // the following raises a compiler warning based on how the macro is defined
    // in numpy.  There doesn't appear to be anything we can do to avoid it.
    PyArrayObject *result = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INTP);
    if (result == NULL) {
        return NULL;
    }

    for (i = 0; i<size; i++) {
        ptr = PyArray_GETPTR1(result, i);
        *ptr = kv_A(*arr, i);
    }

    return (PyArrayObject *) result;
}

static void STRtree_dealloc(STRtreeObject *self)
{
    void *context = geos_context[0];
    size_t i, size;
    // free the tree
    if (self->ptr != NULL) { GEOSSTRtree_destroy_r(context, self->ptr); }
    // free the geometries
    size = kv_size(self->_geoms);
    for (i = 0; i < size; i++) {
        Py_XDECREF(kv_pop(self->_geoms));
    }
    kv_destroy(self->_geoms);
    // free the PyObject
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *STRtree_new(PyTypeObject *type, PyObject *args,
                             PyObject *kwds)
{
    int node_capacity;
    PyObject *arr;
    void *tree, *ptr;
    npy_intp n, i, count = 0;
    GEOSGeometry *geom;
    geom_obj_vec _geoms;
    GeometryObject *obj;
    GEOSContextHandle_t context = geos_context[0];

    if (!PyArg_ParseTuple(args, "Oi", &arr, &node_capacity)) {
        return NULL;
    }
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Not an ndarray");
        return NULL;
    }
    if (!PyArray_ISOBJECT((PyArrayObject *) arr)) {
        PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
        return NULL;
    }
    if (PyArray_NDIM((PyArrayObject *) arr) != 1) {
        PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
        return NULL;
    }

    tree = GEOSSTRtree_create_r(context, (size_t) node_capacity);
    if (tree == NULL) {
        return NULL;
    }

    n = PyArray_SIZE((PyArrayObject *) arr);

    kv_init(_geoms);
    for(i = 0; i < n; i++) {
        /* get the geometry */
        ptr = PyArray_GETPTR1((PyArrayObject *) arr, i);
        obj = *(GeometryObject **) ptr;
        /* fail and cleanup incase obj was no geometry */
        if (!get_geom(obj, &geom)) {
            GEOSSTRtree_destroy_r(context, tree);
            // free the geometries
            count = kv_size(_geoms);
            for (i = 0; i < count; i++) { Py_XDECREF(kv_pop(_geoms)); }
            kv_destroy(_geoms);
            return NULL;
        }
        /* skip incase obj was None */
        if (geom == NULL) {
            kv_push(GeometryObject *, _geoms, NULL);
        } else {
        /* perform the insert */
            Py_INCREF(obj);
            kv_push(GeometryObject *, _geoms, obj);
            count++;
            GEOSSTRtree_insert_r(context, tree, geom, (void *) i );
        }
    }

    STRtreeObject *self = (STRtreeObject *) type->tp_alloc(type, 0);
    if (self == NULL) {
        GEOSSTRtree_destroy_r(context, tree);
        return NULL;
    }
    self->ptr = tree;
    self->count = count;
    self->_geoms = _geoms;
    return (PyObject *) self;
}


/* Callback to give to strtree_query
 * Given the value returned from each intersecting geometry it inserts that
 * value (the index) into the given size_vector */

void query_callback(void *item, void *user_data)
{
    kv_push(npy_intp, *(npy_intp_vec *)user_data, (npy_intp) item);
}


/* Query the tree based on input geometry and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned. */

static PyObject *STRtree_query(STRtreeObject *self, PyObject *args) {
    GEOSContextHandle_t context = geos_context[0];
    GeometryObject *geometry, *target_geometry;
    int predicate = 0; // default no predicate
    GEOSGeometry *geom, *target_geom;
    const GEOSPreparedGeometry *pgeom;
    npy_intp_vec arr, arr2; // Resizable array for matches for each geometry
    npy_intp i, size, index;
    FuncGEOS_YpY_b *predicate_func;
    PyArrayObject *result;

    if (self->ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
        return NULL;
    }
    if (self->count == 0) {
        npy_intp dims[1] = {0};
        return PyArray_SimpleNew(1, dims, NPY_INTP);
    }

    if (!PyArg_ParseTuple(args, "O!i", &GeometryType, &geometry, &predicate)){
        return NULL;
    }

    if (!get_geom(geometry, &geom)) {
        PyErr_SetString(PyExc_TypeError, "Invalid geometry");
        return NULL;
    }

    // query the tree for indices of geometries in the tree with
    // envelopes that intersect the geometry.
    kv_init(arr);
    if (geom != NULL) {
        GEOSSTRtree_query_r(context, self->ptr, geom, query_callback, &arr);
    }

    // No predicate function provided, return all geometry indexes from
    // query.
    // If array is empty, return an empty numpy array
    if (predicate == 0 || kv_size(arr) == 0) {
        result = copy_kvec_to_npy(&arr);
        kv_destroy(arr);
        return (PyObject *) result;
    }

    switch (predicate) {
        case 1: {  // intersects
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedIntersects_r;
            break;
        }
        case 2: { // within
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedWithin_r;
            break;
        }
        case 3: { // contains
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedContains_r;
            break;
        }
        case 4: { // overlaps
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedOverlaps_r;
            break;
        }
        case 5: { // crosses
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedCrosses_r;
            break;
        }
        case 6: { // touches
            predicate_func = (FuncGEOS_YpY_b *)GEOSPreparedTouches_r;
            break;
        }
        default: { // unknown predicate
            PyErr_SetString(PyExc_ValueError, "Invalid query predicate");
            return NULL;
        }
    }

    pgeom = GEOSPrepare_r(context, geom);
    if (pgeom == NULL) {
        kv_destroy(arr);
        return NULL;
    }

    size = kv_size(arr);
    kv_init(arr2);

    for (i = 0; i < size; i++) {
        // get index for right geometries from arr
        index = kv_A(arr, i);

        // get GEOS geometry from pygeos geometry
        target_geometry = kv_A(self->_geoms, index);
        if (target_geometry == NULL) { continue; }
        get_geom((GeometryObject *) target_geometry, &target_geom);

        // keep the index value if it passes the predicate
        if (predicate_func(context, pgeom, target_geom)) {
            kv_push(npy_intp, arr2, index);
        }
    }

    result = copy_kvec_to_npy(&arr2);

    GEOSPreparedGeom_destroy_r(context, pgeom);
    kv_destroy(arr);
    kv_destroy(arr2);

    return (PyObject *) result;
}

static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY, "Pointer to GEOSSTRtree"},
    {"count", T_LONG, offsetof(STRtreeObject, count), READONLY, "The number of geometries inside the tree"},
    {NULL}  /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction) STRtree_query, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometry, and optionally tests them "
     "against predicate function if provided. "
    },
    {NULL}  /* Sentinel */
};

PyTypeObject STRtreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pygeos.lib.STRtree",
    .tp_doc = "A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.",
    .tp_basicsize = sizeof(STRtreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = STRtree_new,
    .tp_dealloc = (destructor) STRtree_dealloc,
    .tp_members = STRtree_members,
    .tp_methods = STRtree_methods
};


int init_strtree_type(PyObject *m)
{
    if (PyType_Ready(&STRtreeType) < 0) {
        return -1;
    }

    Py_INCREF(&STRtreeType);
    PyModule_AddObject(m, "STRtree", (PyObject *) &STRtreeType);
    return 0;
}
