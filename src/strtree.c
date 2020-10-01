#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "geos.h"
#include "kvec.h"
#include "pygeom.h"
#include "strtree.h"

/* GEOS function that takes a prepared geometry and a regular geometry
 * and returns bool value */

typedef char FuncGEOS_YpY_b(void* context, const GEOSPreparedGeometry* a,
                            const GEOSGeometry* b);

/* get predicate function based on ID.  See strtree.py::BinaryPredicate for
 * lookup table of id to function name */

FuncGEOS_YpY_b* get_predicate_func(int predicate_id) {
  switch (predicate_id) {
    case 1: {  // intersects
      return (FuncGEOS_YpY_b*)GEOSPreparedIntersects_r;
    }
    case 2: {  // within
      return (FuncGEOS_YpY_b*)GEOSPreparedWithin_r;
    }
    case 3: {  // contains
      return (FuncGEOS_YpY_b*)GEOSPreparedContains_r;
    }
    case 4: {  // overlaps
      return (FuncGEOS_YpY_b*)GEOSPreparedOverlaps_r;
    }
    case 5: {  // crosses
      return (FuncGEOS_YpY_b*)GEOSPreparedCrosses_r;
    }
    case 6: {  // touches
      return (FuncGEOS_YpY_b*)GEOSPreparedTouches_r;
    }
    case 7: {  // covers
      return (FuncGEOS_YpY_b*)GEOSPreparedCovers_r;
    }
    case 8: {  // covered_by
      return (FuncGEOS_YpY_b*)GEOSPreparedCoveredBy_r;
    }
    case 9: {  // contains_properly
      return (FuncGEOS_YpY_b*)GEOSPreparedContainsProperly_r;
    }
    default: {  // unknown predicate
      PyErr_SetString(PyExc_ValueError, "Invalid query predicate");
      return NULL;
    }
  }
}

/* Copy values from arr to a new numpy integer array.
 *
 * Parameters
 * ----------
 * arr: dynamic vector array to convert to ndarray
 */

static PyArrayObject* copy_kvec_to_npy(npy_intp_vec* arr) {
  npy_intp i;
  npy_intp size = kv_size(*arr);

  npy_intp dims[1] = {size};
  // the following raises a compiler warning based on how the macro is defined
  // in numpy.  There doesn't appear to be anything we can do to avoid it.
  PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INTP);
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  for (i = 0; i < size; i++) {
    // assign value into numpy array
    *(npy_intp*)PyArray_GETPTR1(result, i) = kv_A(*arr, i);
  }

  return (PyArrayObject*)result;
}

static void STRtree_dealloc(STRtreeObject* self) {
  size_t i, size;

  // free the tree
  if (self->ptr != NULL) {
    GEOS_INIT;
    GEOSSTRtree_destroy_r(ctx, self->ptr);
    GEOS_FINISH;
  }
  // free the geometries
  size = kv_size(self->_geoms);
  for (i = 0; i < size; i++) {
    Py_XDECREF(kv_A(self->_geoms, i));
  }
  kv_destroy(self->_geoms);
  // free the PyObject
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* STRtree_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  int node_capacity;
  PyObject* arr;
  void *tree, *ptr;
  npy_intp n, i, count = 0;
  GEOSGeometry* geom;
  pg_geom_obj_vec _geoms;
  GeometryObject* obj;

  if (!PyArg_ParseTuple(args, "Oi", &arr, &node_capacity)) {
    return NULL;
  }
  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }
  if (!PyArray_ISOBJECT((PyArrayObject*)arr)) {
    PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
    return NULL;
  }
  if (PyArray_NDIM((PyArrayObject*)arr) != 1) {
    PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
    return NULL;
  }

  GEOS_INIT;

  tree = GEOSSTRtree_create_r(ctx, (size_t)node_capacity);
  if (tree == NULL) {
    errstate = PGERR_GEOS_EXCEPTION;
    return NULL;
    GEOS_FINISH;
  }

  n = PyArray_SIZE((PyArrayObject*)arr);

  kv_init(_geoms);
  for (i = 0; i < n; i++) {
    /* get the geometry */
    ptr = PyArray_GETPTR1((PyArrayObject*)arr, i);
    obj = *(GeometryObject**)ptr;
    /* fail and cleanup incase obj was no geometry */
    if (!get_geom(obj, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      GEOSSTRtree_destroy_r(ctx, tree);
      // free the geometries
      count = kv_size(_geoms);
      for (i = 0; i < count; i++) {
        Py_XDECREF(kv_A(_geoms, i));
      }
      kv_destroy(_geoms);
      GEOS_FINISH;
      return NULL;
    }
    /* If geometry is None or empty, do not add it to the tree or count.
     * Set it as NULL for the internal geometries used for predicate tests.
     */
    if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
      kv_push(GeometryObject*, _geoms, NULL);
    } else {
      /* perform the insert */
      Py_INCREF(obj);
      kv_push(GeometryObject*, _geoms, obj);
      count++;
      GEOSSTRtree_insert_r(ctx, tree, geom, (void*)i);
    }
  }

  STRtreeObject* self = (STRtreeObject*)type->tp_alloc(type, 0);
  if (self == NULL) {
    GEOSSTRtree_destroy_r(ctx, tree);
    GEOS_FINISH;
    return NULL;
  }
  GEOS_FINISH;
  self->ptr = tree;
  self->count = count;
  self->_geoms = _geoms;
  return (PyObject*)self;
}

/* Callback called by strtree_query with the index of each intersecting geometry
 * and a dynamic vector to push that index onto.
 *
 * Parameters
 * ----------
 * item: index of intersected geometry in the tree
 * user_data: pointer to dynamic vector; index is pushed onto this vector
 * */

void query_callback(void* item, void* user_data) {
  kv_push(npy_intp, *(npy_intp_vec*)user_data, (npy_intp)item);
}

/* Evaluate the predicate function against a prepared version of geom
 * for each geometry in the tree specified by indexes in out_indexes.
 * out_indexes is updated in place with the indexes of the geometries in the
 * tree that meet the predicate.
 *
 * Parameters
 * ----------
 * predicate_func: pointer to a prepared predicate function, e.g.,
 * GEOSPreparedIntersects_r geom: input geometry to prepare and test against each geometry
 * in the tree specified by in_indexes. tree_geometries: pointer to ndarray of all
 * geometries in the tree in_indexes: dynamic vector of indexes of tree geometries that
 * have overlapping envelopes with envelope of input geometry. out_indexes: dynamic vector
 * of indexes of tree geometries that meet predicate function. count: pointer to an
 * integer where the number of geometries that met the predicate will be written
 *
 * Returns PGERR_GEOS_EXCEPTION if an error was encountered or PGERR_SUCCESS otherwise
 * */

static char evaluate_predicate(void* context, FuncGEOS_YpY_b* predicate_func,
                               GEOSGeometry* geom, pg_geom_obj_vec* tree_geometries,
                               npy_intp_vec* in_indexes, npy_intp_vec* out_indexes,
                               npy_intp* count) {
  GeometryObject* pg_geom;
  GEOSGeometry* target_geom;
  const GEOSPreparedGeometry* prepared_geom;
  npy_intp i, size, index;

  // Create prepared geometry
  prepared_geom = GEOSPrepare_r(context, geom);
  if (prepared_geom == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  size = kv_size(*in_indexes);
  *count = 0;
  for (i = 0; i < size; i++) {
    // get index for right geometries from in_indexes
    index = kv_A(*in_indexes, i);

    // get GEOS geometry from pygeos geometry at index in tree geometries
    pg_geom = kv_A(*tree_geometries, index);
    if (pg_geom == NULL) {
      continue;
    }
    get_geom((GeometryObject*)pg_geom, &target_geom);

    // keep the index value if it passes the predicate
    if (predicate_func(context, prepared_geom, target_geom)) {
      kv_push(npy_intp, *out_indexes, index);
      (*count)++;
    }
  }

  GEOSPreparedGeom_destroy_r(context, prepared_geom);

  return PGERR_SUCCESS;
}

/* Query the tree based on input geometry and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 *
 * args must be:
 * - pygeos geometry object
 * - predicate id (see strtree.py for list of ids)
 * */

static PyObject* STRtree_query(STRtreeObject* self, PyObject* args) {
  GeometryObject* geometry;
  int predicate_id = 0;  // default no predicate
  GEOSGeometry* geom;
  npy_intp_vec query_indexes,
      predicate_indexes;  // Resizable array for matches for each geometry
  npy_intp count;
  FuncGEOS_YpY_b* predicate_func = NULL;
  PyArrayObject* result;

  if (self->ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
    return NULL;
  }

  if (!PyArg_ParseTuple(args, "O!i", &GeometryType, &geometry, &predicate_id)) {
    return NULL;
  }

  if (!get_geom(geometry, &geom)) {
    PyErr_SetString(PyExc_TypeError, "Invalid geometry");
    return NULL;
  }

  if (self->count == 0) {
    npy_intp dims[1] = {0};
    return PyArray_SimpleNew(1, dims, NPY_INTP);
  }

  if (predicate_id != 0) {
    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
      return NULL;
    }
  }

  GEOS_INIT;

  // query the tree for indices of geometries in the tree with
  // envelopes that intersect the geometry.
  kv_init(query_indexes);
  if (geom != NULL && !GEOSisEmpty_r(ctx, geom)) {
    GEOSSTRtree_query_r(ctx, self->ptr, geom, query_callback, &query_indexes);
  }

  if (predicate_id == 0 || kv_size(query_indexes) == 0) {
    // No predicate function provided, return all geometry indexes from
    // query.  If array is empty, return an empty numpy array
    result = copy_kvec_to_npy(&query_indexes);
    kv_destroy(query_indexes);
    GEOS_FINISH;
    return (PyObject*)result;
  }

  kv_init(predicate_indexes);
  errstate = evaluate_predicate(ctx, predicate_func, geom, &self->_geoms, &query_indexes,
                                &predicate_indexes, &count);
  if (errstate != PGERR_SUCCESS) {
    // error performing predicate
    kv_destroy(query_indexes);
    kv_destroy(predicate_indexes);
    GEOS_FINISH;
    return NULL;
  }

  result = copy_kvec_to_npy(&predicate_indexes);

  kv_destroy(query_indexes);
  kv_destroy(predicate_indexes);
  GEOS_FINISH;
  return (PyObject*)result;
}

/* Query the tree based on input geometries and predicate function.
 * The index of each geometry in the tree whose envelope intersects the
 * envelope of the input geometry is returned by default.
 * If predicate function is provided, only the index of those geometries that
 * satisfy the predicate function are returned.
 * Returns two arrays of equal length: first is indexes of the source geometries
 * and second is indexes of tree geometries that meet the above conditions.
 *
 * args must be:
 * - ndarray of pygeos geometries
 * - predicate id (see strtree.py for list of ids)
 *
 * */

static PyObject* STRtree_query_bulk(STRtreeObject* self, PyObject* args) {
  PyObject* arr;
  PyArrayObject* pg_geoms;
  GeometryObject* pg_geom;
  int predicate_id = 0;  // default no predicate
  GEOSGeometry* geom;
  npy_intp_vec query_indexes, src_indexes, target_indexes;
  npy_intp i, j, n, size;
  FuncGEOS_YpY_b* predicate_func = NULL;
  PyArrayObject* result;

  if (self->ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Tree is uninitialized");
    return NULL;
  }

  if (!PyArg_ParseTuple(args, "Oi", &arr, &predicate_id)) {
    return NULL;
  }

  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }

  pg_geoms = (PyArrayObject*)arr;
  if (!PyArray_ISOBJECT(pg_geoms)) {
    PyErr_SetString(PyExc_TypeError, "Array should be of object dtype");
    return NULL;
  }

  if (PyArray_NDIM(pg_geoms) != 1) {
    PyErr_SetString(PyExc_TypeError, "Array should be one dimensional");
    return NULL;
  }

  if (predicate_id != 0) {
    predicate_func = get_predicate_func(predicate_id);
    if (predicate_func == NULL) {
      return NULL;
    }
  }

  n = PyArray_SIZE(pg_geoms);

  if (self->count == 0 || n == 0) {
    npy_intp dims[2] = {2, 0};
    return PyArray_SimpleNew(2, dims, NPY_INTP);
  }

  kv_init(src_indexes);
  kv_init(target_indexes);

  GEOS_INIT_THREADS;

  for (i = 0; i < n; i++) {
    // get pygeos geometry from input geometry array
    pg_geom = *(GeometryObject**)PyArray_GETPTR1(pg_geoms, i);
    if (!get_geom(pg_geom, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      break;
    }
    if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
      continue;
    }

    kv_init(query_indexes);
    GEOSSTRtree_query_r(ctx, self->ptr, geom, query_callback, &query_indexes);

    if (kv_size(query_indexes) == 0) {
      // no target geoms in query window, skip this source geom
      kv_destroy(query_indexes);
      continue;
    }

    if (predicate_id == 0) {
      // no predicate, push results directly onto target_indexes
      size = kv_size(query_indexes);
      for (j = 0; j < size; j++) {
        kv_push(npy_intp, src_indexes, i);
        kv_push(npy_intp, target_indexes, kv_A(query_indexes, j));
      }
    } else {
      // this pushes directly onto target_indexes
      errstate = evaluate_predicate(ctx, predicate_func, geom, &self->_geoms,
                                    &query_indexes, &target_indexes, &size);

      if (errstate != PGERR_SUCCESS) {
        kv_destroy(query_indexes);
        kv_destroy(src_indexes);
        kv_destroy(target_indexes);
        break;
      }

      for (j = 0; j < size; j++) {
        kv_push(npy_intp, src_indexes, i);
      }
    }

    kv_destroy(query_indexes);
  }

  GEOS_FINISH_THREADS;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  size = kv_size(src_indexes);
  npy_intp dims[2] = {2, size};

  // the following raises a compiler warning based on how the macro is defined
  // in numpy.  There doesn't appear to be anything we can do to avoid it.
  result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INTP);
  if (result == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "could not allocate numpy array");
    return NULL;
  }

  for (i = 0; i < size; i++) {
    // assign value into numpy arrays
    *(npy_intp*)PyArray_GETPTR2(result, 0, i) = kv_A(src_indexes, i);
    *(npy_intp*)PyArray_GETPTR2(result, 1, i) = kv_A(target_indexes, i);
  }

  kv_destroy(src_indexes);
  kv_destroy(target_indexes);
  return (PyObject*)result;
}

static PyMemberDef STRtree_members[] = {
    {"_ptr", T_PYSSIZET, offsetof(STRtreeObject, ptr), READONLY,
     "Pointer to GEOSSTRtree"},
    {"count", T_LONG, offsetof(STRtreeObject, count), READONLY,
     "The number of geometries inside the tree"},
    {NULL} /* Sentinel */
};

static PyMethodDef STRtree_methods[] = {
    {"query", (PyCFunction)STRtree_query, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search geometry, "
     "and optionally tests them "
     "against predicate function if provided. "},
    {"query_bulk", (PyCFunction)STRtree_query_bulk, METH_VARARGS,
     "Queries the index for all items whose extents intersect the given search "
     "geometries, and optionally tests them "
     "against predicate function if provided. "},
    {NULL} /* Sentinel */
};

PyTypeObject STRtreeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pygeos.lib.STRtree",
    .tp_doc =
        "A query-only R-tree created using the Sort-Tile-Recursive (STR) algorithm.",
    .tp_basicsize = sizeof(STRtreeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = STRtree_new,
    .tp_dealloc = (destructor)STRtree_dealloc,
    .tp_members = STRtree_members,
    .tp_methods = STRtree_methods};

int init_strtree_type(PyObject* m) {
  if (PyType_Ready(&STRtreeType) < 0) {
    return -1;
  }

  Py_INCREF(&STRtreeType);
  PyModule_AddObject(m, "STRtree", (PyObject*)&STRtreeType);
  return 0;
}
