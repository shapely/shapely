#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pygeos_ARRAY_API

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>

#include "geos.h"
#include "pygeom.h"

/* These function prototypes enables that these functions can call themselves */
static char get_coordinates(GEOSContextHandle_t, GEOSGeometry*, PyArrayObject*, npy_intp*,
                            int);
static void* set_coordinates(GEOSContextHandle_t, GEOSGeometry*, PyArrayObject*,
                             npy_intp*, int);

/* Get coordinates from a point, linestring or linearring and puts them at
position `cursor` in the array `out`. Increases the cursor correspondingly.
Returns 0 on error, 1 on success */
static char get_coordinates_simple(GEOSContextHandle_t context, GEOSGeometry* geom,
                                   PyArrayObject* out, npy_intp* cursor, int include_z) {
  unsigned int n, i;
  double *x, *y, *z;
  const GEOSCoordSequence* seq = GEOSGeom_getCoordSeq_r(context, geom);
  if (seq == NULL) {
    return 0;
  }
  if (GEOSCoordSeq_getSize_r(context, seq, &n) == 0) {
    return 0;
  }
  for (i = 0; i < n; i++, *cursor += 1) {
    x = PyArray_GETPTR2(out, *cursor, 0);
    y = PyArray_GETPTR2(out, *cursor, 1);
    if (include_z) {
      z = PyArray_GETPTR2(out, *cursor, 2);
    }
    if (!GEOSCoordSeq_getX_r(context, seq, i, x)) {
      return 0;
    }
    if (!GEOSCoordSeq_getY_r(context, seq, i, y)) {
      return 0;
    }
    if (include_z) {
      if (!GEOSCoordSeq_getZ_r(context, seq, i, z)) {
        return 0;
      }
    }
  }
  return 1;
}

/* Get coordinates from a polygon by calling `get_coordinates_simple` on each
ring (exterior ring, interior ring 1, ..., interior ring N).
Returns 0 on error, 1 on success */
static char get_coordinates_polygon(GEOSContextHandle_t context, GEOSGeometry* geom,
                                    PyArrayObject* out, npy_intp* cursor, int include_z) {
  int n, i;
  GEOSGeometry* ring;

  ring = (GEOSGeometry*)GEOSGetExteriorRing_r(context, geom);
  if (ring == NULL) {
    return 0;
  }
  if (!get_coordinates_simple(context, ring, out, cursor, include_z)) {
    return 0;
  }

  n = GEOSGetNumInteriorRings_r(context, geom);
  if (n == -1) {
    return 0;
  }
  for (i = 0; i < n; i++) {
    ring = (GEOSGeometry*)GEOSGetInteriorRingN_r(context, geom, i);
    if (ring == NULL) {
      return 0;
    }
    if (!get_coordinates_simple(context, ring, out, cursor, include_z)) {
      return 0;
    }
  }
  return 1;
}

/* Get coordinates from a collection by calling `get_coordinates` on each
subgeometry. The call to `get_coordinates` is a recursive call so that nested
collections are allowed. Returns 0 on error, 1 on success */
static char get_coordinates_collection(GEOSContextHandle_t context, GEOSGeometry* geom,
                                       PyArrayObject* out, npy_intp* cursor,
                                       int include_z) {
  int n, i;
  GEOSGeometry* sub_geom;

  n = GEOSGetNumGeometries_r(context, geom);
  if (n == -1) {
    return 0;
  }
  for (i = 0; i < n; i++) {
    sub_geom = (GEOSGeometry*)GEOSGetGeometryN_r(context, geom, i);
    if (sub_geom == NULL) {
      return 0;
    }
    if (!get_coordinates(context, sub_geom, out, cursor, include_z)) {
      return 0;
    }
  }
  return 1;
}

/* Gets coordinates from a geometry and puts them at position `cursor` in the
array `out`. The value of the cursor is increased correspondingly. Returns 0
on error, 1 on success*/
static char get_coordinates(GEOSContextHandle_t context, GEOSGeometry* geom,
                            PyArrayObject* out, npy_intp* cursor, int include_z) {
  int type = GEOSGeomTypeId_r(context, geom);
  if ((type == 0) || (type == 1) || (type == 2)) {
    return get_coordinates_simple(context, geom, out, cursor, include_z);
  } else if (type == 3) {
    return get_coordinates_polygon(context, geom, out, cursor, include_z);
  } else if ((type >= 4) && (type <= 7)) {
    return get_coordinates_collection(context, geom, out, cursor, include_z);
  } else {
    return 0;
  }
}

/* Returns a copy of the input geometry (point, linestring or linearring) with
new coordinates set from position `cursor` in the array `out`. The value of the
cursor is increased correspondingly. Returns NULL on error,*/
static void* set_coordinates_simple(GEOSContextHandle_t context, GEOSGeometry* geom,
                                    int type, PyArrayObject* coords, npy_intp* cursor,
                                    int include_z) {
  unsigned int n, i, dims;
  double *x, *y, *z;
  GEOSGeometry* ret;

  /* Special case for POINT EMPTY (Point coordinate list cannot be 0-length) */
  if ((type == 0) && (GEOSisEmpty_r(context, geom) == 1)) {
    return GEOSGeom_createEmptyPoint_r(context);
  }

  /* Investigate the current (const) CoordSequence */
  const GEOSCoordSequence* seq = GEOSGeom_getCoordSeq_r(context, geom);
  if (seq == NULL) {
    return NULL;
  }
  if (GEOSCoordSeq_getSize_r(context, seq, &n) == 0) {
    return NULL;
  }
  if (GEOSCoordSeq_getDimensions_r(context, seq, &dims) == 0) {
    return NULL;
  }

  /* If we have CoordSeq with z dim, but new coordinates only are 2D,
   * create new CoordSeq that is also only 2D */
  if ((dims == 3) && !include_z) {
    dims = 2;
  }

  /* Create a new one to fill with the new coordinates */
  GEOSCoordSequence* seq_new = GEOSCoordSeq_create_r(context, n, dims);
  if (seq_new == NULL) {
    return NULL;
  }

  for (i = 0; i < n; i++, *cursor += 1) {
    x = PyArray_GETPTR2(coords, *cursor, 0);
    y = PyArray_GETPTR2(coords, *cursor, 1);
    if (GEOSCoordSeq_setX_r(context, seq_new, i, *x) == 0) {
      goto fail;
    }
    if (GEOSCoordSeq_setY_r(context, seq_new, i, *y) == 0) {
      goto fail;
    }
    if (dims == 3) {
      z = PyArray_GETPTR2(coords, *cursor, 2);
      if (GEOSCoordSeq_setZ_r(context, seq_new, i, *z) == 0) {
        goto fail;
      }
    }
  }

  /* Construct a new geometry */
  if (type == 0) {
    ret = GEOSGeom_createPoint_r(context, seq_new);
  } else if (type == 1) {
    ret = GEOSGeom_createLineString_r(context, seq_new);
  } else if (type == 2) {
    ret = GEOSGeom_createLinearRing_r(context, seq_new);
  } else {
    goto fail;
  }
  /* Do not destroy the seq_new if ret is NULL; will lead to segfaults */
  return ret;

fail:
  GEOSCoordSeq_destroy_r(context, seq_new);
  return NULL;
}

/* Returns a copy of the input polygon with new coordinates set by calling
`set_coordinates_simple` on the linearrings that make the polygon.
Returns NULL on error,*/
static void* set_coordinates_polygon(GEOSContextHandle_t context, GEOSGeometry* geom,
                                     PyArrayObject* coords, npy_intp* cursor,
                                     int include_z) {
  int i;
  GEOSGeometry *shell, *hole, *result = NULL;
  int n = GEOSGetNumInteriorRings_r(context, geom);

  if (n == -1) {
    return NULL;
  }
  GEOSGeometry** holes = malloc(sizeof(void*) * n);

  /* create the exterior ring */
  shell = (GEOSGeometry*)GEOSGetExteriorRing_r(context, geom);
  if (shell == NULL) {
    goto finish;
  }
  shell = set_coordinates_simple(context, shell, 2, coords, cursor, include_z);
  if (shell == NULL) {
    goto finish;
  }

  for (i = 0; i < n; i++) {
    hole = (GEOSGeometry*)GEOSGetInteriorRingN_r(context, geom, i);
    if (hole == NULL) {
      goto finish;
    }
    hole = set_coordinates_simple(context, hole, 2, coords, cursor, include_z);
    if (hole == NULL) {
      goto finish;
    }
    holes[i] = hole;
  }

  result = GEOSGeom_createPolygon_r(context, shell, holes, n);

finish:
  if (holes != NULL) {
    free(holes);
  }
  return result;
}

/* Returns a copy of the input collection with new coordinates set by calling
`set_coordinates` on the constituent subgeometries. Returns NULL on error,*/
static void* set_coordinates_collection(GEOSContextHandle_t context, GEOSGeometry* geom,
                                        int type, PyArrayObject* coords, npy_intp* cursor,
                                        int include_z) {
  int i;
  GEOSGeometry *sub_geom, *result = NULL;
  int n = GEOSGetNumGeometries_r(context, geom);
  if (n == -1) {
    return NULL;
  }
  GEOSGeometry** geoms = malloc(sizeof(void*) * n);

  for (i = 0; i < n; i++) {
    sub_geom = (GEOSGeometry*)GEOSGetGeometryN_r(context, geom, i);
    if (sub_geom == NULL) {
      goto finish;
    }
    sub_geom = set_coordinates(context, sub_geom, coords, cursor, include_z);
    if (sub_geom == NULL) {
      goto finish;
    }
    geoms[i] = sub_geom;
  }

  result = GEOSGeom_createCollection_r(context, type, geoms, n);
finish:
  if (geoms != NULL) {
    free(geoms);
  }
  return result;
}

/* Returns a copy of the input geometry with new coordinates set from position
`cursor` in the array `out`. The value of the cursor is increased
correspondingly. Returns NULL on error,*/
static void* set_coordinates(GEOSContextHandle_t context, GEOSGeometry* geom,
                             PyArrayObject* coords, npy_intp* cursor, int include_z) {
  int type = GEOSGeomTypeId_r(context, geom);
  if ((type == 0) || (type == 1) || (type == 2)) {
    return set_coordinates_simple(context, geom, type, coords, cursor, include_z);
  } else if (type == 3) {
    return set_coordinates_polygon(context, geom, coords, cursor, include_z);
  } else if ((type >= 4) && (type <= 7)) {
    return set_coordinates_collection(context, geom, type, coords, cursor, include_z);
  } else {
    return NULL;
  }
}

/* Count the total number of coordinate pairs in an array of Geometry objects */
npy_intp CountCoords(PyArrayObject* arr) {
  NpyIter* iter;
  NpyIter_IterNextFunc* iternext;
  char** dataptr;
  int ret;
  npy_intp result = 0;
  GeometryObject* obj;
  GEOSGeometry* geom;

  /* Handle zero-sized arrays specially */
  if (PyArray_SIZE(arr) == 0) {
    return 0;
  }

  /* We use the Numpy iterator C-API here.
  The iterator exposes an "iternext" function which updates a "dataptr"
  see also: https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html */
  iter = NpyIter_New(arr, NPY_ITER_READONLY | NPY_ITER_REFS_OK, NPY_KEEPORDER,
                     NPY_NO_CASTING, NULL);
  if (iter == NULL) {
    return -1;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return -1;
  }
  dataptr = NpyIter_GetDataPtrArray(iter);

  GEOS_INIT;

  do {
    /* get the geometry */
    obj = *(GeometryObject**)dataptr[0];
    if (!get_geom(obj, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      result = -1;
      goto finish;
    }
    /* skip incase obj was None */
    if (geom == NULL) {
      continue;
    }
    /* count coordinates */
    ret = GEOSGetNumCoordinates_r(ctx, geom);
    if (ret < 0) {
      errstate = PGERR_GEOS_EXCEPTION;
      result = -1;
      goto finish;
    }
    result += ret;
  } while (iternext(iter));

finish:
  GEOS_FINISH;
  NpyIter_Deallocate(iter);
  return result;
}

PyObject* GetCoords(PyArrayObject* arr, int include_z, int return_index) {
  npy_intp coord_dim;
  NpyIter* iter;
  NpyIter_IterNextFunc* iternext;
  char** dataptr;
  npy_intp cursor, i, geom_i;
  GeometryObject* obj;
  GEOSGeometry* geom;
  PyArrayObject* index = NULL;

  /* create a coordinate array with the appropriate dimensions */
  npy_intp size = CountCoords(arr);
  if (size == -1) {
    return NULL;
  }
  if (include_z) {
    coord_dim = 3;
  } else {
    coord_dim = 2;
  }
  npy_intp dims[2] = {size, coord_dim};
  PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (result == NULL) {
    return NULL;
  }
  if (return_index) {
    npy_intp dims_ind[1] = {size};
    index = (PyArrayObject*)PyArray_SimpleNew(1, dims_ind, NPY_INTP);
    if (index == NULL) {
      Py_DECREF(result);
      return NULL;
    }
  }

  /* Handle zero-sized arrays specially */
  if (PyArray_SIZE(arr) == 0) {
    if (return_index) {
      PyObject* result_tpl = PyTuple_New(2);
      PyTuple_SET_ITEM(result_tpl, 0, (PyObject*)result);
      PyTuple_SET_ITEM(result_tpl, 1, (PyObject*)index);
      return result_tpl;
    } else {
      return (PyObject*)result;
    }
  }

  /* We use the Numpy iterator C-API here.
  The iterator exposes an "iternext" function which updates a "dataptr"
  see also: https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html */
  iter = NpyIter_New(arr, NPY_ITER_READONLY | NPY_ITER_REFS_OK, NPY_CORDER,
                     NPY_NO_CASTING, NULL);
  if (iter == NULL) {
    Py_DECREF(result);
    Py_XDECREF(index);
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    Py_DECREF(result);
    Py_XDECREF(index);
    return NULL;
  }
  dataptr = NpyIter_GetDataPtrArray(iter);

  GEOS_INIT;

  /* We work with a "cursor" that tells the get_coordinates function where
  to write the coordinate data into the output array "result" */
  cursor = 0;
  geom_i = -1;
  do {
    /* get the geometry */
    obj = *(GeometryObject**)dataptr[0];
    geom_i++;
    if (!get_geom(obj, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    /* skip None values */
    if (geom == NULL) {
      continue;
    }
    /* keep track of the current cursor in "i" to be able to loop from the
    first to the last coordinate belonging to this geometry later */
    i = cursor;
    /* get the coordinates (updates "cursor") */
    if (!get_coordinates(ctx, geom, result, &cursor, include_z)) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
    if (return_index) {
      /* loop from "i" to "cursor" */
      for (; i < cursor; i++) {
        *(npy_intp*)PyArray_GETPTR1(index, i) = geom_i;
      }
    }
  } while (iternext(iter));

finish:
  GEOS_FINISH;
  NpyIter_Deallocate(iter);

  if (errstate != PGERR_SUCCESS) {
    Py_DECREF(result);
    Py_XDECREF(index);
    return NULL;
  } else if (return_index) {
    PyObject* result_tpl = PyTuple_New(2);
    PyTuple_SET_ITEM(result_tpl, 0, (PyObject*)result);
    PyTuple_SET_ITEM(result_tpl, 1, (PyObject*)index);
    return result_tpl;
  } else {
    return (PyObject*)result;
  }
}

PyObject* SetCoords(PyArrayObject* geoms, PyArrayObject* coords) {
  NpyIter* iter;
  NpyIter_IterNextFunc* iternext;
  char** dataptr;
  npy_intp cursor;
  npy_intp* coords_shape;
  int include_z;
  GeometryObject* obj;
  PyObject* new_obj;
  GEOSGeometry *geom, *new_geom;

  /* SetCoords acts implace: if the array is zero-sized, just return the
  same object */
  if (PyArray_SIZE(geoms) == 0) {
    Py_INCREF((PyObject*)geoms);
    return (PyObject*)geoms;
  }

  coords_shape = PyArray_SHAPE(coords);
  include_z = (coords_shape[1] == 3);

  /* We use the Numpy iterator C-API here.
  The iterator exposes an "iternext" function which updates a "dataptr"
  see also: https://docs.scipy.org/doc/numpy/reference/c-api.iterator.html */
  iter = NpyIter_New(geoms, NPY_ITER_READWRITE | NPY_ITER_REFS_OK, NPY_CORDER,
                     NPY_NO_CASTING, NULL);
  if (iter == NULL) {
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    NpyIter_Deallocate(iter);
    return NULL;
  }
  dataptr = NpyIter_GetDataPtrArray(iter);

  GEOS_INIT;

  /* We work with a "cursor" that tells the set_coordinates function where
  to read the coordinate data from the coordinate array "coords" */
  cursor = 0;
  do {
    /* get the geometry */
    obj = *(GeometryObject**)dataptr[0];
    if (!get_geom(obj, &geom)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    /* skip None values */
    if (geom == NULL) {
      continue;
    }
    /* create a new geometry with coordinates from "coords" array */
    new_geom = set_coordinates(ctx, geom, coords, &cursor, include_z);
    if (new_geom == NULL) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
    /* pack into a GeometryObject and set it to the geometry array */
    new_obj = GeometryObject_FromGEOS(new_geom, ctx);
    Py_XDECREF(obj);
    *(PyObject**)dataptr[0] = new_obj;
  } while (iternext(iter));

finish:
  GEOS_FINISH;
  NpyIter_Deallocate(iter);

  if (errstate == PGERR_SUCCESS) {
    Py_INCREF((PyObject*)geoms);
    return (PyObject*)geoms;
  } else {
    return NULL;
  }
}

PyObject* PyCountCoords(PyObject* self, PyObject* args) {
  PyObject* arr;
  npy_intp ret;

  if (!PyArg_ParseTuple(args, "O", &arr)) {
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
  ret = CountCoords((PyArrayObject*)arr);
  if (ret == -1) {
    return NULL;
  }
  return PyLong_FromSsize_t(ret);
}

PyObject* PyGetCoords(PyObject* self, PyObject* args) {
  PyObject* arr;
  int include_z;
  int return_index;

  if (!PyArg_ParseTuple(args, "Opp", &arr, &include_z, &return_index)) {
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
  return GetCoords((PyArrayObject*)arr, include_z, return_index);
}

PyObject* PySetCoords(PyObject* self, PyObject* args) {
  PyObject* geoms;
  PyObject* coords;

  if (!PyArg_ParseTuple(args, "OO", &geoms, &coords)) {
    return NULL;
  }
  if ((!PyArray_Check(geoms)) || (!PyArray_Check(coords))) {
    PyErr_SetString(PyExc_TypeError, "Not an ndarray");
    return NULL;
  }
  if (!PyArray_ISOBJECT((PyArrayObject*)geoms)) {
    PyErr_SetString(PyExc_TypeError, "Geometry array should be of object dtype");
    return NULL;
  }
  if ((PyArray_TYPE((PyArrayObject*)coords)) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_TypeError, "Coordinate array should be of float64 dtype");
    return NULL;
  }
  if ((PyArray_NDIM((PyArrayObject*)coords)) != 2) {
    PyErr_SetString(PyExc_ValueError, "Coordinate array should be 2-dimensional");
    return NULL;
  }
  geoms = SetCoords((PyArrayObject*)geoms, (PyArrayObject*)coords);
  if (geoms == Py_None) {
    return NULL;
  }
  return geoms;
}
