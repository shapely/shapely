#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>

#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#define PY_ARRAY_UNIQUE_SYMBOL shapely_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL shapely_UFUNC_API
#include <numpy/ndarraytypes.h>
#include <numpy/npy_3kcompat.h>
#include <numpy/ufuncobject.h>

#include "fast_loop_macros.h"
#include "geos.h"
#include "pygeos.h"
#include "pygeom.h"
#include "signal_checks.h"

/* ========================================================================
 * GEOS WRAPPER FUNCTIONS
 * ========================================================================
 *
 * Function signature for GEOS operations that take a geometry and return a bool: Y->b.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry (GEOSGeometry*)
 *   b: Output pointer where the result char will be stored (0 = false, 1 = true)
 *
 * Returns:
 *   1 on success, 0 on error (following GEOS convention)
 */
typedef int FuncGEOS_Y_b(GEOSContextHandle_t context, const GEOSGeometry* a, char* b);

static int IsEmpty(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  char result = GEOSisEmpty_r(context, a);
  if (result == 2) {
    return 0;  // Error
  }
  *(char*)b = result;
  return 1;
}

static int IsSimple(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  // the GEOSisSimple_r function fails on geometrycollections
  int type = GEOSGeomTypeId_r(context, a);
  if (type == -1) {
    return 0;  // Error
  } else if (type == 7) {  // GEOMETRYCOLLECTION
    *(char*)b = 0;  // Not simple
    return 1;
  } else {
    char result = GEOSisSimple_r(context, a);
    if (result == 2) {
      return 0;  // Error
    }
    *(char*)b = result;
    return 1;
  }
}

static int IsRing(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  char result = GEOSisRing_r(context, a);
  if (result == 2) {
    return 0;  // Error
  }
  *(char*)b = result;
  return 1;
}

static int IsClosed(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  // the GEOSisClosed_r function fails on non-linestrings
  int type = GEOSGeomTypeId_r(context, a);
  if (type == -1) {
    return 0;  // Error
  } else if ((type == 1) || (type == 2) || (type == 5)) {  // Point, LineString, LinearRing, MultiLineString
    char result = GEOSisClosed_r(context, a);
    if (result == 2) {
      return 0;  // Error
    }
    *(char*)b = result;
    return 1;
  } else {
    *(char*)b = 0;  // Not closed
    return 1;
  }
}

static int IsValid(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  char result = GEOSisValid_r(context, a);
  if (result == 2) {
    return 0;  // Error
  }
  *(char*)b = result;
  return 1;
}

static int HasZ(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  char result = GEOSHasZ_r(context, a);
  if (result == 2) {
    return 0;  // Error
  }
  *(char*)b = result;
  return 1;
}

#if GEOS_SINCE_3_12_0
static int HasM(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  char result = GEOSHasM_r(context, a);
  if (result == 2) {
    return 0;  // Error
  }
  *(char*)b = result;
  return 1;
}
#endif

static int IsCCW(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  const GEOSCoordSequence* coord_seq;
  char is_ccw = 2;  // return value of 2 means GEOSException
  int i;

  // Return False for non-linear geometries
  i = GEOSGeomTypeId_r(context, a);
  if (i == -1) {
    return 0;  // Error
  }
  if ((i != GEOS_LINEARRING) && (i != GEOS_LINESTRING)) {
    *(char*)b = 0;  // Not CCW
    return 1;
  }

  // Return False for lines with fewer than 4 points
  i = GEOSGeomGetNumPoints_r(context, a);
  if (i == -1) {
    return 0;  // Error
  }
  if (i < 4) {
    *(char*)b = 0;  // Not CCW
    return 1;
  }

  // Get the coordinatesequence and call isCCW()
  coord_seq = GEOSGeom_getCoordSeq_r(context, a);
  if (coord_seq == NULL) {
    return 0;  // Error
  }
  if (!GEOSCoordSeq_isCCW_r(context, coord_seq, &is_ccw)) {
    return 0;  // Error
  }
  *(char*)b = is_ccw;
  return 1;
}

static int IsGeometry(GEOSContextHandle_t context, const GEOSGeometry* a, char* b) {
  // is_geometry just checks if we have a valid geometry
  *(char*)b = (a != NULL) ? 1 : 0;
  return 1;
}

/* ========================================================================
 * UFUNC LOOPS FOR Y -> b operations
 * ======================================================================== */

/* The Y->b ufunc loop implementation is based on the existing Y_b_func in ufuncs.c,
 * but adapted to call our wrapper functions above.
 */
static void Y_b_ufunc_loop(char** args, const npy_intp* dimensions, const npy_intp* steps,
                           void* data) {
  FuncGEOS_Y_b* func = (FuncGEOS_Y_b*)data;
  GEOSGeometry* in1 = NULL;
  char ret;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    /* get the geometry; return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      /* in case of a missing value: return 0 (False) */
      ret = 0;
    } else {
      /* call the GEOS function */
      if (func(geos_context[0], in1, &ret)) {
        // Success, ret contains the result
      } else {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
    *(npy_bool*)op1 = ret;
  }

finish:
  GEOS_FINISH_THREADS;
}

/* ========================================================================
 * SCALAR FUNCTIONS FOR DIRECT ACCESS
 * ======================================================================== */

static PyObject* is_empty_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_TRUE;  // NULL geometry is empty
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsEmpty(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* is_simple_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry is not simple
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsSimple(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* is_ring_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry is not a ring
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsRing(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* is_closed_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry is not closed
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsClosed(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* is_valid_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry is not valid
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsValid(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* has_z_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry has no Z
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (HasZ(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

#if GEOS_SINCE_3_12_0
static PyObject* has_m_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry has no M
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (HasM(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}
#endif

static PyObject* is_ccw_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  if (geom->ptr == NULL) {
    Py_RETURN_FALSE;  // NULL geometry is not CCW
  }

  GEOSContextHandle_t context = geos_context[0];
  char result;
  if (IsCCW(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

static PyObject* is_geometry_scalar(PyObject* self, PyObject* args) {
  PyObject* geom_obj;
  if (!PyArg_ParseTuple(args, "O", &geom_obj)) {
    return NULL;
  }

  GeometryObject* geom = (GeometryObject*)geom_obj;
  if (!PyObject_IsInstance(geom_obj, (PyObject*)&GeometryType)) {
    PyErr_SetString(PyExc_TypeError, "Expected Geometry object");
    return NULL;
  }

  char result;
  GEOSContextHandle_t context = geos_context[0];
  if (IsGeometry(context, geom->ptr, &result)) {
    return PyBool_FromLong(result);
  } else {
    Py_RETURN_FALSE;  // Error -> False
  }
}

/* ========================================================================
 * UFUNC CREATION MACROS (similar to DEFINE_Y_b in ufuncs.c)
 * ======================================================================== */

#define DEFINE_Y_b_UFUNC_AND_SCALAR(NAME, FUNC) \
  static char NAME##_dtypes[2] = {NPY_OBJECT, NPY_BOOL}; \
  static void* NAME##_data[1] = {FUNC}; \
  static PyUFuncGenericFunction NAME##_funcs[1] = {&Y_b_ufunc_loop}; \
  ufunc = PyUFunc_FromFuncAndData(NAME##_funcs, NAME##_data, NAME##_dtypes, 1, 1, 1, \
                                  PyUFunc_None, #NAME, NULL, 0); \
  PyDict_SetItemString(d, #NAME, ufunc); \
  Py_DECREF(ufunc)

/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

int init_geos_funcs_Y_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  /* Define ufuncs */
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_empty, IsEmpty);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_simple, IsSimple);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_ring, IsRing);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_closed, IsClosed);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_valid, IsValid);
  DEFINE_Y_b_UFUNC_AND_SCALAR(has_z, HasZ);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_geometry, IsGeometry);
  DEFINE_Y_b_UFUNC_AND_SCALAR(is_ccw, IsCCW);

#if GEOS_SINCE_3_12_0
  DEFINE_Y_b_UFUNC_AND_SCALAR(has_m, HasM);
#endif

  /* Define scalar functions */
  static PyMethodDef scalar_methods[] = {
    {"is_empty_scalar", is_empty_scalar, METH_VARARGS, "Check if geometry is empty (scalar)"},
    {"is_simple_scalar", is_simple_scalar, METH_VARARGS, "Check if geometry is simple (scalar)"},
    {"is_ring_scalar", is_ring_scalar, METH_VARARGS, "Check if geometry is a ring (scalar)"},
    {"is_closed_scalar", is_closed_scalar, METH_VARARGS, "Check if geometry is closed (scalar)"},
    {"is_valid_scalar", is_valid_scalar, METH_VARARGS, "Check if geometry is valid (scalar)"},
    {"has_z_scalar", has_z_scalar, METH_VARARGS, "Check if geometry has Z coordinate (scalar)"},
    {"is_geometry_scalar", is_geometry_scalar, METH_VARARGS, "Check if object is a geometry (scalar)"},
    {"is_ccw_scalar", is_ccw_scalar, METH_VARARGS, "Check if geometry is counter-clockwise (scalar)"},
#if GEOS_SINCE_3_12_0
    {"has_m_scalar", has_m_scalar, METH_VARARGS, "Check if geometry has M coordinate (scalar)"},
#endif
    {NULL, NULL, 0, NULL}
  };

  /* Add scalar methods to module */
  for (PyMethodDef* method = scalar_methods; method->ml_name != NULL; method++) {
    PyObject* func = PyCFunction_New(method, NULL);
    if (func == NULL) {
      return -1;
    }
    if (PyObject_SetAttrString(m, method->ml_name, func) < 0) {
      Py_DECREF(func);
      return -1;
    }
    Py_DECREF(func);
  }

  return 0;
}
