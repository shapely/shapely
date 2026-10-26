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
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the relate_pattern operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: The `pattern` C string must already be extracted from the Python object
 * before calling this function; the GIL may be released when this is called.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   geom1_obj: First Shapely geometry Python object
 *   geom2_obj: Second Shapely geometry Python object
 *   pattern: DE-9IM pattern string (C string, already extracted from Python object)
 *   result: Pointer where the computed boolean result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_relate_pattern(GEOSContextHandle_t context, PyObject* geom1_obj,
                                  PyObject* geom2_obj, const char* pattern, char* result) {
  const GEOSGeometry* geom1;
  const GEOSGeometry* geom2;
  const GEOSPreparedGeometry* geom1_prepared;

  if (!ShapelyGetGeometryWithPrepared(geom1_obj, &geom1, &geom1_prepared)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom2_obj, &geom2)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  if ((geom1 == NULL) || (geom2 == NULL)) {
    *result = 0;  // False for missing values
    return PGERR_SUCCESS;
  }

#if GEOS_SINCE_3_13_0
  if (geom1_prepared != NULL) {
    *result = GEOSPreparedRelatePattern_r(context, geom1_prepared, geom2, pattern);
  } else {
    *result = GEOSRelatePattern_r(context, geom1, geom2, pattern);
  }
#else
  *result = GEOSRelatePattern_r(context, geom1, geom2, pattern);
#endif

  if (*result == 2) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

static PyObject* PyRelatePattern_Scalar(PyObject* self, PyObject* const* args,
                                          Py_ssize_t nargs) {
  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  // Extract the pattern string from the third argument (must be a Python unicode)
  const char* pattern;
  if (PyUnicode_Check(args[2])) {
    pattern = PyUnicode_AsUTF8(args[2]);
    if (pattern == NULL) {
      return NULL;
    }
  } else {
    PyErr_Format(PyExc_TypeError, "pattern expected string, got %s",
                 Py_TYPE(args[2])->tp_name);
    return NULL;
  }

  char result = 0;

  GEOS_INIT;

  errstate = core_relate_pattern(ctx, args[0], args[1], pattern, &result);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  return PyBool_FromLong(result);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for the relate_pattern operation.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer struct)
 *
 * Note: The string pattern must be scalar and is extracted before releasing the
 * GIL, so the inner loop can access it without holding the GIL.
 */
static void relate_pattern_func(char** args, const npy_intp* dimensions,
                                  const npy_intp* steps, void* data) {
  const char* pattern = NULL;

  // The pattern must be scalar
  if (steps[2] != 0) {
    PyErr_Format(PyExc_ValueError, "pattern keyword only supports scalar argument");
    return;
  }

  PyObject* pattern_obj = *(PyObject**)args[2];
  if (PyUnicode_Check(pattern_obj)) {
    pattern = PyUnicode_AsUTF8(pattern_obj);
    if (pattern == NULL) {
      return;
    }
  } else {
    PyErr_Format(PyExc_TypeError, "pattern expected string, got %s",
                 Py_TYPE(pattern_obj)->tp_name);
    return;
  }

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The TERNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 and ip2 point to current input elements, op1 points to current output element
  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_relate_pattern(ctx, *(PyObject**)ip1, *(PyObject**)ip2, pattern,
                                    (char*)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;
}

static PyUFuncGenericFunction relate_pattern_funcs[1] = {&relate_pattern_func};
static void* relate_pattern_data[1] = {NULL};
static char relate_pattern_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT, NPY_BOOL};


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

int init_geos_funcs_YYO_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  ufunc = PyUFunc_FromFuncAndData(relate_pattern_funcs, relate_pattern_data,
                                   relate_pattern_dtypes, 1, 3, 1,
                                   PyUFunc_None, "relate_pattern", "", 0);
  PyDict_SetItemString(d, "relate_pattern", ufunc);

  static PyMethodDef PyRelatePattern_Scalar_Def = {
      "relate_pattern_scalar",
      (PyCFunction)PyRelatePattern_Scalar,
      METH_FASTCALL,
      "relate_pattern scalar implementation"
  };
  PyObject* PyRelatePattern_Scalar_Func =
      PyCFunction_NewEx(&PyRelatePattern_Scalar_Def, NULL, NULL);
  PyDict_SetItemString(d, "relate_pattern_scalar", PyRelatePattern_Scalar_Func);

  return 0;
}
