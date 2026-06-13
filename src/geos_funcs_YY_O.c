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
 * Core function that performs the relate operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: Creates a Python string object, so must be called while the GIL is held.
 *
 * Parameters:
 *   context: GEOS context handle
 *   geom1_obj: First Shapely geometry Python object
 *   geom2_obj: Second Shapely geometry Python object
 *   result: receives Py_None (for NULL geometry) or a new PyUnicode object
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_relate(GEOSContextHandle_t context, PyObject* geom1_obj,
                         PyObject* geom2_obj, PyObject** result) {
  const GEOSGeometry* geom1;
  const GEOSGeometry* geom2;
  const GEOSPreparedGeometry* geom1_prepared;
  char* str;

  if (!ShapelyGetGeometryWithPrepared(geom1_obj, &geom1, &geom1_prepared)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom2_obj, &geom2)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Missing geometry -> Python None
  if ((geom1 == NULL) || (geom2 == NULL)) {
    Py_XDECREF(*result);
    Py_INCREF(Py_None);
    *result = Py_None;
    return PGERR_SUCCESS;
  }

#if GEOS_SINCE_3_13_0
  if (geom1_prepared != NULL) {
    str = GEOSPreparedRelate_r(context, geom1_prepared, geom2);
  } else {
    str = GEOSRelate_r(context, geom1, geom2);
  }
#else
  str = GEOSRelate_r(context, geom1, geom2);
#endif

  if (str == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  Py_XDECREF(*result);
  *result = PyUnicode_FromString(str);
  GEOSFree_r(context, str);
  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

static PyObject* PyRelate_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  PyObject* result = NULL;

  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  GEOS_INIT;

  errstate = core_relate(ctx, args[0], args[1], &result);

  GEOS_FINISH;

  return result;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * Note: The GIL is held throughout since Python objects are created in the loop.
 */
static void relate_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                         void* data) {
  // Initialize GEOS context (holds GIL: Python objects are created in the loop)
  GEOS_INIT;

  BINARY_LOOP {
    CHECK_SIGNALS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_relate(ctx, *(PyObject**)ip1, *(PyObject**)ip2, (PyObject**)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  GEOS_FINISH;
}

static PyUFuncGenericFunction relate_funcs[1] = {&relate_func};
static void* relate_data[1] = {NULL};
static char relate_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

int init_geos_funcs_YY_O(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  ufunc = PyUFunc_FromFuncAndData(relate_funcs, relate_data, relate_dtypes, 1, 2, 1,
                                   PyUFunc_None, "relate", "", 0);
  PyDict_SetItemString(d, "relate", ufunc);

  static PyMethodDef PyRelate_Scalar_Def = {
      "relate_scalar",
      (PyCFunction)PyRelate_Scalar,
      METH_FASTCALL,
      "relate scalar implementation"
  };
  PyObject* PyRelate_Scalar_Func = PyCFunction_NewEx(&PyRelate_Scalar_Def, NULL, NULL);
  PyDict_SetItemString(d, "relate_scalar", PyRelate_Scalar_Func);

  return 0;
}
