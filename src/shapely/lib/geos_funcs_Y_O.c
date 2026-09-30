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
 * Core function that performs the is_valid_reason operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: Creates a Python string object, so must be called while the GIL is held.
 *
 * Parameters:
 *   context: GEOS context handle
 *   geom_obj: Shapely geometry Python object
 *   result: receives Py_None (for NULL geometry) or a new PyUnicode object
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_is_valid_reason(GEOSContextHandle_t context, PyObject* geom_obj,
                                  PyObject** result) {
  const GEOSGeometry* geom;
  char* str;

  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Missing geometry -> Python None
  if (geom == NULL) {
    Py_XDECREF(*result);
    Py_INCREF(Py_None);
    *result = Py_None;
    return PGERR_SUCCESS;
  }

  str = GEOSisValidReason_r(context, geom);
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

static PyObject* PyIsValidReason_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
  PyObject* result = NULL;

  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError, "expected 1 argument, got %zd", nargs);
    return NULL;
  }

  GEOS_INIT;

  errstate = core_is_valid_reason(ctx, args[0], &result);

  GEOS_FINISH;

  return result;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for the is_valid_reason operation.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer struct)
 *
 * Note: The GIL is held throughout since Python objects are created in the loop.
 */
static void is_valid_reason_func(char** args, const npy_intp* dimensions,
                                  const npy_intp* steps, void* data) {
  // Initialize GEOS context (holds GIL: Python objects are created in the loop)
  GEOS_INIT;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_is_valid_reason(ctx, *(PyObject**)ip1, (PyObject**)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  GEOS_FINISH;
}

static PyUFuncGenericFunction is_valid_reason_funcs[1] = {&is_valid_reason_func};
static void* is_valid_reason_data[1] = {NULL};
static char is_valid_reason_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

int init_geos_funcs_Y_O(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  ufunc = PyUFunc_FromFuncAndData(is_valid_reason_funcs, is_valid_reason_data,
                                   is_valid_reason_dtypes, 1, 1, 1,
                                   PyUFunc_None, "is_valid_reason", "", 0);
  PyDict_SetItemString(d, "is_valid_reason", ufunc);

  static PyMethodDef PyIsValidReason_Scalar_Def = {
      "is_valid_reason_scalar",
      (PyCFunction)PyIsValidReason_Scalar,
      METH_FASTCALL,
      "is_valid_reason scalar implementation"
  };
  PyObject* PyIsValidReason_Scalar_Func =
      PyCFunction_NewEx(&PyIsValidReason_Scalar_Def, NULL, NULL);
  PyDict_SetItemString(d, "is_valid_reason_scalar", PyIsValidReason_Scalar_Func);

  return 0;
}
