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
 * Function signature for GEOS operations that take a geometry and return a
 * string (as a GEOS-allocated char*): Y->O.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *
 * Returns:
 *   GEOS-allocated char* string (caller must GEOSFree_r it), or NULL on error
 */
typedef char* FuncGEOS_Y_str(GEOSContextHandle_t context, const GEOSGeometry* a);

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: Creates a Python string object, so must be called while the GIL is held.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   func: Function pointer to the specific GEOS operation to perform
 *   geom_obj: Shapely geometry Python object
 *   result: receives Py_None (for NULL geometry) or a new PyUnicode object
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Y_O_operation(GEOSContextHandle_t context, FuncGEOS_Y_str* func,
                                PyObject* geom_obj, PyObject** result) {
  const GEOSGeometry* geom;
  char* str;

  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  if (geom == NULL) {
    // NULL geometry -> Python None
    Py_INCREF(Py_None);
    *result = Py_None;
    return PGERR_SUCCESS;
  }

  str = func(context, geom);
  if (str == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  *result = PyUnicode_FromString(str);
  GEOSFree_r(context, str);
  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Y->O operations.
 * This handles a single geometry input (not arrays) and returns a Python string.
 * It should be registered as a METH_FASTCALL method (accepting one argument).
 *
 * This function is used as a template by the DEFINE_Y_O macro to create
 * specific scalar functions like PyGEOSisValidReason_r_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (one geometry)
 *   nargs: Number of arguments (should be 1)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result string (or None), or NULL on error
 */
static PyObject* Py_Y_O_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                FuncGEOS_Y_str* func) {
  PyObject* result = NULL;

  if (nargs != 1) {
    PyErr_Format(PyExc_TypeError, "expected 1 argument, got %zd", nargs);
    return NULL;
  }

  GEOS_INIT;

  errstate = core_Y_O_operation(ctx, func, args[0], &result);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    Py_XDECREF(result);
    return NULL;
  }

  return result;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for Y->O operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Note: The GIL is held throughout since Python objects are created in the loop.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Y_O_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  FuncGEOS_Y_str* func = (FuncGEOS_Y_str*)data;

  // Initialize GEOS context (holds GIL: Python objects are created in the loop)
  GEOS_INIT;

  UNARY_LOOP {
    CHECK_SIGNALS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    PyObject** out = (PyObject**)op1;
    PyObject* new_val = NULL;

    errstate = core_Y_O_operation(ctx, func, *(PyObject**)ip1, &new_val);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }

    Py_XDECREF(*out);
    *out = new_val;
  }

finish:
  GEOS_FINISH;
}

static PyUFuncGenericFunction Y_O_funcs[1] = {&Y_O_func};

/* Type signature: one geometry object -> Python object (string) */
static char Y_O_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ======================================================================== */

/*
 * Macro to create both the scalar function and the data pointer for the ufunc.
 *
 * DEFINE_Y_O(func_name) creates:
 *   - static PyObject* Py##func_name##_Scalar(...)
 */
#define DEFINE_Y_O(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, \
                                           Py_ssize_t nargs) { \
    return Py_Y_O_Scalar(self, args, nargs, (FuncGEOS_Y_str*)func_name); \
  }

DEFINE_Y_O(GEOSisValidReason_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * Macro to register both ufunc and scalar versions.
 */
#define INIT_Y_O(func_name, py_name) do { \
    static void* func_name##_FuncData[1] = {(void*)func_name}; \
    \
    ufunc = PyUFunc_FromFuncAndData(Y_O_funcs, func_name##_FuncData, Y_O_dtypes, 1, 1, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar", \
        (PyCFunction)Py##func_name##_Scalar, \
        METH_FASTCALL, \
        #py_name " scalar implementation" \
    }; \
    PyObject* Py##func_name##_Scalar_Func = PyCFunction_NewEx(&Py##func_name##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func_name##_Scalar_Func); \
} while(0)

int init_geos_funcs_Y_O(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  INIT_Y_O(GEOSisValidReason_r, is_valid_reason);

  return 0;
}
