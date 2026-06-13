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
 * Function signatures for GEOS operations that take two geometries and a
 * string pattern, and return a bool: YYO->b (where O is NPY_OBJECT holding a
 * Python unicode string).
 *
 * Currently only GEOSRelatePattern_r / GEOSPreparedRelatePattern_r use this
 * signature.  The prepared variant is available from GEOS 3.13.0.
 *
 * The string pattern input must always be scalar (enforced in the ufunc).
 *
 * Parameters for the non-prepared variant:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *   pattern: DE-9IM pattern string (C string, null-terminated)
 *
 * Returns:
 *   0 for false, 1 for true, 2 on error (following GEOS convention)
 */
typedef char FuncGEOS_YYs_b(GEOSContextHandle_t context, const GEOSGeometry* a,
                             const GEOSGeometry* b, const char* pattern);
typedef char FuncGEOS_YYs_b_p(GEOSContextHandle_t context,
                               const GEOSPreparedGeometry* a, const GEOSGeometry* b,
                               const char* pattern);

/*
 * Struct to hold function pointers.  func_prepared may be NULL.
 */
typedef struct {
  FuncGEOS_YYs_b* func;
  FuncGEOS_YYs_b_p* func_prepared;
} YYO_b_func_data;

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: The `pattern` C string must already be extracted from the Python object
 * before calling this function; the GIL may be released when this is called.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   data: YYO_b_func_data struct containing function pointers
 *   geom1_obj: First Shapely geometry Python object
 *   geom2_obj: Second Shapely geometry Python object
 *   pattern: DE-9IM pattern string (C string, already extracted from Python object)
 *   result: Pointer where the computed boolean result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YYO_b_operation(GEOSContextHandle_t context, const YYO_b_func_data* data,
                                  PyObject* geom1_obj, PyObject* geom2_obj,
                                  const char* pattern, char* result) {
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
  if (geom1_prepared != NULL && data->func_prepared != NULL) {
    *result = data->func_prepared(context, geom1_prepared, geom2, pattern);
  } else {
    *result = data->func(context, geom1, geom2, pattern);
  }
#else
  *result = data->func(context, geom1, geom2, pattern);
#endif

  if (*result == 2) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YYO->b operations.
 * This handles two geometry inputs and a string pattern (not arrays) and returns a Python bool.
 * It should be registered as a METH_FASTCALL method (accepting three arguments).
 *
 * This function is used as a template by the DEFINE_YYO_b macro to create
 * specific scalar functions like PyGEOSRelatePattern_r_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometries and a string pattern)
 *   nargs: Number of arguments (should be 3)
 *   data: YYO_b_func_data struct containing GEOS function pointers
 *
 * Returns:
 *   PyBool object containing the result, or NULL on error
 */
static PyObject* Py_YYO_b_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   const YYO_b_func_data* data) {
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

  errstate = core_YYO_b_operation(ctx, data, args[0], args[1], pattern, &result);

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
 * NumPy universal function implementation for YYO->b operations.
 * This handles arrays of geometry pairs efficiently by iterating through them.
 *
 * Note: The string pattern must be scalar and is extracted before releasing the
 * GIL, so the inner loop can access it without holding the GIL.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer struct)
 */
static void YYO_b_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
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
      // error already set by Python
      return;
    }
  } else {
    PyErr_Format(PyExc_TypeError, "pattern expected string, got %s",
                 Py_TYPE(pattern_obj)->tp_name);
    return;
  }

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_YYO_b_operation(ctx, (YYO_b_func_data*)data, *(PyObject**)ip1,
                                    *(PyObject**)ip2, pattern, (char*)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;
}

static PyUFuncGenericFunction YYO_b_funcs[1] = {&YYO_b_func};

/* Type signature: two geometry objects + Python object (string) -> bool */
static char YYO_b_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ======================================================================== */

/*
 * Macro to define both the func_data struct and the scalar Python function.
 *
 * DEFINE_YYO_b(func, func_prepared) creates:
 *   - static YYO_b_func_data func##_data
 *   - static PyObject* Py##func##_Scalar(...)
 */
#define DEFINE_YYO_b(func, func_prepared) \
  static YYO_b_func_data func##_data = {func, func_prepared}; \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YYO_b_Scalar(self, args, nargs, &func##_data); \
  }

#if GEOS_SINCE_3_13_0
DEFINE_YYO_b(GEOSRelatePattern_r, GEOSPreparedRelatePattern_r);
#else
DEFINE_YYO_b(GEOSRelatePattern_r, NULL);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * Macro to register both ufunc and scalar versions.
 */
#define INIT_YYO_b(py_name, func, func_prepared) do { \
    static void* func##_udata[1] = {(void*)&func##_data}; \
    \
    ufunc = PyUFunc_FromFuncAndData(YYO_b_funcs, func##_udata, YYO_b_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    static PyMethodDef Py##func##_Scalar_Def = { \
        #py_name "_scalar", \
        (PyCFunction)Py##func##_Scalar, \
        METH_FASTCALL, \
        #py_name " scalar implementation" \
    }; \
    PyObject* Py##func##_Scalar_Func = PyCFunction_NewEx(&Py##func##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func##_Scalar_Func); \
} while(0)

int init_geos_funcs_YYO_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;

#if GEOS_SINCE_3_13_0
  INIT_YYO_b(relate_pattern, GEOSRelatePattern_r, GEOSPreparedRelatePattern_r);
#else
  INIT_YYO_b(relate_pattern, GEOSRelatePattern_r, NULL);
#endif

  return 0;
}
