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
 * Function signatures for GEOS operations that take two geometries and a double,
 * and return a bool: YYd->b.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry (or prepared geometry for the prepared variant)
 *   b: Second input geometry
 *   c: Double parameter
 *
 * Returns:
 *   0 for false, 1 for true, 2 on error (following GEOS convention)
 */
typedef char FuncGEOS_YYd_b(GEOSContextHandle_t context, const GEOSGeometry* a,
                             const GEOSGeometry* b, double c);
typedef char FuncGEOS_YYd_b_p(GEOSContextHandle_t context,
                               const GEOSPreparedGeometry* a, const GEOSGeometry* b,
                               double c);

/*
 * Struct to hold function pointers for both prepared and non-prepared versions.
 * func_prepared may be NULL when no prepared variant exists.
 */
typedef struct {
  FuncGEOS_YYd_b* func;
  FuncGEOS_YYd_b_p* func_prepared;
} YYd_b_func_data;

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   data: YYd_b_func_data struct containing function pointers
 *   geom1_obj: First Shapely geometry object
 *   geom2_obj: Second Shapely geometry object
 *   param: Double parameter
 *   result: Pointer where the computed boolean result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YYd_b_operation(GEOSContextHandle_t context, const YYd_b_func_data* data,
                                  PyObject* geom1_obj, PyObject* geom2_obj, double param,
                                  char* result) {
  const GEOSGeometry* geom1;
  const GEOSGeometry* geom2;
  const GEOSPreparedGeometry* geom1_prepared;

  // Extract the underlying GEOS geometries from Python objects
  if (!ShapelyGetGeometryWithPrepared(geom1_obj, &geom1, &geom1_prepared)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom2_obj, &geom2)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Missing geometry or NaN parameter -> return False as per convention for boolean operations
  if ((geom1 == NULL) || (geom2 == NULL) || npy_isnan(param)) {
    *result = 0;
    return PGERR_SUCCESS;
  }

  // Call the appropriate GEOS function based on whether first geometry is prepared
  // and whether prepared version is available
  if (geom1_prepared == NULL || data->func_prepared == NULL) {
    *result = data->func(context, geom1, geom2, param);
  } else {
    *result = data->func_prepared(context, geom1_prepared, geom2, param);
  }

  if (*result == 2) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YYd->b operations.
 * This handles three inputs (two geometries and a double) and returns a Python bool.
 * It should be registered as a METH_FASTCALL method (accepting three arguments).
 *
 * This function is used as a template to create specific scalar functions
 * like PyGEOSEqualsExact_r_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometries and a double)
 *   nargs: Number of arguments (should be 3)
 *   data: YYd_b_func_data struct containing GEOS function pointers
 *
 * Returns:
 *   PyBool object containing the result, or NULL on error
 */
static PyObject* Py_YYd_b_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   const YYd_b_func_data* data) {
  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  char result = 0;
  double param = PyFloat_AsDouble(args[2]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YYd_b_operation(ctx, data, args[0], args[1], param, &result);

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
 * NumPy universal function implementation for YYd->b operations.
 * This handles arrays of geometry pairs and doubles efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer struct)
 */
static void YYd_b_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The TERNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_YYd_b_operation(ctx, (YYd_b_func_data*)data, *(PyObject**)ip1,
                                    *(PyObject**)ip2, *(double*)ip3, (char*)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  // Clean up GEOS context and handle any errors (reacquires Python GIL)
  GEOS_FINISH_THREADS;
}

/*
 * Function pointer array for NumPy ufunc creation.
 */
static PyUFuncGenericFunction YYd_b_funcs[1] = {&YYd_b_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT x 2, NPY_DOUBLE, returns NPY_BOOL.
 */
static char YYd_b_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates two things:
 *  1. A YYd_b_func_data struct instance containing the GEOS function pointers
 *  2. A Python function that implements the scalar logic.
 *
 * Example: DEFINE_YYd_b(GEOSEqualsExact_r, NULL) creates
 * - static YYd_b_func_data GEOSEqualsExact_r_data
 * - static PyObject* PyGEOSEqualsExact_r_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Parameters:
 *   func: Non-prepared GEOS function
 *   func_prepared: Prepared GEOS function
 */
#define DEFINE_YYd_b(func, func_prepared) \
  static YYd_b_func_data func##_data = {func, func_prepared}; \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YYd_b_Scalar(self, args, nargs, &func##_data); \
  }

DEFINE_YYd_b(GEOSEqualsExact_r, NULL);
DEFINE_YYd_b(GEOSDistanceWithin_r, GEOSPreparedDistanceWithin_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

 /*
 * We use a macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "equals_exact") for array operations
 * 2. A scalar function (e.g., "equals_exact_scalar") for single geometry pair operations
 *
 * Parameters:
 *   py_name: Python function name (e.g., equals_exact)
 *   func: Non-prepared GEOS function
 *   func_prepared: Prepared GEOS function
 *
 */

#define INIT_YYd_b(py_name, func, func_prepared) do { \
    static void* func##_udata[1] = {(void*)&func##_data}; \
    \
    ufunc = PyUFunc_FromFuncAndData(YYd_b_funcs, func##_udata, YYd_b_dtypes, 1, 3, 1, \
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

int init_geos_funcs_YYd_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  INIT_YYd_b(equals_exact, GEOSEqualsExact_r, NULL);
  INIT_YYd_b(dwithin, GEOSDistanceWithin_r, GEOSPreparedDistanceWithin_r);

  return 0;
}
