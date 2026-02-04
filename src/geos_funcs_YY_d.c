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
 * Function signatures for GEOS operations that take two geometries and return a double: YY->d.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *   result: Pointer to output double value
 *
 * Returns:
 *   0 on error, 1 on success (following GEOS convention)
 */
typedef int FuncGEOS_YY_d(GEOSContextHandle_t context, const GEOSGeometry* a, const GEOSGeometry* b, double* result);

/*
 * Wrapper functions for GEOS operations that need special handling
 */

static int GEOSProjectWrapped_r(GEOSContextHandle_t context, const GEOSGeometry* a, const GEOSGeometry* b, double* c) {
  /* Project returns -1.0 instead of an error code */
  *c = GEOSProject_r(context, a, b);
  if (*c == -1.0) {
    return 0;
  } else {
    return 1;
  }
}

static int GEOSProjectNormalizedWrapped_r(GEOSContextHandle_t context, const GEOSGeometry* a, const GEOSGeometry* b, double* c) {
  /* ProjectNormalized returns -1.0 instead of an error code */
  *c = GEOSProjectNormalized_r(context, a, b);
  if (*c == -1.0) {
    return 0;
  } else {
    return 1;
  }
}

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation on two geometries.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   func: The GEOS function to call
 *   geom1_obj: First Shapely geometry object
 *   geom2_obj: Second Shapely geometry object
 *   result: Pointer where the computed double result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YY_d_operation(GEOSContextHandle_t context, FuncGEOS_YY_d* func,
                                PyObject* geom1_obj, PyObject* geom2_obj, double* result) {
  const GEOSGeometry* geom1;
  const GEOSGeometry* geom2;

  // Extract the underlying GEOS geometries from Python objects
  if (!ShapelyGetGeometry(geom1_obj, &geom1)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom2_obj, &geom2)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle missing geometry (None) - return NaN
  if ((geom1 == NULL) || (geom2 == NULL)) {
    *result = NPY_NAN;
    return PGERR_SUCCESS;
  }

  // Handle empty geometries - return NaN
  // Note that GEOS does this too, but some versions and platforms, segfaults are raised
  // "invalid value" warnings are set or 0.0 or Inf are returned.
  if (GEOSisEmpty_r(context, geom1) || GEOSisEmpty_r(context, geom2)) {
    *result = NPY_NAN;
    return PGERR_SUCCESS;
  }

  // Call the GEOS function
  if (func(context, geom1, geom2, result) == 0) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YY->d operations.
 * This handles single geometry pairs (not arrays) and returns a Python float.
 * It should be registered as a METH_FASTCALL method (accepting two geometries).
 *
 * This function is used as a template to create specific scalar functions
 * like PyDistance_Scalar, PyFrechetDistance_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geom1, geom2)
 *   nargs: Number of arguments (should be 2)
 *   func: FuncGEOS_YY_d function pointer
 *
 * Returns:
 *   PyFloat object containing the result, or NULL on error
 */
static PyObject* Py_YY_d_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                 FuncGEOS_YY_d* func) {
  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  double result = 0;

  GEOS_INIT;

  errstate = core_YY_d_operation(ctx, func, args[0], args[1], &result);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_FINISH
  }

  return PyFloat_FromDouble(result);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for YY->d operations.
 * This handles arrays of geometry pairs efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void YY_d_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_YY_d* func = (FuncGEOS_YY_d*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The BINARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 and ip2 point to current input elements, op1 points to current output element
  BINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_YY_d_operation(ctx, func, *(PyObject**)ip1, *(PyObject**)ip2, (double*)op1);
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
 * NumPy requires this format to register different implementations
 * for different type combinations.
 */
static PyUFuncGenericFunction YY_d_funcs[1] = {&YY_d_func};

/*
 * Type signature for the ufunc: takes two NPY_OBJECT (geometries), returns NPY_DOUBLE.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char YY_d_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGEOSDistance_r_Scalar that can be called from Python.
 * The generated function signature is:
 *  static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Example: DEFINE_YY_d(GEOSDistance_r) creates PyGEOSDistance_r_Scalar function
 *
 * Parameters:
 *   func: Name of the C function that performs the GEOS operation
 */
#define DEFINE_YY_d(func) \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YY_d_Scalar(self, args, nargs, func); \
  }

DEFINE_YY_d(GEOSDistance_r);
DEFINE_YY_d(GEOSFrechetDistance_r);
DEFINE_YY_d(GEOSHausdorffDistance_r);
DEFINE_YY_d(GEOSProjectWrapped_r);
DEFINE_YY_d(GEOSProjectNormalizedWrapped_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "distance") for array operations
 * 2. A scalar function (e.g., "distance_scalar") for single geometry pair operations
 *
 * Parameters:
 *   py_name: Python function name (e.g., distance)
 *   func: GEOS function to call
 *
 */

#define INIT_YY_d(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 2 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(YY_d_funcs, func_name##_FuncData, YY_d_dtypes, 1, 2, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        (PyCFunction)Py##func_name##_Scalar,  /* C function pointer */ \
        METH_FASTCALL,                        /* Function takes fast call arguments */ \
        #py_name " scalar implementation"     /* Docstring */ \
    }; \
    PyObject* Py##func_name##_Scalar_Func = PyCFunction_NewEx(&Py##func_name##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func_name##_Scalar_Func); \
} while(0)

/*
 * The init function below is called when the Shapely module is imported.
 *
 * Parameters:
 *   m: The Python module object (unused here)
 *   d: Module dictionary where functions will be registered
 */
int init_geos_funcs_YY_d(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_YY_d(GEOSDistance_r, distance);
  INIT_YY_d(GEOSFrechetDistanceWrapped_r, frechet_distance);
  INIT_YY_d(GEOSHausdorffDistance_r, hausdorff_distance);
  INIT_YY_d(GEOSProjectWrapped_r, line_locate_point);
  INIT_YY_d(GEOSProjectNormalizedWrapped_r, line_locate_point_normalized);
  return 0;
}
