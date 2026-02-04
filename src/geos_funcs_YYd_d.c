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
 * Function signature for GEOS operations that take two geometries and a double,
 * and return a double: YYd->d.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *   c: Double parameter
 *   d: Pointer to output double value
 *
 * Returns:
 *   1 on success, 0 on error (following GEOS convention)
 */
typedef int FuncGEOS_YYd_d(GEOSContextHandle_t context, const GEOSGeometry* a,
                           const GEOSGeometry* b, double c, double* d);

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   func: Function pointer to the specific GEOS operation to perform
 *   geom_obj_a: First Shapely geometry object (Python wrapper around GEOSGeometry)
 *   geom_obj_b: Second Shapely geometry object (Python wrapper around GEOSGeometry)
 *   densify_frac: Double parameter for densification
 *   result: Pointer where the computed distance result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YYd_d_operation(GEOSContextHandle_t context, FuncGEOS_YYd_d* func,
                                 PyObject* geom_obj_a, PyObject* geom_obj_b,
                                 double densify_frac, double* result) {
  const GEOSGeometry* geom_a;
  const GEOSGeometry* geom_b;

  // Extract the underlying GEOS geometries from the Python geometry objects
  if (!ShapelyGetGeometry(geom_obj_a, &geom_a)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom_obj_b, &geom_b)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry, NaN densify_frac, or empty geometry cases - return NaN
  if ((geom_a == NULL) || (geom_b == NULL) || isnan(densify_frac) ||
      GEOSisEmpty_r(context, geom_a) || GEOSisEmpty_r(context, geom_b)) {
    *result = NPY_NAN;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSHausdorffDistanceDensify_r, GEOSFrechetDistanceDensify_r)
  if (func(context, geom_a, geom_b, densify_frac, result) == 0) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YYd->d operations.
 * This handles three inputs (two geometries and a double) and returns a Python float.
 * It should be registered as a METH_FASTCALL method (accepting three arguments).
 *
 * This function is used as a template by the DEFINE_YYd_d macro to create
 * specific scalar functions like PyHausdorffDistanceDensify_Scalar, PyFrechetDistanceDensify_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometries and a double)
 *   nargs: Number of arguments (should be 3)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyFloat object containing the result distance, or NULL on error
 */
static PyObject* Py_YYd_d_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs, FuncGEOS_YYd_d* func) {
  double result = NPY_NAN;
  double densify_frac;

  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  // Extract the densify fraction from the third argument
  densify_frac = PyFloat_AsDouble(args[2]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YYd_d_operation(ctx, func, args[0], args[1], densify_frac, &result);

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
 * NumPy universal function implementation for YYd->d operations.
 * This handles arrays of geometries and doubles efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void YYd_d_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_YYd_d* func = (FuncGEOS_YYd_d*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The TERNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to first input (geometry), ip2 points to second input (geometry),
  // ip3 points to third input (double), op1 points to output (double)
  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    double densify_frac = *(double*)ip3;
    errstate = core_YYd_d_operation(ctx, func, *(PyObject**)ip1, *(PyObject**)ip2,
                                    densify_frac, (double*)op1);
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
static PyUFuncGenericFunction YYd_d_funcs[1] = {&YYd_d_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry) x 2, NPY_DOUBLE, returns NPY_DOUBLE.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char YYd_d_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Example: DEFINE_YYd_d(GEOSHausdorffDistanceDensify_r) creates PyGEOSHausdorffDistanceDensify_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_YYd_d(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YYd_d_Scalar(self, args, nargs, (FuncGEOS_YYd_d*)func_name); \
  }

DEFINE_YYd_d(GEOSHausdorffDistanceDensify_r);
DEFINE_YYd_d(GEOSFrechetDistanceDensify_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "hausdorff_distance_densify") for array operations
 * 2. A scalar function (e.g., "hausdorff_distance_densify_scalar") for three-argument operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSHausdorffDistanceDensify_r)
 *   py_name: Python function name (e.g., hausdorff_distance_densify)
 *
 */
#define INIT_YYd_d(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 3 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(YYd_d_funcs, func_name##_FuncData, YYd_d_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        (PyCFunction)Py##func_name##_Scalar,  /* C function pointer */ \
        METH_FASTCALL,                        /* Fast call convention */ \
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
int init_geos_funcs_YYd_d(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_YYd_d(GEOSHausdorffDistanceDensify_r, hausdorff_distance_densify);
  INIT_YYd_d(GEOSFrechetDistanceDensify_r, frechet_distance_densify);

  return 0;
}
