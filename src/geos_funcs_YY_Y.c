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
#include "geom_arr.h"
#include "geos.h"
#include "pygeos.h"
#include "pygeom.h"
#include "signal_checks.h"

/* ========================================================================
 * GEOS WRAPPER FUNCTIONS
 * ========================================================================
 *
 * Function signature for GEOS operations that take two geometries and return a geometry: YY->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error or for missing values
 */
typedef GEOSGeometry* FuncGEOS_YY_Y(GEOSContextHandle_t context, const GEOSGeometry* a, const GEOSGeometry* b);

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
 *   last_error: Error buffer to check for GEOS exceptions
 *   result_ptr: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YY_Y_operation(GEOSContextHandle_t context, FuncGEOS_YY_Y* func,
                                PyObject* geom_obj_a, PyObject* geom_obj_b,
                                GEOSGeometry** result) {
  const GEOSGeometry* geom_a;
  const GEOSGeometry* geom_b;

  // Extract the underlying GEOS geometries from the Python geometry objects
  if (!ShapelyGetGeometry(geom_obj_a, &geom_a)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (geom_a == NULL) {
    // Handle NULL geometry case - return NULL (None)
    *result = NULL;
    return PGERR_SUCCESS;
  }
  if (!ShapelyGetGeometry(geom_obj_b, &geom_b)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (geom_b == NULL) {
    // Handle NULL geometry case - return NULL (None)
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSIntersection_r, GEOSUnion_r, etc.)
  *result = func(context, geom_a, geom_b);

  if (*result == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YY->Y operations.
 * This handles two geometry inputs (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting only two arguments).
 *
 * This function is used as a template by the DEFINE_YY_Y macro to create
 * specific scalar functions like PyIntersection_Scalar, PyUnion_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometry objects)
 *   nargs: Number of arguments (should be 2)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_YY_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs, FuncGEOS_YY_Y* func) {
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YY_Y_operation(ctx, func, args[0], args[1], &ret_ptr);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_FINISH
  }

  return GeometryObject_FromGEOS(ret_ptr, ctx);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for YY->Y operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void YY_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_YY_Y* func = (FuncGEOS_YY_Y*)data;
  GEOSGeometry** geom_arr = NULL;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  if ((steps[2] == 0) && (dimensions[0] > 1)) {
    // In case of zero-strided output, raise an error
    errstate = PGERR_INPLACE_OUTPUT;
    goto finish;
  }

  // allocate a temporary array to store output GEOSGeometry objects
  geom_arr = malloc(sizeof(void*) * dimensions[0]);
  if (geom_arr == NULL) {
    errstate = PGERR_NO_MALLOC;
    goto finish;
  }

  // The BINARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to first input element, ip2 points to second input element, op1 points to output element
  BINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_YY_Y_operation(ctx, func, *(PyObject**)ip1, *(PyObject**)ip2, &geom_arr[i]);
    if (errstate != PGERR_SUCCESS) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
  }

  finish:

  // Clean up GEOS context and handle any errors (reacquires Python GIL)
  GEOS_FINISH_THREADS;

  // fill the numpy array with PyObjects while holding the GIL
  if (errstate == PGERR_SUCCESS) {
    geom_arr_to_npy(geom_arr, args[2], steps[2], dimensions[0]);
  }
  if (geom_arr != NULL) {
    free(geom_arr);
  }
}

/*
 * Function pointer array for NumPy ufunc creation.
 * NumPy requires this format to register different implementations
 * for different type combinations.
 */
static PyUFuncGenericFunction YY_Y_funcs[1] = {&YY_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry) x 2, returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char YY_Y_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyIntersection_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Example: DEFINE_YY_Y(GEOSIntersection_r) creates PyGEOSIntersection_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_YY_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YY_Y_Scalar(self, args, nargs, (FuncGEOS_YY_Y*)func_name); \
  }

DEFINE_YY_Y(GEOSIntersection_r);
DEFINE_YY_Y(GEOSDifference_r);
DEFINE_YY_Y(GEOSSymDifference_r);
DEFINE_YY_Y(GEOSUnion_r);
DEFINE_YY_Y(GEOSSharedPaths_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "intersection") for array operations
 * 2. A scalar function (e.g., "intersection_scalar") for two geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSIntersection_r)
 *   py_name: Python function name (e.g., intersection)
 *
 */
#define INIT_YY_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 2 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(YY_Y_funcs, func_name##_FuncData, YY_Y_dtypes, 1, 2, 1, \
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
int init_geos_funcs_YY_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_YY_Y(GEOSIntersection_r, intersection);
  INIT_YY_Y(GEOSDifference_r, difference);
  INIT_YY_Y(GEOSSymDifference_r, symmetric_difference);
  INIT_YY_Y(GEOSUnion_r, union);
  INIT_YY_Y(GEOSSharedPaths_r, shared_paths);

  return 0;
}
