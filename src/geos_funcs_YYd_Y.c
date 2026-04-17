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
 * Function signature for GEOS operations that take two geometries and a double,
 * and return a geometry: YYd->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *   c: Double parameter
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error
 */
typedef GEOSGeometry* FuncGEOS_YYd_Y(GEOSContextHandle_t context,
                                      const GEOSGeometry* a,
                                      const GEOSGeometry* b, double c);

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
 *   double_param: Double parameter required for the operation
 *   result: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YYd_Y_operation(GEOSContextHandle_t context, FuncGEOS_YYd_Y* func,
                                 PyObject* geom_obj_a, PyObject* geom_obj_b,
                                 double double_param, GEOSGeometry** result) {
  const GEOSGeometry* geom_a;
  const GEOSGeometry* geom_b;

  // Extract the underlying GEOS geometries from the Python geometry objects
  if (!ShapelyGetGeometry(geom_obj_a, &geom_a)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom_obj_b, &geom_b)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry or NaN double parameter cases - return NULL (None)
  if ((geom_a == NULL) || (geom_b == NULL) || npy_isnan(double_param)) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSIntersectionPrec_r, GEOSDifferencePrec_r, etc.)
  *result = func(context, geom_a, geom_b, double_param);

  if (*result == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YYd->Y operations.
 * This handles three inputs (two geometries and a double) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting three arguments).
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometries and a double)
 *   nargs: Number of arguments (should be 3)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_YYd_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                  FuncGEOS_YYd_Y* func) {
  GEOSGeometry* ret_ptr = NULL;
  double double_param;

  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  // Extract the double parameter from the third argument
  double_param = PyFloat_AsDouble(args[2]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YYd_Y_operation(ctx, func, args[0], args[1], double_param, &ret_ptr);

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
 * NumPy universal function implementation for YYd->Y operations.
 * This handles arrays of geometries and doubles efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void YYd_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_YYd_Y* func = (FuncGEOS_YYd_Y*)data;
  GEOSGeometry** geom_arr = NULL;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  if ((steps[3] == 0) && (dimensions[0] > 1)) {
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

  // The TERNARY_LOOP macro unpacks args, dimensions, and steps and iterates through
  // ip1 points to first input (geometry), ip2 points to second input (geometry),
  // ip3 points to third input (double), op1 points to output (geometry)
  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    double double_param = *(double*)ip3;
    errstate = core_YYd_Y_operation(ctx, func, *(PyObject**)ip1, *(PyObject**)ip2,
                                    double_param, &geom_arr[i]);
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
    geom_arr_to_npy(geom_arr, args[3], steps[3], dimensions[0]);
  }
  if (geom_arr != NULL) {
    free(geom_arr);
  }
}

/*
 * Function pointer array for NumPy ufunc creation.
 */
static PyUFuncGenericFunction YYd_Y_funcs[1] = {&YYd_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry) x 2, NPY_DOUBLE, returns NPY_OBJECT.
 */
static char YYd_Y_dtypes[4] = {NPY_OBJECT, NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ======================================================================== */

#define DEFINE_YYd_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YYd_Y_Scalar(self, args, nargs, (FuncGEOS_YYd_Y*)func_name); \
  }

DEFINE_YYd_Y(GEOSIntersectionPrec_r);
DEFINE_YYd_Y(GEOSDifferencePrec_r);
DEFINE_YYd_Y(GEOSSymDifferencePrec_r);
DEFINE_YYd_Y(GEOSUnionPrec_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

#define INIT_YYd_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 3 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(YYd_Y_funcs, func_name##_FuncData, YYd_Y_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create scalar Python function */ \
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
int init_geos_funcs_YYd_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_YYd_Y(GEOSIntersectionPrec_r, intersection_prec);
  INIT_YYd_Y(GEOSDifferencePrec_r, difference_prec);
  INIT_YYd_Y(GEOSSymDifferencePrec_r, symmetric_difference_prec);
  INIT_YYd_Y(GEOSUnionPrec_r, union_prec);

  return 0;
}
