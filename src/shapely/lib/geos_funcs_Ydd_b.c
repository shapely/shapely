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
 * Function signatures for GEOS operations that take a geometry and two doubles,
 * and return a bool: Ydd->b.
 *
 * On GEOS >= 3.12, dedicated XY variants take a prepared geometry and (x, y).
 * On older GEOS, we fall back to constructing a point and calling the regular
 * prepared predicate.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   pg:      Prepared geometry (first argument)
 *   x, y:    Coordinate values
 *
 * Returns:
 *   0 for false, 1 for true, 2 on error (following GEOS convention)
 */
#if GEOS_SINCE_3_12_0
typedef char FuncGEOS_Ydd_b(GEOSContextHandle_t context, const GEOSPreparedGeometry* pg,
                             double x, double y);
#else
typedef char FuncGEOS_Ydd_b(GEOSContextHandle_t context, const GEOSPreparedGeometry* pg,
                             const GEOSGeometry* point);
#endif

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation on a geometry and an (x, y) point.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * The first input geometry is always prepared on-the-fly when it hasn't already been
 * prepared by the user.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   func:    Function pointer to the specific GEOS prepared operation
 *   geom_obj: Shapely geometry object (Python wrapper around GEOSGeometry)
 *   x, y:    Coordinate values of the point to test against
 *   result:  Pointer where the computed boolean result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Ydd_b_operation(GEOSContextHandle_t context, FuncGEOS_Ydd_b* func,
                                  PyObject* geom_obj, double x, double y, char* result) {
  const GEOSGeometry* geom;
  const GEOSPreparedGeometry* geom_prepared;

  if (!ShapelyGetGeometryWithPrepared(geom_obj, &geom, &geom_prepared)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  /* NULL geometry or NaN coordinate -> return False */
  if ((geom == NULL) || npy_isnan(x) || npy_isnan(y)) {
    *result = 0;
    return PGERR_SUCCESS;
  }

  /* Prepare on-the-fly if the geometry was not already prepared */
  char destroy_prepared = 0;
  if (geom_prepared == NULL) {
    geom_prepared = GEOSPrepare_r(context, geom);
    if (geom_prepared == NULL) {
      return PGERR_GEOS_EXCEPTION;
    }
    destroy_prepared = 1;
  }

#if GEOS_SINCE_3_12_0
  *result = func(context, geom_prepared, x, y);
#else
  GEOSGeometry* point = NULL;
  if (create_point(context, x, y, NULL, SHAPELY_HANDLE_NAN_ALLOW, &point) == PGERR_SUCCESS) {
    *result = func(context, geom_prepared, point);
    GEOSGeom_destroy_r(context, point);
  } else {
    *result = 2;  // GEOS convention for error
  }
#endif

  if (destroy_prepared) {
    GEOSPreparedGeom_destroy_r(context, geom_prepared);
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
 * Generic scalar Python function implementation for Ydd->b operations.
 * This handles a single (geometry, x, y) call and returns a Python bool.
 * It should be registered as a METH_FASTCALL method.
 *
 * Parameters:
 *   self:  Module object (unused, required by Python C API)
 *   args:  Array of arguments [geom, x, y]
 *   nargs: Number of arguments (must be 3)
 *   func:  Function pointer to the specific GEOS prepared operation
 *
 * Returns:
 *   PyBool object containing the result, or NULL on error
 */
static PyObject* Py_Ydd_b_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   FuncGEOS_Ydd_b* func) {
  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "Expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  double x = PyFloat_AsDouble(args[1]);
  if (PyErr_Occurred()) {
    return NULL;
  }
  double y = PyFloat_AsDouble(args[2]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  char result = 0;

  GEOS_INIT;

  errstate = core_Ydd_b_operation(ctx, func, args[0], x, y, &result);

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
 * NumPy universal function implementation for Ydd->b operations.
 * This handles arrays of (geometry, double, double) efficiently by iterating
 * through them, using the prepared geometry path for best performance.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Ydd_b_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Ydd_b* func = (FuncGEOS_Ydd_b*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The TERNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to first input (geometry), ip2 points to second input (x coordinate),
  // ip3 points to third input (y coordinate), op1 points to output (boolean)
  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_Ydd_b_operation(ctx, func, *(PyObject**)ip1, *(double*)ip2, *(double*)ip3, (char*)op1);
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
static PyUFuncGenericFunction Ydd_b_funcs[1] = {&Ydd_b_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), NPY_DOUBLE x 2, returns NPY_BOOL.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Ydd_b_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each Ydd->b GEOS operation.
 *
 * This creates a function that can be called from Python.
 *
 * Example: DEFINE_Ydd_b(GEOSPreparedContainsXY_r) creates:
 *   static PyObject* PyGEOSPreparedContainsXY_r_Scalar(...)
 *
 * Parameters:
 *   func: Name of the C functtion that performs the GEOS operation
 */
#define DEFINE_Ydd_b(func) \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_Ydd_b_Scalar(self, args, nargs, (FuncGEOS_Ydd_b*)func); \
  }

#if GEOS_SINCE_3_12_0
DEFINE_Ydd_b(GEOSPreparedContainsXY_r);
DEFINE_Ydd_b(GEOSPreparedIntersectsXY_r);
#else
DEFINE_Ydd_b(GEOSPreparedContains_r);
DEFINE_Ydd_b(GEOSPreparedIntersects_r);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * Macro to register both the ufunc and the scalar Python function for a Ydd->b operation.
 *
 * This creates two Python-callable objects in the module dictionary:
 *   1. A NumPy ufunc (e.g. "contains_xy") for array operations
 *   2. A scalar function (e.g. "contains_xy_scalar") for single-element operations
 *
 * Parameters:
 *   py_name:  Python name (e.g. contains_xy)
 *   func:     GEOS function pointer (version-dependent)
 */
#define INIT_Ydd_b(py_name, func) do { \
    static void* func##_FuncData[1] = {(void*)(FuncGEOS_Ydd_b*)func}; \
    \
    ufunc = PyUFunc_FromFuncAndData(Ydd_b_funcs, func##_FuncData, Ydd_b_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    static PyMethodDef Py##func##_Scalar_Def = { \
        #py_name "_scalar",                   \
        (PyCFunction)Py##func##_Scalar,       \
        METH_FASTCALL,                        \
        #py_name " scalar implementation"     \
    }; \
    PyObject* Py##func##_Scalar_Func = PyCFunction_NewEx(&Py##func##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func##_Scalar_Func); \
} while(0)

int init_geos_funcs_Ydd_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;

#if GEOS_SINCE_3_12_0
  INIT_Ydd_b(contains_xy, GEOSPreparedContainsXY_r);
  INIT_Ydd_b(intersects_xy, GEOSPreparedIntersectsXY_r);
#else
  INIT_Ydd_b(contains_xy, GEOSPreparedContains_r);
  INIT_Ydd_b(intersects_xy, GEOSPreparedIntersects_r);
#endif

  return 0;
}
