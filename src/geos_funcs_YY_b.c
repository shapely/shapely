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
 * Function signatures for GEOS operations that take two geometries and return a bool: YY->b.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry (for FuncGEOS_YY_b) or prepared geometry (for FuncGEOS_YY_b_p)
 *   b: Second input geometry
 *
 * Returns:
 *   0 for false, 1 for true, 2 on error (following GEOS convention)
 */
typedef char FuncGEOS_YY_b(GEOSContextHandle_t context, const GEOSGeometry* a, const GEOSGeometry* b);
typedef char FuncGEOS_YY_b_p(GEOSContextHandle_t context, const GEOSPreparedGeometry* a, const GEOSGeometry* b);

static char ShapelyContainsProperly(GEOSContextHandle_t context, const GEOSGeometry* g1, const GEOSGeometry* g2) {
  const GEOSPreparedGeometry* prepared_geom_tmp = NULL;
  char ret;

  prepared_geom_tmp = GEOSPrepare_r(context, g1);
  if (prepared_geom_tmp == NULL) {
    return 2;
  }
  ret = GEOSPreparedContainsProperly_r(context, prepared_geom_tmp, g2);
  GEOSPreparedGeom_destroy_r(context, prepared_geom_tmp);
  return ret;
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
 *   func: Function pointer to the GEOS operation to perform
 *   func_prepared: Prepared geometry version of the function (called if first geom has is prepared)
 *   geom1_obj: First Shapely geometry object
 *   geom2_obj: Second Shapely geometry object
 *   result: Pointer where the computed boolean result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YY_b_operation(GEOSContextHandle_t context, FuncGEOS_YY_b* func,
                                FuncGEOS_YY_b_p* func_prepared, PyObject* geom1_obj,
                                PyObject* geom2_obj, char* result) {
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

  // Handle NULL geometry case - return False as per convention for boolean operations
  if ((geom1 == NULL) || (geom2 == NULL)) {
    *result = 0;  // False
    return PGERR_SUCCESS;
  }

  // Call the appropriate GEOS function based on whether first geometry is prepared
  if (geom1_prepared == NULL) {
    *result = func(context, geom1, geom2);
  } else {
    *result = func_prepared(context, geom1_prepared, geom2);
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
 * Generic scalar Python function implementation for YY->b operations.
 * This handles single geometry pairs (not arrays) and returns a Python bool.
 * It should be registered as a METH_VARARGS method (accepting two arguments).
 *
 * This function is used as a template to create specific scalar functions
 * like PyDisjoint_Scalar, PyIntersects_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Tuple containing (geom1, geom2)
 *   func: Function pointer to the non-prepared GEOS operation
 *   func_prepared: Prepared geometry version of the function
 *
 * Returns:
 *   PyBool object containing the result, or NULL on error
 */
static PyObject* Py_YY_b_Scalar(PyObject* self, PyObject* args,
                                 FuncGEOS_YY_b* func, FuncGEOS_YY_b_p* func_prepared) {
  PyObject *geom1_obj, *geom2_obj;
  char result = 0;

  if (!PyArg_ParseTuple(args, "OO", &geom1_obj, &geom2_obj)) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YY_b_operation(ctx, func, func_prepared, geom1_obj, geom2_obj, &result);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_FINISH
  }

  return PyBool_FromLong(result);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for YY->b operations with prepared support.
 * This handles arrays of geometry pairs efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer tuple)
 */
static void YY_b_p_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS functions from the user data tuple
  FuncGEOS_YY_b* func = ((FuncGEOS_YY_b**)data)[0];
  FuncGEOS_YY_b_p* func_prepared = ((FuncGEOS_YY_b_p**)data)[1];

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The BINARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 and ip2 point to current input elements, op1 points to current output element
  BINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_YY_b_operation(ctx, func, func_prepared, *(PyObject**)ip1,
                                   *(PyObject**)ip2, (char*)op1);
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
static PyUFuncGenericFunction YY_b_p_funcs[1] = {&YY_b_p_func};

/*
 * Type signature for the ufunc: takes two NPY_OBJECT (geometries), returns NPY_BOOL.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char YY_b_p_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyDisjoint_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func}_Scalar(PyObject* self, PyObject* args)
 *
 * Example: DEFINE_YY_b(Disjoint, GEOSDisjoint_r, GEOSPreparedDisjoint_r)
 *          creates PyDisjoint_Scalar function
 *
 * Parameters:
 *   func: Non-prepared GEOS function (FuncGEOS_YY_b*)
 *   func_prepared: Prepared GEOS function (FuncGEOS_YY_b_p*)
 */
#define DEFINE_YY_b(func, func_prepared) \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* args) { \
    return Py_YY_b_Scalar(self, args, (FuncGEOS_YY_b*)func, (FuncGEOS_YY_b_p*)func_prepared); \
  }

DEFINE_YY_b(GEOSDisjoint_r, GEOSPreparedDisjoint_r);
DEFINE_YY_b(GEOSTouches_r, GEOSPreparedTouches_r);
DEFINE_YY_b(GEOSIntersects_r, GEOSPreparedIntersects_r);
DEFINE_YY_b(GEOSCrosses_r, GEOSPreparedCrosses_r);
DEFINE_YY_b(GEOSWithin_r, GEOSPreparedWithin_r);
DEFINE_YY_b(GEOSContains_r, GEOSPreparedContains_r);
DEFINE_YY_b(ShapelyContainsProperly, GEOSPreparedContainsProperly_r);
DEFINE_YY_b(GEOSOverlaps_r, GEOSPreparedOverlaps_r);
DEFINE_YY_b(GEOSCovers_r, GEOSPreparedCovers_r);
DEFINE_YY_b(GEOSCoveredBy_r, GEOSPreparedCoveredBy_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "disjoint") for array operations
 * 2. A scalar function (e.g., "disjoint_scalar") for single geometry pair operations
 *
 * Parameters:
 *   py_name: Python function name (e.g., disjoint)
 *   func: Non-prepared GEOS function
 *   func_prepared: Prepared GEOS function
 *
 */

#define INIT_YY_b(py_name, func, func_prepared) do { \
    /* Create data array with tuple of function pointers to pass to the ufunc */ \
    static void* func##_FuncData[2] = {func, func_prepared}; \
    static void* func##_FuncDataTuple[1] = {func##_FuncData}; \
    \
    /* Create NumPy ufunc: 2 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(YY_b_p_funcs, func##_FuncDataTuple, YY_b_p_dtypes, 1, 2, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        (PyCFunction)Py##func##_Scalar,       /* C function pointer */ \
        METH_VARARGS,                         /* Function takes two arguments */ \
        #py_name " scalar implementation"     /* Docstring */ \
    }; \
    PyObject* Py##func##_Scalar_Func = PyCFunction_NewEx(&Py##func##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func##_Scalar_Func); \
} while(0)

/*
 * The init function below is called when the Shapely module is imported.
 *
 * Parameters:
 *   m: The Python module object (unused here)
 *   d: Module dictionary where functions will be registered
 */
int init_geos_funcs_YY_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_YY_b(disjoint, GEOSDisjoint_r, GEOSPreparedDisjoint_r);
  INIT_YY_b(touches, GEOSTouches_r, GEOSPreparedTouches_r);
  INIT_YY_b(intersects, GEOSIntersects_r, GEOSPreparedIntersects_r);
  INIT_YY_b(crosses, GEOSCrosses_r, GEOSPreparedCrosses_r);
  INIT_YY_b(within, GEOSWithin_r, GEOSPreparedWithin_r);
  INIT_YY_b(contains, GEOSContains_r, GEOSPreparedContains_r);
  INIT_YY_b(contains_properly, ShapelyContainsProperly, GEOSPreparedContainsProperly_r);
  INIT_YY_b(overlaps, GEOSOverlaps_r, GEOSPreparedOverlaps_r);
  INIT_YY_b(covers, GEOSCovers_r, GEOSPreparedCovers_r);
  INIT_YY_b(covered_by, GEOSCoveredBy_r, GEOSPreparedCoveredBy_r);
  return 0;
}
