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
 * Function signature for GEOS operations that take a geometry and return a bool: Y->b.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *
 * Returns:
 *   0 for false, 1 for true, 2 on error (following GEOS convention)
 */
typedef char FuncGEOS_Y_b(GEOSContextHandle_t context, const GEOSGeometry* a);

/* the GEOSisSimple_r function fails on geometrycollections */
static char GEOSisSimpleAllTypes_r(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  int type = GEOSGeomTypeId_r(context, geom);
  if (type == -1) {
    return 2;  // Predicates use a return value of 2 for errors
  } else if (type == GEOS_GEOMETRYCOLLECTION) {
    return 0;
  } else {
    return GEOSisSimple_r(context, geom);
  }
}

/* the GEOSisClosed_r function fails on non-linestrings */
static char GEOSisClosedAllTypes_r(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  int type = GEOSGeomTypeId_r(context, geom);
  if (type == -1) {
    return 2;  // Predicates use a return value of 2 for errors
  } else if ((type == GEOS_LINESTRING) || (type == GEOS_LINEARRING) || (type == GEOS_MULTILINESTRING)) {
    return GEOSisClosed_r(context, geom);
  } else {
    return 0;
  }
}

static char GEOSGeom_isCCW_r(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  const GEOSCoordSequence* coord_seq;
  char is_ccw = 2;  // return value of 2 means GEOSException
  int i;

  // Return False for non-linear geometries
  i = GEOSGeomTypeId_r(context, geom);
  if (i == -1) {
    return 2;
  }
  if ((i != GEOS_LINEARRING) && (i != GEOS_LINESTRING)) {
    return 0;
  }

  // Return False for lines with fewer than 4 points
  i = GEOSGeomGetNumPoints_r(context, geom);
  if (i == -1) {
    return 2;
  }
  if (i < 4) {
    return 0;
  }

  // Get the coordinatesequence and call isCCW()
  coord_seq = GEOSGeom_getCoordSeq_r(context, geom);
  if (coord_seq == NULL) {
    return 2;
  }
  if (!GEOSCoordSeq_isCCW_r(context, coord_seq, &is_ccw)) {
    return 2;
  }
  return is_ccw;
}

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
 *   geom_obj: Shapely geometry object (Python wrapper around GEOSGeometry)
 *   result: Pointer where the computed char result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Y_b_operation(GEOSContextHandle_t context, FuncGEOS_Y_b* func,
                               PyObject* geom_obj, char* result) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case - return False as per convention for boolean operations
  if (geom == NULL) {
    *result = 0;  // False
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSisEmpty_r, GEOSisValid_r, etc.)
  *result = func(context, (void*)geom);
  if (*result == 2) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Y->b operations.
 * This handles single geometry inputs (not arrays) and returns a Python bool.
 * It should be registered as a METH_O method (accepting only a single argument).
 *
 * This function is used as a template by the DEFINE_Y_b macro to create
 * specific scalar functions like PyIsEmpty_Scalar, PyIsValid_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   obj: Input geometry object (should be a GeometryObject)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyBool object containing the result, or NULL on error
 */
static PyObject* Py_Y_b_Scalar(PyObject* self, PyObject* obj, FuncGEOS_Y_b* func) {
  char result = 0;

  GEOS_INIT;

  errstate = core_Y_b_operation(ctx, func, obj, &result);

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
 * NumPy universal function implementation for Y->b operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Y_b_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Y_b* func = (FuncGEOS_Y_b*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_Y_b_operation(ctx, func, *(PyObject**)ip1, (char*)op1);
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
static PyUFuncGenericFunction Y_b_funcs[1] = {&Y_b_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), returns NPY_BOOL.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Y_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyIsEmpty_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 *
 * Example: DEFINE_Y_b(IsEmpty) creates PyIsEmpty_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Y_b(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return Py_Y_b_Scalar(self, obj, (FuncGEOS_Y_b*)func_name); \
  }

DEFINE_Y_b(GEOSGeom_isCCW_r);
DEFINE_Y_b(GEOSisEmpty_r);
DEFINE_Y_b(GEOSisSimpleAllTypes_r);
DEFINE_Y_b(GEOSisRing_r);
DEFINE_Y_b(GEOSHasZ_r);
DEFINE_Y_b(GEOSisClosedAllTypes_r);
DEFINE_Y_b(GEOSisValid_r);

#if GEOS_SINCE_3_12_0
DEFINE_Y_b(GEOSHasM_r);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "is_empty") for array operations
 * 2. A scalar function (e.g., "is_empty_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., IsEmpty)
 *   py_name: Python function name (e.g., is_empty)
 *
 */


#define INIT_Y_b(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 1 input, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Y_b_funcs, func_name##_FuncData, Y_b_dtypes, 1, 1, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        Py##func_name##_Scalar,               /* C function pointer */ \
        METH_O,                               /* Function takes one argument */ \
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
int init_geos_funcs_Y_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Y_b(GEOSGeom_isCCW_r, is_ccw);
  INIT_Y_b(GEOSisEmpty_r, is_empty);
  INIT_Y_b(GEOSisSimpleAllTypes_r, is_simple);
  INIT_Y_b(GEOSisRing_r, is_ring);
  INIT_Y_b(GEOSHasZ_r, has_z);
  INIT_Y_b(GEOSisClosedAllTypes_r, is_closed);
  INIT_Y_b(GEOSisValid_r, is_valid);

#if GEOS_SINCE_3_12_0
  INIT_Y_b(GEOSHasM_r, has_m);
#endif

  return 0;
}
