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

/* ========================================================================
 * GEOS WRAPPER FUNCTIONS
 * ========================================================================
 *
 * Function signature for GEOS operations that take a geometry and return a double: Y->d.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry (GEOSGeometry*)
 *   b: Output pointer where the result double will be stored
 *
 * Returns:
 *   1 on success, 0 on error (following GEOS convention)
 */
typedef int FuncGEOS_Y_d(GEOSContextHandle_t context, const GEOSGeometry* a, double* b);

static int GetX(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != GEOS_POINT) {
    *(double*)b = NPY_NAN;
    return 1;  // Success, but result is NaN for non-Points
  } else {
    return GEOSGeomGetX_r(context, a, b);
  }
}
static int GetY(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != GEOS_POINT) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetY_r(context, a, b);
  }
}
static int GetZ(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != GEOS_POINT) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetZ_r(context, a, b);
  }
}

#if GEOS_SINCE_3_12_0
static int GetM(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != GEOS_POINT) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetM_r(context, a, b);
  }
}
#endif

static int GetPrecision(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  // GEOS returns -1 on error; 0 indicates double precision; > 0 indicates a precision
  // grid size was set for this geometry.
  double out = GEOSGeom_getPrecision_r(context, a);
  if (out == -1) {
    return 0;
  }
  *(double*)b = out;
  return 1;
}
static int MinimumClearance(GEOSContextHandle_t context, const GEOSGeometry* a, double* b) {
  // GEOSMinimumClearance deviates from the standard pattern:
  // - Most GEOS functions return 0 on error, 1 on success
  // - This function returns 2 on error, 0/1 on success
  int retcode = GEOSMinimumClearance_r(context, a, b);
  if (retcode == 2) {
    return 0;
  } else {
    return 1;
  }
}
static int GEOSMinimumBoundingRadius(GEOSContextHandle_t context, const GEOSGeometry* geom, double* radius) {
  GEOSGeometry* center = NULL;

  // GEOSMinimumBoundingCircle_r computes both center and radius
  // We only need the radius, but must clean up the center geometry
  GEOSGeometry* ret = GEOSMinimumBoundingCircle_r(context, geom, radius, &center);

  if (ret == NULL) {
    return 0;  // Error occurred
  }

  // Clean up temporary geometries
  GEOSGeom_destroy_r(context, center);
  GEOSGeom_destroy_r(context, ret);

  return 1;  // Success - radius is now set
}


/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Parameters:
 *   ctx: GEOS context handle for thread-safe operations
 *   func: Function pointer to the specific GEOS operation to perform
 *   geom_obj: Shapely geometry object (Python wrapper around GEOSGeometry)
 *   result: Pointer where the computed double result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Y_d_operation(GEOSContextHandle_t ctx, FuncGEOS_Y_d* func,
                               PyObject* geom_obj, double* result) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case - return NaN as per NumPy convention
  if (geom == NULL) {
    *result = NPY_NAN;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSArea_r, GEOSLength_r, etc.)
  // GEOS functions return 0 on error, 1 on success
  if (func(ctx, geom, result) == 0) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Y->d operations.
 * This handles single geometry inputs (not arrays) and returns a Python float.
 * It should be registered as a METH_O method (accepting only a single argument).
 *
 * This function is used as a template by the DEFINE_Y_d macro to create
 * specific scalar functions like PyGetX_Scalar, PyGetY_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   obj: Input geometry object (should be a GeometryObject)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyFloat object containing the result, or NULL on error
 */
static PyObject* Py_Y_d_Scalar(PyObject* self, PyObject* obj, FuncGEOS_Y_d* func) {
  double result = NPY_NAN;
  char errstate = PGERR_SUCCESS;

  // These pointers allow the GEOS_HANDLE_ERR macro to access error messages
  char* last_error = geos_last_error;      // Global error message buffer
  char* last_warning = geos_last_warning;  // Global warning message buffer

  // Perform the actual GEOS operation using shared core logic
  errstate = core_Y_d_operation(geos_context[0], func, obj, &result);

  // Handle any errors or warnings that occurred during the operation
  // This macro checks errstate and sets appropriate Python exceptions
  GEOS_HANDLE_ERR;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_HANDLE_ERR
  }

  return PyFloat_FromDouble(result);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for Y->d operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Y_d_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Y_d* func = (FuncGEOS_Y_d*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    errstate = core_Y_d_operation(ctx, func, *(PyObject**)ip1, (double*)op1);
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
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), returns NPY_DOUBLE.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGetX_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 *
 * Example: DEFINE_Y_d(GetX) creates PyGetX_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Y_d(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return Py_Y_d_Scalar(self, obj, (FuncGEOS_Y_d*)func_name); \
  }

DEFINE_Y_d(GetX);
DEFINE_Y_d(GetY);
DEFINE_Y_d(GetZ);
#if GEOS_SINCE_3_12_0
DEFINE_Y_d(GetM);
#endif
DEFINE_Y_d(GEOSArea_r);
DEFINE_Y_d(GEOSLength_r);
DEFINE_Y_d(GetPrecision);
DEFINE_Y_d(MinimumClearance);
DEFINE_Y_d(GEOSMinimumBoundingRadius);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "get_x") for array operations
 * 2. A scalar function (e.g., "get_x_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSArea_r)
 *   py_name: Python function name (e.g., area)
 *
 */


#define INIT_Y_d(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 1 input, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, func_name##_FuncData, Y_d_dtypes, 1, 1, 1, \
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
int init_geos_funcs_Y_d(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Y_d(GetX, get_x);
  INIT_Y_d(GetY, get_y);
  INIT_Y_d(GetZ, get_z);

#if GEOS_SINCE_3_12_0
  INIT_Y_d(GetM, get_m);
#endif

  INIT_Y_d(GEOSArea_r, area);
  INIT_Y_d(GEOSLength_r, length);
  INIT_Y_d(GetPrecision, get_precision);
  INIT_Y_d(MinimumClearance, minimum_clearance);
  INIT_Y_d(GEOSMinimumBoundingRadius, minimum_bounding_radius);

  return 0;
}
