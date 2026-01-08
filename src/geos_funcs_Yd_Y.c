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
 * Function signature for GEOS operations that take a geometry and a double, returning a geometry: Yd->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *   b: Input double parameter
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error (GEOS error or wrong geometry type)
 */
typedef GEOSGeometry* FuncGEOS_Yd_Y(GEOSContextHandle_t context, const GEOSGeometry* a, double b);

/* Wrapper functions to handle special cases */

static GEOSGeometry* GEOSInterpolateProtectEmpty_r(GEOSContextHandle_t context, const GEOSGeometry* geom, double d) {
  char errstate = geos_interpolate_checker(context, geom);
  if (errstate == PGERR_SUCCESS) {
    return GEOSInterpolate_r(context, geom, d);
  } else if (errstate == PGERR_EMPTY_GEOMETRY) {
    return GEOSGeom_createEmptyPoint_r(context);
  } else {
    return NULL;
  }
}

static GEOSGeometry* GEOSInterpolateNormalizedProtectEmpty_r(GEOSContextHandle_t context, const GEOSGeometry* geom,
                                                     double d) {
  char errstate = geos_interpolate_checker(context, geom);
  if (errstate == PGERR_SUCCESS) {
    return GEOSInterpolateNormalized_r(context, geom, d);
  } else if (errstate == PGERR_EMPTY_GEOMETRY) {
    return GEOSGeom_createEmptyPoint_r(context);
  } else {
    return NULL;
  }
}

static GEOSGeometry* GEOSMaximumInscribedCircleWithDefaultTolerance(GEOSContextHandle_t context, const GEOSGeometry* a, double b) {
  double tolerance;
  if (b == 0.0 && !GEOSisEmpty_r(context, a)) {
    double xmin, xmax, ymin, ymax;
    double width, height, size;

#if GEOS_SINCE_3_11_0
    if (!GEOSGeom_getExtent_r(context, a, &xmin, &ymin, &xmax, &ymax)) {
      return NULL;
    }
#else
    if (!GEOSGeom_getXMin_r(context, a, &xmin)) {
      return NULL;
    }
    if (!GEOSGeom_getYMin_r(context, a, &ymin)) {
      return NULL;
    }
    if (!GEOSGeom_getXMax_r(context, a, &xmax)) {
      return NULL;
    }
    if (!GEOSGeom_getYMax_r(context, a, &ymax)) {
      return NULL;
    }
#endif
    width = xmax - xmin;
    height = ymax - ymin;
    size = width > height ? width : height;
    tolerance = size / 1000.0;
  } else {
    tolerance = b;
  }
  return GEOSMaximumInscribedCircle_r(context, a, tolerance);
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
 *   d: Double parameter for the operation
 *   last_error: Error buffer to check for GEOS exceptions
 *   result: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Yd_Y_operation(GEOSContextHandle_t context, FuncGEOS_Yd_Y* func,
                                PyObject* geom_obj, double d, char* last_error, GEOSGeometry** result) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  if (geom == NULL || npy_isnan(d)) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSInterpolate_r, GEOSSimplify_r, etc.)
  *result = func(context, geom, d);

  // Interpolate functions return NULL on PGERR_GEOMETRY_TYPE and on
  // PGERR_GEOS_EXCEPTION. Distinguish these by the state of last_error.
  if (*result == NULL) {
    if (last_error[0] != 0) {
      return PGERR_GEOS_EXCEPTION;
    } else {
      return PGERR_GEOMETRY_TYPE;
    }
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Yd->Y operations.
 * This handles single geometry inputs (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting geometry and double).
 *
 * This function is used as a template by the DEFINE_Yd_Y macro to create
 * specific scalar functions like PySimplify_Scalar, PyForce3D_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geometry, double_param)
 *   nargs: Number of arguments (should be 2)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Yd_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs, FuncGEOS_Yd_Y* func) {
  PyObject* geom_obj;
  double double_param;
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  geom_obj = args[0];
  if (!PyFloat_Check(args[1])) {
    PyErr_Format(PyExc_TypeError, "Expected float as second argument, got %s", Py_TYPE(args[1])->tp_name);
    return NULL;
  }
  double_param = PyFloat_AsDouble(args[1]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_Yd_Y_operation(ctx, func, geom_obj, double_param, last_error, &ret_ptr);

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
 * NumPy universal function implementation for Yd->Y operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Yd_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Yd_Y* func = (FuncGEOS_Yd_Y*)data;
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
  // ip1 points to current geometry element, ip2 points to current double element, op1 points to current output element
  BINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Yd_Y_operation(ctx, func, *(PyObject**)ip1, *(double*)ip2, last_error, &geom_arr[i]);
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
static PyUFuncGenericFunction Yd_Y_funcs[1] = {&Yd_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry) and NPY_DOUBLE, returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Yd_Y_dtypes[3] = {NPY_OBJECT, NPY_DOUBLE, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PySimplify_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Example: DEFINE_Yd_Y(GEOSSimplify_r) creates PyGEOSSimplify_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Yd_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_Yd_Y_Scalar(self, args, nargs, (FuncGEOS_Yd_Y*)func_name); \
  }

DEFINE_Yd_Y(GEOSInterpolateProtectEmpty_r);
DEFINE_Yd_Y(GEOSInterpolateNormalizedProtectEmpty_r);
DEFINE_Yd_Y(GEOSSimplify_r);
DEFINE_Yd_Y(GEOSTopologyPreserveSimplify_r);
DEFINE_Yd_Y(PyGEOSForce3D);
DEFINE_Yd_Y(GEOSUnaryUnionPrec_r);
DEFINE_Yd_Y(GEOSMaximumInscribedCircleWithDefaultTolerance);
DEFINE_Yd_Y(GEOSDensify_r);

#if GEOS_SINCE_3_11_0
DEFINE_Yd_Y(GEOSRemoveRepeatedPoints_r);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "simplify") for array operations
 * 2. A scalar function (e.g., "simplify_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSSimplify_r)
 *   py_name: Python function name (e.g., simplify)
 *
 */
#define INIT_Yd_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 2 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Yd_Y_funcs, func_name##_FuncData, Yd_Y_dtypes, 1, 2, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        (PyCFunction)Py##func_name##_Scalar,  /* C function pointer */ \
        METH_FASTCALL,                        /* Function takes two arguments */ \
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
int init_geos_funcs_Yd_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Yd_Y(GEOSInterpolateProtectEmpty_r, line_interpolate_point);
  INIT_Yd_Y(GEOSInterpolateNormalizedProtectEmpty_r, line_interpolate_point_normalized);
  INIT_Yd_Y(GEOSSimplify_r, simplify);
  INIT_Yd_Y(GEOSTopologyPreserveSimplify_r, simplify_preserve_topology);
  INIT_Yd_Y(PyGEOSForce3D, force_3d);
  INIT_Yd_Y(GEOSUnaryUnionPrec_r, unary_union_prec);
  INIT_Yd_Y(GEOSMaximumInscribedCircleWithDefaultTolerance, maximum_inscribed_circle);
  INIT_Yd_Y(GEOSDensify_r, segmentize);

#if GEOS_SINCE_3_11_0
  INIT_Yd_Y(GEOSRemoveRepeatedPoints_r, remove_repeated_points);
#endif

  return 0;
}
