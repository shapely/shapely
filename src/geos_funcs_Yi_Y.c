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
 * Function signature for GEOS operations that take a geometry and an int, returning a geometry: Yi->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *   b: Input integer parameter
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error or for missing values
 */
typedef GEOSGeometry* FuncGEOS_Yi_Y(GEOSContextHandle_t context, const GEOSGeometry* a, int b);

/* Wrapper functions to handle special cases */

static GEOSGeometry* GetPointN(GEOSContextHandle_t context, const GEOSGeometry* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  int size, i;
  if ((typ != GEOS_LINESTRING) && (typ != GEOS_LINEARRING)) {
    return NULL;
  }
  size = GEOSGeomGetNumPoints_r(context, geom);
  if (size == -1) {
    return NULL;
  }
  if (n < 0) {
    /* Negative indexing: we get it for free */
    i = size + n;
  } else {
    i = n;
  }
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  return GEOSGeomGetPointN_r(context, geom, i);
}

static GEOSGeometry* GetInteriorRingN(GEOSContextHandle_t context, const GEOSGeometry* geom, int n) {
  char typ = GEOSGeomTypeId_r(context, geom);
  int size, i;
  const GEOSGeometry* ring;

  if (typ != GEOS_POLYGON) {
    return NULL;
  }
  size = GEOSGetNumInteriorRings_r(context, geom);
  if (size == -1) {
    return NULL;
  }
  if (n < 0) {
    /* Negative indexing: we get it for free */
    i = size + n;
  } else {
    i = n;
  }
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  ring = GEOSGetInteriorRingN_r(context, geom, i);
  if (ring == NULL) {
    return NULL;
  }
  /* Create a copy of the obtained geometry */
  return GEOSGeom_clone_r(context, ring);
}

static GEOSGeometry* GetGeometryN(GEOSContextHandle_t context, const GEOSGeometry* geom, int n) {
  int size, i;
  const GEOSGeometry* subgeom;

  size = GEOSGetNumGeometries_r(context, geom);
  if (size == -1) {
    return NULL;
  }
  if (n < 0) {
    /* Negative indexing: we get it for free */
    i = size + n;
  } else {
    i = n;
  }
  if ((i < 0) || (i >= size)) {
    /* Important, could give segfaults else */
    return NULL;
  }
  subgeom = GEOSGetGeometryN_r(context, geom, i);
  if (subgeom == NULL) {
    return NULL;
  }
  /* Create a copy of the obtained geometry */
  return GEOSGeom_clone_r(context, subgeom);
}

/* the set srid function acts in-place */
static GEOSGeometry* GEOSSetSRID_r_with_clone(GEOSContextHandle_t context, const GEOSGeometry* geom, int srid) {
  GEOSGeometry* ret = GEOSGeom_clone_r(context, geom);
  if (ret == NULL) {
    return NULL;
  }
  GEOSSetSRID_r(context, ret, srid);
  return ret;
}

#if GEOS_SINCE_3_12_0
static GEOSGeometry* GEOSOrientPolygons_r_with_clone(GEOSContextHandle_t context, const GEOSGeometry* geom, int exteriorCW) {
  int ret;
  GEOSGeometry* cloned = GEOSGeom_clone_r(context, geom);
  if (cloned == NULL) {
    return NULL;
  }
  ret = GEOSOrientPolygons_r(context, cloned, exteriorCW);
  if (ret == -1) {
    return NULL;
  }
  return cloned;
}
#endif

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
 *   i: Integer parameter for the operation
 *   last_error: Error buffer to check for GEOS exceptions
 *   result_ptr: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Yi_Y_operation(GEOSContextHandle_t context, FuncGEOS_Yi_Y* func,
                                PyObject* geom_obj, int i, char* last_error, GEOSGeometry** result) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case - return NULL (None)
  if (geom == NULL) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GetPointN, GetInteriorRingN, etc.)
  *result = func(context, geom, i);

  // NULL can mean either error or a valid "missing value" for some functions
  // (GetPointN, GetInteriorRingN, GetGeometryN) so check the last_error before
  // setting error state
  if ((*result == NULL) && (last_error[0] != 0)) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Yi->Y operations.
 * This handles single geometry inputs (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting geometry and int).
 *
 * This function is used as a template by the DEFINE_Yi_Y macro to create
 * specific scalar functions like PyGetPointN_Scalar, PySetSRID_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geometry, int_param)
 *   nargs: Number of arguments (should be 2)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Yi_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs, FuncGEOS_Yi_Y* func) {
  PyObject* geom_obj;
  int int_param;
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "Expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  geom_obj = args[0];
  if (!PyLong_Check(args[1])) {
    PyErr_Format(PyExc_TypeError, "Expected int as second argument, got %s", Py_TYPE(args[1])->tp_name);
    return NULL;
  }
  int_param = PyLong_AsLong(args[1]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_Yi_Y_operation(ctx, func, geom_obj, int_param, last_error, &ret_ptr);

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
 * NumPy universal function implementation for Yi->Y operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Yi_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Yi_Y* func = (FuncGEOS_Yi_Y*)data;
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
  // ip1 points to current geometry element, ip2 points to current int element, op1 points to current output
  BINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Yi_Y_operation(ctx, func, *(PyObject**)ip1, *(int*)ip2, last_error, &geom_arr[i]);
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
static PyUFuncGenericFunction Yi_Y_funcs[1] = {&Yi_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry) and NPY_INT, returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Yi_Y_dtypes[3] = {NPY_OBJECT, NPY_INT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGetPointN_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* args)
 *
 * Example: DEFINE_Yi_Y(GetPointN) creates PyGetPointN_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Yi_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_Yi_Y_Scalar(self, args, nargs, (FuncGEOS_Yi_Y*)func_name); \
  }

DEFINE_Yi_Y(GetPointN);
DEFINE_Yi_Y(GetInteriorRingN);
DEFINE_Yi_Y(GetGeometryN);
DEFINE_Yi_Y(GEOSSetSRID_r_with_clone);
#if GEOS_SINCE_3_12_0
DEFINE_Yi_Y(GEOSOrientPolygons_r_with_clone);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "get_point") for array operations
 * 2. A scalar function (e.g., "get_point_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GetPointN)
 *   py_name: Python function name (e.g., get_point)
 *
 */
#define INIT_Yi_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 2 inputs (geom, int), 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Yi_Y_funcs, func_name##_FuncData, Yi_Y_dtypes, 1, 2, 1, \
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
int init_geos_funcs_Yi_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Yi_Y(GetPointN, get_point);
  INIT_Yi_Y(GetInteriorRingN, get_interior_ring);
  INIT_Yi_Y(GetGeometryN, get_geometry);
  INIT_Yi_Y(GEOSSetSRID_r_with_clone, set_srid);

#if GEOS_SINCE_3_12_0
  INIT_Yi_Y(GEOSOrientPolygons_r_with_clone, orient_polygons);
#endif

  return 0;
}
