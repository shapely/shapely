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
 * Function signature for GEOS operations that take a geometry, a double,
 * and a bool (passed as int), returning a geometry: Ydb->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *   b: Input double parameter
 *   c: Input bool parameter (as int)
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error (GEOS error or wrong geometry type)
 */
typedef GEOSGeometry* FuncGEOS_Ydb_Y(GEOSContextHandle_t context, const GEOSGeometry* a,
                                      double b, int c);

/* Wrapper functions to handle special cases */

#if GEOS_SINCE_3_11_0
static GEOSGeometry* ShapelyConcaveHull_r(GEOSContextHandle_t ctx,
                                           const GEOSGeometry* g,
                                           double ratio,
                                           int allow_holes) {
  return GEOSConcaveHull_r(ctx, g, ratio, (unsigned int)allow_holes);
}
#endif  // GEOS_SINCE_3_11_0

#if GEOS_SINCE_3_12_0
static GEOSGeometry* ShapelyCoverageSimplifyVWValidated_r(GEOSContextHandle_t ctx,
                                                           const GEOSGeometry* g,
                                                           double tolerance,
                                                           int simplify_boundary) {
  int num_geoms = GEOSGetNumGeometries_r(ctx, g);
  for (int j = 0; j < num_geoms; j++) {
    const GEOSGeometry* sub = GEOSGetGeometryN_r(ctx, g, j);
    int geom_type = GEOSGeomTypeId_r(ctx, sub);
    if (geom_type != GEOS_POLYGON && geom_type != GEOS_MULTIPOLYGON) {
      /* Return NULL without setting a GEOS error so that the caller can
       * distinguish this from a real GEOS exception (via last_error). */
      return NULL;
    }
  }
  return GEOSCoverageSimplifyVW_r(ctx, g, tolerance, !simplify_boundary);
}
#endif  // GEOS_SINCE_3_12_0

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
 *   d: Double parameter for the GEOS operation
 *   flag: Boolean flag parameter for the GEOS operation (as int)
 *   last_error: Error buffer to check for GEOS exceptions
 *   result: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Ydb_Y_operation(GEOSContextHandle_t context, FuncGEOS_Ydb_Y* func,
                                  PyObject* geom_obj, double d, int flag,
                                  char* last_error, GEOSGeometry** result) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  if (geom == NULL || npy_isnan(d)) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function
  *result = func(context, geom, d, flag);

  // Functions return NULL on PGERR_GEOMETRY_TYPE and on
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
 * Generic scalar Python function implementation for Ydb->Y operations.
 * This handles a geometry input, a double, and a boolean flag (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting only three arguments).
 *
 * This function is used as a template by the DEFINE_Ydb_Y macro to create
 * specific scalar functions like PyCoverageSimplify_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geometry, double, bool)
 *   nargs: Number of arguments (should be 3)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Ydb_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   FuncGEOS_Ydb_Y* func) {
  PyObject* geom_obj;
  double double_param;
  int bool_param;
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "Expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  geom_obj = args[0];

  if (!PyFloat_Check(args[1])) {
    PyErr_Format(PyExc_TypeError, "Expected float as second argument, got %s",
                 Py_TYPE(args[1])->tp_name);
    return NULL;
  }
  double_param = PyFloat_AsDouble(args[1]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  bool_param = PyObject_IsTrue(args[2]);
  if (bool_param < 0) {
    return NULL;  // PyObject_IsTrue sets an exception on error
  }

  GEOS_INIT;

  errstate = core_Ydb_Y_operation(ctx, func, geom_obj, double_param,
                                   bool_param, last_error, &ret_ptr);

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
 * NumPy universal function implementation for Ydb->Y operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Ydb_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Ydb_Y* func = (FuncGEOS_Ydb_Y*)data;
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

  TERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Ydb_Y_operation(ctx, func, *(PyObject**)ip1, *(double*)ip2,
                                     (int)(*(npy_bool*)ip3),
                                     last_error, &geom_arr[i]);
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
 * NumPy requires this format to register different implementations
 * for different type combinations.
 */
static PyUFuncGenericFunction Ydb_Y_funcs[1] = {&Ydb_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), NPY_DOUBLE, and NPY_BOOL,
 * returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Ydb_Y_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_BOOL, NPY_OBJECT};

/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGEOSDelaunayTriangulation_r_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
 *
 * Example: DEFINE_Ydb_Y(GEOSDelaunayTriangulation_r) creates PyGEOSDelaunayTriangulation_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Ydb_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, \
                                           Py_ssize_t nargs) { \
    return Py_Ydb_Y_Scalar(self, args, nargs, (FuncGEOS_Ydb_Y*)func_name); \
  }

DEFINE_Ydb_Y(GEOSDelaunayTriangulation_r)

#if GEOS_SINCE_3_11_0
DEFINE_Ydb_Y(ShapelyConcaveHull_r)
#endif

#if GEOS_SINCE_3_12_0
DEFINE_Ydb_Y(ShapelyCoverageSimplifyVWValidated_r)
#endif

/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * Macro to register both ufunc and scalar versions of a Ydb->Y function.
 *
 * Creates:
 *   1. A NumPy ufunc (e.g., "delaunay_triangles") for array operations.
 *   2. A scalar function (e.g., "delaunay_triangles_scalar") for single geometries.
 */
#define INIT_Ydb_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 3 inputs, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Ydb_Y_funcs, func_name##_FuncData, Ydb_Y_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    /* Create Python function */ \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar",                   /* Function name */ \
        (PyCFunction)Py##func_name##_Scalar,  /* C function pointer */ \
        METH_FASTCALL,                        /* Function takes three arguments */ \
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
int init_geos_funcs_Ydb_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Ydb_Y(GEOSDelaunayTriangulation_r, delaunay_triangles);

#if GEOS_SINCE_3_11_0
  INIT_Ydb_Y(ShapelyConcaveHull_r, concave_hull);
#endif

#if GEOS_SINCE_3_12_0
  INIT_Ydb_Y(ShapelyCoverageSimplifyVWValidated_r, coverage_simplify);
#endif

  return 0;
}
