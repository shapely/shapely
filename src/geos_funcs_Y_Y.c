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
 * UTILITY FUNCTIONS
 * ========================================================================
 */

/* ========================================================================
 * GEOS WRAPPER FUNCTIONS
 * ========================================================================
 *
 * Function signature for GEOS operations that take a geometry and return a geometry: Y->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *
 * Returns:
 *   New GEOSGeometry* object, or NULL on error or for missing values
 */
typedef GEOSGeometry* FuncGEOS_Y_Y(GEOSContextHandle_t context, const GEOSGeometry* a);

/* Wrapper functions to handle special cases */

static GEOSGeometry* GEOSBoundaryAllTypes_r(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ == GEOS_GEOMETRYCOLLECTION) {
    return NULL;
  } else {
    return GEOSBoundary_r(context, geom);
  }
}

/* the normalize function acts in-place */
static GEOSGeometry* GEOSNormalize_r_with_clone(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  int ret;
  GEOSGeometry* new_geom = GEOSGeom_clone_r(context, geom);
  if (new_geom == NULL) {
    return NULL;
  }
  ret = GEOSNormalize_r(context, new_geom);
  if (ret == -1) {
    GEOSGeom_destroy_r(context, new_geom);
    return NULL;
  }
  return new_geom;
}

static GEOSGeometry* GetExteriorRing(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ != GEOS_POLYGON) {
    return NULL;
  }
  const GEOSGeometry* ring = GEOSGetExteriorRing_r(context, geom);
  /* Create a copy of the obtained geometry */
  if (ring == NULL) {
    return NULL;
  }
  return GEOSGeom_clone_r(context, ring);
}

static GEOSGeometry* GEOSMinimumBoundingCircleWithReturn(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  GEOSGeometry* center = NULL;
  double radius;
  GEOSGeometry* ret = GEOSMinimumBoundingCircle_r(context, geom, &radius, &center);
  if (ret == NULL) {
    return NULL;
  }
  GEOSGeom_destroy_r(context, center);
  return ret;
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
 *   last_error: Error buffer to check for GEOS exceptions
 *   result_ptr: Pointer where the computed GEOSGeometry result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Y_Y_operation(GEOSContextHandle_t context, FuncGEOS_Y_Y* func,
                               PyObject* geom_obj, char* last_error, GEOSGeometry** result) {
  GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!get_geom((GeometryObject*)geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case - return NULL (None)
  if (geom == NULL) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function (e.g., GEOSEnvelope_r, GEOSConvexHull_r, etc.)
  *result = func(context, geom);

  // NULL can mean either error or a valid "missing value" for some functions
  // (GetExteriorRing, GEOSBoundaryAllTypes_r) so check the last_error before
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
 * Generic scalar Python function implementation for Y->Y operations.
 * This handles single geometry inputs (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_O method (accepting only a single argument).
 *
 * This function is used as a template by the DEFINE_Y_Y macro to create
 * specific scalar functions like PyEnvelope_Scalar, PyConvexHull_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   obj: Input geometry object (should be a GeometryObject)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Y_Y_Scalar(PyObject* self, PyObject* obj, FuncGEOS_Y_Y* func) {
  GEOSGeometry* ret_ptr = NULL;
  PyObject* ret;

  GEOS_INIT;

  errstate = core_Y_Y_operation(ctx, func, obj, last_error, &ret_ptr);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_FINISH
  }

  ret = GeometryObject_FromGEOS(ret_ptr, ctx);
  return ret;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for Y->Y operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Y_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  FuncGEOS_Y_Y* func = (FuncGEOS_Y_Y*)data;
  GEOSGeometry** geom_arr = NULL;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  if (steps[1] == 0) {
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

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Y_Y_operation(ctx, func, *(PyObject**)ip1, last_error, &geom_arr[i]);
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
    geom_arr_to_npy(geom_arr, args[1], steps[1], dimensions[0]);
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
static PyUFuncGenericFunction Y_Y_funcs[1] = {&Y_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Y_Y_dtypes[2] = {NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyEnvelope_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 *
 * Example: DEFINE_Y_Y(GEOSEnvelope_r) creates PyGEOSEnvelope_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Y_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return Py_Y_Y_Scalar(self, obj, (FuncGEOS_Y_Y*)func_name); \
  }

DEFINE_Y_Y(GEOSEnvelope_r);
DEFINE_Y_Y(GEOSConvexHull_r);
DEFINE_Y_Y(GEOSBoundaryAllTypes_r);
DEFINE_Y_Y(GEOSUnaryUnion_r);
DEFINE_Y_Y(GEOSPointOnSurface_r);
DEFINE_Y_Y(GEOSGetCentroid_r);
DEFINE_Y_Y(GEOSLineMerge_r);
DEFINE_Y_Y(GEOSMinimumClearanceLine_r);
DEFINE_Y_Y(GEOSNode_r);
DEFINE_Y_Y(GEOSGeom_extractUniquePoints_r);
DEFINE_Y_Y(GetExteriorRing);
DEFINE_Y_Y(GEOSNormalize_r_with_clone);
DEFINE_Y_Y(PyGEOSForce2D);
DEFINE_Y_Y(GEOSMinimumRotatedRectangle_r);
DEFINE_Y_Y(GEOSReverse_r);
DEFINE_Y_Y(GEOSBuildArea_r);
DEFINE_Y_Y(GEOSCoverageUnion_r);
DEFINE_Y_Y(GEOSMinimumBoundingCircleWithReturn);
DEFINE_Y_Y(GEOSMinimumWidth_r);
DEFINE_Y_Y(GEOSConstrainedDelaunayTriangulation_r);
#if GEOS_SINCE_3_11_0
DEFINE_Y_Y(GEOSLineMergeDirected_r);
#endif
#if GEOS_SINCE_3_12_0
DEFINE_Y_Y(GEOSDisjointSubsetUnion_r);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "envelope") for array operations
 * 2. A scalar function (e.g., "envelope_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSEnvelope_r)
 *   py_name: Python function name (e.g., envelope)
 *
 */
#define INIT_Y_Y(func_name, py_name) do { \
    /* Create data array to pass GEOS function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 1 input, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Y_Y_funcs, func_name##_FuncData, Y_Y_dtypes, 1, 1, 1, \
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
int init_geos_funcs_Y_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Y_Y(GEOSEnvelope_r, envelope);
  INIT_Y_Y(GEOSConvexHull_r, convex_hull);
  INIT_Y_Y(GEOSBoundaryAllTypes_r, boundary);
  INIT_Y_Y(GEOSUnaryUnion_r, unary_union);
  INIT_Y_Y(GEOSPointOnSurface_r, point_on_surface);
  INIT_Y_Y(GEOSGetCentroid_r, centroid);
  INIT_Y_Y(GEOSLineMerge_r, line_merge);
  INIT_Y_Y(GEOSMinimumClearanceLine_r, minimum_clearance_line);
  INIT_Y_Y(GEOSNode_r, node);
  INIT_Y_Y(GEOSGeom_extractUniquePoints_r, extract_unique_points);
  INIT_Y_Y(GetExteriorRing, get_exterior_ring);
  INIT_Y_Y(GEOSNormalize_r_with_clone, normalize);
  INIT_Y_Y(PyGEOSForce2D, force_2d);
  INIT_Y_Y(GEOSMinimumRotatedRectangle_r, oriented_envelope);
  INIT_Y_Y(GEOSReverse_r, reverse);
  INIT_Y_Y(GEOSBuildArea_r, build_area);
  INIT_Y_Y(GEOSCoverageUnion_r, coverage_union);
  INIT_Y_Y(GEOSMinimumBoundingCircleWithReturn, minimum_bounding_circle);
  INIT_Y_Y(GEOSMinimumWidth_r, minimum_width);
  INIT_Y_Y(GEOSConstrainedDelaunayTriangulation_r, constrained_delaunay_triangles);

#if GEOS_SINCE_3_11_0
  INIT_Y_Y(GEOSLineMergeDirected_r, line_merge_directed);
#endif

#if GEOS_SINCE_3_12_0
  INIT_Y_Y(GEOSDisjointSubsetUnion_r, disjoint_subset_union);
#endif

  return 0;
}
