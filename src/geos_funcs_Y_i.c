#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>
#include <stdint.h>

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
 * Function signature for GEOS operations that take a geometry and return an int: Y->i.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry (GEOSGeometry*)
 *
 * Returns:
 *   The requested integer value, or an error code on error
 */
typedef int FuncGEOS_Y_i(GEOSContextHandle_t context, const GEOSGeometry* a);

/* Wrapper for GEOSGeomGetNumPoints_r - returns 0 for non-linear geometries */
static int GetNumPoints(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if ((typ == GEOS_LINESTRING) || (typ == GEOS_LINEARRING)) {
    return GEOSGeomGetNumPoints_r(context, geom);
  } else {
    return 0;
  }
}

/* Wrapper for GEOSGetNumInteriorRings_r - returns 0 for non-polygon geometries */
static int GetNumInteriorRings(GEOSContextHandle_t context, const GEOSGeometry* geom) {
  char typ = GEOSGeomTypeId_r(context, geom);
  if (typ == GEOS_POLYGON) {
    return GEOSGetNumInteriorRings_r(context, geom);
  } else {
    return 0;
  }
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
 *   errcode: Error code to check against (indicates failure)
 *   none_value: Value to return when input is None
 *   geom_obj: Shapely geometry object (Python wrapper around GEOSGeometry)
 *   result: Pointer where the computed int result will be stored
 *   last_error_ptr: Pointer to the last_error buffer (available in GEOS_INIT context)
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Y_i_operation(GEOSContextHandle_t context, FuncGEOS_Y_i* func,
                               int errcode, int none_value, PyObject* geom_obj,
                               int* result, char* last_error) {
  const GEOSGeometry* geom;

  // Extract the underlying GEOS geometry from the Python geometry object
  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case - return none_value as per convention
  if (geom == NULL) {
    *result = none_value;
    return PGERR_SUCCESS;
  }

  // Call the specific GEOS function
  *result = func(context, geom);

  // Check if the result equals the error code
  // We also check if GEOS actually set an error (we can't be sure otherwise)
  if ((*result == errcode) && (last_error[0] != 0)) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Y->i operations.
 * This handles single geometry inputs (not arrays) and returns a Python int.
 * It should be registered as a METH_O method (accepting only a single argument).
 *
 * This function is used as a template by the DEFINE_Y_i macro to create
 * specific scalar functions like PyGetTypeId_Scalar, PyGetNumPoints_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   obj: Input geometry object (should be a GeometryObject)
 *   func: Function pointer to the specific GEOS operation
 *   errcode: Error code for the GEOS operation
 *   none_value: Value to return for None inputs
 *
 * Returns:
 *   PyLong object containing the result, or NULL on error
 */
static PyObject* Py_Y_i_Scalar(PyObject* self, PyObject* obj, FuncGEOS_Y_i* func,
                               int errcode, int none_value) {
  int result = 0;

  GEOS_INIT;

  errstate = core_Y_i_operation(ctx, func, errcode, none_value, obj, &result, last_error);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;  // Python exception was set by GEOS_FINISH
  }

  return PyLong_FromLong(result);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for Y->i operations.
 * This handles arrays of geometries efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointers and error codes)
 */
static void Y_i_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific GEOS function from the user data
  // Data format: [func_ptr, errcode_ptr, none_value_ptr]
  FuncGEOS_Y_i* func = (FuncGEOS_Y_i*)((void**)data)[0];
  // Extract and convert error code from opaque data pointer:
  //   ((void**)data)[1]     - Get the second element as a void* pointer
  //   (intptr_t)(...)       - Convert void* to intptr_t (pointer-sized integer)
  //                           intptr_t is the safe, portable way to hold any pointer value
  //   (int)(...)            - Convert intptr_t to the actual int we need
  int errcode = (int)(intptr_t)(((void**)data)[1]);
  int none_value = (int)(intptr_t)(((void**)data)[2]);

  int result = 0;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    errstate = core_Y_i_operation(ctx, func, errcode, none_value, *(PyObject**)ip1,
                                  &result, last_error);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
    *(npy_int*)op1 = result;
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
static PyUFuncGenericFunction Y_i_funcs[1] = {&Y_i_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT (geometry), returns NPY_INT.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Y_i_dtypes[2] = {NPY_OBJECT, NPY_INT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGetTypeId_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 *
 * Example: DEFINE_Y_i(GetTypeId, -1, -1) creates PyGetTypeId_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 *   errcode: Error code that indicates failure
 *   none_value: Value to return when input is None
 *
 * This macro also generates the data tuple for the ufunc registration.
 */
#define DEFINE_Y_i(func_name, errcode, none_value) \
  static void* func_name##_func_tuple[3] = {func_name, (void*)(intptr_t)errcode, (void*)(intptr_t)none_value}; \
  static void* func_name##_data[1] = {func_name##_func_tuple}; \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return Py_Y_i_Scalar(self, obj, (FuncGEOS_Y_i*)func_name, errcode, none_value); \
  }

DEFINE_Y_i(GEOSGeomTypeId_r, -1, -1);
DEFINE_Y_i(GEOSGeom_getDimensions_r, 0, -1);
DEFINE_Y_i(GEOSGeom_getCoordinateDimension_r, -1, -1);
DEFINE_Y_i(GEOSGetSRID_r, 0, -1);
DEFINE_Y_i(GetNumPoints, -1, 0);
DEFINE_Y_i(GetNumInteriorRings, -1, 0);
DEFINE_Y_i(GEOSGetNumGeometries_r, -1, 0);
DEFINE_Y_i(GEOSGetNumCoordinates_r, -1, 0);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "get_type_id") for array operations
 * 2. A scalar function (e.g., "get_type_id_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSGeomTypeId_r)
 *   py_name: Python function name (e.g., get_type_id)
 *
 */
#define INIT_Y_i(func_name, py_name) do { \
    /* Create NumPy ufunc: 1 input, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(Y_i_funcs, func_name##_data, Y_i_dtypes, 1, 1, 1, \
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
int init_geos_funcs_Y_i(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_Y_i(GEOSGeomTypeId_r, get_type_id);
  INIT_Y_i(GEOSGeom_getDimensions_r, get_dimensions);
  INIT_Y_i(GEOSGeom_getCoordinateDimension_r, get_coordinate_dimension);
  INIT_Y_i(GEOSGetSRID_r, get_srid);
  INIT_Y_i(GetNumPoints, get_num_points);
  INIT_Y_i(GetNumInteriorRings, get_num_interior_rings);
  INIT_Y_i(GEOSGetNumGeometries_r, get_num_geometries);
  INIT_Y_i(GEOSGetNumCoordinates_r, get_num_coordinates);

  return 0;
}
