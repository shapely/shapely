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
 * Function signature for GEOS operations that take a geometry, a double, and
 * an int, and return a geometry: Ydi->Y.
 * *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *   b: Double parameter
 *   c: Integer parameter
 *
 * Returns:
 *   New GEOSGeometry*, or NULL on error
 */
typedef GEOSGeometry* FuncGEOS_Ydi_Y(GEOSContextHandle_t context, const GEOSGeometry* a,
                                      double b, int c);

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
 *   geom_obj: Shapely geometry Python object
 *   b: Double parameter
 *   c: Integer parameter
 *   result: receives the new GEOSGeometry* (or NULL for missing/NaN input)
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Ydi_Y_operation(GEOSContextHandle_t context, FuncGEOS_Ydi_Y* func,
                                  PyObject* geom_obj, double b, int c,
                                  GEOSGeometry** result) {
  const GEOSGeometry* geom;

  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // NULL geometry or NaN -> return NULL (None)
  if ((geom == NULL) || npy_isnan(b)) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  *result = func(context, geom, b, c);
  if (*result == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Ydi->Y operations.
 * This handles a geometry input, a double, and an integer (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting three arguments).
 *
 * This function is used as a template by the DEFINE_Ydi_Y macro to create
 * specific scalar functions like PyGEOSGeom_setPrecision_r_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geometry, double, int)
 *   nargs: Number of arguments (should be 3)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Ydi_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                   FuncGEOS_Ydi_Y* func) {
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 3) {
    PyErr_Format(PyExc_TypeError, "expected 3 arguments, got %zd", nargs);
    return NULL;
  }

  double b = PyFloat_AsDouble(args[1]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  int c = (int)PyLong_AsLong(args[2]);
  if (PyErr_Occurred()) {
    return NULL;
  }

  GEOS_INIT;

  errstate = core_Ydi_Y_operation(ctx, func, args[0], b, c, &ret_ptr);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  return GeometryObject_FromGEOS(ret_ptr, ctx);
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for Ydi->Y operations.
 * This handles arrays of geometries and doubles efficiently by iterating through them.
 *
 * Note: The integer parameter must be scalar and is read before entering the GEOS context.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Ydi_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                        void* data) {
  FuncGEOS_Ydi_Y* func = (FuncGEOS_Ydi_Y*)data;
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

  QUATERNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Ydi_Y_operation(ctx, func, *(PyObject**)ip1, *(double*)ip2, *(int*)ip3,
                                    &geom_arr[i]);
    if (errstate != PGERR_SUCCESS) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;

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
static PyUFuncGenericFunction Ydi_Y_funcs[1] = {&Ydi_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT, NPY_DOUBLE, and NPY_INT, returns NPY_OBJECT (geometry).
 * This tells NumPy what input and output types this ufunc supports.
 */
static char Ydi_Y_dtypes[4] = {NPY_OBJECT, NPY_DOUBLE, NPY_INT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * We use a macro to define a scalar Python function for each GEOS operation.
 *
 * This creates a function like PyGEOSGeom_setPrecision_r_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* args)
 *
 * Example: DEFINE_Ydi_Y(GEOSGeom_setPrecision_r) creates PyGEOSGeom_setPrecision_r_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the GEOS operation
 */
#define DEFINE_Ydi_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, \
                                           Py_ssize_t nargs) { \
    return Py_Ydi_Y_Scalar(self, args, nargs, (FuncGEOS_Ydi_Y*)func_name); \
  }

DEFINE_Ydi_Y(GEOSGeom_setPrecision_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "set_precision") for array operations
 * 2. A scalar function (e.g., "set_precision_scalar") for single geometry operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., GEOSGeom_setPrecision_r)
 *   py_name: Python function name (e.g., set_precision)
 *
 */
#define INIT_Ydi_Y(func_name, py_name) do { \
    static void* func_name##_FuncData[1] = {(void*)func_name}; \
    \
    ufunc = PyUFunc_FromFuncAndData(Ydi_Y_funcs, func_name##_FuncData, Ydi_Y_dtypes, 1, 3, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    static PyMethodDef Py##func_name##_Scalar_Def = { \
        #py_name "_scalar", \
        (PyCFunction)Py##func_name##_Scalar, \
        METH_FASTCALL, \
        #py_name " scalar implementation" \
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
int init_geos_funcs_Ydi_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  INIT_Ydi_Y(GEOSGeom_setPrecision_r, set_precision);

  return 0;
}
