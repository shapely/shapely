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
 * Function signature for GEOS operations that take a geometry and four doubles
 * and return a geometry: Ydddd->Y.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: Input geometry
 *   d1, d2, d3, d4: Four double parameters
 *
 * Returns:
 *   New GEOSGeometry*, or NULL on error
 */
typedef GEOSGeometry* FuncGEOS_Ydddd_Y(GEOSContextHandle_t context, const GEOSGeometry* a,
                                        double d1, double d2, double d3, double d4);

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Parameters:
 *   context: GEOS context handle
 *   func: GEOS function pointer
 *   geom_obj: Shapely geometry Python object
 *   d1, d2, d3, d4: four double parameters
 *   result: receives the new GEOSGeometry* (or NULL for missing input)
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_Ydddd_Y_operation(GEOSContextHandle_t context, FuncGEOS_Ydddd_Y* func,
                                    PyObject* geom_obj, double d1, double d2,
                                    double d3, double d4, GEOSGeometry** result) {
  const GEOSGeometry* geom;

  if (!ShapelyGetGeometry(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Missing geometry -> return NULL (None)
  if (geom == NULL) {
    *result = NULL;
    return PGERR_SUCCESS;
  }

  *result = func(context, geom, d1, d2, d3, d4);
  if (*result == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for Ydddd->Y operations.
 * This handles a geometry input and four double parameters (not arrays) and returns a Python Geometry.
 * It should be registered as a METH_FASTCALL method (accepting five arguments).
 *
 * This function is used as a template by the DEFINE_Ydddd_Y macro to create
 * specific scalar functions like PyClipByRect_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (geometry + four doubles)
 *   nargs: Number of arguments (should be 5)
 *   func: Function pointer to the specific GEOS operation
 *
 * Returns:
 *   PyObject* containing the result geometry, or NULL on error
 */
static PyObject* Py_Ydddd_Y_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                     FuncGEOS_Ydddd_Y* func) {
  GEOSGeometry* ret_ptr = NULL;

  if (nargs != 5) {
    PyErr_Format(PyExc_TypeError, "expected 5 arguments, got %zd", nargs);
    return NULL;
  }

  double d1 = PyFloat_AsDouble(args[1]);
  if (PyErr_Occurred()) return NULL;
  double d2 = PyFloat_AsDouble(args[2]);
  if (PyErr_Occurred()) return NULL;
  double d3 = PyFloat_AsDouble(args[3]);
  if (PyErr_Occurred()) return NULL;
  double d4 = PyFloat_AsDouble(args[4]);
  if (PyErr_Occurred()) return NULL;

  GEOS_INIT;

  errstate = core_Ydddd_Y_operation(ctx, func, args[0], d1, d2, d3, d4, &ret_ptr);

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
 * NumPy universal function implementation for Ydddd->Y operations.
 * This handles arrays of geometries and doubles efficiently by iterating through them.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void Ydddd_Y_func(char** args, const npy_intp* dimensions, const npy_intp* steps,
                          void* data) {
  FuncGEOS_Ydddd_Y* func = (FuncGEOS_Ydddd_Y*)data;
  GEOSGeometry** geom_arr = NULL;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  if ((steps[5] == 0) && (dimensions[0] > 1)) {
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

  QUINARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      destroy_geom_arr(ctx, geom_arr, i - 1);
      goto finish;
    }
    errstate = core_Ydddd_Y_operation(ctx, func, *(PyObject**)ip1,
                                      *(double*)ip2, *(double*)ip3,
                                      *(double*)ip4, *(double*)ip5,
                                      &geom_arr[i]);
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
    geom_arr_to_npy(geom_arr, args[5], steps[5], dimensions[0]);
  }
  if (geom_arr != NULL) {
    free(geom_arr);
  }
}

/*
 * Function pointer array for NumPy ufunc creation.
 */
static PyUFuncGenericFunction Ydddd_Y_funcs[1] = {&Ydddd_Y_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT, NPY_DOUBLE x 4, returns NPY_OBJECT.
 */
static char Ydddd_Y_dtypes[6] = {NPY_OBJECT, NPY_DOUBLE, NPY_DOUBLE,
                                  NPY_DOUBLE, NPY_DOUBLE, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ======================================================================== */

#define DEFINE_Ydddd_Y(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* const* args, \
                                           Py_ssize_t nargs) { \
    return Py_Ydddd_Y_Scalar(self, args, nargs, (FuncGEOS_Ydddd_Y*)func_name); \
  }

DEFINE_Ydddd_Y(GEOSClipByRect_r);


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

#define INIT_Ydddd_Y(func_name, py_name) do { \
    static void* func_name##_FuncData[1] = {(void*)func_name}; \
    \
    ufunc = PyUFunc_FromFuncAndData(Ydddd_Y_funcs, func_name##_FuncData, Ydddd_Y_dtypes, 1, 5, 1, \
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

int init_geos_funcs_Ydddd_Y(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  INIT_Ydddd_Y(GEOSClipByRect_r, clip_by_rect);

  return 0;
}
