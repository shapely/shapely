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
 * Function signatures for GEOS operations that take two geometries and return
 * a string (as a GEOS-allocated char*): YY->O.
 *
 * Currently only GEOSRelate_r / GEOSPreparedRelate_r use this signature.
 * The prepared variant is available from GEOS 3.13.0.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   a: First input geometry
 *   b: Second input geometry
 *
 * Returns:
 *   GEOS-allocated char* string (caller must GEOSFree_r it), or NULL on error
 */
typedef char* FuncGEOS_YY_str(GEOSContextHandle_t context, const GEOSGeometry* a,
                               const GEOSGeometry* b);
typedef char* FuncGEOS_YY_str_p(GEOSContextHandle_t context,
                                 const GEOSPreparedGeometry* a, const GEOSGeometry* b);

/*
 * Struct to hold function pointers for both prepared and non-prepared versions.
 * func_prepared may be NULL when no prepared variant exists.
 */
typedef struct {
  FuncGEOS_YY_str* func;
  FuncGEOS_YY_str_p* func_prepared;
} YY_O_func_data;

/* ========================================================================
 * CORE OPERATION LOGIC
 * ======================================================================== */

/*
 * Core function that performs the actual GEOS operation.
 * This is shared between both ufunc and scalar implementations to avoid code duplication.
 *
 * Note: Creates a Python string object, so must be called while the GIL is held.
 *
 * Parameters:
 *   context: GEOS context handle for thread-safe operations
 *   data: YY_O_func_data struct containing function pointers
 *   geom1_obj: First Shapely geometry Python object
 *   geom2_obj: Second Shapely geometry Python object
 *   result: receives Py_None (for NULL geometry) or a new PyUnicode object
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
static char core_YY_O_operation(GEOSContextHandle_t context, const YY_O_func_data* data,
                                 PyObject* geom1_obj, PyObject* geom2_obj,
                                 PyObject** result) {
  const GEOSGeometry* geom1;
  const GEOSGeometry* geom2;
  const GEOSPreparedGeometry* geom1_prepared;
  char* str;

  if (!ShapelyGetGeometryWithPrepared(geom1_obj, &geom1, &geom1_prepared)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (!ShapelyGetGeometry(geom2_obj, &geom2)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  if ((geom1 == NULL) || (geom2 == NULL)) {
    // NULL geometry -> Python None
    Py_INCREF(Py_None);
    *result = Py_None;
    return PGERR_SUCCESS;
  }

#if GEOS_SINCE_3_13_0
  if (geom1_prepared != NULL && data->func_prepared != NULL) {
    str = data->func_prepared(context, geom1_prepared, geom2);
  } else {
    str = data->func(context, geom1, geom2);
  }
#else
  str = data->func(context, geom1, geom2);
#endif

  if (str == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }

  *result = PyUnicode_FromString(str);
  GEOSFree_r(context, str);
  return PGERR_SUCCESS;
}

/* ========================================================================
 * SCALAR PYTHON FUNCTION
 * ======================================================================== */

/*
 * Generic scalar Python function implementation for YY->O operations.
 * This handles two geometry inputs (not arrays) and returns a Python string.
 * It should be registered as a METH_FASTCALL method (accepting two arguments).
 *
 * This function is used as a template by the DEFINE_YY_O macro to create
 * specific scalar functions like PyGEOSRelate_r_Scalar, etc.
 *
 * Parameters:
 *   self: Module object (unused, required by Python C API)
 *   args: Array of arguments (two geometries)
 *   nargs: Number of arguments (should be 2)
 *   data: YY_O_func_data struct containing GEOS function pointers
 *
 * Returns:
 *   PyObject* containing the result string (or None), or NULL on error
 */
static PyObject* Py_YY_O_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs,
                                  const YY_O_func_data* data) {
  PyObject* result = NULL;

  if (nargs != 2) {
    PyErr_Format(PyExc_TypeError, "expected 2 arguments, got %zd", nargs);
    return NULL;
  }

  GEOS_INIT;

  errstate = core_YY_O_operation(ctx, data, args[0], args[1], &result);

  GEOS_FINISH;

  if (errstate != PGERR_SUCCESS) {
    Py_XDECREF(result);
    return NULL;
  }

  return result;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for YY->O operations.
 * This handles arrays of geometry pairs efficiently by iterating through them.
 *
 * Note: The GIL is held throughout since Python objects are created in the loop.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer struct)
 */
static void YY_O_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Initialize GEOS context (holds GIL: Python objects are created in the loop)
  GEOS_INIT;

  BINARY_LOOP {
    CHECK_SIGNALS(i);
    if (errstate == PGERR_PYSIGNAL) {
      goto finish;
    }
    PyObject** out = (PyObject**)op1;
    PyObject* new_val = NULL;

    errstate = core_YY_O_operation(ctx, (YY_O_func_data*)data, *(PyObject**)ip1,
                                   *(PyObject**)ip2, &new_val);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }

    Py_XDECREF(*out);
    *out = new_val;
  }

finish:
  GEOS_FINISH;
}

static PyUFuncGenericFunction YY_O_funcs[1] = {&YY_O_func};

/* Type signature: two geometry objects -> Python object (string) */
static char YY_O_dtypes[3] = {NPY_OBJECT, NPY_OBJECT, NPY_OBJECT};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ======================================================================== */

/*
 * Macro to define both the func_data struct and the scalar Python function.
 *
 * DEFINE_YY_O(func, func_prepared) creates:
 *   - static YY_O_func_data func##_data
 *   - static PyObject* Py##func##_Scalar(...)
 */
#define DEFINE_YY_O(func, func_prepared) \
  static YY_O_func_data func##_data = {func, func_prepared}; \
  static PyObject* Py##func##_Scalar(PyObject* self, PyObject* const* args, Py_ssize_t nargs) { \
    return Py_YY_O_Scalar(self, args, nargs, &func##_data); \
  }

#if GEOS_SINCE_3_13_0
DEFINE_YY_O(GEOSRelate_r, GEOSPreparedRelate_r);
#else
DEFINE_YY_O(GEOSRelate_r, NULL);
#endif


/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * Macro to register both ufunc and scalar versions.
 */
#define INIT_YY_O(py_name, func, func_prepared) do { \
    static void* func##_udata[1] = {(void*)&func##_data}; \
    \
    ufunc = PyUFunc_FromFuncAndData(YY_O_funcs, func##_udata, YY_O_dtypes, 1, 2, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    \
    static PyMethodDef Py##func##_Scalar_Def = { \
        #py_name "_scalar", \
        (PyCFunction)Py##func##_Scalar, \
        METH_FASTCALL, \
        #py_name " scalar implementation" \
    }; \
    PyObject* Py##func##_Scalar_Func = PyCFunction_NewEx(&Py##func##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func##_Scalar_Func); \
} while(0)

int init_geos_funcs_YY_O(PyObject* m, PyObject* d) {
  PyObject* ufunc;

#if GEOS_SINCE_3_13_0
  INIT_YY_O(relate, GEOSRelate_r, GEOSPreparedRelate_r);
#else
  INIT_YY_O(relate, GEOSRelate_r, NULL);
#endif

  return 0;
}
