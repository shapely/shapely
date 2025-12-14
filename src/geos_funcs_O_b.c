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
 * WRAPPER FUNCTIONS
 * ========================================================================
 *
 * Function signature for GEOS operations that take any object and return a bool: O->b.
 *
 * Parameters:
 *   context: GEOS context handle for thread safety
 *   obj: Input Python object (may be a geometry or any other type)
 *   result: Pointer where the computed double result will be stored
 *
 * Returns:
 *   Error state code (PGERR_SUCCESS, PGERR_NOT_A_GEOMETRY, etc.)
 */
typedef char FuncO_b(GEOSContextHandle_t context, PyObject* obj, char* result);

static char IsMissing(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g = NULL;
  if (!ShapelyGetGeometry(obj, &g)) {
    *result = 0;
  // ShapelyGetGeometry sets g to NULL for None input
  } else if (g == NULL) {
    *result = 1;
  } else {
    *result = 0;
  }
  return PGERR_SUCCESS;
}

static char IsGeometry(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g = NULL;
  if (!ShapelyGetGeometry(obj, &g)) {
    *result = 0;
  // ShapelyGetGeometry sets g to NULL for None input
  } else if (g == NULL) {
    *result = 0;
  } else {
    *result = 1;
  }
  return PGERR_SUCCESS;
}

static char IsValidInput(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g;
  *result = ShapelyGetGeometry(obj, &g);
  return PGERR_SUCCESS;
}

static char IsPrepared(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g;
  const GEOSPreparedGeometry* prep;
  if (!ShapelyGetGeometryWithPrepared(obj, &g, &prep)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  *result = prep != NULL;
  return PGERR_SUCCESS;
}

static char PrepareGeometry(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g;
  const GEOSPreparedGeometry* prep;
  if (!ShapelyGetGeometryWithPrepared(obj, &g, &prep)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if ((g == NULL) | (prep != NULL)) {
    // Nothing to do; set result to False
    *result = 0;
  } else {
    prep = GEOSPrepare_r(context, g);
    if (prep == NULL) {
      return PGERR_GEOS_EXCEPTION;
    }
    ((GeometryObject*)obj)->ptr_prepared = prep;
    *result = 1;
  }
  return PGERR_SUCCESS;
}
static char DestroyPreparedGeometryObject(GEOSContextHandle_t context, PyObject* obj, char* result) {
  const GEOSGeometry* g;
  const GEOSPreparedGeometry* prep;
  if (!ShapelyGetGeometryWithPrepared(obj, &g, &prep)) {
    return PGERR_NOT_A_GEOMETRY;
  }
  if (prep == NULL) {
    // Nothing to do; set result to False
    *result = 0;
  } else {
    GEOSPreparedGeom_destroy_r(context, prep);
    ((GeometryObject*)obj)->ptr_prepared = NULL;
    *result = 1;
  }
  return PGERR_SUCCESS;
}

/* ========================================================================
 * NUMPY UFUNC
 * ======================================================================== */

/*
 * NumPy universal function implementation for O->b operations.
 * This handles arrays of objects efficiently by iterating through them.
 * Since these operations don't call GEOS, we only need to manage the GIL.
 *
 * Parameters:
 *   args, dimensions, steps: Standard ufunc loop parameters (see NumPy docs)
 *   data: User data passed from ufunc creation (contains function pointer)
 */
static void O_b_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  // Extract the specific function from the user data
  FuncO_b* func = (FuncO_b*)data;

  // Initialize GEOS context with thread support (releases Python GIL)
  GEOS_INIT_THREADS;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      break;
    }
    errstate = func(ctx, *(PyObject**)ip1, (char*)op1);
    if (errstate != PGERR_SUCCESS) {
      break;
    }
  }

  // Clean up GEOS context and handle any errors (reacquires Python GIL)
  GEOS_FINISH_THREADS;
}

/*
 * Function pointer array for NumPy ufunc creation.
 * NumPy requires this format to register different implementations
 * for different type combinations.
 */
static PyUFuncGenericFunction O_b_funcs[1] = {&O_b_func};

/*
 * Type signature for the ufunc: takes NPY_OBJECT, returns NPY_BOOL.
 * This tells NumPy what input and output types this ufunc supports.
 */
static char O_b_dtypes[2] = {NPY_OBJECT, NPY_BOOL};


/* ========================================================================
 * PYTHON FUNCTION DEFINITIONS
 * ========================================================================
 *
 * Define functions for the O->b operations. We don't use a macro here because
 * they are all slightly different.
 *
 * The function signatures are:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 */
static PyObject* PyIsMissing_Scalar(PyObject* self, PyObject* obj) {
    char result;
    IsMissing(NULL, obj, &result);
    return PyBool_FromLong(result);
}

static PyObject* PyIsGeometry_Scalar(PyObject* self, PyObject* obj) {
    char result;
    IsGeometry(NULL, obj, &result);
    return PyBool_FromLong(result);
}

static PyObject* PyIsValidInput_Scalar(PyObject* self, PyObject* obj) {
    char result;
    IsValidInput(NULL, obj, &result);
    return PyBool_FromLong(result);
}

static PyObject* PyIsPrepared_Scalar(PyObject* self, PyObject* obj) {
  char result;
  GEOS_INIT;
  errstate = IsPrepared(ctx, obj, &result);
  GEOS_FINISH;
  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }
  return PyBool_FromLong(result);
}

static PyObject* PyPrepareGeometry_Scalar(PyObject* self, PyObject* obj) {
  char result;
  GEOS_INIT;
  errstate = PrepareGeometry(ctx, obj, &result);
  GEOS_FINISH;
  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }
  return PyBool_FromLong(result);
  }

static PyObject* PyDestroyPreparedGeometryObject_Scalar(PyObject* self, PyObject* obj) {
  char result;
  GEOS_INIT;
  errstate = DestroyPreparedGeometryObject(ctx, obj, &result);
  GEOS_FINISH;
  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }
  return PyBool_FromLong(result);
}

/* ========================================================================
 * MODULE INITIALIZATION
 * ======================================================================== */

/*
 * We use a single macro to register both ufunc and scalar versions of a function with Python.
 *
 * This creates two Python-callable functions:
 * 1. A NumPy ufunc (e.g., "is_missing") for array operations
 * 2. A scalar function (e.g., "is_missing_scalar") for single object operations
 *
 * Parameters:
 *   func_name: Name of the C function (e.g., IsMissing)
 *   py_name: Python function name (e.g., is_missing)
 *
 */

#define INIT_O_b(func_name, py_name) do { \
    /* Create data array to pass function pointer to the 'data' parameter of the ufunc */ \
    static void* func_name##_FuncData[1] = {func_name}; \
    \
    /* Create NumPy ufunc: 1 input, 1 output, 1 type signature */ \
    ufunc = PyUFunc_FromFuncAndData(O_b_funcs, func_name##_FuncData, O_b_dtypes, 1, 1, 1, \
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
int init_geos_funcs_O_b(PyObject* m, PyObject* d) {
  PyObject* ufunc;  // Temporary variable for ufunc creation

  INIT_O_b(IsMissing, is_missing);
  INIT_O_b(IsGeometry, is_geometry);
  INIT_O_b(IsValidInput, is_valid_input);
  INIT_O_b(IsPrepared, is_prepared);
  INIT_O_b(PrepareGeometry, prepare);
  INIT_O_b(DestroyPreparedGeometryObject, destroy_prepared);

  return 0;
}
