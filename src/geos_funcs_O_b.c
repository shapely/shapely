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
 * Function signature for operations that take a Python object and return a bool: O->b.
 * These functions do not raise errors on non-geometry objects (unlike Y_b functions).
 *
 * Parameters:
 *   obj: Input Python object (may be a geometry or any other type)
 *
 * Returns:
 *   0 for false, 1 for true
 */
typedef char FuncO_b(PyObject* obj);

static char IsMissing(PyObject* obj) {
  const GEOSGeometry* g = NULL;
  if (!ShapelyGetGeometry(obj, &g)) {
    return 0;
  };
  return g == NULL;
}

static char IsGeometry(PyObject* obj) {
  const GEOSGeometry* g = NULL;
  if (!ShapelyGetGeometry(obj, &g)) {
    return 0;
  }
  return g != NULL;
}

static char IsValidInput(PyObject* obj) {
  const GEOSGeometry* g;
  return ShapelyGetGeometry(obj, &g);
}

static char IsPrepared(PyObject* obj) {
  const GEOSGeometry* g;
  if (!ShapelyGetGeometry(obj, &g)) {
    return 2;
  }
  if (g == NULL) {
    return 0;  // Valid input (None), but not prepared
  }
  // Now we know that obj is a GeometryObject
  return ((GeometryObject*)obj)->ptr_prepared != NULL;
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

  // Release GIL for the loop (but don't init GEOS context)
  GEOS_INIT_THREADS;

  (void)ctx;   // dims warning;

  // The UNARY_LOOP macro unpacks args, dimensions, and steps and iterates through input/output arrays
  // ip1 points to current input element, op1 points to current output element
  UNARY_LOOP {
    CHECK_SIGNALS_THREADS(i);
    if (errstate == PGERR_PYSIGNAL) {
      break;
    }
    *(npy_bool*)op1 = func(*(PyObject**)ip1);
    if (*(npy_bool*)op1 == 2) {
      errstate = PGERR_NOT_A_GEOMETRY;
      break;
    }
  }

  // Reacquire GIL and handle errors
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
 * We use a macro to define a scalar Python function for each operation.
 *
 * This creates a function like PyIsMissing_Scalar that can be called from Python.
 * The generated function signature is:
 *   static PyObject* Py{func_name}_Scalar(PyObject* self, PyObject* obj)
 *
 * Example: DEFINE_O_b(IsMissing) creates PyIsMissing_Scalar function
 *
 * Parameters:
 *   func_name: Name of the C function that performs the operation
 */
#define DEFINE_O_b(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return PyBool_FromLong(func_name(obj)); \
  }

DEFINE_O_b(IsMissing);
DEFINE_O_b(IsGeometry);
DEFINE_O_b(IsValidInput);

// IsPrepared is different because it can return 2 (not a geometry)
static PyObject* PyIsPrepared_Scalar(PyObject* self, PyObject* obj) {
  char result = IsPrepared(obj);
  if (result == 2) {
        PyErr_SetString(PyExc_TypeError,
                      "One of the arguments is of incorrect type. Please provide only "
                      "Geometry objects.");
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

  return 0;
}
