/* geos_funcs_Y_d.c: ufuncs for GEOS functions with Geometry->double signature */

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
#include "ufuncs.h"


// The Y->d GOES function type definition
typedef int FuncGEOS_Y_d(void* context, void* a, double* b);


// Core function that is used by both ufunc and scalar implementations
static char core_Y_d_operation(GEOSContextHandle_t ctx, FuncGEOS_Y_d* func, GeometryObject* geom_obj, double* result) {
  GEOSGeometry* geom = NULL;

  // Extract geometry from GeometryObject
  if (!get_geom(geom_obj, &geom)) {
    return PGERR_NOT_A_GEOMETRY;
  }

  // Handle NULL geometry case
  if (geom == NULL) {
    *result = NPY_NAN;
    return PGERR_SUCCESS;
  }

  // Call the GEOS function
  if (func(ctx, geom, result) == 0) {
    return PGERR_GEOS_EXCEPTION;
  }

  return PGERR_SUCCESS;
}

// Python function for the (scalar) call to the GEOS function
static PyObject* Py_Y_d_Scalar(PyObject* self, PyObject* obj, FuncGEOS_Y_d* func) {
  double result = NPY_NAN;
  char errstate = PGERR_SUCCESS;
  char* last_error = geos_last_error;  // Pointer to the global last error message
  char* last_warning = geos_last_warning;  // Pointer to the global last warning message

  errstate = core_Y_d_operation(geos_context[0], func, (GeometryObject*)obj, &result);

  GEOS_HANDLE_ERR;

  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  return PyFloat_FromDouble(result);
}


// NumPy ufunc implementation
static void Y_d_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  FuncGEOS_Y_d* func = (FuncGEOS_Y_d*)data;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    errstate = core_Y_d_operation(ctx, func, *(GeometryObject**)ip1, (double*)op1);
    if (errstate != PGERR_SUCCESS) {
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;
}

// A pointer to the above function
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};

// The input and output types for the ufunc
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};

// Macro to define actual python function (scalar)
#define DEFINE_Y_d(func_name) \
  static PyObject* Py##func_name##_Scalar(PyObject* self, PyObject* obj) { \
    return Py_Y_d_Scalar(self, obj, (FuncGEOS_Y_d*)func_name); \
  }

// Macro to init both ufunc and scalar functions
#define INIT_Y_d(func_name, py_name) do { \
    static void* func_name##_FuncData[1] = {func_name}; \
    ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, func_name##_FuncData, Y_d_dtypes, 1, 1, 1, \
                                    PyUFunc_None, #py_name, "", 0); \
    PyDict_SetItemString(d, #py_name, ufunc); \
    static PyMethodDef Py##func_name##_Scalar_Def = {#py_name "_scalar", Py##func_name##_Scalar, METH_O, #py_name " scalar implementation"}; \
    PyObject* Py##func_name##_Scalar_Func = PyCFunction_NewEx(&Py##func_name##_Scalar_Def, NULL, NULL); \
    PyDict_SetItemString(d, #py_name "_scalar", Py##func_name##_Scalar_Func); \
} while(0)

// Now follow all the actual GEOS functions
static int GetX(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetX_r(context, a, b);
  }
}
DEFINE_Y_d(GetX);
static int GetY(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetY_r(context, a, b);
  }
}
DEFINE_Y_d(GetY);
static int GetZ(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetZ_r(context, a, b);
  }
}
DEFINE_Y_d(GetZ);
#if GEOS_SINCE_3_12_0
static int GetM(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetM_r(context, a, b);
  }
}
DEFINE_Y_d(GetM);
#endif
DEFINE_Y_d(GEOSArea_r);
DEFINE_Y_d(GEOSLength_r);

static int GetPrecision(void* context, void* a, double* b) {
  // GEOS returns -1 on error; 0 indicates double precision; > 0 indicates a precision
  // grid size was set for this geometry.
  double out = GEOSGeom_getPrecision_r(context, a);
  if (out == -1) {
    return 0;
  }
  *(double*)b = out;
  return 1;
}
DEFINE_Y_d(GetPrecision);
static int MinimumClearance(void* context, void* a, double* b) {
  // GEOSMinimumClearance deviates from the pattern of returning 0 on exception and 1 on
  // success for functions that return an int (it follows pattern for boolean functions
  // returning char 0/1 and 2 on exception)
  int retcode = GEOSMinimumClearance_r(context, a, b);
  if (retcode == 2) {
    return 0;
  } else {
    return 1;
  }
}
DEFINE_Y_d(MinimumClearance);
static int GEOSMinimumBoundingRadius(void* context, GEOSGeometry* geom, double* radius) {
  GEOSGeometry* center = NULL;
  GEOSGeometry* ret = GEOSMinimumBoundingCircle_r(context, geom, radius, &center);
  if (ret == NULL) {
    return 0;  // exception code
  }
  GEOSGeom_destroy_r(context, center);
  GEOSGeom_destroy_r(context, ret);
  return 1;  // success code
}
DEFINE_Y_d(GEOSMinimumBoundingRadius);


int init_geos_funcs_Y_d(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  INIT_Y_d(GetPrecision, get_precision);
  INIT_Y_d(GetX, get_x);
  INIT_Y_d(GetY, get_y);
  INIT_Y_d(GetZ, get_z);
  INIT_Y_d(GEOSArea_r, area);
  INIT_Y_d(GEOSLength_r, length);
  INIT_Y_d(MinimumClearance, minimum_clearance);
  INIT_Y_d(GEOSMinimumBoundingRadius, minimum_bounding_radius);

  #if GEOS_SINCE_3_12_0
  INIT_Y_d(GetM, get_m);
  #endif

  return 0;
}
