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

static int GetX(void* context, void* a, double* b) {
  char typ = GEOSGeomTypeId_r(context, a);
  if (typ != 0) {
    *(double*)b = NPY_NAN;
    return 1;
  } else {
    return GEOSGeomGetX_r(context, a, b);
  }
}
static void* get_x_data[1] = {GetX};

typedef int FuncGEOS_Y_d(void* context, void* a, double* b);
static char Y_d_dtypes[2] = {NPY_OBJECT, NPY_DOUBLE};
static void Y_d_func(char** args, const npy_intp* dimensions, const npy_intp* steps, void* data) {
  FuncGEOS_Y_d* func = (FuncGEOS_Y_d*)data;
  GEOSGeometry* in1 = NULL;

  GEOS_INIT_THREADS;

  UNARY_LOOP {
    /* get the geometry: return on error */
    if (!get_geom(*(GeometryObject**)ip1, &in1)) {
      errstate = PGERR_NOT_A_GEOMETRY;
      goto finish;
    }
    if (in1 == NULL) {
      *(double*)op1 = NPY_NAN;
    } else {
      /* let the GEOS function set op1; return on error */
      if (func(ctx, in1, (npy_double*)op1) == 0) {
        errstate = PGERR_GEOS_EXCEPTION;
        goto finish;
      }
    }
  }

finish:
  GEOS_FINISH_THREADS;
}
static PyUFuncGenericFunction Y_d_funcs[1] = {&Y_d_func};


PyObject* PyGetX1(PyObject* self, PyObject* obj) {
  double result = NPY_NAN;

  // Set up arguments for Y_d_func to simulate scalar operation
  char* input_ptr = (char*)&obj;
  char* output_ptr = (char*)&result;
  char* args[2] = {input_ptr, output_ptr};

  // Single element operation
  npy_intp dimensions[1] = {1};
  npy_intp steps[2] = {sizeof(PyObject*), sizeof(double)};

  // Call Y_d_func with GetX function
  Y_d_func(args, dimensions, steps, get_x_data[0]);

  // Check if we have a Python exception set by Y_d_func
  if (PyErr_Occurred()) {
    return NULL;
  }

  return PyFloat_FromDouble(result);
}

PyObject* PyGetX2(PyObject* self, PyObject* obj) {
  FuncGEOS_Y_d* func = (FuncGEOS_Y_d*)get_x_data[0];
  GEOSGeometry* in1 = NULL;
  double result = NPY_NAN;

  GEOS_INIT_THREADS;

  /* get the geometry: return on error */
  if (!get_geom((GeometryObject*)obj, &in1)) {
    errstate = PGERR_NOT_A_GEOMETRY;
    goto finish;
  }
  if (in1 == NULL) {
    result = NPY_NAN;
  } else {
    /* let the GEOS function set result; return on error */
    if (func(ctx, in1, &result) == 0) {
      errstate = PGERR_GEOS_EXCEPTION;
      goto finish;
    }
  }

finish:
  GEOS_FINISH_THREADS;

  /* Return NULL on error, otherwise return the value */
  if (errstate != PGERR_SUCCESS) {
    return NULL;
  }

  return PyFloat_FromDouble(result);
}


static PyMethodDef GetXMethods[] = {
    {"get_x_1", PyGetX1, METH_O,
     ""},
    {"get_x_2", PyGetX2, METH_O,
     ""},
    {NULL, NULL, 0, NULL}};


int init_ufuncs_Y_d(PyObject* m, PyObject* d) {
  PyObject* ufunc;

  ufunc = PyUFunc_FromFuncAndData(Y_d_funcs, get_x_data, Y_d_dtypes, 1, 1, 1,
                                  PyUFunc_None, "get_x", "", 0);
  PyDict_SetItemString(d, "get_x_ufunc", ufunc);

  // Attach PyGetX1 and PyGetX2 to module (using METH_O for single object arg)
  PyObject* get_x1 = PyCFunction_NewEx(&GetXMethods[0], NULL, NULL);
  PyDict_SetItemString(d, "get_x_1", get_x1);

  PyObject* get_x2 = PyCFunction_NewEx(&GetXMethods[1], NULL, NULL);
  PyDict_SetItemString(d, "get_x_2", get_x2);
}
