#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL shapely_ARRAY_API
#include <numpy/ndarraytypes.h>

#include "geos.h"
#include "geom_arr.h"
#include "pygeom.h"

/* Convert a geometry array to a NumPy array of Geometry objects */
void geom_arr_to_npy(GEOSGeometry** array, char* ptr, npy_intp stride,
                     npy_intp count) {
  npy_intp i;
  PyObject* ret;
  PyObject** out;

  GEOS_INIT;

  for (i = 0; i < count; i++, ptr += stride) {
    ret = GeometryObject_FromGEOS(array[i], ctx);
    out = (PyObject**)ptr;
    Py_XDECREF(*out);
    *out = ret;
  }

  GEOS_FINISH;
}

/* Destroy a geometry array */
void destroy_geom_arr(GEOSContextHandle_t context, GEOSGeometry** array, int length) {
  int i;
  for (i = 0; i < length; i++) {
    if (array[i] != NULL) {
      GEOSGeom_destroy_r(context, array[i]);
    }
  }
}
