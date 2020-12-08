#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>

#include "geos.h"
#include "pygeom.h"
#include "vector.h"


typedef struct {
  PyObject_HEAD void* ptr;
  npy_intp count;
  geom_obj_vec_t _geoms;
} STRtreeObject;

extern PyTypeObject STRtreeType;

extern int init_strtree_type(PyObject* m);

#endif
