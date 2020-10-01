#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>

#include "geos.h"
#include "pygeom.h"

/* A resizable vector with numpy indices */
typedef struct {
  size_t n, m;
  npy_intp* a;
} npy_intp_vec;

/* A resizable vector with pointers to pygeos GeometryObjects */
typedef struct {
  size_t n, m;
  GeometryObject** a;
} pg_geom_obj_vec;

typedef struct {
  PyObject_HEAD void* ptr;
  npy_intp count;
  pg_geom_obj_vec _geoms;
} STRtreeObject;

extern PyTypeObject STRtreeType;

extern int init_strtree_type(PyObject* m);

#endif
