#ifndef _RTREE_H
#define _RTREE_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
    PyObject *geometries;
    long count;
} STRtreeObject;

/* A resizable vector with numpy indices */
typedef struct
{
    size_t n, m;
    npy_intp *a;
} npy_intp_vec;

PyTypeObject STRtreeType;

int init_strtree_type(PyObject *m);

#endif
