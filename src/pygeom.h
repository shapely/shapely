#ifndef _PYGEOM_H
#define _PYGEOM_H

#include <Python.h>
#include "geos.h"


typedef struct {
    PyObject_HEAD
    void *ptr;
} GeometryObject;


GeometryObject *Geom_Empty;

PyTypeObject GeometryType;

/* Initializes a new geometry object */
PyObject *GeometryObject_FROMGEOS(PyTypeObject *type, GEOSGeometry *ptr);
PyObject *GeometryObject_FromGEOS(PyTypeObject *type, GEOSGeometry *ptr);

int init_geom_type(PyObject *m);

#endif
