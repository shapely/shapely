#ifndef _PYGEOM_H
#define _PYGEOM_H

#include <Python.h>

#include "geos.h"

typedef struct {
  PyObject_HEAD void* ptr;
} GeometryObject;

extern PyTypeObject GeometryType;

/* Initializes a new geometry object */
extern PyObject* GeometryObject_FromGEOS(GEOSGeometry* ptr, GEOSContextHandle_t ctx);
/* Get a GEOSGeometry from a GeometryObject */
extern char get_geom(GeometryObject* obj, GEOSGeometry** out);

extern int init_geom_type(PyObject* m);

#endif
