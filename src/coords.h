#ifndef _PYGEOSCOORDS_H
#define _PYGEOSCOORDS_H

#include <Python.h>
#include <geos_c.h>

extern PyObject *PyCountCoords(PyObject *self, PyObject *args);
extern PyObject *PyGetCoords(PyObject *self, PyObject *args);
extern PyObject *PySetCoords(PyObject *self, PyObject *args);

#endif
