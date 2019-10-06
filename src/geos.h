#ifndef _GEOS_H
#define _GEOS_H

#include <Python.h>
#include <geos_c.h>

#define RAISE_ILLEGAL_GEOS if (!PyErr_Occurred()) {PyErr_Format(PyExc_RuntimeError, "Uncaught GEOS exception");}

/* This declares a global GEOS Context */
extern void *geos_context[1];

int init_geos(PyObject *m);

#endif
