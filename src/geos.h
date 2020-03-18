#ifndef _GEOS_H
#define _GEOS_H

#include <Python.h>

/* To avoid accidental use of non reentrant GEOS API. */
#define GEOS_USE_ONLY_R_API

// wrap geos.h import to silence geos gcc warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include <geos_c.h>
#pragma GCC diagnostic pop


#define RAISE_ILLEGAL_GEOS if (!PyErr_Occurred()) {PyErr_Format(PyExc_RuntimeError, "Uncaught GEOS exception");}
#define GEOS_SINCE_350 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 5))
#define GEOS_SINCE_360 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 6))
#define GEOS_SINCE_370 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 7))
#define GEOS_SINCE_380 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 8))
/* This declares a global GEOS Context */
extern void *geos_context[1];

extern int init_geos(PyObject *m);

#endif
