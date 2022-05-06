#ifndef _UFUNCS_H
#define _UFUNCS_H

#include <Python.h>

extern int init_ufuncs(PyObject* m, PyObject* d);
extern PyObject* PySetInterruptInterval(PyObject* self, PyObject* args);
extern PyObject* PySetMainThreadId(PyObject* self, PyObject* args);

#endif
