#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "pygeos.h"  // for PGERR_PYSIGNAL
#include "signal_checks.h"

/* Global values for interrupt checking */
int check_signals_interval[1] = {10000};
unsigned long main_thread_id[1] = {0};

PyObject* PySetupSignalChecks(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, "ik", check_signals_interval, main_thread_id)) {
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}
