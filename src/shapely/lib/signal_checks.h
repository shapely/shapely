#ifndef _SIGNAL_CHECKS_H
#define _SIGNAL_CHECKS_H

#include <Python.h>

/* Global variables for interrupt checking */
extern int check_signals_interval[1];
extern unsigned long main_thread_id[1];

/* Macro for checking signals and threads in loops
 *
 * This macro checks for Python signals at regular intervals during loops.
 * It should be called inside loops that may run for a long time to ensure
 * that KeyboardInterrupt and other signals are handled properly.
 *
 * The macro checks if the current iteration (I) is a multiple of the
 * check_signals_interval. If so, and if we're on the main thread, it
 * temporarily reacquires the GIL to check for pending signals.
 *
 * Note: errstate must be defined in the calling scope.
 * The macro will set errstate to PGERR_PYSIGNAL
 * if a signal is detected.
 */
#define CHECK_SIGNALS_THREADS(I)                            \
  if (((I + 1) % check_signals_interval[0]) == 0) {         \
    if (PyThread_get_thread_ident() == main_thread_id[0]) { \
      Py_BLOCK_THREADS;                                     \
      if (PyErr_CheckSignals() == -1) {                     \
        errstate = PGERR_PYSIGNAL;                          \
      }                                                     \
      Py_UNBLOCK_THREADS;                                   \
    }                                                       \
  }

/* Function to setup signal checking parameters */
PyObject* PySetupSignalChecks(PyObject* self, PyObject* args);

#endif /* _SIGNAL_CHECKS_H */
