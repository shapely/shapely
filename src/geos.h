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

/* Macros to setup GEOS Context and error handlers

Typical PyGEOS pattern in a function that uses GEOS:


// GEOS_INIT will do three things:
// 1. Make the GEOS context available in the variable ``ctx``
// 2. Initialize a variable ``errstate`` to PGERR_SUCCESS.
// 3. Set up GEOS error and warning buffers, respectively ``last_error`` and
``last_warning``

GEOS_INIT;   // or GEOS_INIT_THREADS if you use no CPython calls


// call a GEOS function using the context 'ctx'
result = SomeGEOSFunc(ctx, ...);

// handle an error state
if (result == NULL) { errstate = PGERR_GEOS_EXCEPTION; goto finish; }


// GEOS_FINISH will remove the GEOS context and set python errors in case
// errstate != PGERR_SUCCESS.

finish:
  GEOS_FINISH;  //  or GEOS_FINISH_THREADS if you use no CPython calls

*/

// Define the error states
enum {
  PGERR_SUCCESS,
  PGERR_NOT_A_GEOMETRY,
  PGERR_GEOS_EXCEPTION,
  PGERR_NO_MALLOC,
  PGERR_GEOMETRY_TYPE,
  PGERR_MULTIPOINT_WITH_POINT_EMPTY,
  PGERR_EMPTY_GEOMETRY
};

// Define how the states are handled by CPython
#define GEOS_HANDLE_ERR                                                                  \
  if (last_warning[0] != 0) {                                                            \
    PyErr_WarnEx(PyExc_Warning, last_warning, 0);                                        \
  }                                                                                      \
  switch (errstate) {                                                                    \
    case PGERR_SUCCESS:                                                                  \
      break;                                                                             \
    case PGERR_NOT_A_GEOMETRY:                                                           \
      PyErr_SetString(PyExc_TypeError,                                                   \
                      "One of the arguments is of incorrect type. Please provide only "  \
                      "Geometry objects.");                                              \
      break;                                                                             \
    case PGERR_GEOS_EXCEPTION:                                                           \
      PyErr_SetString(geos_exception[0], last_error);                                    \
      break;                                                                             \
    case PGERR_NO_MALLOC:                                                                \
      PyErr_SetString(PyExc_MemoryError, "Could not allocate memory");                   \
      break;                                                                             \
    case PGERR_GEOMETRY_TYPE:                                                            \
      PyErr_SetString(PyExc_TypeError,                                                   \
                      "One of the Geometry inputs is of incorrect geometry type.");      \
      break;                                                                             \
    case PGERR_MULTIPOINT_WITH_POINT_EMPTY:                                              \
      PyErr_SetString(PyExc_ValueError,                                                  \
                      "WKT output of multipoints with an empty point is unsupported on " \
                      "this version of GEOS.");                                          \
      break;                                                                             \
    case PGERR_EMPTY_GEOMETRY:                                                           \
      PyErr_SetString(PyExc_ValueError, "One of the Geometry inputs is empty.");         \
      break;                                                                             \
    default:                                                                             \
      PyErr_Format(PyExc_RuntimeError,                                                   \
                   "Pygeos ufunc returned with unknown error state code %d.", errstate); \
      break;                                                                             \
  }

// Define initialization / finalization macros
#define _GEOS_INIT_DEF           \
  char errstate = PGERR_SUCCESS; \
  char last_error[1024] = "";    \
  char last_warning[1024] = "";  \
  GEOSContextHandle_t ctx

#define _GEOS_INIT                                                           \
  ctx = GEOS_init_r();                                                       \
  GEOSContext_setErrorMessageHandler_r(ctx, geos_error_handler, last_error); \
  GEOSContext_setNoticeMessageHandler_r(ctx, geos_notice_handler, last_warning)

#define GEOS_INIT \
  _GEOS_INIT_DEF; \
  _GEOS_INIT

#define GEOS_INIT_THREADS \
  _GEOS_INIT_DEF;         \
  Py_BEGIN_ALLOW_THREADS _GEOS_INIT

#define GEOS_FINISH   \
  GEOS_finish_r(ctx); \
  GEOS_HANDLE_ERR

#define GEOS_FINISH_THREADS \
  GEOS_finish_r(ctx);       \
  Py_END_ALLOW_THREADS GEOS_HANDLE_ERR

#define GEOS_SINCE_3_5_0 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 5))
#define GEOS_SINCE_3_6_0 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 6))
#define GEOS_SINCE_3_7_0 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 7))
#define GEOS_SINCE_3_8_0 ((GEOS_VERSION_MAJOR >= 3) && (GEOS_VERSION_MINOR >= 8))

extern PyObject* geos_exception[1];

extern void geos_error_handler(const char* message, void* userdata);
extern void geos_notice_handler(const char* message, void* userdata);
extern void destroy_geom_arr(void* context, GEOSGeometry** array, int length);
extern char has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry* geom);
extern GEOSGeometry* point_empty_to_nan_all_geoms(GEOSContextHandle_t ctx,
                                                  GEOSGeometry* geom);
extern char check_to_wkt_compatible(GEOSContextHandle_t ctx, GEOSGeometry* geom);
extern char geos_interpolate_checker(GEOSContextHandle_t ctx, GEOSGeometry* geom);

extern int init_geos(PyObject* m);

#endif
