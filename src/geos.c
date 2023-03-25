#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "geos.h"

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/npy_math.h>
#include <structmember.h>

/* This initializes a globally accessible GEOS Context, only to be used when holding the
 * GIL */
void* geos_context[1] = {NULL};

/* This initializes a globally accessible GEOSException object */
PyObject* geos_exception[1] = {NULL};

int init_geos(PyObject* m) {
  PyObject* base_class = PyErr_NewException("shapely.errors.ShapelyError", NULL, NULL);
  PyModule_AddObject(m, "ShapelyError", base_class);
  geos_exception[0] =
      PyErr_NewException("shapely.errors.GEOSException", base_class, NULL);
  PyModule_AddObject(m, "GEOSException", geos_exception[0]);

  void* context_handle = GEOS_init_r();
  // TODO: the error handling is not yet set up for the global context (it is right now
  // only used where error handling is not used)
  // GEOSContext_setErrorMessageHandler_r(context_handle, geos_error_handler, last_error);
  geos_context[0] = context_handle;

  return 0;
}

void destroy_geom_arr(void* context, GEOSGeometry** array, int length) {
  int i;
  for (i = 0; i < length; i++) {
    if (array[i] != NULL) {
      GEOSGeom_destroy_r(context, array[i]);
    }
  }
}

/* These functions are used to workaround two GEOS issues (in WKB writer for
 * GEOS < 3.9, in WKT writer for GEOS < 3.9 and in GeoJSON writer for GEOS 3.10.0):
 * - POINT EMPTY was not handled correctly (we do it ourselves)
 * - MULTIPOINT (EMPTY) resulted in segfault (we check for it and raise)
 */

/* Returns 1 if a multipoint has an empty point, 0 otherwise, 2 on error.
 */
char multipoint_has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int n, i;
  char is_empty;
  const GEOSGeometry* sub_geom;

  n = GEOSGetNumGeometries_r(ctx, geom);
  if (n == -1) {
    return 2;
  }
  for (i = 0; i < n; i++) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
    if (sub_geom == NULL) {
      return 2;
    }
    is_empty = GEOSisEmpty_r(ctx, sub_geom);
    if (is_empty != 0) {
      // If empty encountered, or on exception, return:
      return is_empty;
    }
  }
  return 0;
}

/* Returns 1 if geometry is an empty point, 0 otherwise, 2 on error.
 */
char is_point_empty(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int geom_type;

  geom_type = GEOSGeomTypeId_r(ctx, geom);
  if (geom_type == GEOS_POINT) {
    return GEOSisEmpty_r(ctx, geom);
  } else if (geom_type == -1) {
    return 2;  // GEOS exception
  } else {
    return 0;  // No empty point
  }
}

/* Returns 1 if a geometrycollection has an empty point, 0 otherwise, 2 on error.
Checks recursively (geometrycollections may contain multipoints / geometrycollections)
*/
char geometrycollection_has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int n, i;
  char has_empty;
  const GEOSGeometry* sub_geom;

  n = GEOSGetNumGeometries_r(ctx, geom);
  if (n == -1) {
    return 2;
  }
  for (i = 0; i < n; i++) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
    if (sub_geom == NULL) {
      return 2;
    }
    has_empty = has_point_empty(ctx, (GEOSGeometry*)sub_geom);
    if (has_empty != 0) {
      // If empty encountered, or on exception, return:
      return has_empty;
    }
  }
  return 0;
}

/* Returns 1 if geometry is / has an empty point, 0 otherwise, 2 on error.
 */
char has_point_empty(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int geom_type;

  geom_type = GEOSGeomTypeId_r(ctx, geom);
  if (geom_type == GEOS_POINT) {
    return GEOSisEmpty_r(ctx, geom);
  } else if (geom_type == GEOS_MULTIPOINT) {
    return multipoint_has_point_empty(ctx, geom);
  } else if (geom_type == GEOS_GEOMETRYCOLLECTION) {
    return geometrycollection_has_point_empty(ctx, geom);
  } else if (geom_type == -1) {
    return 2;  // GEOS exception
  } else {
    return 0;  // No empty point
  }
}

/* Creates a POINT (nan, nan[, nan)] from a POINT EMPTY template

   Returns NULL on error
*/
GEOSGeometry* point_empty_to_nan(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int j, ndim;
  GEOSCoordSequence* coord_seq;
  GEOSGeometry* result;

  ndim = GEOSGeom_getCoordinateDimension_r(ctx, geom);
  if (ndim == 0) {
    return NULL;
  }

  coord_seq = GEOSCoordSeq_create_r(ctx, 1, ndim);
  if (coord_seq == NULL) {
    return NULL;
  }
  for (j = 0; j < ndim; j++) {
    if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, 0, j, Py_NAN)) {
      GEOSCoordSeq_destroy_r(ctx, coord_seq);
      return NULL;
    }
  }
  result = GEOSGeom_createPoint_r(ctx, coord_seq);
  if (result == NULL) {
    GEOSCoordSeq_destroy_r(ctx, coord_seq);
    return NULL;
  }
  GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
  return result;
}

/* Creates a new multipoint, replacing empty points with POINT (nan, nan[, nan)]

   Returns NULL on error
*/
GEOSGeometry* multipoint_empty_to_nan(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int n, i;
  GEOSGeometry* result;
  const GEOSGeometry* sub_geom;

  n = GEOSGetNumGeometries_r(ctx, geom);
  if (n == -1) {
    return NULL;
  }

  GEOSGeometry** geoms = malloc(sizeof(void*) * n);
  for (i = 0; i < n; i++) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
    if (GEOSisEmpty_r(ctx, sub_geom)) {
      geoms[i] = point_empty_to_nan(ctx, (GEOSGeometry*)sub_geom);
    } else {
      geoms[i] = GEOSGeom_clone_r(ctx, (GEOSGeometry*)sub_geom);
    }
    // If the function errored: cleanup and return
    if (geoms[i] == NULL) {
      destroy_geom_arr(ctx, geoms, i);
      free(geoms);
      return NULL;
    }
  }

  result = GEOSGeom_createCollection_r(ctx, GEOS_MULTIPOINT, geoms, n);
  // If the function errored: cleanup and return
  if (result == NULL) {
    destroy_geom_arr(ctx, geoms, i);
    free(geoms);
    return NULL;
  }

  free(geoms);
  GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
  return result;
}

/* Creates a new geometrycollection, replacing all empty points with POINT (nan, nan[,
   nan)]

   Returns NULL on error
*/
GEOSGeometry* geometrycollection_empty_to_nan(GEOSContextHandle_t ctx,
                                              GEOSGeometry* geom) {
  int n, i;
  GEOSGeometry* result = NULL;
  const GEOSGeometry* sub_geom;

  n = GEOSGetNumGeometries_r(ctx, geom);
  if (n == -1) {
    return NULL;
  }

  GEOSGeometry** geoms = malloc(sizeof(void*) * n);
  for (i = 0; i < n; i++) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
    geoms[i] = point_empty_to_nan_all_geoms(ctx, (GEOSGeometry*)sub_geom);
    // If the function errored: cleanup and return
    if (geoms[i] == NULL) {
      goto finish;
    }
  }

  result = GEOSGeom_createCollection_r(ctx, GEOS_GEOMETRYCOLLECTION, geoms, n);

finish:

  // If the function errored: cleanup, else set SRID
  if (result == NULL) {
    destroy_geom_arr(ctx, geoms, i);
  } else {
    GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
  }
  free(geoms);
  return result;
}

/* Creates a new geometry, replacing empty points with POINT (nan, nan[, nan)]

   Returns NULL on error.
*/
GEOSGeometry* point_empty_to_nan_all_geoms(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  int geom_type;
  GEOSGeometry* result;

  geom_type = GEOSGeomTypeId_r(ctx, geom);
  if (geom_type == -1) {
    result = NULL;
  } else if (is_point_empty(ctx, geom)) {
    result = point_empty_to_nan(ctx, geom);
  } else if (geom_type == GEOS_MULTIPOINT) {
    result = multipoint_empty_to_nan(ctx, geom);
  } else if (geom_type == GEOS_GEOMETRYCOLLECTION) {
    result = geometrycollection_empty_to_nan(ctx, geom);
  } else {
    result = GEOSGeom_clone_r(ctx, geom);
  }

  GEOSSetSRID_r(ctx, result, GEOSGetSRID_r(ctx, geom));
  return result;
}

/* Checks whether the geometry is a multipoint with an empty point in it
 *
 * According to https://github.com/libgeos/geos/issues/305, this check is not
 * necessary for GEOS 3.7.3, 3.8.2, or 3.9. When these versions are out, we
 * should add version conditionals and test.
 *
 * The return value is one of:
 * - PGERR_SUCCESS
 * - PGERR_MULTIPOINT_WITH_POINT_EMPTY
 * - PGERR_GEOS_EXCEPTION
 */
char check_to_wkt_compatible(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  char geom_type, is_empty;

  geom_type = GEOSGeomTypeId_r(ctx, geom);
  if (geom_type == -1) {
    return PGERR_GEOS_EXCEPTION;
  }
  if (geom_type != GEOS_MULTIPOINT) {
    return PGERR_SUCCESS;
  }

  is_empty = multipoint_has_point_empty(ctx, geom);
  if (is_empty == 0) {
    return PGERR_SUCCESS;
  } else if (is_empty == 1) {
    return PGERR_MULTIPOINT_WITH_POINT_EMPTY;
  } else {
    return PGERR_GEOS_EXCEPTION;
  }
}

#if GEOS_SINCE_3_9_0

/* Checks whether the geometry is a 3D empty geometry and, if so, create the WKT string
 *
 * GEOS 3.9.* is able to distiguish 2D and 3D simple geometries (non-collections). But the
 * but the WKT serialization never writes a 3D empty geometry. This function fixes that.
 * It only makes sense to use this for GEOS versions >= 3.9.
 *
 * Pending GEOS ticket: https://trac.osgeo.org/geos/ticket/1129
 *
 * If the geometry is a 3D empty, then the (char**) wkt argument is filled with the
 * correct WKT string. Else, wkt becomes NULL and the GEOS WKT writer should be used.
 *
 * The return value is one of:
 * - PGERR_SUCCESS
 * - PGERR_GEOS_EXCEPTION
 */
char wkt_empty_3d_geometry(GEOSContextHandle_t ctx, GEOSGeometry* geom, char** wkt) {
  char is_empty;
  int geom_type;

  is_empty = GEOSisEmpty_r(ctx, geom);
  if (is_empty == 2) {
    return PGERR_GEOS_EXCEPTION;
  } else if (is_empty == 0) {
    *wkt = NULL;
    return PGERR_SUCCESS;
  }
  if (GEOSGeom_getCoordinateDimension_r(ctx, geom) == 2) {
    *wkt = NULL;
    return PGERR_SUCCESS;
  }
  geom_type = GEOSGeomTypeId_r(ctx, geom);
  switch (geom_type) {
    case GEOS_POINT:
      *wkt = "POINT Z EMPTY";
      return PGERR_SUCCESS;
    case GEOS_LINESTRING:
      *wkt = "LINESTRING Z EMPTY";
      break;
    case GEOS_LINEARRING:
      *wkt = "LINEARRING Z EMPTY";
      break;
    case GEOS_POLYGON:
      *wkt = "POLYGON Z EMPTY";
      break;
    // Note: Empty collections cannot be 3D in GEOS.
    // We do include the options in case of future support.
    case GEOS_MULTIPOINT:
      *wkt = "MULTIPOINT Z EMPTY";
      break;
    case GEOS_MULTILINESTRING:
      *wkt = "MULTILINESTRING Z EMPTY";
      break;
    case GEOS_MULTIPOLYGON:
      *wkt = "MULTIPOLYGON Z EMPTY";
      break;
    case GEOS_GEOMETRYCOLLECTION:
      *wkt = "GEOMETRYCOLLECTION Z EMPTY";
      break;
    default:
      return PGERR_GEOS_EXCEPTION;
  }
  return PGERR_SUCCESS;
}

#endif  // GEOS_SINCE_3_9_0

/* GEOSInterpolate_r and GEOSInterpolateNormalized_r segfault on empty
 * geometries and also on collections with the first geometry empty.
 *
 * This function returns:
 * - PGERR_GEOMETRY_TYPE on non-linear geometries
 * - PGERR_EMPTY_GEOMETRY on empty linear geometries
 * - PGERR_EXCEPTIONS on GEOS exceptions
 * - PGERR_SUCCESS on a non-empty and linear geometry
 *
 * Note that GEOS 3.8 fixed this situation for empty LINESTRING/LINEARRING,
 * but it still segfaults on other empty geometries.
 */
char geos_interpolate_checker(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  char type;
  char is_empty;
  const GEOSGeometry* sub_geom;

  // Check if the geometry is linear
  type = GEOSGeomTypeId_r(ctx, geom);
  if (type == -1) {
    return PGERR_GEOS_EXCEPTION;
  } else if ((type == GEOS_POINT) || (type == GEOS_POLYGON) ||
             (type == GEOS_MULTIPOINT) || (type == GEOS_MULTIPOLYGON)) {
    return PGERR_GEOMETRY_TYPE;
  }

  // Check if the geometry is empty
  is_empty = GEOSisEmpty_r(ctx, geom);
  if (is_empty == 1) {
    return PGERR_EMPTY_GEOMETRY;
  } else if (is_empty == 2) {
    return PGERR_GEOS_EXCEPTION;
  }

  // For collections: also check the type and emptyness of the first geometry
  if ((type == GEOS_MULTILINESTRING) || (type == GEOS_GEOMETRYCOLLECTION)) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, 0);
    if (sub_geom == NULL) {
      return PGERR_GEOS_EXCEPTION;  // GEOSException
    }
    type = GEOSGeomTypeId_r(ctx, sub_geom);
    if (type == -1) {
      return PGERR_GEOS_EXCEPTION;
    } else if ((type != GEOS_LINESTRING) && (type != GEOS_LINEARRING)) {
      return PGERR_GEOMETRY_TYPE;
    }
    is_empty = GEOSisEmpty_r(ctx, sub_geom);
    if (is_empty == 1) {
      return PGERR_EMPTY_GEOMETRY;
    } else if (is_empty == 2) {
      return PGERR_GEOS_EXCEPTION;
    }
  }
  return PGERR_SUCCESS;
}

/* Define the GEOS error handler. See GEOS_INIT / GEOS_FINISH macros in geos.h*/
void geos_error_handler(const char* message, void* userdata) {
  snprintf(userdata, 1024, "%s", message);
}

/* Extract bounds from geometry.
 *
 * Bounds coordinates will be set to NPY_NAN if geom is NULL, empty, or does not have an
 * envelope.
 *
 * Parameters
 * ----------
 * ctx: GEOS context handle
 * geom: pointer to GEOSGeometry; can be NULL
 * xmin: pointer to xmin value
 * ymin: pointer to ymin value
 * xmax: pointer to xmax value
 * ymax: pointer to ymax value
 *
 * Must be called from within a GEOS_INIT_THREADS / GEOS_FINISH_THREADS
 * or GEOS_INIT / GEOS_FINISH block.
 *
 * Returns
 * -------
 * 1 on success; 0 on error
 */
int get_bounds(GEOSContextHandle_t ctx, GEOSGeometry* geom, double* xmin, double* ymin,
               double* xmax, double* ymax) {
  int retval = 1;

  if (geom == NULL || GEOSisEmpty_r(ctx, geom)) {
    *xmin = *ymin = *xmax = *ymax = NPY_NAN;
    return 1;
  }

#if GEOS_SINCE_3_7_0
  // use min / max coordinates

  if (!(GEOSGeom_getXMin_r(ctx, geom, xmin) && GEOSGeom_getYMin_r(ctx, geom, ymin) &&
        GEOSGeom_getXMax_r(ctx, geom, xmax) && GEOSGeom_getYMax_r(ctx, geom, ymax))) {
    return 0;
  }

#else
  // extract coordinates from envelope

  GEOSGeometry* envelope = NULL;
  const GEOSGeometry* ring = NULL;
  const GEOSCoordSequence* coord_seq = NULL;
  int size;

  /* construct the envelope */
  envelope = GEOSEnvelope_r(ctx, geom);
  if (envelope == NULL) {
    return 0;
  }

  size = GEOSGetNumCoordinates_r(ctx, envelope);

  /* get the bbox depending on the number of coordinates in the envelope */
  if (size == 0) { /* Envelope is empty */
    *xmin = *ymin = *xmax = *ymax = NPY_NAN;
  } else if (size == 1) { /* Envelope is a point */
    if (!GEOSGeomGetX_r(ctx, envelope, xmin)) {
      retval = 0;
      goto finish;
    }
    if (!GEOSGeomGetY_r(ctx, envelope, ymin)) {
      retval = 0;
      goto finish;
    }
    *xmax = *xmin;
    *ymax = *ymin;
  } else if (size == 5) { /* Envelope is a box */
    ring = GEOSGetExteriorRing_r(ctx, envelope);
    if (ring == NULL) {
      retval = 0;
      goto finish;
    }
    coord_seq = GEOSGeom_getCoordSeq_r(ctx, ring);
    if (coord_seq == NULL) {
      retval = 0;
      goto finish;
    }
    if (!GEOSCoordSeq_getX_r(ctx, coord_seq, 0, xmin)) {
      retval = 0;
      goto finish;
    }
    if (!GEOSCoordSeq_getY_r(ctx, coord_seq, 0, ymin)) {
      retval = 0;
      goto finish;
    }
    if (!GEOSCoordSeq_getX_r(ctx, coord_seq, 2, xmax)) {
      retval = 0;
      goto finish;
    }
    if (!GEOSCoordSeq_getY_r(ctx, coord_seq, 2, ymax)) {
      retval = 0;
      goto finish;
    }
  }

finish:
  if (envelope != NULL) {
    GEOSGeom_destroy_r(ctx, envelope);
  }

#endif

  return retval;
}

/* Create a Polygon from bounding coordinates.
 *
 * Must be called from within a GEOS_INIT_THREADS / GEOS_FINISH_THREADS
 * or GEOS_INIT / GEOS_FINISH block.
 *
 * Parameters
 * ----------
 * ctx: GEOS context handle
 * xmin: minimum X value
 * ymin: minimum Y value
 * xmax: maximum X value
 * ymax: maximum Y value
 * ccw: if 1, box will be created in counterclockwise direction from bottom right;
 *  otherwise will be created in clockwise direction from bottom left.
 *
 * Returns
 * -------
 * GEOSGeometry* on success (owned by caller) or
 * NULL on failure or NPY_NAN coordinates
 */
GEOSGeometry* create_box(GEOSContextHandle_t ctx, double xmin, double ymin, double xmax,
                         double ymax, char ccw) {
  if (npy_isnan(xmin) || npy_isnan(ymin) || npy_isnan(xmax) || npy_isnan(ymax)) {
    return NULL;
  }

  GEOSCoordSequence* coords = NULL;
  GEOSGeometry* geom = NULL;
  GEOSGeometry* ring = NULL;

  // Construct coordinate sequence and set vertices
  coords = GEOSCoordSeq_create_r(ctx, 5, 2);
  if (coords == NULL) {
    return NULL;
  }

  if (ccw) {
    // Start from bottom right (xmax, ymin) to match shapely
    if (!(GEOSCoordSeq_setX_r(ctx, coords, 0, xmax) &&
          GEOSCoordSeq_setY_r(ctx, coords, 0, ymin) &&
          GEOSCoordSeq_setX_r(ctx, coords, 1, xmax) &&
          GEOSCoordSeq_setY_r(ctx, coords, 1, ymax) &&
          GEOSCoordSeq_setX_r(ctx, coords, 2, xmin) &&
          GEOSCoordSeq_setY_r(ctx, coords, 2, ymax) &&
          GEOSCoordSeq_setX_r(ctx, coords, 3, xmin) &&
          GEOSCoordSeq_setY_r(ctx, coords, 3, ymin) &&
          GEOSCoordSeq_setX_r(ctx, coords, 4, xmax) &&
          GEOSCoordSeq_setY_r(ctx, coords, 4, ymin))) {
      if (coords != NULL) {
        GEOSCoordSeq_destroy_r(ctx, coords);
      }

      return NULL;
    }
  } else {
    // Start from bottom left (min, ymin) to match shapely
    if (!(GEOSCoordSeq_setX_r(ctx, coords, 0, xmin) &&
          GEOSCoordSeq_setY_r(ctx, coords, 0, ymin) &&
          GEOSCoordSeq_setX_r(ctx, coords, 1, xmin) &&
          GEOSCoordSeq_setY_r(ctx, coords, 1, ymax) &&
          GEOSCoordSeq_setX_r(ctx, coords, 2, xmax) &&
          GEOSCoordSeq_setY_r(ctx, coords, 2, ymax) &&
          GEOSCoordSeq_setX_r(ctx, coords, 3, xmax) &&
          GEOSCoordSeq_setY_r(ctx, coords, 3, ymin) &&
          GEOSCoordSeq_setX_r(ctx, coords, 4, xmin) &&
          GEOSCoordSeq_setY_r(ctx, coords, 4, ymin))) {
      if (coords != NULL) {
        GEOSCoordSeq_destroy_r(ctx, coords);
      }

      return NULL;
    }
  }

  // Construct linear ring then use to construct polygon
  // Note: coords are owned by ring; if ring fails to construct, it will
  // automatically clean up the coords
  ring = GEOSGeom_createLinearRing_r(ctx, coords);
  if (ring == NULL) {
    return NULL;
  }

  // Note: ring is owned by polygon; if polygon fails to construct, it will
  // automatically clean up the ring
  geom = GEOSGeom_createPolygon_r(ctx, ring, NULL, 0);
  if (geom == NULL) {
    return NULL;
  }

  return geom;
}

/* Create a Point from x and y coordinates.
 *
 * Must be called from within a GEOS_INIT_THREADS / GEOS_FINISH_THREADS
 * or GEOS_INIT / GEOS_FINISH block.
 *
 * Helper function for quickly creating a Point for older GEOS versions.
 *
 * Parameters
 * ----------
 * ctx: GEOS context handle
 * x: X value
 * y: Y value
 *
 * Returns
 * -------
 * GEOSGeometry* on success (owned by caller) or NULL on failure
 */
GEOSGeometry* create_point(GEOSContextHandle_t ctx, double x, double y) {
#if GEOS_SINCE_3_8_0
  return GEOSGeom_createPointFromXY_r(ctx, x, y);
#else
  GEOSCoordSequence* coord_seq = NULL;
  GEOSGeometry* geom = NULL;

  coord_seq = GEOSCoordSeq_create_r(ctx, 1, 2);
  if (coord_seq == NULL) {
    return NULL;
  }
  if (!GEOSCoordSeq_setX_r(ctx, coord_seq, 0, x)) {
    GEOSCoordSeq_destroy_r(ctx, coord_seq);
    return NULL;
  }
  if (!GEOSCoordSeq_setY_r(ctx, coord_seq, 0, y)) {
    GEOSCoordSeq_destroy_r(ctx, coord_seq);
    return NULL;
  }
  geom = GEOSGeom_createPoint_r(ctx, coord_seq);
  if (geom == NULL) {
    GEOSCoordSeq_destroy_r(ctx, coord_seq);
    return NULL;
  }
  return geom;
#endif
}

/* Create a 3D empty Point
 *
 * Works around a limitation of the GEOS C API by constructing the point
 * from its WKT representation (POINT Z EMPTY).
 *
 * Returns
 * -------
 * GEOSGeometry* on success (owned by caller) or NULL on failure
 */
GEOSGeometry* PyGEOS_create3DEmptyPoint(GEOSContextHandle_t ctx) {
  const char* wkt = "POINT Z EMPTY";
  GEOSGeometry* geom;
  GEOSWKTReader* reader;

  reader = GEOSWKTReader_create_r(ctx);
  if (reader == NULL) {
    return NULL;
  }
  geom = GEOSWKTReader_read_r(ctx, reader, wkt);
  GEOSWKTReader_destroy_r(ctx, reader);
  return geom;
}

/* Force the coordinate dimensionality (2D / 3D) of any geometry
 *
 * Parameters
 * ----------
 * ctx: GEOS context handle
 * geom: geometry
 * dims: dimensions to force (2 or 3)
 * z: Z coordinate (ignored if dims==2)
 *
 * Returns
 * -------
 * GEOSGeometry* on success (owned by caller) or NULL on failure
 */
GEOSGeometry* force_dims(GEOSContextHandle_t, GEOSGeometry*, unsigned int, double);

GEOSGeometry* force_dims_simple(GEOSContextHandle_t ctx, GEOSGeometry* geom, int type,
                                unsigned int dims, double z) {
  unsigned int actual_dims, n, i, j;
  double coord;
  const GEOSCoordSequence* seq = GEOSGeom_getCoordSeq_r(ctx, geom);

  /* Special case for POINT EMPTY (Point coordinate list cannot be 0-length) */
  if ((type == 0) && (GEOSisEmpty_r(ctx, geom) == 1)) {
    if (dims == 2) {
      return GEOSGeom_createEmptyPoint_r(ctx);
    } else if (dims == 3) {
      return PyGEOS_create3DEmptyPoint(ctx);
    } else {
      return NULL;
    }
  }

  /* Investigate the coordinate sequence, return when already of correct dimensionality */
  if (GEOSCoordSeq_getDimensions_r(ctx, seq, &actual_dims) == 0) {
    return NULL;
  }
  if (actual_dims == dims) {
    return GEOSGeom_clone_r(ctx, geom);
  }
  if (GEOSCoordSeq_getSize_r(ctx, seq, &n) == 0) {
    return NULL;
  }

  /* Create a new one to fill with the new coordinates */
  GEOSCoordSequence* seq_new = GEOSCoordSeq_create_r(ctx, n, dims);
  if (seq_new == NULL) {
    return NULL;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < 2; j++) {
      if (!GEOSCoordSeq_getOrdinate_r(ctx, seq, i, j, &coord)) {
        GEOSCoordSeq_destroy_r(ctx, seq_new);
        return NULL;
      }
      if (!GEOSCoordSeq_setOrdinate_r(ctx, seq_new, i, j, coord)) {
        GEOSCoordSeq_destroy_r(ctx, seq_new);
        return NULL;
      }
    }
    if (dims == 3) {
      if (!GEOSCoordSeq_setZ_r(ctx, seq_new, i, z)) {
        GEOSCoordSeq_destroy_r(ctx, seq_new);
        return NULL;
      }
    }
  }

  /* Construct a new geometry */
  if (type == 0) {
    return GEOSGeom_createPoint_r(ctx, seq_new);
  } else if (type == 1) {
    return GEOSGeom_createLineString_r(ctx, seq_new);
  } else if (type == 2) {
    return GEOSGeom_createLinearRing_r(ctx, seq_new);
  } else {
    return NULL;
  }
}

GEOSGeometry* force_dims_polygon(GEOSContextHandle_t ctx, GEOSGeometry* geom,
                                 unsigned int dims, double z) {
  int i, n;
  const GEOSGeometry *shell, *hole;
  GEOSGeometry *new_shell, *new_hole, *result = NULL;
  GEOSGeometry** new_holes;

  n = GEOSGetNumInteriorRings_r(ctx, geom);
  if (n == -1) {
    return NULL;
  }

  /* create the exterior ring */
  shell = GEOSGetExteriorRing_r(ctx, geom);
  if (shell == NULL) {
    return NULL;
  }
  new_shell = force_dims_simple(ctx, (GEOSGeometry*)shell, 2, dims, z);
  if (new_shell == NULL) {
    return NULL;
  }

  new_holes = malloc(sizeof(void*) * n);
  if (new_holes == NULL) {
    GEOSGeom_destroy_r(ctx, new_shell);
    return NULL;
  }

  for (i = 0; i < n; i++) {
    hole = GEOSGetInteriorRingN_r(ctx, geom, i);
    if (hole == NULL) {
      GEOSGeom_destroy_r(ctx, new_shell);
      destroy_geom_arr(ctx, new_holes, i - 1);
      goto finish;
    }
    new_hole = force_dims_simple(ctx, (GEOSGeometry*)hole, 2, dims, z);
    if (hole == NULL) {
      GEOSGeom_destroy_r(ctx, new_shell);
      destroy_geom_arr(ctx, new_holes, i - 1);
      goto finish;
    }
    new_holes[i] = new_hole;
  }

  result = GEOSGeom_createPolygon_r(ctx, new_shell, new_holes, n);

finish:
  if (new_holes != NULL) {
    free(new_holes);
  }
  return result;
}

GEOSGeometry* force_dims_collection(GEOSContextHandle_t ctx, GEOSGeometry* geom, int type,
                                    unsigned int dims, double z) {
  int i, n;
  const GEOSGeometry* sub_geom;
  GEOSGeometry *new_sub_geom, *result = NULL;
  GEOSGeometry** geoms;

  n = GEOSGetNumGeometries_r(ctx, geom);
  if (n == -1) {
    return NULL;
  }

  geoms = malloc(sizeof(void*) * n);
  if (geoms == NULL) {
    return NULL;
  }

  for (i = 0; i < n; i++) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, i);
    if (sub_geom == NULL) {
      destroy_geom_arr(ctx, geoms, i - i);
      goto finish;
    }
    new_sub_geom = force_dims(ctx, (GEOSGeometry*)sub_geom, dims, z);
    if (new_sub_geom == NULL) {
      destroy_geom_arr(ctx, geoms, i - i);
      goto finish;
    }
    geoms[i] = new_sub_geom;
  }

  result = GEOSGeom_createCollection_r(ctx, type, geoms, n);
finish:
  if (geoms != NULL) {
    free(geoms);
  }
  return result;
}

GEOSGeometry* force_dims(GEOSContextHandle_t ctx, GEOSGeometry* geom, unsigned int dims,
                         double z) {
  int type = GEOSGeomTypeId_r(ctx, geom);
  if ((type == 0) || (type == 1) || (type == 2)) {
    return force_dims_simple(ctx, geom, type, dims, z);
  } else if (type == 3) {
    return force_dims_polygon(ctx, geom, dims, z);
  } else if ((type >= 4) && (type <= 7)) {
    return force_dims_collection(ctx, geom, type, dims, z);
  } else {
    return NULL;
  }
}

GEOSGeometry* PyGEOSForce2D(GEOSContextHandle_t ctx, GEOSGeometry* geom) {
  return force_dims(ctx, geom, 2, 0.0);
}

GEOSGeometry* PyGEOSForce3D(GEOSContextHandle_t ctx, GEOSGeometry* geom, double z) {
  return force_dims(ctx, geom, 3, z);
}

/* Count the number of finite coordinates in a buffer
 *
 * A coordinate is finite if x, y and optionally z are all not NaN or Inf.
 */
unsigned int count_finite(const double* buf, unsigned int size, unsigned int dims,
                          npy_intp cs1, npy_intp cs2) {
  char *cp1, *cp2;
  unsigned int result = 0;
  char has_non_finite = 0;
  unsigned int i, j;

  cp1 = (char*)buf;
  for (i = 0; i < size; i++, cp1 += cs1) {
    cp2 = cp1;
    has_non_finite = 0;
    for (j = 0; j < dims; j++, cp2 += cs2) {
      if (!(npy_isfinite(*(double*)cp2))) {
        has_non_finite = 1;
        break;
      }
    }
    if (!has_non_finite) {
      result++;
    }
  }
  return result;
}

/* Compare the first and last coordinates in the buffer, skipping nonfinite coordinates.
 */
char ending_coordinates_equal(const double* buf, unsigned int size, unsigned int dims,
                              npy_intp cs1, npy_intp cs2, int handle_nans) {
  char* cp_first = (char*)buf;
  char* cp_last = (char*)buf + cs1 * (size - 1);
  char* cp_inner;
  char has_non_finite = 0;
  unsigned int i, j;

  if (handle_nans == PYGEOS_HANDLE_NANS_IGNORE) {
    /* Find the first finite coordinate index */
    for (i = 0; i < size; i++, cp_first += cs1) {
      cp_inner = cp_first;
      has_non_finite = 0;
      for (j = 0; j < dims; j++, cp_inner += cs2) {
        if (!(npy_isfinite(*(double*)cp_inner))) {
          has_non_finite = 1;
          break;
        }
      }
      if (!has_non_finite) {
        break;
      }
    }
    /* Find the last finite coordinate index */
    for (i = size - 1; i >= 0; i--, cp_last -= cs1) {
      cp_inner = cp_last;
      has_non_finite = 0;
      for (j = 0; j < dims; j++, cp_inner += cs2) {
        if (!(npy_isfinite(*(double*)cp_inner))) {
          has_non_finite = 1;
          break;
        }
      }
      if (!has_non_finite) {
        break;
      }
    }
  }

  /* Compare the coordinates */
  for (j = 0; j < dims; j++, cp_first += cs2, cp_last += cs2) {
    if (*(double*)cp_first != *(double*)cp_last) {
      return 1;
    }
  }

  return 0;
}

/* Create a GEOSCoordSequence from an array
 *
 * Note: this function assumes that the dimension of the buffer is already
 * checked before calling this function, so the buffer and the dims argument
 * is only 2D or 3D.
 *
 * handle_nans: 0 means 'allow', 1 means 'ignore', 2 means 'raise'
 *
 * Returns an error state (PGERR_SUCCESS / PGERR_GEOS_EXCEPTION / PGERR_NAN_COORD).
 */
int coordseq_from_buffer(GEOSContextHandle_t ctx, const double* buf, unsigned int size,
                         unsigned int dims, char is_ring, int handle_nans, npy_intp cs1,
                         npy_intp cs2, GEOSCoordSequence** ret_ptr) {
  GEOSCoordSequence* coord_seq;
  char *cp1, *cp2;
  unsigned int i, j, current, first, actual_size;
  double coord;
  char all_finite;
  char ring_closure = 0;

  switch (handle_nans) {
    case PYGEOS_HANDLE_NANS_ALLOW:
      actual_size = size;
      break;
    case PYGEOS_HANDLE_NANS_IGNORE:
      actual_size = count_finite(buf, size, dims, cs1, cs2);
      break;
    case PYGEOS_HANDLE_NANS_RAISE:
      actual_size = count_finite(buf, size, dims, cs1, cs2);
      if (actual_size != size) {
        return PGERR_NAN_COORD;
      }
      break;
    default:
      return PGERR_NAN_COORD;
  }

  /* Rings automatically get an extra (closing) coordinate if they have
     only 3 or if the first and last are not equal. */
  if (is_ring && (actual_size > 0)) {
    if (actual_size <= 3) {
      ring_closure = 1;
    } else {
      ring_closure = ending_coordinates_equal(buf, size, dims, cs1, cs2, handle_nans);
    }
  }

#if GEOS_SINCE_3_10_0

  if ((!ring_closure) && (actual_size == size)) {
    if ((cs1 == dims * 8) && (cs2 == 8)) {
      /* C-contiguous memory */
      int hasZ = dims == 3;
      coord_seq = GEOSCoordSeq_copyFromBuffer_r(ctx, buf, size, hasZ, 0);
      if (coord_seq == NULL) {
        return PGERR_GEOS_EXCEPTION;
      }
      *ret_ptr = coord_seq;
      return PGERR_SUCCESS;
    } else if ((cs1 == 8) && (cs2 == size * 8)) {
      /* F-contiguous memory (note: this for the subset, so we don't necessarily
      end up here if the full array is F-contiguous) */
      const double* x = buf;
      const double* y = (double*)((char*)buf + cs2);
      const double* z = (dims == 3) ? (double*)((char*)buf + 2 * cs2) : NULL;
      coord_seq = GEOSCoordSeq_copyFromArrays_r(ctx, x, y, z, NULL, size);
      if (coord_seq == NULL) {
        return PGERR_GEOS_EXCEPTION;
      }
      *ret_ptr = coord_seq;
      return PGERR_SUCCESS;
    }
  }

#endif

  coord_seq = GEOSCoordSeq_create_r(ctx, actual_size + ring_closure, dims);
  if (coord_seq == NULL) {
    return PGERR_GEOS_EXCEPTION;
  }
  current = 0;
  first = size + 1;
  cp1 = (char*)buf;
  for (i = 0; i < size; i++, cp1 += cs1) {
    cp2 = cp1;
    all_finite = 1;
    for (j = 0; j < dims; j++, cp2 += cs2) {
      coord = *(double*)cp2;
      if ((size != actual_size) && !npy_isfinite(coord)) {
        all_finite = 0;
        break;
      }
      if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, current, j, coord)) {
        GEOSCoordSeq_destroy_r(ctx, coord_seq);
        return PGERR_GEOS_EXCEPTION;
      }
    }
    if (all_finite) {
      current++;
      if (first == size + 1) {
        first = i;
      }
    }
  }
  /* add the closing coordinate if necessary */
  if (ring_closure) {
    for (j = 0; j < dims; j++) {
      coord = *(double*)((char*)buf + first * cs1 + j * cs2);
      if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, actual_size, j, coord)) {
        GEOSCoordSeq_destroy_r(ctx, coord_seq);
        return PGERR_GEOS_EXCEPTION;
      }
    }
  }
  *ret_ptr = coord_seq;
  return PGERR_SUCCESS;
}

/* Copy coordinates of a GEOSCoordSequence to an array
 *
 * Note: this function assumes that the buffer is from a C-contiguous array,
 * and that the dimension of the buffer is only 2D or 3D.
 *
 * Returns 0 on error, 1 on success.
 */
int coordseq_to_buffer(GEOSContextHandle_t ctx, const GEOSCoordSequence* coord_seq,
                       double* buf, unsigned int size, unsigned int dims) {
#if GEOS_SINCE_3_10_0

  int hasZ = dims == 3;
  return GEOSCoordSeq_copyToBuffer_r(ctx, coord_seq, buf, hasZ, 0);

#else

  char *cp1, *cp2;
  unsigned int i, j;

  cp1 = (char*)buf;
  for (i = 0; i < size; i++, cp1 += 8 * dims) {
    cp2 = cp1;
    for (j = 0; j < dims; j++, cp2 += 8) {
      if (!GEOSCoordSeq_getOrdinate_r(ctx, coord_seq, i, j, (double*)cp2)) {
        return 0;
      }
    }
  }
  return 1;

#endif
}
