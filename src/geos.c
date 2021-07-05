#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "geos.h"

#include <Python.h>
#include <numpy/npy_math.h>
#include <structmember.h>

/* This initializes a globally accessible GEOSException object */
PyObject* geos_exception[1] = {NULL};

int init_geos(PyObject* m) {
  geos_exception[0] = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
  PyModule_AddObject(m, "GEOSException", geos_exception[0]);
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

// POINT EMPTY is converted to POINT (nan nan)
// by GEOS >= 3.10.0. Before that, we do it ourselves here.
#if !GEOS_SINCE_3_10_0

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

#endif  // !GEOS_SINCE_3_10_0

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
  } else if ((type == GEOS_POINT) || (type == GEOS_POLYGON) || (type == GEOS_MULTIPOINT) ||
             (type == GEOS_MULTIPOLYGON)) {
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

/* Define GEOS error handlers. See GEOS_INIT / GEOS_FINISH macros in geos.h*/
void geos_error_handler(const char* message, void* userdata) {
  snprintf(userdata, 1024, "%s", message);
}

void geos_notice_handler(const char* message, void* userdata) {
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
  for (int j = 0; j < 2; j++) {
    if (!GEOSCoordSeq_setOrdinate_r(ctx, coord_seq, 0, j, 0.0)) {
      GEOSCoordSeq_destroy_r(ctx, coord_seq);
      return NULL;
    }
  }
  geom = GEOSGeom_createPoint_r(ctx, coord_seq);
  if (geom == NULL) {
    GEOSCoordSeq_destroy_r(ctx, coord_seq);
    return NULL;
  }
  return geom;
#endif
}
