#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "geos.h"

#include <Python.h>
#include <structmember.h>

/* This initializes a globally accessible GEOSException object */
PyObject* geos_exception[1] = {NULL};

int init_geos(PyObject* m) {
  geos_exception[0] = PyErr_NewException("pygeos.GEOSException", NULL, NULL);
  PyModule_AddObject(m, "GEOSException", geos_exception[0]);
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

void destroy_geom_arr(void* context, GEOSGeometry** array, int length) {
  int i;
  for (i = 0; i < length; i++) {
    if (array[i] != NULL) {
      GEOSGeom_destroy_r(context, array[i]);
    }
  }
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
  } else if ((type == GEOS_POINT) | (type == GEOS_POLYGON) | (type == GEOS_MULTIPOINT) |
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
  if ((type == GEOS_MULTILINESTRING) | (type == GEOS_GEOMETRYCOLLECTION)) {
    sub_geom = GEOSGetGeometryN_r(ctx, geom, 0);
    if (sub_geom == NULL) {
      return PGERR_GEOS_EXCEPTION;  // GEOSException
    }
    type = GEOSGeomTypeId_r(ctx, sub_geom);
    if (type == -1) {
      return PGERR_GEOS_EXCEPTION;
    } else if ((type != GEOS_LINESTRING) & (type != GEOS_LINEARRING)) {
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
