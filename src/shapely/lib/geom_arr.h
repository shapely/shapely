#ifndef GEOM_ARR_H
#define GEOM_ARR_H

#include <geos_c.h>
#include <numpy/ndarraytypes.h>

/* Convert a geometry array to a NumPy array of Geometry objects */
void geom_arr_to_npy(GEOSGeometry** array, char* ptr, npy_intp stride,
                     npy_intp count);

/* Destroy a geometry array */
void destroy_geom_arr(GEOSContextHandle_t context, GEOSGeometry** array, int length);

#endif /* GEOM_ARR_H */
