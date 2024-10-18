
#ifndef GEOARROW_GEOS_H_INCLUDED
#define GEOARROW_GEOS_H_INCLUDED

#include <geos_c.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#endif

#define GEOARROW_GEOS_OK 0

enum GeoArrowGEOSEncoding {
  GEOARROW_GEOS_ENCODING_UNKNOWN = 0,
  GEOARROW_GEOS_ENCODING_WKT,
  GEOARROW_GEOS_ENCODING_WKB,
  GEOARROW_GEOS_ENCODING_GEOARROW,
  GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED
};

typedef int GeoArrowGEOSErrorCode;

const char* GeoArrowGEOSVersionGEOS(void);

const char* GeoArrowGEOSVersionGeoArrow(void);

struct GeoArrowGEOSArrayBuilder;

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderCreate(
    GEOSContextHandle_t handle, struct ArrowSchema* schema,
    struct GeoArrowGEOSArrayBuilder** out);

void GeoArrowGEOSArrayBuilderDestroy(struct GeoArrowGEOSArrayBuilder* builder);

const char* GeoArrowGEOSArrayBuilderGetLastError(
    struct GeoArrowGEOSArrayBuilder* builder);

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderAppend(
    struct GeoArrowGEOSArrayBuilder* builder, const GEOSGeometry** geom, size_t geom_size,
    size_t* n_appended);

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderFinish(
    struct GeoArrowGEOSArrayBuilder* builder, struct ArrowArray* out);

struct GeoArrowGEOSArrayReader;

GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderCreate(GEOSContextHandle_t handle,
                                                    struct ArrowSchema* schema,
                                                    struct GeoArrowGEOSArrayReader** out);

const char* GeoArrowGEOSArrayReaderGetLastError(struct GeoArrowGEOSArrayReader* reader);

GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderRead(struct GeoArrowGEOSArrayReader* reader,
                                                  struct ArrowArray* array, size_t offset,
                                                  size_t length, GEOSGeometry** out,
                                                  size_t* n_out);

void GeoArrowGEOSArrayReaderDestroy(struct GeoArrowGEOSArrayReader* reader);

struct GeoArrowGEOSSchemaCalculator;

GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorCreate(
    struct GeoArrowGEOSSchemaCalculator** out);

void GeoArrowGEOSSchemaCalculatorIngest(struct GeoArrowGEOSSchemaCalculator* calc,
                                        const int32_t* wkb_type, size_t n);

GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorFinish(
    struct GeoArrowGEOSSchemaCalculator* calc, enum GeoArrowGEOSEncoding encoding,
    struct ArrowSchema* out);

void GeoArrowGEOSSchemaCalculatorDestroy(struct GeoArrowGEOSSchemaCalculator* calc);

GeoArrowGEOSErrorCode GeoArrowGEOSMakeSchema(int32_t encoding, int32_t wkb_type,
                                             struct ArrowSchema* out);

static inline int32_t GeoArrowGEOSWKBType(GEOSContextHandle_t handle,
                                          const GEOSGeometry* geom) {
  if (geom == NULL || GEOSGetNumCoordinates_r(handle, geom) == 0) {
    return 0;
  }

  int n_dim = GEOSGeom_getCoordinateDimension_r(handle, geom);

  // Not sure how GEOS handles M in newer versions
  int32_t wkb_type;
  if (n_dim == 3) {
    wkb_type = 2000;
  } else {
    wkb_type = 0;
  }

  int type_id = GEOSGeomTypeId_r(handle, geom);
  switch (type_id) {
    case GEOS_POINT:
      wkb_type += 1;
      break;
    case GEOS_LINEARRING:
    case GEOS_LINESTRING:
      wkb_type += 2;
      break;
    case GEOS_POLYGON:
      wkb_type += 3;
      break;
    case GEOS_MULTIPOINT:
      wkb_type += 4;
      break;
    case GEOS_MULTILINESTRING:
      wkb_type += 5;
      break;
    case GEOS_MULTIPOLYGON:
      wkb_type += 6;
      break;
    case GEOS_GEOMETRYCOLLECTION:
      wkb_type += 7;
      break;
    default:
      break;
  }

  return wkb_type;
}

#ifdef __cplusplus
}
#endif

#endif
