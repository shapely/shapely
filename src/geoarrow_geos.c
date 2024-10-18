
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define GEOS_USE_ONLY_R_API
#include <geoarrow.h>
#include <geos_c.h>

#include "geoarrow_geos.h"

const char* GeoArrowGEOSVersionGEOS(void) { return GEOSversion(); }

const char* GeoArrowGEOSVersionGeoArrow(void) { return GeoArrowVersion(); }

struct GeoArrowGEOSArrayBuilder {
  GEOSContextHandle_t handle;
  struct GeoArrowError error;
  struct GeoArrowBuilder builder;
  struct GeoArrowWKTWriter wkt_writer;
  struct GeoArrowWKBWriter wkb_writer;
  struct GeoArrowVisitor v;
  struct GeoArrowCoordView coords_view;
  double* coords;
};

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderCreate(
    GEOSContextHandle_t handle, struct ArrowSchema* schema,
    struct GeoArrowGEOSArrayBuilder** out) {
  struct GeoArrowGEOSArrayBuilder* builder =
      (struct GeoArrowGEOSArrayBuilder*)malloc(sizeof(struct GeoArrowGEOSArrayBuilder));
  if (builder == NULL) {
    *out = NULL;
    return ENOMEM;
  }

  memset(builder, 0, sizeof(struct GeoArrowGEOSArrayBuilder));
  *out = builder;

  struct GeoArrowSchemaView schema_view;
  GEOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, &builder->error));
  switch (schema_view.type) {
    case GEOARROW_TYPE_WKT:
      GEOARROW_RETURN_NOT_OK(GeoArrowWKTWriterInit(&builder->wkt_writer));
      GeoArrowWKTWriterInitVisitor(&builder->wkt_writer, &builder->v);
      break;
    case GEOARROW_TYPE_WKB:
      GEOARROW_RETURN_NOT_OK(GeoArrowWKBWriterInit(&builder->wkb_writer));
      GeoArrowWKBWriterInitVisitor(&builder->wkb_writer, &builder->v);
      break;
    default:
      GEOARROW_RETURN_NOT_OK(
          GeoArrowBuilderInitFromSchema(&builder->builder, schema, &builder->error));
      GEOARROW_RETURN_NOT_OK(GeoArrowBuilderInitVisitor(&builder->builder, &builder->v));
      break;
  }

  builder->handle = handle;
  builder->v.error = &builder->error;
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowGEOSArrayBuilderEnsureCoords(
    struct GeoArrowGEOSArrayBuilder* builder, uint32_t n_coords, int n_dims) {
  int64_t n_required = n_coords * n_dims;
  int64_t n_current = builder->coords_view.n_coords * builder->coords_view.n_values;
  if (n_required > n_current) {
    if ((n_current * 2) > n_required) {
      n_required = n_current * 2;
    }

    builder->coords = (double*)realloc(builder->coords, n_required * sizeof(double));
    if (builder->coords == NULL) {
      builder->coords_view.n_coords = 0;
      return ENOMEM;
    }
  }

  builder->coords_view.n_coords = n_coords;
  builder->coords_view.n_values = n_dims;
  builder->coords_view.coords_stride = n_dims;
  for (int i = 0; i < n_dims; i++) {
    builder->coords_view.values[i] = builder->coords + i;
  }

  return GEOARROW_OK;
}

void GeoArrowGEOSArrayBuilderDestroy(struct GeoArrowGEOSArrayBuilder* builder) {
  if (builder->coords != NULL) {
    free(builder->coords);
  }

  if (builder->builder.private_data != NULL) {
    GeoArrowBuilderReset(&builder->builder);
  }

  if (builder->wkt_writer.private_data != NULL) {
    GeoArrowWKTWriterReset(&builder->wkt_writer);
  }

  if (builder->wkb_writer.private_data != NULL) {
    GeoArrowWKBWriterReset(&builder->wkb_writer);
  }

  free(builder);
}

const char* GeoArrowGEOSArrayBuilderGetLastError(
    struct GeoArrowGEOSArrayBuilder* builder) {
  return builder->error.message;
}

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderFinish(
    struct GeoArrowGEOSArrayBuilder* builder, struct ArrowArray* out) {
  if (builder->wkt_writer.private_data != NULL) {
    return GeoArrowWKTWriterFinish(&builder->wkt_writer, out, &builder->error);
  } else if (builder->wkb_writer.private_data != NULL) {
    return GeoArrowWKBWriterFinish(&builder->wkb_writer, out, &builder->error);
  } else if (builder->builder.private_data != NULL) {
    return GeoArrowBuilderFinish(&builder->builder, out, &builder->error);
  } else {
    GeoArrowErrorSet(&builder->error, "Invalid state");
    return EINVAL;
  }
}

static GeoArrowErrorCode VisitCoords(struct GeoArrowGEOSArrayBuilder* builder,
                                     const GEOSCoordSequence* seq,
                                     struct GeoArrowVisitor* v) {
  unsigned int size = 0;
  int result = GEOSCoordSeq_getSize_r(builder->handle, seq, &size);
  if (result == 0) {
    GeoArrowErrorSet(v->error, "GEOSCoordSeq_getSize_r() failed");
    return ENOMEM;
  }

  if (size == 0) {
    return GEOARROW_OK;
  }

  unsigned int dims = 0;
  result = GEOSCoordSeq_getDimensions_r(builder->handle, seq, &dims);
  if (result == 0) {
    GeoArrowErrorSet(v->error, "GEOSCoordSeq_getDimensions_r() failed");
    return ENOMEM;
  }

  // Make sure we have enough space to copy the coordinates into
  GEOARROW_RETURN_NOT_OK(GeoArrowGEOSArrayBuilderEnsureCoords(builder, size, dims));

  // Not sure exactly how M coordinates work in GEOS yet
  result =
      GEOSCoordSeq_copyToBuffer_r(builder->handle, seq, builder->coords, dims == 3, 0);
  if (result == 0) {
    GeoArrowErrorSet(v->error, "GEOSCoordSeq_copyToBuffer_r() failed");
    return ENOMEM;
  }

  // Call the visitor method
  GEOARROW_RETURN_NOT_OK(v->coords(v, &builder->coords_view));

  return GEOARROW_OK;
}

static GeoArrowErrorCode VisitGeometry(struct GeoArrowGEOSArrayBuilder* builder,
                                       const GEOSGeometry* geom,
                                       struct GeoArrowVisitor* v) {
  if (geom == NULL) {
    GEOARROW_RETURN_NOT_OK(v->null_feat(v));
    return GEOARROW_OK;
  }

  int type_id = GEOSGeomTypeId_r(builder->handle, geom);
  int coord_dimension = GEOSGeom_getCoordinateDimension_r(builder->handle, geom);

  enum GeoArrowGeometryType geoarrow_type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  enum GeoArrowDimensions geoarrow_dims = GEOARROW_DIMENSIONS_UNKNOWN;

  // Not sure how M dimensions work yet
  switch (coord_dimension) {
    case 2:
      geoarrow_dims = GEOARROW_DIMENSIONS_XY;
      break;
    case 3:
      geoarrow_dims = GEOARROW_DIMENSIONS_XYZ;
      break;
    default:
      GeoArrowErrorSet(v->error, "Unexpected GEOSGeom_getCoordinateDimension_r: %d",
                       coord_dimension);
      return EINVAL;
  }

  switch (type_id) {
    case GEOS_POINT:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_POINT;
      break;
    case GEOS_LINESTRING:
    case GEOS_LINEARRING:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_LINESTRING;
      break;
    case GEOS_POLYGON:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_POLYGON;
      break;
    case GEOS_MULTIPOINT:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
      break;
    case GEOS_MULTILINESTRING:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
      break;
    case GEOS_MULTIPOLYGON:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
      break;
    case GEOS_GEOMETRYCOLLECTION:
      geoarrow_type = GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION;
      break;
    default:
      GeoArrowErrorSet(v->error, "Unexpected GEOSGeomTypeId: %d", type_id);
      return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(v->geom_start(v, geoarrow_type, geoarrow_dims));

  switch (type_id) {
    case GEOS_POINT:
    case GEOS_LINESTRING:
    case GEOS_LINEARRING: {
      const GEOSCoordSequence* seq = GEOSGeom_getCoordSeq_r(builder->handle, geom);
      if (seq == NULL) {
        GeoArrowErrorSet(v->error, "GEOSGeom_getCoordSeq_r() failed");
        return ENOMEM;
      }

      GEOARROW_RETURN_NOT_OK(VisitCoords(builder, seq, v));
      break;
    }

    case GEOS_POLYGON: {
      if (GEOSisEmpty_r(builder->handle, geom)) {
        break;
      }

      const GEOSGeometry* ring = GEOSGetExteriorRing_r(builder->handle, geom);
      if (ring == NULL) {
        GeoArrowErrorSet(v->error, "GEOSGetExteriorRing_r() failed");
        return ENOMEM;
      }

      GEOARROW_RETURN_NOT_OK(v->ring_start(v));
      const GEOSCoordSequence* seq = GEOSGeom_getCoordSeq_r(builder->handle, ring);
      if (seq == NULL) {
        GeoArrowErrorSet(v->error, "GEOSGeom_getCoordSeq_r() failed");
        return ENOMEM;
      }

      GEOARROW_RETURN_NOT_OK(VisitCoords(builder, seq, v));
      GEOARROW_RETURN_NOT_OK(v->ring_end(v));

      int size = GEOSGetNumInteriorRings_r(builder->handle, geom);
      for (int i = 0; i < size; i++) {
        ring = GEOSGetInteriorRingN_r(builder->handle, geom, i);
        if (ring == NULL) {
          GeoArrowErrorSet(v->error, "GEOSGetInteriorRingN_r() failed");
          return ENOMEM;
        }

        GEOARROW_RETURN_NOT_OK(v->ring_start(v));
        seq = GEOSGeom_getCoordSeq_r(builder->handle, ring);
        if (seq == NULL) {
          GeoArrowErrorSet(v->error, "GEOSGeom_getCoordSeq_r() failed");
          return ENOMEM;
        }

        GEOARROW_RETURN_NOT_OK(VisitCoords(builder, seq, v));
        GEOARROW_RETURN_NOT_OK(v->ring_end(v));
      }

      break;
    }

    case GEOS_MULTIPOINT:
    case GEOS_MULTILINESTRING:
    case GEOS_MULTIPOLYGON:
    case GEOS_GEOMETRYCOLLECTION: {
      int size = GEOSGetNumGeometries_r(builder->handle, geom);
      for (int i = 0; i < size; i++) {
        const GEOSGeometry* child = GEOSGetGeometryN_r(builder->handle, geom, i);
        if (child == NULL) {
          GeoArrowErrorSet(v->error, "GEOSGetGeometryN_r() failed");
          return ENOMEM;
        }

        GEOARROW_RETURN_NOT_OK(VisitGeometry(builder, child, v));
      }

      break;
    }
    default:
      GeoArrowErrorSet(v->error, "Unexpected GEOSGeomTypeId: %d", type_id);
      return EINVAL;
  }

  GEOARROW_RETURN_NOT_OK(v->geom_end(v));
  return GEOARROW_OK;
}

GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderAppend(
    struct GeoArrowGEOSArrayBuilder* builder, const GEOSGeometry** geom, size_t geom_size,
    size_t* n_appended) {
  *n_appended = 0;

  for (size_t i = 0; i < geom_size; i++) {
    GEOARROW_RETURN_NOT_OK(builder->v.feat_start(&builder->v));
    GEOARROW_RETURN_NOT_OK(VisitGeometry(builder, geom[i], &builder->v));
    GEOARROW_RETURN_NOT_OK(builder->v.feat_end(&builder->v));
    *n_appended = i + 1;
  }

  return GEOARROW_OK;
}

// This should really be in nanoarrow and/or geoarrow
struct GeoArrowGEOSBitmapReader {
  const uint8_t* bits;
  int64_t byte_i;
  int bit_i;
  uint8_t byte;
};

static inline void GeoArrowGEOSBitmapReaderInit(
    struct GeoArrowGEOSBitmapReader* bitmap_reader, const uint8_t* bits, int64_t offset) {
  memset(bitmap_reader, 0, sizeof(struct GeoArrowGEOSBitmapReader));
  bitmap_reader->bits = bits;

  if (bits != NULL) {
    bitmap_reader->byte_i = offset / 8;
    bitmap_reader->bit_i = offset % 8;
    if (bitmap_reader->bit_i == 0) {
      bitmap_reader->bit_i = 7;
      bitmap_reader->byte_i--;
    } else {
      bitmap_reader->bit_i--;
    }
  }
}

static inline int8_t GeoArrowGEOSBitmapReaderNextIsNull(
    struct GeoArrowGEOSBitmapReader* bitmap_reader) {
  if (bitmap_reader->bits == NULL) {
    return 0;
  }

  if (++bitmap_reader->bit_i == 8) {
    bitmap_reader->byte = bitmap_reader->bits[++bitmap_reader->byte_i];
    bitmap_reader->bit_i = 0;
  }

  return (bitmap_reader->byte & (1 << bitmap_reader->bit_i)) == 0;
}

struct GeoArrowGEOSArrayReader {
  GEOSContextHandle_t handle;
  struct GeoArrowError error;
  struct GeoArrowArrayView array_view;
  // In order to use GeoArrow's read capability we need to write a visitor-based
  // constructor for GEOS geometries, which is complicated and may or may not be
  // faster than GEOS' own readers.
  GEOSWKTReader* wkt_reader;
  GEOSWKBReader* wkb_reader;
  // In-progress items that we might need to clean up if an error was returned
  int64_t n_geoms[2];
  GEOSGeometry** geoms[2];
  struct GeoArrowGEOSBitmapReader bitmap_reader;
  // GEOS' WKT reader needs null-terminated strings, but Arrow stores them in
  // buffers without the null terminator. Thus, we need a bounce buffer to copy
  // each WKT item into before passing to GEOS' reader.
  size_t wkt_temp_size;
  char* wkt_temp;
};

static GeoArrowErrorCode GeoArrowGEOSArrayReaderEnsureScratch(
    struct GeoArrowGEOSArrayReader* reader, int64_t n_geoms, int level) {
  if (n_geoms <= reader->n_geoms[level]) {
    return GEOARROW_OK;
  }

  if ((reader->n_geoms[level] * 2) > n_geoms) {
    n_geoms = reader->n_geoms[level] * 2;
  }

  reader->geoms[level] =
      (GEOSGeometry**)realloc(reader->geoms[level], n_geoms * sizeof(GEOSGeometry*));
  if (reader->geoms[level] == NULL) {
    reader->n_geoms[level] = 0;
    return ENOMEM;
  }

  memset(reader->geoms[level], 0, n_geoms * sizeof(GEOSGeometry*));
  return GEOARROW_OK;
}

static void GeoArrowGEOSArrayReaderResetScratch(struct GeoArrowGEOSArrayReader* reader) {
  for (int level = 0; level < 2; level++) {
    for (int64_t i = 0; i < reader->n_geoms[level]; i++) {
      if (reader->geoms[level][i] != NULL) {
        GEOSGeom_destroy_r(reader->handle, reader->geoms[level][i]);
        reader->geoms[level][i] = NULL;
      }
    }
  }
}

static GeoArrowErrorCode GeoArrowGEOSArrayReaderEnsureWKTTemp(
    struct GeoArrowGEOSArrayReader* reader, int64_t item_size) {
  if (item_size <= reader->wkt_temp_size) {
    return GEOARROW_OK;
  }

  if ((reader->wkt_temp_size * 2) > item_size) {
    item_size = reader->wkt_temp_size * 2;
  }

  reader->wkt_temp = (char*)realloc(reader->wkt_temp, item_size);
  if (reader->wkt_temp == NULL) {
    reader->wkt_temp_size = 0;
    return ENOMEM;
  }

  return GEOARROW_OK;
}

GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderCreate(
    GEOSContextHandle_t handle, struct ArrowSchema* schema,
    struct GeoArrowGEOSArrayReader** out) {
  struct GeoArrowGEOSArrayReader* reader =
      (struct GeoArrowGEOSArrayReader*)malloc(sizeof(struct GeoArrowGEOSArrayReader));
  if (reader == NULL) {
    *out = NULL;
    return ENOMEM;
  }

  memset(reader, 0, sizeof(struct GeoArrowGEOSArrayReader));
  *out = reader;

  reader->handle = handle;
  GEOARROW_RETURN_NOT_OK(
      GeoArrowArrayViewInitFromSchema(&reader->array_view, schema, &reader->error));

  return GEOARROW_OK;
}

const char* GeoArrowGEOSArrayReaderGetLastError(struct GeoArrowGEOSArrayReader* reader) {
  return reader->error.message;
}

static GeoArrowErrorCode MakeGeomFromWKB(struct GeoArrowGEOSArrayReader* reader,
                                         size_t offset, size_t length, GEOSGeometry** out,
                                         size_t* n_out) {
  offset += reader->array_view.offset[0];

  GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader, reader->array_view.validity_bitmap,
                               offset);

  for (size_t i = 0; i < length; i++) {
    if (GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    int64_t data_offset = reader->array_view.offsets[0][i];
    int64_t data_size = reader->array_view.offsets[0][i + 1] - data_offset;

    out[i] = GEOSWKBReader_read_r(reader->handle, reader->wkb_reader,
                                  reader->array_view.data + data_offset, data_size);
    if (out[i] == NULL) {
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSWKBReader_read_r() failed", (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode MakeGeomFromWKT(struct GeoArrowGEOSArrayReader* reader,
                                         size_t offset, size_t length, GEOSGeometry** out,
                                         size_t* n_out) {
  offset += reader->array_view.offset[0];

  GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader, reader->array_view.validity_bitmap,
                               offset);

  for (size_t i = 0; i < length; i++) {
    if (GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    int64_t data_offset = reader->array_view.offsets[0][i];
    int64_t data_size = reader->array_view.offsets[0][i + 1] - data_offset;

    // GEOSWKTReader_read_r() requires a null-terminated string. To ensure that, we
    // copy into memory we own and add the null-terminator ourselves.
    GEOARROW_RETURN_NOT_OK(GeoArrowGEOSArrayReaderEnsureWKTTemp(reader, data_size + 1));
    memcpy(reader->wkt_temp, reader->array_view.data + data_offset, data_size);
    reader->wkt_temp[data_size] = '\0';

    out[i] = GEOSWKTReader_read_r(reader->handle, reader->wkt_reader, reader->wkt_temp);
    if (out[i] == NULL) {
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSWKBReader_read_r() failed", (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode MakeCoordSeq(struct GeoArrowGEOSArrayReader* reader,
                                      size_t offset, size_t length,
                                      GEOSCoordSequence** out) {
  offset += reader->array_view.offset[reader->array_view.n_offsets];
  struct GeoArrowCoordView* coords = &reader->array_view.coords;
  const double* z = NULL;
  const double* m = NULL;

  switch (reader->array_view.schema_view.dimensions) {
    case GEOARROW_DIMENSIONS_XYZ:
      z = coords->values[2];
      break;
    case GEOARROW_DIMENSIONS_XYM:
      m = coords->values[2];
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      z = coords->values[2];
      m = coords->values[3];
      break;
    default:
      break;
  }

  GEOSCoordSequence* seq;

  switch (reader->array_view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      seq = GEOSCoordSeq_copyFromArrays_r(reader->handle, coords->values[0] + offset,
                                          coords->values[1] + offset, z, m, length);
      break;
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      seq = GEOSCoordSeq_copyFromBuffer_r(reader->handle,
                                          coords->values[0] + (offset * coords->n_values),
                                          length, z != NULL, m != NULL);
      break;
    default:
      GeoArrowErrorSet(&reader->error, "Unsupported coord type");
      return ENOTSUP;
  }

  if (seq == NULL) {
    GeoArrowErrorSet(&reader->error, "GEOSCoordSeq_copyFromArrays_r() failed");
    return ENOMEM;
  }

  *out = seq;
  return GEOARROW_OK;
}

static GeoArrowErrorCode MakePoints(struct GeoArrowGEOSArrayReader* reader, size_t offset,
                                    size_t length, GEOSGeometry** out, size_t* n_out) {
  int top_level =
      reader->array_view.schema_view.geometry_type == GEOARROW_GEOMETRY_TYPE_POINT;
  if (top_level) {
    GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader,
                                 reader->array_view.validity_bitmap,
                                 reader->array_view.offset[0] + offset);
  }

  GEOSCoordSequence* seq = NULL;
  for (size_t i = 0; i < length; i++) {
    if (top_level && GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    GEOARROW_RETURN_NOT_OK(MakeCoordSeq(reader, offset + i, 1, &seq));
    out[i] = GEOSGeom_createPoint_r(reader->handle, seq);
    if (out[i] == NULL) {
      GEOSCoordSeq_destroy_r(reader->handle, seq);
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSGeom_createPoint_r() failed", (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode MakeLinestrings(struct GeoArrowGEOSArrayReader* reader,
                                         size_t offset, size_t length, GEOSGeometry** out,
                                         size_t* n_out) {
  offset += reader->array_view.offset[reader->array_view.n_offsets - 1];
  const int32_t* coord_offsets =
      reader->array_view.offsets[reader->array_view.n_offsets - 1];

  int top_level =
      reader->array_view.schema_view.geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING;
  if (top_level) {
    GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader,
                                 reader->array_view.validity_bitmap, offset);
  }

  GEOSCoordSequence* seq = NULL;
  for (size_t i = 0; i < length; i++) {
    if (top_level && GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    GEOARROW_RETURN_NOT_OK(
        MakeCoordSeq(reader, coord_offsets[offset + i],
                     coord_offsets[offset + i + 1] - coord_offsets[offset + i], &seq));
    out[i] = GEOSGeom_createLineString_r(reader->handle, seq);
    if (out[i] == NULL) {
      GEOSCoordSeq_destroy_r(reader->handle, seq);
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSGeom_createLineString_r() failed",
                       (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode MakeLinearrings(struct GeoArrowGEOSArrayReader* reader,
                                         size_t offset, size_t length,
                                         GEOSGeometry** out) {
  offset += reader->array_view.offset[reader->array_view.n_offsets - 1];
  const int32_t* coord_offsets =
      reader->array_view.offsets[reader->array_view.n_offsets - 1];

  GEOSCoordSequence* seq = NULL;
  for (size_t i = 0; i < length; i++) {
    GEOARROW_RETURN_NOT_OK(
        MakeCoordSeq(reader, coord_offsets[offset + i],
                     coord_offsets[offset + i + 1] - coord_offsets[offset + i], &seq));
    out[i] = GEOSGeom_createLinearRing_r(reader->handle, seq);
    if (out[i] == NULL) {
      GEOSCoordSeq_destroy_r(reader->handle, seq);
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSGeom_createLinearRing_r() failed",
                       (long)i);
      return ENOMEM;
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode MakePolygons(struct GeoArrowGEOSArrayReader* reader,
                                      size_t offset, size_t length, GEOSGeometry** out,
                                      size_t* n_out) {
  offset += reader->array_view.offset[reader->array_view.n_offsets - 2];
  const int32_t* ring_offsets =
      reader->array_view.offsets[reader->array_view.n_offsets - 2];

  int top_level =
      reader->array_view.schema_view.geometry_type == GEOARROW_GEOMETRY_TYPE_POLYGON;
  if (top_level) {
    GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader,
                                 reader->array_view.validity_bitmap, offset);
  }

  for (size_t i = 0; i < length; i++) {
    if (top_level && GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    int64_t ring_offset = ring_offsets[offset + i];
    int64_t n_rings = ring_offsets[offset + i + 1] - ring_offset;

    if (n_rings == 0) {
      out[i] = GEOSGeom_createEmptyPolygon_r(reader->handle);
    } else {
      GEOARROW_RETURN_NOT_OK(GeoArrowGEOSArrayReaderEnsureScratch(reader, n_rings, 0));
      GEOARROW_RETURN_NOT_OK(
          MakeLinearrings(reader, ring_offset, n_rings, reader->geoms[0]));
      out[i] = GEOSGeom_createPolygon_r(reader->handle, reader->geoms[0][0],
                                        reader->geoms[0] + 1, n_rings - 1);
      memset(reader->geoms[0], 0, n_rings * sizeof(GEOSGeometry*));
    }

    if (out[i] == NULL) {
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSGeom_createPolygon_r() failed",
                       (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

typedef GeoArrowErrorCode (*GeoArrowGEOSPartMaker)(struct GeoArrowGEOSArrayReader* reader,
                                                   size_t offset, size_t length,
                                                   GEOSGeometry** out, size_t* n_out);

static GeoArrowErrorCode MakeCollection(struct GeoArrowGEOSArrayReader* reader,
                                        size_t offset, size_t length, GEOSGeometry** out,
                                        int geom_level, int offset_level, int geos_type,
                                        GeoArrowGEOSPartMaker part_maker, size_t* n_out) {
  offset += reader->array_view.offset[reader->array_view.n_offsets - offset_level];
  const int32_t* part_offsets =
      reader->array_view.offsets[reader->array_view.n_offsets - offset_level];

  // Currently collections are always outer geometries
  GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader, reader->array_view.validity_bitmap,
                               offset);

  size_t part_n_out = 0;
  for (size_t i = 0; i < length; i++) {
    if (GeoArrowGEOSBitmapReaderNextIsNull(&reader->bitmap_reader)) {
      out[i] = NULL;
      *n_out += 1;
      continue;
    }

    int64_t part_offset = part_offsets[offset + i];
    int64_t n_parts = part_offsets[offset + i + 1] - part_offset;

    if (n_parts == 0) {
      out[i] = GEOSGeom_createEmptyCollection_r(reader->handle, geos_type);
    } else {
      GEOARROW_RETURN_NOT_OK(
          GeoArrowGEOSArrayReaderEnsureScratch(reader, n_parts, geom_level));
      GEOARROW_RETURN_NOT_OK(part_maker(reader, part_offset, n_parts,
                                        reader->geoms[geom_level], &part_n_out));
      out[i] = GEOSGeom_createCollection_r(reader->handle, geos_type,
                                           reader->geoms[geom_level], n_parts);
      memset(reader->geoms[geom_level], 0, n_parts * sizeof(GEOSGeometry*));
    }

    if (out[i] == NULL) {
      GeoArrowErrorSet(&reader->error, "[%ld] GEOSGeom_createEmptyCollection_r() failed",
                       (long)i);
      return ENOMEM;
    }

    *n_out += 1;
  }

  return GEOARROW_OK;
}

GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderRead(struct GeoArrowGEOSArrayReader* reader,
                                                  struct ArrowArray* array, size_t offset,
                                                  size_t length, GEOSGeometry** out,
                                                  size_t* n_out) {
  GeoArrowGEOSArrayReaderResetScratch(reader);

  GEOARROW_RETURN_NOT_OK(
      GeoArrowArrayViewSetArray(&reader->array_view, array, &reader->error));

  GeoArrowGEOSBitmapReaderInit(&reader->bitmap_reader, NULL, 0);

  memset(out, 0, sizeof(GEOSGeometry*) * length);
  *n_out = 0;

  GeoArrowErrorCode result;
  switch (reader->array_view.schema_view.type) {
    case GEOARROW_TYPE_WKB:
      if (reader->wkb_reader == NULL) {
        reader->wkb_reader = GEOSWKBReader_create_r(reader->handle);
        if (reader->wkb_reader == NULL) {
          GeoArrowErrorSet(&reader->error, "GEOSWKBReader_create_r() failed");
          return ENOMEM;
        }
      }

      result = MakeGeomFromWKB(reader, offset, length, out, n_out);
      break;
    case GEOARROW_TYPE_WKT:
      if (reader->wkt_reader == NULL) {
        reader->wkt_reader = GEOSWKTReader_create_r(reader->handle);
        if (reader->wkt_reader == NULL) {
          GeoArrowErrorSet(&reader->error, "GEOSWKTReader_create_r() failed");
          return ENOMEM;
        }
      }

      result = MakeGeomFromWKT(reader, offset, length, out, n_out);
      break;
    default:
      switch (reader->array_view.schema_view.geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POINT:
          result = MakePoints(reader, offset, length, out, n_out);
          break;
        case GEOARROW_GEOMETRY_TYPE_LINESTRING:
          result = MakeLinestrings(reader, offset, length, out, n_out);
          break;
        case GEOARROW_GEOMETRY_TYPE_POLYGON:
          result = MakePolygons(reader, offset, length, out, n_out);
          break;
        case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
          result = MakeCollection(reader, offset, length, out, 0, 1, GEOS_MULTIPOINT,
                                  &MakePoints, n_out);
          break;
        case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
          result = MakeCollection(reader, offset, length, out, 0, 2, GEOS_MULTILINESTRING,
                                  &MakeLinestrings, n_out);
          break;
        case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
          result = MakeCollection(reader, offset, length, out, 1, 3, GEOS_MULTIPOLYGON,
                                  &MakePolygons, n_out);
          break;
        default:
          GeoArrowErrorSet(&reader->error,
                           "GeoArrowGEOSArrayReaderRead not implemented for array type");
          return ENOTSUP;
      }
  }

  return result;
}

void GeoArrowGEOSArrayReaderDestroy(struct GeoArrowGEOSArrayReader* reader) {
  if (reader->wkt_reader != NULL) {
    GEOSWKTReader_destroy_r(reader->handle, reader->wkt_reader);
  }

  if (reader->wkb_reader != NULL) {
    GEOSWKBReader_destroy_r(reader->handle, reader->wkb_reader);
  }

  GeoArrowGEOSArrayReaderResetScratch(reader);

  for (int i = 0; i < 2; i++) {
    if (reader->geoms[i] != NULL) {
      free(reader->geoms[i]);
    }
  }

  if (reader->wkt_temp != NULL) {
    free(reader->wkt_temp);
  }

  free(reader);
}

struct GeoArrowGEOSSchemaCalculator {
  int geometry_type;
  int dimensions;
};

GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorCreate(
    struct GeoArrowGEOSSchemaCalculator** out) {
  struct GeoArrowGEOSSchemaCalculator* calc =
      (struct GeoArrowGEOSSchemaCalculator*)malloc(
          sizeof(struct GeoArrowGEOSSchemaCalculator));
  if (calc == NULL) {
    *out = NULL;
    return ENOMEM;
  }

  calc->geometry_type = -1;
  calc->dimensions = GEOARROW_DIMENSIONS_UNKNOWN;
  *out = calc;

  return GEOARROW_OK;
}

static int GeometryType2(int x, int y) {
  switch (x) {
    case -1:
      return y;
    case GEOARROW_GEOMETRY_TYPE_GEOMETRY:
      return x;
    case GEOARROW_GEOMETRY_TYPE_POINT:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_POINT:
        case GEOARROW_TYPE_MULTIPOINT:
          return y;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_LINESTRING:
        case GEOARROW_TYPE_MULTILINESTRING:
          return y;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_POLYGON:
        case GEOARROW_TYPE_MULTIPOLYGON:
          return y;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_POINT:
        case GEOARROW_TYPE_MULTIPOINT:
          return x;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_LINESTRING:
        case GEOARROW_TYPE_MULTILINESTRING:
          return x;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_TYPE_POLYGON:
        case GEOARROW_TYPE_MULTIPOLYGON:
          return x;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      switch (y) {
        case -1:
          return x;
        case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
          return x;
        default:
          return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
      }
    default:
      return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  }
}

static int Dimensions2(int x, int y) {
  switch (x) {
    case GEOARROW_DIMENSIONS_UNKNOWN:
      return y;
    case GEOARROW_DIMENSIONS_XY:
      switch (y) {
        case GEOARROW_DIMENSIONS_UNKNOWN:
          return x;
        default:
          return y;
      }
    case GEOARROW_DIMENSIONS_XYZ:
      switch (y) {
        case GEOARROW_DIMENSIONS_UNKNOWN:
          return x;
        case GEOARROW_DIMENSIONS_XYM:
          return GEOARROW_DIMENSIONS_XYZM;
        default:
          return y;
      }
    case GEOARROW_DIMENSIONS_XYM:
      switch (y) {
        case GEOARROW_DIMENSIONS_UNKNOWN:
          return x;
        case GEOARROW_DIMENSIONS_XYZ:
          return GEOARROW_DIMENSIONS_XYZM;
        default:
          return y;
      }
    default:
      return GEOARROW_DIMENSIONS_XYZM;
  }
}

void GeoArrowGEOSSchemaCalculatorIngest(struct GeoArrowGEOSSchemaCalculator* calc,
                                        const int32_t* wkb_type, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (wkb_type[i] == 0) {
      continue;
    }

    calc->geometry_type = GeometryType2(calc->geometry_type, wkb_type[i] % 1000);
    calc->dimensions = Dimensions2(calc->dimensions, wkb_type[i] / 1000);
  }
}

GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorFinish(
    struct GeoArrowGEOSSchemaCalculator* calc, enum GeoArrowGEOSEncoding encoding,
    struct ArrowSchema* out) {
  enum GeoArrowCoordType coord_type;
  switch (encoding) {
    case GEOARROW_GEOS_ENCODING_WKT:
    case GEOARROW_GEOS_ENCODING_WKB:
      return GeoArrowGEOSMakeSchema(encoding, 0, out);
    case GEOARROW_GEOS_ENCODING_GEOARROW:
      coord_type = GEOARROW_COORD_TYPE_SEPARATE;
      break;
    case GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED:
      coord_type = GEOARROW_COORD_TYPE_INTERLEAVED;
      break;
    default:
      return EINVAL;
  }

  enum GeoArrowGeometryType geometry_type;
  switch (calc->geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      geometry_type = (enum GeoArrowGeometryType)calc->geometry_type;
      break;
    case -1:
      // We don't have an "empty"/"null" type to return, but "POINT" is also
      // not quite right.
    default:
      return GeoArrowGEOSMakeSchema(GEOARROW_GEOS_ENCODING_WKB, 0, out);
  }

  enum GeoArrowDimensions dimensions;
  switch (calc->dimensions) {
    case GEOARROW_DIMENSIONS_UNKNOWN:
      dimensions = GEOARROW_DIMENSIONS_XY;
      break;
    case GEOARROW_DIMENSIONS_XY:
    case GEOARROW_DIMENSIONS_XYZ:
    case GEOARROW_DIMENSIONS_XYM:
    case GEOARROW_DIMENSIONS_XYZM:
      dimensions = (enum GeoArrowDimensions)calc->dimensions;
      break;
    default:
      return GeoArrowGEOSMakeSchema(GEOARROW_GEOS_ENCODING_WKB, 0, out);
  }

  enum GeoArrowType type = GeoArrowMakeType(geometry_type, dimensions, coord_type);
  GEOARROW_RETURN_NOT_OK(GeoArrowSchemaInitExtension(out, type));
  return GEOARROW_OK;
}

void GeoArrowGEOSSchemaCalculatorDestroy(struct GeoArrowGEOSSchemaCalculator* calc) {
  free(calc);
}

GeoArrowGEOSErrorCode GeoArrowGEOSMakeSchema(int32_t encoding, int32_t wkb_type,
                                             struct ArrowSchema* out) {
  enum GeoArrowType type = GEOARROW_TYPE_UNINITIALIZED;
  enum GeoArrowGeometryType geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_UNKNOWN;
  enum GeoArrowCoordType coord_type = GEOARROW_COORD_TYPE_UNKNOWN;

  switch (encoding) {
    case GEOARROW_GEOS_ENCODING_WKT:
      type = GEOARROW_TYPE_WKT;
      break;
    case GEOARROW_GEOS_ENCODING_WKB:
      type = GEOARROW_TYPE_WKB;
      break;
    case GEOARROW_GEOS_ENCODING_GEOARROW:
      coord_type = GEOARROW_COORD_TYPE_SEPARATE;
      break;
    case GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED:
      coord_type = GEOARROW_COORD_TYPE_INTERLEAVED;
      break;
    default:
      return EINVAL;
  }

  if (type == GEOARROW_TYPE_UNINITIALIZED) {
    geometry_type = wkb_type % 1000;
    dimensions = wkb_type / 1000 + 1;
    type = GeoArrowMakeType(geometry_type, dimensions, coord_type);
  }

  GEOARROW_RETURN_NOT_OK(GeoArrowSchemaInitExtension(out, type));
  return GEOARROW_OK;
}
