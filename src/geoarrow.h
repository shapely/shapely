
#ifndef GEOARROW_CONFIG_H_INCLUDED
#define GEOARROW_CONFIG_H_INCLUDED

#define GEOARROW_VERSION_MAJOR 0
#define GEOARROW_VERSION_MINOR 2
#define GEOARROW_VERSION_PATCH 0
#define GEOARROW_VERSION "0.2.0-SNAPSHOT"

#define GEOARROW_VERSION_INT \
  (GEOARROW_VERSION_MAJOR * 10000 + GEOARROW_VERSION_MINOR * 100 + GEOARROW_VERSION_PATCH)

#define GEOARROW_USE_FAST_FLOAT 0

#define GEOARROW_USE_RYU 0

// #define GEOARROW_NAMESPACE YourNamespaceHere

#if defined(GEOARROW_NAMESPACE)
#define NANOARROW_NAMESPACE GEOARROW_NAMESPACE
#endif

#endif

#ifndef GEOARROW_GEOARROW_TYPES_H_INCLUDED
#define GEOARROW_GEOARROW_TYPES_H_INCLUDED

#include <stddef.h>
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

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callback to get the stream type
  // (will be the same for all arrays in the stream).
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowSchema must be released independently from the stream.
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

  // Callback to get the next array
  // (if no error and the array is released, the stream has ended)
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowArray must be released independently from the stream.
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

  // Callback to get optional detailed error information.
  // This must only be called if the last stream operation failed
  // with a non-0 return code.
  //
  // Return value: pointer to a null-terminated character array describing
  // the last error, or NULL if no description is available.
  //
  // The returned pointer is only valid until the next operation on this stream
  // (including release).
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback: release the stream's own resources.
  // Note that arrays returned by `get_next` must be individually released.
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE
#endif  // ARROW_FLAG_DICTIONARY_ORDERED

/// \brief Return code for success
/// \ingroup geoarrow-utility
#define GEOARROW_OK 0

#define _GEOARROW_CONCAT(x, y) x##y
#define _GEOARROW_MAKE_NAME(x, y) _GEOARROW_CONCAT(x, y)

#define _GEOARROW_RETURN_NOT_OK_IMPL(NAME, EXPR) \
  do {                                           \
    const int NAME = (EXPR);                     \
    if (NAME) return NAME;                       \
  } while (0)

/// \brief Macro helper for error handling
/// \ingroup geoarrow-utility
#define GEOARROW_RETURN_NOT_OK(EXPR) \
  _GEOARROW_RETURN_NOT_OK_IMPL(_GEOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR)

/// \brief Represents an errno-compatible error code
/// \ingroup geoarrow-utility
typedef int GeoArrowErrorCode;

struct GeoArrowError {
  char message[1024];
};

/// \brief A read-only view of a string
/// \ingroup geoarrow-utility
struct GeoArrowStringView {
  /// \brief Pointer to the beginning of the string. May be NULL if size_bytes is 0.
  /// there is no requirement that the strig is null-terminated.
  const char* data;

  /// \brief The size of the string in bytes
  int64_t size_bytes;
};

/// \brief A read-only view of a buffer
/// \ingroup geoarrow-utility
struct GeoArrowBufferView {
  /// \brief Pointer to the beginning of the string. May be NULL if size_bytes is 0.
  const uint8_t* data;

  /// \brief The size of the buffer in bytes
  int64_t size_bytes;
};

/// \brief Type identifier for types supported by this library
/// \ingroup geoarrow-schema
///
/// It is occasionally useful to represent each unique memory layout
/// with a single type identifier. These types include both the serialized
/// representations and the GeoArrow-native representations. Type identifiers
/// for GeoArrow-native representations can be decomposed into or reconstructed
/// from GeoArrowGeometryType, GeoArrowDimensions, and GeoArrowCoordType.
///
/// The values of this enum are chosen to support efficient decomposition
/// and/or reconstruction into the components that make up this value; however,
/// these values are not guaranteed to be stable.
enum GeoArrowType {
  GEOARROW_TYPE_UNINITIALIZED = 0,

  GEOARROW_TYPE_WKB = 100001,
  GEOARROW_TYPE_LARGE_WKB = 100002,

  GEOARROW_TYPE_WKT = 100003,
  GEOARROW_TYPE_LARGE_WKT = 100004,

  GEOARROW_TYPE_POINT = 1,
  GEOARROW_TYPE_LINESTRING = 2,
  GEOARROW_TYPE_POLYGON = 3,
  GEOARROW_TYPE_MULTIPOINT = 4,
  GEOARROW_TYPE_MULTILINESTRING = 5,
  GEOARROW_TYPE_MULTIPOLYGON = 6,

  GEOARROW_TYPE_POINT_Z = 1001,
  GEOARROW_TYPE_LINESTRING_Z = 1002,
  GEOARROW_TYPE_POLYGON_Z = 1003,
  GEOARROW_TYPE_MULTIPOINT_Z = 1004,
  GEOARROW_TYPE_MULTILINESTRING_Z = 1005,
  GEOARROW_TYPE_MULTIPOLYGON_Z = 1006,

  GEOARROW_TYPE_POINT_M = 2001,
  GEOARROW_TYPE_LINESTRING_M = 2002,
  GEOARROW_TYPE_POLYGON_M = 2003,
  GEOARROW_TYPE_MULTIPOINT_M = 2004,
  GEOARROW_TYPE_MULTILINESTRING_M = 2005,
  GEOARROW_TYPE_MULTIPOLYGON_M = 2006,

  GEOARROW_TYPE_POINT_ZM = 3001,
  GEOARROW_TYPE_LINESTRING_ZM = 3002,
  GEOARROW_TYPE_POLYGON_ZM = 3003,
  GEOARROW_TYPE_MULTIPOINT_ZM = 3004,
  GEOARROW_TYPE_MULTILINESTRING_ZM = 3005,
  GEOARROW_TYPE_MULTIPOLYGON_ZM = 3006,

  GEOARROW_TYPE_INTERLEAVED_POINT = 10001,
  GEOARROW_TYPE_INTERLEAVED_LINESTRING = 10002,
  GEOARROW_TYPE_INTERLEAVED_POLYGON = 10003,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOINT = 10004,
  GEOARROW_TYPE_INTERLEAVED_MULTILINESTRING = 10005,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOLYGON = 10006,
  GEOARROW_TYPE_INTERLEAVED_POINT_Z = 11001,
  GEOARROW_TYPE_INTERLEAVED_LINESTRING_Z = 11002,
  GEOARROW_TYPE_INTERLEAVED_POLYGON_Z = 11003,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOINT_Z = 11004,
  GEOARROW_TYPE_INTERLEAVED_MULTILINESTRING_Z = 11005,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOLYGON_Z = 11006,
  GEOARROW_TYPE_INTERLEAVED_POINT_M = 12001,
  GEOARROW_TYPE_INTERLEAVED_LINESTRING_M = 12002,
  GEOARROW_TYPE_INTERLEAVED_POLYGON_M = 12003,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOINT_M = 12004,
  GEOARROW_TYPE_INTERLEAVED_MULTILINESTRING_M = 12005,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOLYGON_M = 12006,
  GEOARROW_TYPE_INTERLEAVED_POINT_ZM = 13001,
  GEOARROW_TYPE_INTERLEAVED_LINESTRING_ZM = 13002,
  GEOARROW_TYPE_INTERLEAVED_POLYGON_ZM = 13003,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOINT_ZM = 13004,
  GEOARROW_TYPE_INTERLEAVED_MULTILINESTRING_ZM = 13005,
  GEOARROW_TYPE_INTERLEAVED_MULTIPOLYGON_ZM = 13006
};

/// \brief Geometry type identifiers supported by GeoArrow
/// \ingroup geoarrow-schema
///
/// The values of this enum are intentionally chosen to be equivalent to
/// well-known binary type identifiers.
enum GeoArrowGeometryType {
  GEOARROW_GEOMETRY_TYPE_GEOMETRY = 0,
  GEOARROW_GEOMETRY_TYPE_POINT = 1,
  GEOARROW_GEOMETRY_TYPE_LINESTRING = 2,
  GEOARROW_GEOMETRY_TYPE_POLYGON = 3,
  GEOARROW_GEOMETRY_TYPE_MULTIPOINT = 4,
  GEOARROW_GEOMETRY_TYPE_MULTILINESTRING = 5,
  GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON = 6,
  GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION = 7
};

/// \brief Dimension combinations supported by GeoArrow
/// \ingroup geoarrow-schema
enum GeoArrowDimensions {
  GEOARROW_DIMENSIONS_UNKNOWN = 0,
  GEOARROW_DIMENSIONS_XY = 1,
  GEOARROW_DIMENSIONS_XYZ = 2,
  GEOARROW_DIMENSIONS_XYM = 3,
  GEOARROW_DIMENSIONS_XYZM = 4
};

/// \brief Coordinate types supported by GeoArrow
/// \ingroup geoarrow-schema
enum GeoArrowCoordType {
  GEOARROW_COORD_TYPE_UNKNOWN = 0,
  GEOARROW_COORD_TYPE_SEPARATE = 1,
  GEOARROW_COORD_TYPE_INTERLEAVED = 2
};

/// \brief Edge types/interpolations supported by GeoArrow
/// \ingroup geoarrow-schema
enum GeoArrowEdgeType { GEOARROW_EDGE_TYPE_PLANAR, GEOARROW_EDGE_TYPE_SPHERICAL };

/// \brief Coordinate reference system types supported by GeoArrow
/// \ingroup geoarrow-schema
enum GeoArrowCrsType {
  GEOARROW_CRS_TYPE_NONE,
  GEOARROW_CRS_TYPE_UNKNOWN,
  GEOARROW_CRS_TYPE_PROJJSON
};

/// \brief Parsed view of an ArrowSchema representation of a GeoArrowType
///
/// This structure can be initialized from an ArrowSchema or a GeoArrowType.
/// It provides a structured view of memory held by an ArrowSchema or other
/// object but does not hold any memory of its own.
struct GeoArrowSchemaView {
  /// \brief The optional ArrowSchema used to populate these values
  struct ArrowSchema* schema;

  /// \brief The Arrow extension name for this type
  struct GeoArrowStringView extension_name;

  /// \brief The serialized extension metadata for this type
  ///
  /// May be NULL if there is no metadata (i.e., the JSON object representing
  /// this type would have no keys/values).
  struct GeoArrowStringView extension_metadata;

  /// \brief The GeoArrowType representing this memory layout
  enum GeoArrowType type;

  /// \brief The GeoArrowGeometryType representing this memory layout
  enum GeoArrowGeometryType geometry_type;

  /// \brief The GeoArrowDimensions representing this memory layout
  enum GeoArrowDimensions dimensions;

  /// \brief The GeoArrowCoordType representing this memory layout
  enum GeoArrowCoordType coord_type;
};

/// \brief Parsed view of GeoArrow extension metadata
struct GeoArrowMetadataView {
  /// \brief A view of the serialized metadata if this was used to populate the view
  struct GeoArrowStringView metadata;

  /// \brief The GeoArrowEdgeType represented by metadata
  enum GeoArrowEdgeType edge_type;

  /// \brief The GeoArrowEdgeType represented by metadata
  enum GeoArrowCrsType crs_type;

  /// \brief The CRS represented by metadata
  ///
  /// Because this value is a view of memory from within a JSON metadata string,
  /// it may contain the outer quotes and have escaped quotes inside it. Use
  /// GeoArrowUnescapeCrs() to sanitize this value if you need to pass it elsewhere.
  struct GeoArrowStringView crs;
};

/// \brief Union type representing a pointer to modifiable data
/// \ingroup geoarrow-builder
union GeoArrowWritableBufferViewData {
  void* data;
  char* as_char;
  uint8_t* as_uint8;
  int32_t* as_int32;
  double* as_double;
};

/// \brief A view of a modifiable buffer
/// \ingroup geoarrow-builder
struct GeoArrowWritableBufferView {
  /// \brief Pointer to the beginning of the data. May be NULL if capacity_bytes is 0.
  union GeoArrowWritableBufferViewData data;

  /// \brief The size of the buffer in bytes
  int64_t size_bytes;

  /// \brief The modifiable capacity of the buffer in bytes
  int64_t capacity_bytes;
};

/// \brief A generic view of coordinates from a GeoArrow array
/// \ingroup geoarrow-array_view
///
/// This view is capable of representing both struct and interleaved coordinates.
/// Use GEOARROW_COORD_VIEW_VALUE() to generically access an ordinate.
struct GeoArrowCoordView {
  /// \brief Pointers to the beginning of each coordinate buffer
  ///
  /// May be NULL if n_coords is 0. For interleaved coordinates, these
  /// will point to the first n_values elements of the same buffer.
  const double* values[4];

  /// \brief The number of coordinates in this view
  int64_t n_coords;

  /// \brief The number of pointers in the values array (i.e., number of dimensions)
  int32_t n_values;

  /// \brief The number of elements to advance a given value pointer to the next ordinate
  ///
  /// For interleaved coordinates, coords_stride will equal n_values; for
  /// struct coordinates, coords_stride will be 1.
  int32_t coords_stride;
};

/// \brief A generic view of a writable vector of coordinates
///
/// This view is capable of representing both struct and interleaved coordinates.
/// Use GEOARROW_COORD_VIEW_VALUE() to generically access or set an ordinate
/// from a pointer to this view.
struct GeoArrowWritableCoordView {
  /// \brief Pointers to the beginning of each coordinate buffer
  double* values[4];

  /// \brief The number of coordinates in this view
  int64_t size_coords;

  /// \brief The modifiable number of coordinates in this view
  int64_t capacity_coords;

  /// \brief The number of pointers in the values array (i.e., number of dimensions)
  int32_t n_values;

  /// \brief The number of elements to advance a given value pointer to the next ordinate
  int32_t coords_stride;
};

/// \brief Generically get or set an ordinate from a GeoArrowWritableCoordView or
/// a GeoArrowCoordView.
/// \ingroup geoarrow-array_view
#define GEOARROW_COORD_VIEW_VALUE(coords_, row_, col_) \
  (coords_)->values[(col_)][(row_) * (coords_)->coords_stride]

/// \brief A parsed view of memory from a GeoArrow-encoded array
/// \ingroup geoarrow-array_view
///
/// This definition may change to more closely match the GeoArrowWritableArrayView
/// in the future.
struct GeoArrowArrayView {
  /// \brief Type information for the array represented by this view
  struct GeoArrowSchemaView schema_view;

  /// \brief The logical offset to apply into each level of nesting
  int64_t offset[4];

  /// \brief The number of elements in each level of nesting
  int64_t length[4];

  /// \brief The validity bitmap for this array
  const uint8_t* validity_bitmap;

  /// \brief The number of offset buffers for the type represented by this array
  int32_t n_offsets;

  /// \brief Pointers to the beginning of each offset buffer
  const int32_t* offsets[3];

  /// \brief The first offset value in each offset bufer
  int32_t first_offset[3];

  /// \brief The last offset value in each offset bufer
  int32_t last_offset[3];

  /// \brief For serialized types, a pointer to the start of the data buffer
  const uint8_t* data;

  /// \brief Generic view of the coordinates in this array
  struct GeoArrowCoordView coords;
};

/// \brief Structured view of writable memory managed by the GeoArrowBuilder
/// \ingroup geoarrow-builder
struct GeoArrowWritableArrayView {
  /// \brief Type information for the array being built
  struct GeoArrowSchemaView schema_view;

  /// \brief The number of elements that have been added to this array
  int64_t length;

  /// \brief The number of buffers required to represent this type
  int64_t n_buffers;

  /// \brief The number of offset buffers for the array being built
  int32_t n_offsets;

  /// \brief Views into writable memory managed by the GeoArrowBuilder
  struct GeoArrowWritableBufferView buffers[8];

  /// \brief View of writable coordinate memory managed by the GeoArrowBuilder
  struct GeoArrowWritableCoordView coords;
};

/// \brief Builder for GeoArrow-encoded arrays
/// \ingroup geoarrow-builder
struct GeoArrowBuilder {
  /// \brief Structured view of the memory managed privately in private_data
  struct GeoArrowWritableArrayView view;

  /// \brief Implementation-specific data
  void* private_data;
};

/// \brief Visitor for an array of geometries
/// \ingroup geoarrow-visitor
///
/// A structure of function pointers and implementation-specific data used
/// to allow geometry input from an abstract source. The visitor itself
/// does not have a release callback and is not responsible for the
/// lifecycle of any of its members. The order of method calls is essentially
/// the same as the order these pieces of information would be encountered
/// when parsing well-known text or well-known binary.
///
/// Implementations should perform enough checks to ensure that they do not
/// crash if a reader calls its methods in an unexpected order; however, they
/// are free to generate non-sensical output in this case.
///
/// For example: visiting the well-known text "MULTIPOINT (0 1, 2 3)" would
/// result in the following visitor calls:
///
/// - feat_start
/// - geom_start(GEOARROW_GEOMETRY_TYPE_MULTIPOINT, GEOARROW_DIMENSIONS_XY)
/// - geom_start(GEOARROW_GEOMETRY_TYPE_POINT, GEOARROW_DIMENSIONS_XY)
/// - coords(0 1)
/// - geom_end()
/// - geom_start(GEOARROW_GEOMETRY_TYPE_POINT, GEOARROW_DIMENSIONS_XY)
/// - coords(2 3)
/// - geom_end()
/// - geom_end()
/// - feat_end()
///
/// Most visitor implementations consume the entire input; however, some
/// return early once they have all the information they need to compute
/// a value for a given feature. In this case, visitors return EAGAIN
/// and readers must pass this value back to the caller who in turn must
/// provide a call to feat_end() to finish the feature.
struct GeoArrowVisitor {
  /// \brief Called when starting to iterate over a new feature
  int (*feat_start)(struct GeoArrowVisitor* v);

  /// \brief Called after feat_start for a null_feature
  int (*null_feat)(struct GeoArrowVisitor* v);

  /// \brief Called after feat_start for a new geometry
  ///
  /// Every non-null feature will have at least one call to geom_start.
  /// Collections (including multi-geometry types) will have nested calls to geom_start.
  int (*geom_start)(struct GeoArrowVisitor* v, enum GeoArrowGeometryType geometry_type,
                    enum GeoArrowDimensions dimensions);

  /// \brief For polygon geometries, called after geom_start at the beginning of a ring
  int (*ring_start)(struct GeoArrowVisitor* v);

  /// \brief Called when a sequence of coordinates is encountered
  ///
  /// This callback may be called more than once (i.e., readers are free to chunk
  /// coordinates however they see fit). The GeoArrowCoordView may represent
  /// either interleaved of struct coordinates depending on the reader implementation.
  int (*coords)(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords);

  /// \brief For polygon geometries, called at the end of a ring
  ///
  /// Every call to ring_start must have a matching call to ring_end
  int (*ring_end)(struct GeoArrowVisitor* v);

  /// \brief Called at the end of a geometry
  ///
  /// Every call to geom_start must have a matching call to geom_end.
  int (*geom_end)(struct GeoArrowVisitor* v);

  /// \brief Called at the end of a feature, including null features
  ///
  /// Every call to feat_start must have a matching call to feat_end.
  int (*feat_end)(struct GeoArrowVisitor* v);

  /// \brief Opaque visitor-specific data
  void* private_data;

  /// \brief The error into which the reader and/or visitor can place a detailed
  /// message.
  ///
  /// When a visitor is initializing callbacks and private_data it should take care
  /// to not change the value of error. This value can be NULL.
  struct GeoArrowError* error;
};

/// \brief Generalized compute kernel
///
/// Callers are responsible for calling the release callback when finished
/// using the kernel.
struct GeoArrowKernel {
  /// \brief Called before any batches are pushed to compute the output schema
  /// based on the input schema.
  int (*start)(struct GeoArrowKernel* kernel, struct ArrowSchema* schema,
               const char* options, struct ArrowSchema* out, struct GeoArrowError* error);

  /// \brief Push a batch into the kernel
  ///
  /// Scalar kernels will populate out with the compute result; aggregate kernels
  /// will not.
  int (*push_batch)(struct GeoArrowKernel* kernel, struct ArrowArray* array,
                    struct ArrowArray* out, struct GeoArrowError* error);

  /// \brief Compute the final result
  ///
  /// For aggreate kernels, compute the result based on previous batches.
  /// In theory, aggregate kernels should allow more than one call to
  /// finish; however, this is not tested in any existing code.
  int (*finish)(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                struct GeoArrowError* error);

  /// \brief Release resources held by the kernel
  ///
  /// Implementations must set the kernel->release member to NULL.
  void (*release)(struct GeoArrowKernel* kernel);

  /// \brief Opaque, implementation-specific data
  void* private_data;
};

#ifdef __cplusplus
}
#endif

#endif

#ifndef GEOARROW_H_INCLUDED
#define GEOARROW_H_INCLUDED

#include <stdint.h>



#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup geoarrow geoarrow C library
///
/// Except where noted, objects are not thread-safe and clients should
/// take care to serialize accesses to methods.
///
/// Because this library is intended to be vendored, it provides full type
/// definitions and encourages clients to stack or statically allocate
/// where convenient.

/// \defgroup geoarrow-utility Utilities and error handling
///
/// The geoarrow C library follows the same error idioms as the nanoarrow C
/// library: GEOARROW_OK is returned on success, and a GeoArrowError is populated
/// with a null-terminated error message otherwise if there is an opportunity to
/// provide one. The provided GeoArrowError can always be NULL if a detailed message
/// is not important to the caller. Pointer output arguments are not modified unless
/// GEOARROW_OK is returned.
///
/// @{

/// \brief Return a version string in the form "major.minor.patch"
const char* GeoArrowVersion(void);

/// \brief Return an integer that can be used to compare versions sequentially
int GeoArrowVersionInt(void);

/// \brief Populate a GeoArrowError using a printf-style format string
GeoArrowErrorCode GeoArrowErrorSet(struct GeoArrowError* error, const char* fmt, ...);

/// \brief Parse a string into a double
GeoArrowErrorCode GeoArrowFromChars(const char* first, const char* last, double* out);

/// \brief Print a double to a buffer
int64_t GeoArrowPrintDouble(double f, uint32_t precision, char* result);

/// @}

/// \defgroup geoarrow-schema Data type creation and inspection
///
/// The ArrowSchema is the ABI-stable way to communicate type information using the
/// Arrow C Data interface. These functions export ArrowSchema objects or parse
/// their content into a more easily inspectable object. All unique memory layouts
/// have a GeoArrowType identifier, most of which can be decomposed into
/// GeoArrowGeometryType, GeoArrowDimensions, and GeoArrowCoordType.
///
/// In addition to memory layout, these functions provide a mechanism to serialize
/// and deserialize Arrow extension type information. The serialization format
/// is a JSON object and three keys are currently encoded: crs_type, crs, and
/// edge_type. The embedded parser is not a complete JSON parser and in some
/// circumstances will accept or transport invalid JSON without erroring.
///
/// Serializing extension type information into an ArrowSchema and parsing an
/// ArrowSchema is expensive and should be avoided where possible.
///
/// @{

/// \brief Initialize an ArrowSchema with a geoarrow storage type
GeoArrowErrorCode GeoArrowSchemaInit(struct ArrowSchema* schema, enum GeoArrowType type);

/// \brief Initialize an ArrowSchema with a geoarrow extension type
GeoArrowErrorCode GeoArrowSchemaInitExtension(struct ArrowSchema* schema,
                                              enum GeoArrowType type);

/// \brief Parse an ArrowSchema extension type into a GeoArrowSchemaView
GeoArrowErrorCode GeoArrowSchemaViewInit(struct GeoArrowSchemaView* schema_view,
                                         const struct ArrowSchema* schema,
                                         struct GeoArrowError* error);

/// \brief Parse an ArrowSchema storage type into a GeoArrowSchemaView
GeoArrowErrorCode GeoArrowSchemaViewInitFromStorage(
    struct GeoArrowSchemaView* schema_view, const struct ArrowSchema* schema,
    struct GeoArrowStringView extension_name, struct GeoArrowError* error);

/// \brief Initialize a GeoArrowSchemaView directly from a GeoArrowType identifier
GeoArrowErrorCode GeoArrowSchemaViewInitFromType(struct GeoArrowSchemaView* schema_view,
                                                 enum GeoArrowType type);

/// \brief Initialize a GeoArrowSchemaView directly from a GeoArrowType identifier
GeoArrowErrorCode GeoArrowMetadataViewInit(struct GeoArrowMetadataView* metadata_view,
                                           struct GeoArrowStringView metadata,
                                           struct GeoArrowError* error);

/// \brief Serialize parsed metadata into JSON
int64_t GeoArrowMetadataSerialize(const struct GeoArrowMetadataView* metadata_view,
                                  char* out, int64_t n);

/// \brief Update extension metadata associated with an existing ArrowSchema
GeoArrowErrorCode GeoArrowSchemaSetMetadata(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view);

/// \brief Deprecated function used for backward compatability with very early
/// versions of geoarrow
GeoArrowErrorCode GeoArrowSchemaSetMetadataDeprecated(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view);

/// \brief Update extension metadata associated with an existing ArrowSchema
/// based on the extension metadata of another
GeoArrowErrorCode GeoArrowSchemaSetMetadataFrom(struct ArrowSchema* schema,
                                                const struct ArrowSchema* schema_src);

/// \brief Unescape a coordinate reference system value
///
/// The crs member of the GeoArrowMetadataView is a view into the extension metadata;
/// however, in some cases this will be a quoted string (i.e., `"EPSG:4326"`) and in
/// others it will be a JSON object (i.e., PROJJSON like
/// `{"some key": "some value", ..}`). When passing this string elsewhere, you will
/// almost always want the quoted value to be unescaped (i.e., the JSON string value),
/// but the JSON object to remain as-is. GeoArrowUnescapeCrs() performs this logic
/// based on the value of the first character.
int64_t GeoArrowUnescapeCrs(struct GeoArrowStringView crs, char* out, int64_t n);

/// @}

/// \defgroup geoarrow-array_view Array inspection
///
/// The GeoArrowArrayView is the primary means by which an ArrowArray of a
/// valid type can be inspected. The GeoArrowArrayView is intended to be
/// initialized once for a given type and re-used for multiple arrays
/// (e.g., in a stream).
///
/// @{

/// \brief Initialize a GeoArrowArrayView from a GeoArrowType identifier
GeoArrowErrorCode GeoArrowArrayViewInitFromType(struct GeoArrowArrayView* array_view,
                                                enum GeoArrowType type);

/// \brief Initialize a GeoArrowArrayView from an ArrowSchema
GeoArrowErrorCode GeoArrowArrayViewInitFromSchema(struct GeoArrowArrayView* array_view,
                                                  const struct ArrowSchema* schema,
                                                  struct GeoArrowError* error);

/// \brief Populate the members of the GeoArrowArrayView from an ArrowArray
GeoArrowErrorCode GeoArrowArrayViewSetArray(struct GeoArrowArrayView* array_view,
                                            const struct ArrowArray* array,
                                            struct GeoArrowError* error);

/// @}

/// \defgroup geoarrow-builder Array creation
///
/// The GeoArrowBuilder supports creating GeoArrow-encoded arrays. There are
/// three ways to do so:
///
/// - Build the individual buffers yourself and transfer ownership to the
///   array for each using GeoArrowBuilderSetOwnedBuffer()
/// - Append the appropriate values to each buffer in-place using
///   GeoArrowBuilderAppendBuffer()
/// - Use GeoArrowBuilderInitVisitor() and let the visitor build the buffers
///   for you.
///
/// For all methods you can re-use the builder object for multiple batches
/// and call GeoArrowBuilderFinish() multiple times. You should
/// use the same mechanism for building an array when reusing a builder
/// object.
///
/// The GeoArrowBuilder models GeoArrow arrays as a sequence of buffers numbered
/// from the outer array inwards. The 0th buffer is always the validity buffer
/// and can be omitted for arrays that contain no null features. This is followed
/// by between 0 (point) and 3 (multipolygon) int32 offset buffers and between
/// 1 (interleaved) and 4 (xyzm struct) double buffers representing coordinate
/// values. The GeoArrowBuilder omits validity buffers for inner arrays since
/// the GeoArrow specification states that these arrays must contain zero nulls.
///
/// @{

/// \brief Initialize memory for a GeoArrowBuilder based on a GeoArrowType identifier
GeoArrowErrorCode GeoArrowBuilderInitFromType(struct GeoArrowBuilder* builder,
                                              enum GeoArrowType type);

/// \brief Initialize memory for a GeoArrowBuilder based on an ArrowSchema
GeoArrowErrorCode GeoArrowBuilderInitFromSchema(struct GeoArrowBuilder* builder,
                                                const struct ArrowSchema* schema,
                                                struct GeoArrowError* error);

/// \brief Reserve additional space for a buffer in a GeoArrowBuilder
GeoArrowErrorCode GeoArrowBuilderReserveBuffer(struct GeoArrowBuilder* builder, int64_t i,
                                               int64_t additional_size_bytes);

/// \brief Append data to a buffer in a GeoArrowBuilder without checking if a reserve
/// is needed
static inline void GeoArrowBuilderAppendBufferUnsafe(struct GeoArrowBuilder* builder,
                                                     int64_t i,
                                                     struct GeoArrowBufferView value);

/// \brief Append data to a buffer in a GeoArrowBuilder
static inline GeoArrowErrorCode GeoArrowBuilderAppendBuffer(
    struct GeoArrowBuilder* builder, int64_t i, struct GeoArrowBufferView value);

/// \brief Replace a buffer with one whose lifecycle is externally managed.
GeoArrowErrorCode GeoArrowBuilderSetOwnedBuffer(
    struct GeoArrowBuilder* builder, int64_t i, struct GeoArrowBufferView value,
    void (*custom_free)(uint8_t* ptr, int64_t size, void* private_data),
    void* private_data);

/// \brief Finish an ArrowArray containing the built input
///
/// This function can be called more than once to support multiple batches.
GeoArrowErrorCode GeoArrowBuilderFinish(struct GeoArrowBuilder* builder,
                                        struct ArrowArray* array,
                                        struct GeoArrowError* error);

/// \brief Free resources held by a GeoArrowBuilder
void GeoArrowBuilderReset(struct GeoArrowBuilder* builder);

/// @}

/// \defgroup geoarrow-kernels Transform Arrays
///
/// The GeoArrow C library provides limited support for transforming arrays.
/// Notably, it provides support for parsing WKT and WKB into GeoArrow
/// native encoding and serializing GeoArrow arrays to WKT and/or WKB.
///
/// The GeoArrowKernel is a generalization of the compute operations available
/// in this build of the GeoArrow C library. Two types of kernels are implemented:
/// scalar and aggregate. Scalar kernels always output an `ArrowArray` of the same
/// length as the input from `push_batch()` and do not output an `ArrowArray` from
/// `finish()`; aggregate kernels do not output an `ArrowArray` from `push_batch()`
/// and output a single `ArrowArray` from `finish()` with no constraint on the length
/// of the array that is produced. For both kernel types, the `ArrowSchema` of the
/// output is returned by the `start()` method, where `options` (serialized in the
/// same form as the `ArrowSchema` metadata member) can also be passed. Current
/// implementations do not validate options except to the extent needed to avoid
/// a crash.
///
/// This is intended to minimize the number of patterns needed in wrapper code rather than
/// be a perfect abstraction of a compute function. Similarly, these kernels are optimized
/// for type coverage rather than performance.
///
/// - void: Scalar kernel that outputs a null array of the same length as the input
///   for each batch.
/// - void_agg: Aggregate kernel that outputs a null array of length 1 for any number
///   of inputs.
/// - visit_void_agg: Aggregate kernel that visits every coordinate of every feature
///   of the input, outputting a null array of length 1 for any number of inputs.
///   This is useful for validating well-known text and well-known binary as it will
///   error for input that cannot be visited completely.
/// - as_wkt: Scalar kernel that outputs the well-known text version of the input
///   as faithfully as possible (including transferring metadata from the input).
///   Arrays with valid `GeoArrowType`s are supported.
/// - as_wkb: Scalar kernel that outputs the well-known binary version of the input
///   as faithfully as possible (including transferring metadata from the input).
///   Arrays with valid `GeoArrowType`s are supported.
/// - as_geoarrow: Scalar kernel that outputs the GeoArrow version of the input
///   as faithfully as possible (including transferring metadata from the input).
///   Arrays with valid `GeoArrowType`s are supported. The type of the output is
///   controlled by the `type` option, specified as a `GeoArrowType` cast to integer.
/// - format_wkt: A variation on as_wkt that supports options `precision`
///   and `max_element_size_bytes`. This kernel is lazy and does not visit an entire
///   feature beyond that required for `max_element_size_bytes`.
/// - unique_geometry_types_agg: An aggregate kernel that collects unique geometry
///   types in the input. The output is a single int32 array of ISO WKB type codes.
/// - box: A scalar kernel that returns the 2-dimensional bounding box by feature.
///   the output bounding box is represented as a struct array with column order
///   xmin, xmax, ymin, ymax. Null features are recorded as a null item in the
///   output; empty features are recorded as Inf, -Inf, Inf, -Inf.
/// - box_agg: An aggregate kernel that returns the 2-dimensional bounding box
///   containing all features of the input in the same form as the box kernel.
///   the result is always length one and is never null. For the purposes of this
///   kernel, nulls are treated as empty.
///
/// @{

/// \brief Initialize memory for a GeoArrowKernel
///
/// If GEOARROW_OK is returned, the caller is responsible for calling the embedded
/// release callback to free any resources that were allocated.
GeoArrowErrorCode GeoArrowKernelInit(struct GeoArrowKernel* kernel, const char* name,
                                     const char* options);

/// @}

/// \defgroup geoarrow-visitor Low-level reader/visitor interfaces
///
/// The GeoArrow specification defines memory layouts for many types.
/// Whereas it is more performant to write dedicated conversions
/// between each source and destination type, the number of conversions
/// required prohibits a compact and maintainable general-purpose
/// library.
///
/// Instead, we define the GeoArrowVisitor and provide a means
/// by which to "visit" each feature in an array of geometries for every
/// supported type. Conversely, we provide a GeoArrowVisitor implementation
/// to create arrays of each supported type upon visitation of an arbitrary
/// source. This design also facilitates reusing the readers and writers
/// provided here by other libraries.
///
/// @{

/// \brief Initialize a GeoArrowVisitor with a visitor that does nothing
void GeoArrowVisitorInitVoid(struct GeoArrowVisitor* v);

/// \brief Populate a GeoArrowVisitor pointing to a GeoArrowBuilder
GeoArrowErrorCode GeoArrowBuilderInitVisitor(struct GeoArrowBuilder* builder,
                                             struct GeoArrowVisitor* v);

/// \brief Visit the features of a GeoArrowArrayView
///
/// The caller must have initialized the GeoArrowVisitor with the appropriate
/// writer before calling this function.
GeoArrowErrorCode GeoArrowArrayViewVisit(const struct GeoArrowArrayView* array_view,
                                         int64_t offset, int64_t length,
                                         struct GeoArrowVisitor* v);

/// \brief Well-known text writer
///
/// This struct also contains options for well-known text serialization.
/// These options can be modified from the defaults after
/// GeoArrowWKTWriterInit() and before GeoArrowWKTWriterInitVisitor().
struct GeoArrowWKTWriter {
  /// \brief The number of significant digits to include in the output (default: 16)
  int precision;

  /// \brief Set to 0 to use the verbose (but still technically valid) MULTIPOINT
  /// representation (i.e., MULTIPOINT((0 1), (2 3))).
  int use_flat_multipoint;

  /// \brief Constrain the maximum size of each element in the returned array
  ///
  /// Use -1 to denote an unlimited size for each element. When the limit is
  /// reached or shortly after, the called handler method will return EAGAIN,
  /// after which it is safe to call feat_end to end the feature. This ensures
  /// that a finite amount of input is consumed if this elemtn is set.
  int64_t max_element_size_bytes;

  /// \brief Implementation-specific details
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowWKTWriter
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKTWriterReset().
GeoArrowErrorCode GeoArrowWKTWriterInit(struct GeoArrowWKTWriter* writer);

/// \brief Populate a GeoArrowVisitor pointing to this writer
void GeoArrowWKTWriterInitVisitor(struct GeoArrowWKTWriter* writer,
                                  struct GeoArrowVisitor* v);

/// \brief Finish an ArrowArray containing elements from the visited input
///
/// This function can be called more than once to support multiple batches.
GeoArrowErrorCode GeoArrowWKTWriterFinish(struct GeoArrowWKTWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error);

/// \brief Free resources held by a GeoArrowWKTWriter
void GeoArrowWKTWriterReset(struct GeoArrowWKTWriter* writer);

/// \brief Well-known text reader
struct GeoArrowWKTReader {
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowWKTReader
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKTReaderReset().
GeoArrowErrorCode GeoArrowWKTReaderInit(struct GeoArrowWKTReader* reader);

/// \brief Visit well-known text
///
/// The caller must have initialized the GeoArrowVisitor with the appropriate
/// writer before calling this function.
GeoArrowErrorCode GeoArrowWKTReaderVisit(struct GeoArrowWKTReader* reader,
                                         struct GeoArrowStringView s,
                                         struct GeoArrowVisitor* v);

/// \brief Free resources held by a GeoArrowWKTReader
void GeoArrowWKTReaderReset(struct GeoArrowWKTReader* reader);

/// \brief ISO well-known binary writer
struct GeoArrowWKBWriter {
  /// \brief Implmentation-specific data
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowWKBWriter
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKBWriterReset().
GeoArrowErrorCode GeoArrowWKBWriterInit(struct GeoArrowWKBWriter* writer);

/// \brief Populate a GeoArrowVisitor pointing to this writer
void GeoArrowWKBWriterInitVisitor(struct GeoArrowWKBWriter* writer,
                                  struct GeoArrowVisitor* v);

/// \brief Finish an ArrowArray containing elements from the visited input
///
/// This function can be called more than once to support multiple batches.
GeoArrowErrorCode GeoArrowWKBWriterFinish(struct GeoArrowWKBWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error);

/// \brief Free resources held by a GeoArrowWKBWriter
void GeoArrowWKBWriterReset(struct GeoArrowWKBWriter* writer);

/// \brief Well-known binary (ISO or EWKB) reader
struct GeoArrowWKBReader {
  /// \brief Implmentation-specific data
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowWKBReader
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKBReaderReset().
GeoArrowErrorCode GeoArrowWKBReaderInit(struct GeoArrowWKBReader* reader);

/// \brief Visit well-known binary
///
/// The caller must have initialized the GeoArrowVisitor with the appropriate
/// writer before calling this function.
GeoArrowErrorCode GeoArrowWKBReaderVisit(struct GeoArrowWKBReader* reader,
                                         struct GeoArrowBufferView src,
                                         struct GeoArrowVisitor* v);

/// \brief Free resources held by a GeoArrowWKBWriter
void GeoArrowWKBReaderReset(struct GeoArrowWKBReader* reader);

/// \brief Array reader for any geoarrow extension array
struct GeoArrowArrayReader {
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowArrayReader
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowArrayReaderReset().
GeoArrowErrorCode GeoArrowArrayReaderInit(struct GeoArrowArrayReader* reader);

/// \brief Visit a GeoArrowArray
///
/// The caller must have initialized the GeoArrowVisitor with the appropriate
/// writer before calling this function.
GeoArrowErrorCode GeoArrowArrayReaderVisit(struct GeoArrowArrayReader* reader,
                                           const struct GeoArrowArrayView* array_view,
                                           int64_t offset, int64_t length,
                                           struct GeoArrowVisitor* v);

/// \brief Free resources held by a GeoArrowArrayReader
void GeoArrowArrayReaderReset(struct GeoArrowArrayReader* reader);

/// \brief Generc GeoArrow array writer
struct GeoArrowArrayWriter {
  void* private_data;
};

/// \brief Initialize the memory of a GeoArrowArrayWriter from a GeoArrowType
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKTWriterReset().
GeoArrowErrorCode GeoArrowArrayWriterInitFromType(struct GeoArrowArrayWriter* writer,
                                                  enum GeoArrowType type);

/// \brief Initialize the memory of a GeoArrowArrayWriter from an ArrowSchema
///
/// If GEOARROW_OK is returned, the caller is responsible for calling
/// GeoArrowWKTWriterReset().
GeoArrowErrorCode GeoArrowArrayWriterInitFromSchema(struct GeoArrowArrayWriter* writer,
                                                    const struct ArrowSchema* schema);

/// \brief Populate a GeoArrowVisitor pointing to this writer
GeoArrowErrorCode GeoArrowArrayWriterInitVisitor(struct GeoArrowArrayWriter* writer,
                                                 struct GeoArrowVisitor* v);

/// \brief Finish an ArrowArray containing elements from the visited input
///
/// This function can be called more than once to support multiple batches.
GeoArrowErrorCode GeoArrowArrayWriterFinish(struct GeoArrowArrayWriter* writer,
                                            struct ArrowArray* array,
                                            struct GeoArrowError* error);

/// \brief Free resources held by a GeoArrowArrayWriter
void GeoArrowArrayWriterReset(struct GeoArrowArrayWriter* writer);

/// @}

#ifdef __cplusplus
}
#endif



#endif

#ifndef GEOARROW_GEOARROW_TYPES_INLINE_H_INCLUDED
#define GEOARROW_GEOARROW_TYPES_INLINE_H_INCLUDED

#include <stddef.h>
#include <string.h>



#ifdef __cplusplus
extern "C" {
#endif

/// \brief Extract GeometryType from a GeoArrowType
/// \ingroup geoarrow-schema
static inline enum GeoArrowGeometryType GeoArrowGeometryTypeFromType(
    enum GeoArrowType type) {
  switch (type) {
    case GEOARROW_TYPE_UNINITIALIZED:
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_LARGE_WKT:
      return GEOARROW_GEOMETRY_TYPE_GEOMETRY;

    default:
      break;
  }

  int type_int = type;

  if (type_int >= GEOARROW_TYPE_INTERLEAVED_POINT) {
    type_int -= 10000;
  }

  if (type_int >= 4000) {
    type_int -= 4000;
  } else if (type_int >= 3000) {
    type_int -= 3000;
  } else if (type_int >= 2000) {
    type_int -= 2000;
  } else if (type_int >= 1000) {
    type_int -= 1000;
  }

  if (type_int > 6 || type_int < 1) {
    return GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  } else {
    return (enum GeoArrowGeometryType)type_int;
  }
}

/// \brief Returns the Arrow extension name for a given GeoArrowType
/// \ingroup geoarrow-schema
static inline const char* GeoArrowExtensionNameFromType(enum GeoArrowType type) {
  switch (type) {
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_LARGE_WKB:
      return "geoarrow.wkb";
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_LARGE_WKT:
      return "geoarrow.wkt";

    default:
      break;
  }

  int geometry_type = GeoArrowGeometryTypeFromType(type);
  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      return "geoarrow.point";
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      return "geoarrow.linestring";
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      return "geoarrow.polygon";
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      return "geoarrow.multipoint";
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      return "geoarrow.multilinestring";
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      return "geoarrow.multipolygon";
    default:
      return NULL;
  }
}

/// \brief Extract GeoArrowDimensions from a GeoArrowType
/// \ingroup geoarrow-schema
static inline enum GeoArrowDimensions GeoArrowDimensionsFromType(enum GeoArrowType type) {
  switch (type) {
    case GEOARROW_TYPE_UNINITIALIZED:
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_LARGE_WKT:
      return GEOARROW_DIMENSIONS_UNKNOWN;

    default:
      break;
  }

  int geometry_type = GeoArrowGeometryTypeFromType(type);
  int type_int = type;
  type_int -= geometry_type;
  if (type_int > 5000) {
    type_int -= 10000;
  }

  switch (type_int) {
    case 0:
      return GEOARROW_DIMENSIONS_XY;
    case 1000:
      return GEOARROW_DIMENSIONS_XYZ;
    case 2000:
      return GEOARROW_DIMENSIONS_XYM;
    case 3000:
      return GEOARROW_DIMENSIONS_XYZM;
    default:
      return GEOARROW_DIMENSIONS_UNKNOWN;
  }
}

/// \brief Extract GeoArrowCoordType from a GeoArrowType
/// \ingroup geoarrow-schema
static inline enum GeoArrowCoordType GeoArrowCoordTypeFromType(enum GeoArrowType type) {
  if (type >= GEOARROW_TYPE_WKB) {
    return GEOARROW_COORD_TYPE_UNKNOWN;
  } else if (type >= GEOARROW_TYPE_INTERLEAVED_POINT) {
    return GEOARROW_COORD_TYPE_INTERLEAVED;
  } else if (type >= GEOARROW_TYPE_POINT) {
    return GEOARROW_COORD_TYPE_SEPARATE;
  } else {
    return GEOARROW_COORD_TYPE_UNKNOWN;
  }
}

/// \brief Construct a GeometryType from a GeoArrowGeometryType, GeoArrowDimensions,
/// and GeoArrowCoordType.
/// \ingroup geoarrow-schema
static inline enum GeoArrowType GeoArrowMakeType(enum GeoArrowGeometryType geometry_type,
                                                 enum GeoArrowDimensions dimensions,
                                                 enum GeoArrowCoordType coord_type) {
  if (geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRY) {
    return GEOARROW_TYPE_UNINITIALIZED;
  } else if (dimensions == GEOARROW_DIMENSIONS_UNKNOWN) {
    return GEOARROW_TYPE_UNINITIALIZED;
  } else if (coord_type == GEOARROW_COORD_TYPE_UNKNOWN) {
    return GEOARROW_TYPE_UNINITIALIZED;
  }

  int type_int = (dimensions - 1) * 1000 + (coord_type - 1) * 10000 + geometry_type;
  return (enum GeoArrowType)type_int;
}

/// \brief The all-caps string associated with a given GeometryType (e.g., POINT)
/// \ingroup geoarrow-schema
static inline const char* GeoArrowGeometryTypeString(
    enum GeoArrowGeometryType geometry_type) {
  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      return "POINT";
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      return "LINESTRING";
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      return "POLYGON";
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      return "MULTIPOINT";
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      return "MULTILINESTRING";
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      return "MULTIPOLYGON";
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      return "GEOMETRYCOLLECTION";
    default:
      return NULL;
  }
}

// Such that kNumOffsets[geometry_type] gives the right answer
static const int _GeoArrowkNumOffsets[] = {-1, 0, 1, 2, 1, 2, 3, -1};

// Such that kNumDimensions[dimensions] gives the right answer
static const int _GeoArrowkNumDimensions[] = {-1, 2, 3, 3, 4};

static inline int GeoArrowBuilderBufferCheck(struct GeoArrowBuilder* builder, int64_t i,
                                             int64_t additional_size_bytes) {
  return builder->view.buffers[i].capacity_bytes >=
         (builder->view.buffers[i].size_bytes + additional_size_bytes);
}

static inline void GeoArrowBuilderAppendBufferUnsafe(struct GeoArrowBuilder* builder,
                                                     int64_t i,
                                                     struct GeoArrowBufferView value) {
  struct GeoArrowWritableBufferView* buffer = builder->view.buffers + i;
  memcpy(buffer->data.as_uint8 + buffer->size_bytes, value.data, value.size_bytes);
  buffer->size_bytes += value.size_bytes;
}

// This could probably be or use a lookup table at some point
static inline void GeoArrowMapDimensions(enum GeoArrowDimensions src_dim,
                                         enum GeoArrowDimensions dst_dim, int* dim_map) {
  dim_map[0] = 0;
  dim_map[1] = 1;
  dim_map[2] = -1;
  dim_map[3] = -1;

  switch (dst_dim) {
    case GEOARROW_DIMENSIONS_XYM:
      switch (src_dim) {
        case GEOARROW_DIMENSIONS_XYM:
          dim_map[2] = 2;
          break;
        case GEOARROW_DIMENSIONS_XYZM:
          dim_map[2] = 3;
          break;
        default:
          break;
      }
      break;

    case GEOARROW_DIMENSIONS_XYZ:
      switch (src_dim) {
        case GEOARROW_DIMENSIONS_XYZ:
        case GEOARROW_DIMENSIONS_XYZM:
          dim_map[2] = 2;
          break;
        default:
          break;
      }
      break;

    case GEOARROW_DIMENSIONS_XYZM:
      switch (src_dim) {
        case GEOARROW_DIMENSIONS_XYZ:
          dim_map[2] = 2;
          break;
        case GEOARROW_DIMENSIONS_XYM:
          dim_map[3] = 2;
          break;
        case GEOARROW_DIMENSIONS_XYZM:
          dim_map[2] = 2;
          dim_map[3] = 3;
          break;
        default:
          break;
      }
      break;

    default:
      break;
  }
}

// Four little-endian NANs
static uint8_t _GeoArrowkEmptyPointCoords[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};

// Copies coordinates from one view to another keeping dimensions the same.
// This function fills dimensions in dst but not in src with NAN; dimensions
// in src but not in dst are dropped. This is useful for generic copying of
// small sequences (e.g., the builder) but shouldn't be used when there is some
// prior knowledge of the coordinate type.
static inline void GeoArrowCoordViewCopy(const struct GeoArrowCoordView* src,
                                         enum GeoArrowDimensions src_dim,
                                         int64_t src_offset,
                                         struct GeoArrowWritableCoordView* dst,
                                         enum GeoArrowDimensions dst_dim,
                                         int64_t dst_offset, int64_t n) {
  // Copy the XYs
  for (int64_t i = 0; i < n; i++) {
    GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 0) =
        GEOARROW_COORD_VIEW_VALUE(src, src_offset + i, 0);
    GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 1) =
        GEOARROW_COORD_VIEW_VALUE(src, src_offset + i, 1);
  }

  if (dst->n_values == 2) {
    return;
  }

  int dst_dim_map[4];
  GeoArrowMapDimensions(src_dim, dst_dim, dst_dim_map);

  if (dst_dim_map[2] == -1) {
    for (int64_t i = 0; i < n; i++) {
      memcpy(&(GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 2)),
             _GeoArrowkEmptyPointCoords, sizeof(double));
    }
  } else {
    for (int64_t i = 0; i < n; i++) {
      GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 2) =
          GEOARROW_COORD_VIEW_VALUE(src, src_offset + i, dst_dim_map[2]);
    }
  }

  if (dst->n_values == 3) {
    return;
  }

  if (dst_dim_map[3] == -1) {
    for (int64_t i = 0; i < n; i++) {
      memcpy(&(GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 3)),
             _GeoArrowkEmptyPointCoords, sizeof(double));
    }
  } else {
    for (int64_t i = 0; i < n; i++) {
      GEOARROW_COORD_VIEW_VALUE(dst, dst_offset + i, 3) =
          GEOARROW_COORD_VIEW_VALUE(src, src_offset + i, dst_dim_map[3]);
    }
  }
}

static inline int GeoArrowBuilderCoordsCheck(struct GeoArrowBuilder* builder,
                                             int64_t additional_size_coords) {
  return builder->view.coords.capacity_coords >=
         (builder->view.coords.size_coords + additional_size_coords);
}

static inline void GeoArrowBuilderCoordsAppendUnsafe(
    struct GeoArrowBuilder* builder, const struct GeoArrowCoordView* coords,
    enum GeoArrowDimensions dimensions, int64_t offset, int64_t n) {
  GeoArrowCoordViewCopy(coords, dimensions, offset, &builder->view.coords,
                        builder->view.schema_view.dimensions,
                        builder->view.coords.size_coords, n);
  builder->view.coords.size_coords += n;
}

static inline int GeoArrowBuilderOffsetCheck(struct GeoArrowBuilder* builder, int32_t i,
                                             int64_t additional_size_elements) {
  return (builder->view.buffers[i + 1].capacity_bytes / sizeof(int32_t)) >=
         ((builder->view.buffers[i + 1].size_bytes / sizeof(int32_t)) +
          additional_size_elements);
}

static inline void GeoArrowBuilderOffsetAppendUnsafe(struct GeoArrowBuilder* builder,
                                                     int32_t i, int32_t* data,
                                                     int64_t additional_size_elements) {
  struct GeoArrowWritableBufferView* buf = &builder->view.buffers[i + 1];
  memcpy(buf->data.as_uint8 + buf->size_bytes, data,
         additional_size_elements * sizeof(int32_t));
  buf->size_bytes += additional_size_elements * sizeof(int32_t);
}

struct _GeoArrowFindBufferResult {
  struct ArrowArray* array;
  int level;
  int64_t i;
};

static inline int64_t _GeoArrowArrayFindBuffer(struct ArrowArray* array,
                                               struct _GeoArrowFindBufferResult* res,
                                               int64_t i, int level, int skip_first) {
  int64_t total_buffers = (array->n_buffers - skip_first);
  if (i < total_buffers) {
    res->array = array;
    res->i = i + skip_first;
    res->level = level;
    return total_buffers;
  }

  i -= total_buffers;

  for (int64_t child_id = 0; child_id < array->n_children; child_id++) {
    int64_t child_buffers =
        _GeoArrowArrayFindBuffer(array->children[child_id], res, i, level + 1, 1);
    total_buffers += child_buffers;
    if (i < child_buffers) {
      return total_buffers;
    }
    i -= child_buffers;
  }

  return total_buffers;
}

static inline GeoArrowErrorCode GeoArrowBuilderAppendBuffer(
    struct GeoArrowBuilder* builder, int64_t i, struct GeoArrowBufferView value) {
  if (!GeoArrowBuilderBufferCheck(builder, i, value.size_bytes)) {
    int result = GeoArrowBuilderReserveBuffer(builder, i, value.size_bytes);
    if (result != GEOARROW_OK) {
      return result;
    }
  }

  GeoArrowBuilderAppendBufferUnsafe(builder, i, value);
  return GEOARROW_OK;
}

static inline GeoArrowErrorCode GeoArrowBuilderCoordsReserve(
    struct GeoArrowBuilder* builder, int64_t additional_size_coords) {
  if (GeoArrowBuilderCoordsCheck(builder, additional_size_coords)) {
    return GEOARROW_OK;
  }

  struct GeoArrowWritableCoordView* writable_view = &builder->view.coords;
  int result;
  int64_t last_buffer = builder->view.n_buffers - 1;
  int n_values = writable_view->n_values;

  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      // Sync the coord view size back to the buffer size
      builder->view.buffers[last_buffer].size_bytes =
          writable_view->size_coords * sizeof(double) * n_values;

      // Use the normal reserve
      result = GeoArrowBuilderReserveBuffer(
          builder, last_buffer, additional_size_coords * sizeof(double) * n_values);
      if (result != GEOARROW_OK) {
        return result;
      }

      // Sync the capacity and pointers back to the writable view
      writable_view->capacity_coords =
          builder->view.buffers[last_buffer].capacity_bytes / sizeof(double) / n_values;
      for (int i = 0; i < n_values; i++) {
        writable_view->values[i] = builder->view.buffers[last_buffer].data.as_double + i;
      }

      return GEOARROW_OK;

    case GEOARROW_COORD_TYPE_SEPARATE:
      for (int64_t i = last_buffer - n_values + 1; i <= last_buffer; i++) {
        // Sync the coord view size back to the buffer size
        builder->view.buffers[i].size_bytes = writable_view->size_coords * sizeof(double);

        // Use the normal reserve
        result = GeoArrowBuilderReserveBuffer(builder, i,
                                              additional_size_coords * sizeof(double));
        if (result != GEOARROW_OK) {
          return result;
        }
      }

      // Sync the capacity and pointers back to the writable view
      writable_view->capacity_coords =
          builder->view.buffers[last_buffer].capacity_bytes / sizeof(double);
      for (int i = 0; i < n_values; i++) {
        writable_view->values[i] =
            builder->view.buffers[last_buffer - n_values + 1 + i].data.as_double;
      }

      return GEOARROW_OK;
    default:
      // Beacuse there is no include <errno.h> here yet
      return -1;
  }
}

static inline GeoArrowErrorCode GeoArrowBuilderCoordsAppend(
    struct GeoArrowBuilder* builder, const struct GeoArrowCoordView* coords,
    enum GeoArrowDimensions dimensions, int64_t offset, int64_t n) {
  if (!GeoArrowBuilderCoordsCheck(builder, n)) {
    int result = GeoArrowBuilderCoordsReserve(builder, n);
    if (result != GEOARROW_OK) {
      return result;
    }
  }

  GeoArrowBuilderCoordsAppendUnsafe(builder, coords, dimensions, offset, n);
  return GEOARROW_OK;
}

static inline GeoArrowErrorCode GeoArrowBuilderOffsetReserve(
    struct GeoArrowBuilder* builder, int32_t i, int64_t additional_size_elements) {
  if (GeoArrowBuilderOffsetCheck(builder, i, additional_size_elements)) {
    return GEOARROW_OK;
  }

  return GeoArrowBuilderReserveBuffer(builder, i + 1,
                                      additional_size_elements * sizeof(int32_t));
}

static inline GeoArrowErrorCode GeoArrowBuilderOffsetAppend(
    struct GeoArrowBuilder* builder, int32_t i, int32_t* data,
    int64_t additional_size_elements) {
  if (!GeoArrowBuilderOffsetCheck(builder, i, additional_size_elements)) {
    int result = GeoArrowBuilderOffsetReserve(builder, i, additional_size_elements);
    if (result != GEOARROW_OK) {
      return result;
    }
  }

  GeoArrowBuilderOffsetAppendUnsafe(builder, i, data, additional_size_elements);
  return GEOARROW_OK;
}

#ifdef __cplusplus
}
#endif

#endif
