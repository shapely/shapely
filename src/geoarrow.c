#include "geoarrow.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_BUILD_ID_H_INCLUDED
#define NANOARROW_BUILD_ID_H_INCLUDED

#define NANOARROW_VERSION_MAJOR 0
#define NANOARROW_VERSION_MINOR 4
#define NANOARROW_VERSION_PATCH 0
#define NANOARROW_VERSION "0.4.0-SNAPSHOT"

#define NANOARROW_VERSION_INT                                        \
  (NANOARROW_VERSION_MAJOR * 10000 + NANOARROW_VERSION_MINOR * 100 + \
   NANOARROW_VERSION_PATCH)

// When testing we use nanoarrow.h, but geoarrow_config.h won't exist in bundled
// mode. In the tests we just have to make sure geoarrow.h is always included first.
#if !defined(GEOARROW_CONFIG_H_INCLUDED)

#endif

#endif
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_NANOARROW_TYPES_H_INCLUDED
#define NANOARROW_NANOARROW_TYPES_H_INCLUDED

#include <stdint.h>
#include <string.h>

#if defined(NANOARROW_DEBUG) && !defined(NANOARROW_PRINT_AND_DIE)
#include <stdio.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

/// \defgroup nanoarrow-arrow-cdata Arrow C Data interface
///
/// The Arrow C Data (https://arrow.apache.org/docs/format/CDataInterface.html)
/// and Arrow C Stream (https://arrow.apache.org/docs/format/CStreamInterface.html)
/// interfaces are part of the
/// Arrow Columnar Format specification
/// (https://arrow.apache.org/docs/format/Columnar.html). See the Arrow documentation for
/// documentation of these structures.
///
/// @{

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

/// \brief Move the contents of src into dst and set src->release to NULL
static inline void ArrowSchemaMove(struct ArrowSchema* src, struct ArrowSchema* dst) {
  memcpy(dst, src, sizeof(struct ArrowSchema));
  src->release = NULL;
}

/// \brief Move the contents of src into dst and set src->release to NULL
static inline void ArrowArrayMove(struct ArrowArray* src, struct ArrowArray* dst) {
  memcpy(dst, src, sizeof(struct ArrowArray));
  src->release = NULL;
}

/// \brief Move the contents of src into dst and set src->release to NULL
static inline void ArrowArrayStreamMove(struct ArrowArrayStream* src,
                                        struct ArrowArrayStream* dst) {
  memcpy(dst, src, sizeof(struct ArrowArrayStream));
  src->release = NULL;
}

/// @}

// Utility macros
#define _NANOARROW_CONCAT(x, y) x##y
#define _NANOARROW_MAKE_NAME(x, y) _NANOARROW_CONCAT(x, y)

#define _NANOARROW_RETURN_NOT_OK_IMPL(NAME, EXPR) \
  do {                                            \
    const int NAME = (EXPR);                      \
    if (NAME) return NAME;                        \
  } while (0)

#define _NANOARROW_CHECK_RANGE(x_, min_, max_) \
  NANOARROW_RETURN_NOT_OK((x_ >= min_ && x_ <= max_) ? NANOARROW_OK : EINVAL)

#define _NANOARROW_CHECK_UPPER_LIMIT(x_, max_) \
  NANOARROW_RETURN_NOT_OK((x_ <= max_) ? NANOARROW_OK : EINVAL)

#if defined(NANOARROW_DEBUG)
#define _NANOARROW_RETURN_NOT_OK_WITH_ERROR_IMPL(NAME, EXPR, ERROR_PTR_EXPR, EXPR_STR) \
  do {                                                                                 \
    const int NAME = (EXPR);                                                           \
    if (NAME) {                                                                        \
      ArrowErrorSet((ERROR_PTR_EXPR), "%s failed with errno %d\n* %s:%d", EXPR_STR,    \
                    NAME, __FILE__, __LINE__);                                         \
      return NAME;                                                                     \
    }                                                                                  \
  } while (0)
#else
#define _NANOARROW_RETURN_NOT_OK_WITH_ERROR_IMPL(NAME, EXPR, ERROR_PTR_EXPR, EXPR_STR) \
  do {                                                                                 \
    const int NAME = (EXPR);                                                           \
    if (NAME) {                                                                        \
      ArrowErrorSet((ERROR_PTR_EXPR), "%s failed with errno %d", EXPR_STR, NAME);      \
      return NAME;                                                                     \
    }                                                                                  \
  } while (0)
#endif

/// \brief Return code for success.
/// \ingroup nanoarrow-errors
#define NANOARROW_OK 0

/// \brief Represents an errno-compatible error code
/// \ingroup nanoarrow-errors
typedef int ArrowErrorCode;

/// \brief Check the result of an expression and return it if not NANOARROW_OK
/// \ingroup nanoarrow-errors
#define NANOARROW_RETURN_NOT_OK(EXPR) \
  _NANOARROW_RETURN_NOT_OK_IMPL(_NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR)

/// \brief Check the result of an expression and return it if not NANOARROW_OK,
/// adding an auto-generated message to an ArrowError.
/// \ingroup nanoarrow-errors
///
/// This macro is used to ensure that functions that accept an ArrowError
/// as input always set its message when returning an error code (e.g., when calling
/// a nanoarrow function that does *not* accept ArrowError).
#define NANOARROW_RETURN_NOT_OK_WITH_ERROR(EXPR, ERROR_EXPR) \
  _NANOARROW_RETURN_NOT_OK_WITH_ERROR_IMPL(                  \
      _NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR, ERROR_EXPR, #EXPR)

#if defined(NANOARROW_DEBUG) && !defined(NANOARROW_PRINT_AND_DIE)
#define NANOARROW_PRINT_AND_DIE(VALUE, EXPR_STR)                                  \
  do {                                                                            \
    fprintf(stderr, "%s failed with errno %d\n* %s:%d\n", EXPR_STR, (int)(VALUE), \
            __FILE__, (int)__LINE__);                                             \
    abort();                                                                      \
  } while (0)
#endif

#if defined(NANOARROW_DEBUG)
#define _NANOARROW_ASSERT_OK_IMPL(NAME, EXPR, EXPR_STR) \
  do {                                                  \
    const int NAME = (EXPR);                            \
    if (NAME) NANOARROW_PRINT_AND_DIE(NAME, EXPR_STR);  \
  } while (0)

/// \brief Assert that an expression's value is NANOARROW_OK
/// \ingroup nanoarrow-errors
///
/// If nanoarrow was built in debug mode (i.e., defined(NANOARROW_DEBUG) is true),
/// print a message to stderr and abort. If nanoarrow was built in release mode,
/// this statement has no effect. You can customize fatal error behaviour
/// be defining the NANOARROW_PRINT_AND_DIE macro before including nanoarrow.h
/// This macro is provided as a convenience for users and is not used internally.
#define NANOARROW_ASSERT_OK(EXPR) \
  _NANOARROW_ASSERT_OK_IMPL(_NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR, #EXPR)
#else
#define NANOARROW_ASSERT_OK(EXPR) EXPR
#endif

static char _ArrowIsLittleEndian(void) {
  uint32_t check = 1;
  char first_byte;
  memcpy(&first_byte, &check, sizeof(char));
  return first_byte;
}

/// \brief Arrow type enumerator
/// \ingroup nanoarrow-utils
///
/// These names are intended to map to the corresponding arrow::Type::type
/// enumerator; however, the numeric values are specifically not equal
/// (i.e., do not rely on numeric comparison).
enum ArrowType {
  NANOARROW_TYPE_UNINITIALIZED = 0,
  NANOARROW_TYPE_NA = 1,
  NANOARROW_TYPE_BOOL,
  NANOARROW_TYPE_UINT8,
  NANOARROW_TYPE_INT8,
  NANOARROW_TYPE_UINT16,
  NANOARROW_TYPE_INT16,
  NANOARROW_TYPE_UINT32,
  NANOARROW_TYPE_INT32,
  NANOARROW_TYPE_UINT64,
  NANOARROW_TYPE_INT64,
  NANOARROW_TYPE_HALF_FLOAT,
  NANOARROW_TYPE_FLOAT,
  NANOARROW_TYPE_DOUBLE,
  NANOARROW_TYPE_STRING,
  NANOARROW_TYPE_BINARY,
  NANOARROW_TYPE_FIXED_SIZE_BINARY,
  NANOARROW_TYPE_DATE32,
  NANOARROW_TYPE_DATE64,
  NANOARROW_TYPE_TIMESTAMP,
  NANOARROW_TYPE_TIME32,
  NANOARROW_TYPE_TIME64,
  NANOARROW_TYPE_INTERVAL_MONTHS,
  NANOARROW_TYPE_INTERVAL_DAY_TIME,
  NANOARROW_TYPE_DECIMAL128,
  NANOARROW_TYPE_DECIMAL256,
  NANOARROW_TYPE_LIST,
  NANOARROW_TYPE_STRUCT,
  NANOARROW_TYPE_SPARSE_UNION,
  NANOARROW_TYPE_DENSE_UNION,
  NANOARROW_TYPE_DICTIONARY,
  NANOARROW_TYPE_MAP,
  NANOARROW_TYPE_EXTENSION,
  NANOARROW_TYPE_FIXED_SIZE_LIST,
  NANOARROW_TYPE_DURATION,
  NANOARROW_TYPE_LARGE_STRING,
  NANOARROW_TYPE_LARGE_BINARY,
  NANOARROW_TYPE_LARGE_LIST,
  NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO
};

/// \brief Get a string value of an enum ArrowType value
/// \ingroup nanoarrow-utils
///
/// Returns NULL for invalid values for type
static inline const char* ArrowTypeString(enum ArrowType type);

static inline const char* ArrowTypeString(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_NA:
      return "na";
    case NANOARROW_TYPE_BOOL:
      return "bool";
    case NANOARROW_TYPE_UINT8:
      return "uint8";
    case NANOARROW_TYPE_INT8:
      return "int8";
    case NANOARROW_TYPE_UINT16:
      return "uint16";
    case NANOARROW_TYPE_INT16:
      return "int16";
    case NANOARROW_TYPE_UINT32:
      return "uint32";
    case NANOARROW_TYPE_INT32:
      return "int32";
    case NANOARROW_TYPE_UINT64:
      return "uint64";
    case NANOARROW_TYPE_INT64:
      return "int64";
    case NANOARROW_TYPE_HALF_FLOAT:
      return "half_float";
    case NANOARROW_TYPE_FLOAT:
      return "float";
    case NANOARROW_TYPE_DOUBLE:
      return "double";
    case NANOARROW_TYPE_STRING:
      return "string";
    case NANOARROW_TYPE_BINARY:
      return "binary";
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      return "fixed_size_binary";
    case NANOARROW_TYPE_DATE32:
      return "date32";
    case NANOARROW_TYPE_DATE64:
      return "date64";
    case NANOARROW_TYPE_TIMESTAMP:
      return "timestamp";
    case NANOARROW_TYPE_TIME32:
      return "time32";
    case NANOARROW_TYPE_TIME64:
      return "time64";
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      return "interval_months";
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      return "interval_day_time";
    case NANOARROW_TYPE_DECIMAL128:
      return "decimal128";
    case NANOARROW_TYPE_DECIMAL256:
      return "decimal256";
    case NANOARROW_TYPE_LIST:
      return "list";
    case NANOARROW_TYPE_STRUCT:
      return "struct";
    case NANOARROW_TYPE_SPARSE_UNION:
      return "sparse_union";
    case NANOARROW_TYPE_DENSE_UNION:
      return "dense_union";
    case NANOARROW_TYPE_DICTIONARY:
      return "dictionary";
    case NANOARROW_TYPE_MAP:
      return "map";
    case NANOARROW_TYPE_EXTENSION:
      return "extension";
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      return "fixed_size_list";
    case NANOARROW_TYPE_DURATION:
      return "duration";
    case NANOARROW_TYPE_LARGE_STRING:
      return "large_string";
    case NANOARROW_TYPE_LARGE_BINARY:
      return "large_binary";
    case NANOARROW_TYPE_LARGE_LIST:
      return "large_list";
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      return "interval_month_day_nano";
    default:
      return NULL;
  }
}

/// \brief Arrow time unit enumerator
/// \ingroup nanoarrow-utils
///
/// These names and values map to the corresponding arrow::TimeUnit::type
/// enumerator.
enum ArrowTimeUnit {
  NANOARROW_TIME_UNIT_SECOND = 0,
  NANOARROW_TIME_UNIT_MILLI = 1,
  NANOARROW_TIME_UNIT_MICRO = 2,
  NANOARROW_TIME_UNIT_NANO = 3
};

/// \brief Validation level enumerator
/// \ingroup nanoarrow-array
enum ArrowValidationLevel {
  /// \brief Do not validate buffer sizes or content.
  NANOARROW_VALIDATION_LEVEL_NONE = 0,

  /// \brief Validate buffer sizes that depend on array length but do not validate buffer
  /// sizes that depend on buffer data access.
  NANOARROW_VALIDATION_LEVEL_MINIMAL = 1,

  /// \brief Validate all buffer sizes, including those that require buffer data access,
  /// but do not perform any checks that are O(1) along the length of the buffers.
  NANOARROW_VALIDATION_LEVEL_DEFAULT = 2,

  /// \brief Validate all buffer sizes and all buffer content. This is useful in the
  /// context of untrusted input or input that may have been corrupted in transit.
  NANOARROW_VALIDATION_LEVEL_FULL = 3
};

/// \brief Get a string value of an enum ArrowTimeUnit value
/// \ingroup nanoarrow-utils
///
/// Returns NULL for invalid values for time_unit
static inline const char* ArrowTimeUnitString(enum ArrowTimeUnit time_unit);

static inline const char* ArrowTimeUnitString(enum ArrowTimeUnit time_unit) {
  switch (time_unit) {
    case NANOARROW_TIME_UNIT_SECOND:
      return "s";
    case NANOARROW_TIME_UNIT_MILLI:
      return "ms";
    case NANOARROW_TIME_UNIT_MICRO:
      return "us";
    case NANOARROW_TIME_UNIT_NANO:
      return "ns";
    default:
      return NULL;
  }
}

/// \brief Functional types of buffers as described in the Arrow Columnar Specification
/// \ingroup nanoarrow-array-view
enum ArrowBufferType {
  NANOARROW_BUFFER_TYPE_NONE,
  NANOARROW_BUFFER_TYPE_VALIDITY,
  NANOARROW_BUFFER_TYPE_TYPE_ID,
  NANOARROW_BUFFER_TYPE_UNION_OFFSET,
  NANOARROW_BUFFER_TYPE_DATA_OFFSET,
  NANOARROW_BUFFER_TYPE_DATA
};

/// \brief The maximum number of buffers in an ArrowArrayView or ArrowLayout
/// \ingroup nanoarrow-array-view
///
/// All currently supported types have 3 buffers or fewer; however, future types
/// may involve a variable number of buffers (e.g., string view). These buffers
/// will be represented by separate members of the ArrowArrayView or ArrowLayout.
#define NANOARROW_MAX_FIXED_BUFFERS 3

/// \brief An non-owning view of a string
/// \ingroup nanoarrow-utils
struct ArrowStringView {
  /// \brief A pointer to the start of the string
  ///
  /// If size_bytes is 0, this value may be NULL.
  const char* data;

  /// \brief The size of the string in bytes,
  ///
  /// (Not including the null terminator.)
  int64_t size_bytes;
};

/// \brief Return a view of a const C string
/// \ingroup nanoarrow-utils
static inline struct ArrowStringView ArrowCharView(const char* value);

static inline struct ArrowStringView ArrowCharView(const char* value) {
  struct ArrowStringView out;

  out.data = value;
  if (value) {
    out.size_bytes = (int64_t)strlen(value);
  } else {
    out.size_bytes = 0;
  }

  return out;
}

union ArrowBufferViewData {
  const void* data;
  const int8_t* as_int8;
  const uint8_t* as_uint8;
  const int16_t* as_int16;
  const uint16_t* as_uint16;
  const int32_t* as_int32;
  const uint32_t* as_uint32;
  const int64_t* as_int64;
  const uint64_t* as_uint64;
  const double* as_double;
  const float* as_float;
  const char* as_char;
};

/// \brief An non-owning view of a buffer
/// \ingroup nanoarrow-utils
struct ArrowBufferView {
  /// \brief A pointer to the start of the buffer
  ///
  /// If size_bytes is 0, this value may be NULL.
  union ArrowBufferViewData data;

  /// \brief The size of the buffer in bytes
  int64_t size_bytes;
};

/// \brief Array buffer allocation and deallocation
/// \ingroup nanoarrow-buffer
///
/// Container for allocate, reallocate, and free methods that can be used
/// to customize allocation and deallocation of buffers when constructing
/// an ArrowArray.
struct ArrowBufferAllocator {
  /// \brief Reallocate a buffer or return NULL if it cannot be reallocated
  uint8_t* (*reallocate)(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                         int64_t old_size, int64_t new_size);

  /// \brief Deallocate a buffer allocated by this allocator
  void (*free)(struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t size);

  /// \brief Opaque data specific to the allocator
  void* private_data;
};

/// \brief An owning mutable view of a buffer
/// \ingroup nanoarrow-buffer
struct ArrowBuffer {
  /// \brief A pointer to the start of the buffer
  ///
  /// If capacity_bytes is 0, this value may be NULL.
  uint8_t* data;

  /// \brief The size of the buffer in bytes
  int64_t size_bytes;

  /// \brief The capacity of the buffer in bytes
  int64_t capacity_bytes;

  /// \brief The allocator that will be used to reallocate and/or free the buffer
  struct ArrowBufferAllocator allocator;
};

/// \brief An owning mutable view of a bitmap
/// \ingroup nanoarrow-bitmap
struct ArrowBitmap {
  /// \brief An ArrowBuffer to hold the allocated memory
  struct ArrowBuffer buffer;

  /// \brief The number of bits that have been appended to the bitmap
  int64_t size_bits;
};

/// \brief A description of an arrangement of buffers
/// \ingroup nanoarrow-utils
///
/// Contains the minimum amount of information required to
/// calculate the size of each buffer in an ArrowArray knowing only
/// the length and offset of the array.
struct ArrowLayout {
  /// \brief The function of each buffer
  enum ArrowBufferType buffer_type[NANOARROW_MAX_FIXED_BUFFERS];

  /// \brief The data type of each buffer
  enum ArrowType buffer_data_type[NANOARROW_MAX_FIXED_BUFFERS];

  /// \brief The size of an element each buffer or 0 if this size is variable or unknown
  int64_t element_size_bits[NANOARROW_MAX_FIXED_BUFFERS];

  /// \brief The number of elements in the child array per element in this array for a
  /// fixed-size list
  int64_t child_size_elements;
};

/// \brief A non-owning view of an ArrowArray
/// \ingroup nanoarrow-array-view
///
/// This data structure provides access to the values contained within
/// an ArrowArray with fields provided in a more readily-extractible
/// form. You can re-use an ArrowArrayView for multiple ArrowArrays
/// with the same storage type, use it to represent a hypothetical
/// ArrowArray that does not exist yet, or use it to validate the buffers
/// of a future ArrowArray.
struct ArrowArrayView {
  /// \brief The underlying ArrowArray or NULL if it has not been set or
  /// if the buffers in this ArrowArrayView are not backed by an ArrowArray.
  const struct ArrowArray* array;

  /// \brief The number of elements from the physical start of the buffers.
  int64_t offset;

  /// \brief The number of elements in this view.
  int64_t length;

  /// \brief A cached null count or -1 to indicate that this value is unknown.
  int64_t null_count;

  /// \brief The type used to store values in this array
  ///
  /// This type represents only the minimum required information to
  /// extract values from the array buffers (e.g., for a Date32 array,
  /// this value will be NANOARROW_TYPE_INT32). For dictionary-encoded
  /// arrays, this will be the index type.
  enum ArrowType storage_type;

  /// \brief The buffer types, strides, and sizes of this Array's buffers
  struct ArrowLayout layout;

  /// \brief This Array's buffers as ArrowBufferView objects
  struct ArrowBufferView buffer_views[NANOARROW_MAX_FIXED_BUFFERS];

  /// \brief The number of children of this view
  int64_t n_children;

  /// \brief Pointers to views of this array's children
  struct ArrowArrayView** children;

  /// \brief Pointer to a view of this array's dictionary
  struct ArrowArrayView* dictionary;

  /// \brief Union type id to child index mapping
  ///
  /// If storage_type is a union type, a 256-byte ArrowMalloc()ed buffer
  /// such that child_index == union_type_id_map[type_id] and
  /// type_id == union_type_id_map[128 + child_index]. This value may be
  /// NULL in the case where child_id == type_id.
  int8_t* union_type_id_map;
};

// Used as the private data member for ArrowArrays allocated here and accessed
// internally within inline ArrowArray* helpers.
struct ArrowArrayPrivateData {
  // Holder for the validity buffer (or first buffer for union types, which are
  // the only type whose first buffer is not a valdiity buffer)
  struct ArrowBitmap bitmap;

  // Holder for additional buffers as required
  struct ArrowBuffer buffers[NANOARROW_MAX_FIXED_BUFFERS - 1];

  // The array of pointers to buffers. This must be updated after a sequence
  // of appends to synchronize its values with the actual buffer addresses
  // (which may have ben reallocated uring that time)
  const void* buffer_data[NANOARROW_MAX_FIXED_BUFFERS];

  // The storage data type, or NANOARROW_TYPE_UNINITIALIZED if unknown
  enum ArrowType storage_type;

  // The buffer arrangement for the storage type
  struct ArrowLayout layout;

  // Flag to indicate if there are non-sequence union type ids.
  // In the future this could be replaced with a type id<->child mapping
  // to support constructing unions in append mode where type_id != child_index
  int8_t union_type_id_is_child_index;
};

/// \brief A representation of an interval.
/// \ingroup nanoarrow-utils
struct ArrowInterval {
  /// \brief The type of interval being used
  enum ArrowType type;
  /// \brief The number of months represented by the interval
  int32_t months;
  /// \brief The number of days represented by the interval
  int32_t days;
  /// \brief The number of ms represented by the interval
  int32_t ms;
  /// \brief The number of ns represented by the interval
  int64_t ns;
};

/// \brief Zero initialize an Interval with a given unit
/// \ingroup nanoarrow-utils
static inline void ArrowIntervalInit(struct ArrowInterval* interval,
                                     enum ArrowType type) {
  memset(interval, 0, sizeof(struct ArrowInterval));
  interval->type = type;
}

/// \brief A representation of a fixed-precision decimal number
/// \ingroup nanoarrow-utils
///
/// This structure should be initialized with ArrowDecimalInit() once and
/// values set using ArrowDecimalSetInt(), ArrowDecimalSetBytes128(),
/// or ArrowDecimalSetBytes256().
struct ArrowDecimal {
  /// \brief An array of 64-bit integers of n_words length defined in native-endian order
  uint64_t words[4];

  /// \brief The number of significant digits this decimal number can represent
  int32_t precision;

  /// \brief The number of digits after the decimal point. This can be negative.
  int32_t scale;

  /// \brief The number of words in the words array
  int n_words;

  /// \brief Cached value used by the implementation
  int high_word_index;

  /// \brief Cached value used by the implementation
  int low_word_index;
};

/// \brief Initialize a decimal with a given set of type parameters
/// \ingroup nanoarrow-utils
static inline void ArrowDecimalInit(struct ArrowDecimal* decimal, int32_t bitwidth,
                                    int32_t precision, int32_t scale) {
  memset(decimal->words, 0, sizeof(decimal->words));
  decimal->precision = precision;
  decimal->scale = scale;
  decimal->n_words = bitwidth / 8 / sizeof(uint64_t);

  if (_ArrowIsLittleEndian()) {
    decimal->low_word_index = 0;
    decimal->high_word_index = decimal->n_words - 1;
  } else {
    decimal->low_word_index = decimal->n_words - 1;
    decimal->high_word_index = 0;
  }
}

/// \brief Get a signed integer value of a sufficiently small ArrowDecimal
///
/// This does not check if the decimal's precision sufficiently small to fit
/// within the signed 64-bit integer range (A precision less than or equal
/// to 18 is sufficiently small).
static inline int64_t ArrowDecimalGetIntUnsafe(const struct ArrowDecimal* decimal) {
  return (int64_t)decimal->words[decimal->low_word_index];
}

/// \brief Copy the bytes of this decimal into a sufficiently large buffer
/// \ingroup nanoarrow-utils
static inline void ArrowDecimalGetBytes(const struct ArrowDecimal* decimal,
                                        uint8_t* out) {
  memcpy(out, decimal->words, decimal->n_words * sizeof(uint64_t));
}

/// \brief Returns 1 if the value represented by decimal is >= 0 or -1 otherwise
/// \ingroup nanoarrow-utils
static inline int64_t ArrowDecimalSign(const struct ArrowDecimal* decimal) {
  return 1 | ((int64_t)(decimal->words[decimal->high_word_index]) >> 63);
}

/// \brief Sets the integer value of this decimal
/// \ingroup nanoarrow-utils
static inline void ArrowDecimalSetInt(struct ArrowDecimal* decimal, int64_t value) {
  if (value < 0) {
    memset(decimal->words, 0xff, decimal->n_words * sizeof(uint64_t));
  } else {
    memset(decimal->words, 0, decimal->n_words * sizeof(uint64_t));
  }

  decimal->words[decimal->low_word_index] = value;
}

/// \brief Copy bytes from a buffer into this decimal
/// \ingroup nanoarrow-utils
static inline void ArrowDecimalSetBytes(struct ArrowDecimal* decimal,
                                        const uint8_t* value) {
  memcpy(decimal->words, value, decimal->n_words * sizeof(uint64_t));
}

#ifdef __cplusplus
}
#endif

#endif
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_H_INCLUDED
#define NANOARROW_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// If using CMake, optionally pass -DNANOARROW_NAMESPACE=MyNamespace which will set this
// define in nanoarrow_config.h. If not, you can optionally #define NANOARROW_NAMESPACE
// MyNamespace here.

// This section remaps the non-prefixed symbols to the prefixed symbols so that
// code written against this build can be used independent of the value of
// NANOARROW_NAMESPACE.
#ifdef NANOARROW_NAMESPACE
#define NANOARROW_CAT(A, B) A##B
#define NANOARROW_SYMBOL(A, B) NANOARROW_CAT(A, B)

#define ArrowNanoarrowVersion NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowNanoarrowVersion)
#define ArrowNanoarrowVersionInt \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowNanoarrowVersionInt)
#define ArrowErrorMessage NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowErrorMessage)
#define ArrowMalloc NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMalloc)
#define ArrowRealloc NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowRealloc)
#define ArrowFree NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowFree)
#define ArrowBufferAllocatorDefault \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowBufferAllocatorDefault)
#define ArrowBufferDeallocator \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowBufferDeallocator)
#define ArrowErrorSet NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowErrorSet)
#define ArrowLayoutInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowLayoutInit)
#define ArrowSchemaInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaInit)
#define ArrowSchemaInitFromType \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaInitFromType)
#define ArrowSchemaSetType NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetType)
#define ArrowSchemaSetTypeStruct \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetTypeStruct)
#define ArrowSchemaSetTypeFixedSize \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetTypeFixedSize)
#define ArrowSchemaSetTypeDecimal \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetTypeDecimal)
#define ArrowSchemaSetTypeDateTime \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetTypeDateTime)
#define ArrowSchemaSetTypeUnion \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetTypeUnion)
#define ArrowSchemaDeepCopy NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaDeepCopy)
#define ArrowSchemaSetFormat NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetFormat)
#define ArrowSchemaSetName NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetName)
#define ArrowSchemaSetMetadata \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaSetMetadata)
#define ArrowSchemaAllocateChildren \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaAllocateChildren)
#define ArrowSchemaAllocateDictionary \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaAllocateDictionary)
#define ArrowMetadataReaderInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataReaderInit)
#define ArrowMetadataReaderRead \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataReaderRead)
#define ArrowMetadataSizeOf NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataSizeOf)
#define ArrowMetadataHasKey NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataHasKey)
#define ArrowMetadataGetValue NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataGetValue)
#define ArrowMetadataBuilderInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataBuilderInit)
#define ArrowMetadataBuilderAppend \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataBuilderAppend)
#define ArrowMetadataBuilderSet \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataBuilderSet)
#define ArrowMetadataBuilderRemove \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowMetadataBuilderRemove)
#define ArrowSchemaViewInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaViewInit)
#define ArrowSchemaToString NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowSchemaToString)
#define ArrowArrayInitFromType \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayInitFromType)
#define ArrowArrayInitFromSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayInitFromSchema)
#define ArrowArrayInitFromArrayView \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayInitFromArrayView)
#define ArrowArrayInitFromArrayView \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayInitFromArrayView)
#define ArrowArrayAllocateChildren \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayAllocateChildren)
#define ArrowArrayAllocateDictionary \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayAllocateDictionary)
#define ArrowArraySetValidityBitmap \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArraySetValidityBitmap)
#define ArrowArraySetBuffer NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArraySetBuffer)
#define ArrowArrayReserve NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayReserve)
#define ArrowArrayFinishBuilding \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayFinishBuilding)
#define ArrowArrayFinishBuildingDefault \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayFinishBuildingDefault)
#define ArrowArrayViewInitFromType \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewInitFromType)
#define ArrowArrayViewInitFromSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewInitFromSchema)
#define ArrowArrayViewAllocateChildren \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewAllocateChildren)
#define ArrowArrayViewAllocateDictionary \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewAllocateDictionary)
#define ArrowArrayViewSetLength \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewSetLength)
#define ArrowArrayViewSetArray \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewSetArray)
#define ArrowArrayViewSetArrayMinimal \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewSetArrayMinimal)
#define ArrowArrayViewValidate \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewValidate)
#define ArrowArrayViewReset NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowArrayViewReset)
#define ArrowBasicArrayStreamInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowBasicArrayStreamInit)
#define ArrowBasicArrayStreamSetArray \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowBasicArrayStreamSetArray)
#define ArrowBasicArrayStreamValidate \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowBasicArrayStreamValidate)

#endif

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup nanoarrow Nanoarrow C library
///
/// Except where noted, objects are not thread-safe and clients should
/// take care to serialize accesses to methods.
///
/// Because this library is intended to be vendored, it provides full type
/// definitions and encourages clients to stack or statically allocate
/// where convenient.

/// \defgroup nanoarrow-malloc Memory management
///
/// Non-buffer members of a struct ArrowSchema and struct ArrowArray
/// must be allocated using ArrowMalloc() or ArrowRealloc() and freed
/// using ArrowFree() for schemas and arrays allocated here. Buffer members
/// are allocated using an ArrowBufferAllocator.
///
/// @{

/// \brief Allocate like malloc()
void* ArrowMalloc(int64_t size);

/// \brief Reallocate like realloc()
void* ArrowRealloc(void* ptr, int64_t size);

/// \brief Free a pointer allocated using ArrowMalloc() or ArrowRealloc().
void ArrowFree(void* ptr);

/// \brief Return the default allocator
///
/// The default allocator uses ArrowMalloc(), ArrowRealloc(), and
/// ArrowFree().
struct ArrowBufferAllocator ArrowBufferAllocatorDefault(void);

/// \brief Create a custom deallocator
///
/// Creates a buffer allocator with only a free method that can be used to
/// attach a custom deallocator to an ArrowBuffer. This may be used to
/// avoid copying an existing buffer that was not allocated using the
/// infrastructure provided here (e.g., by an R or Python object).
struct ArrowBufferAllocator ArrowBufferDeallocator(
    void (*custom_free)(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                        int64_t size),
    void* private_data);

/// @}

/// \defgroup nanoarrow-errors Error handling
///
/// Functions generally return an errno-compatible error code; functions that
/// need to communicate more verbose error information accept a pointer
/// to an ArrowError. This can be stack or statically allocated. The
/// content of the message is undefined unless an error code has been
/// returned. If a nanoarrow function is passed a non-null ArrowError pointer, the
/// ArrowError pointed to by the argument will be propagated with a
/// null-terminated error message. It is safe to pass a NULL ArrowError anywhere
/// in the nanoarrow API.
///
/// Except where documented, it is generally not safe to continue after a
/// function has returned a non-zero ArrowErrorCode. The NANOARROW_RETURN_NOT_OK and
/// NANOARROW_ASSERT_OK macros are provided to help propagate errors. C++ clients can use
/// the helpers provided in the nanoarrow.hpp header to facilitate using C++ idioms
/// for memory management and error propgagtion.
///
/// @{

/// \brief Error type containing a UTF-8 encoded message.
struct ArrowError {
  /// \brief A character buffer with space for an error message.
  char message[1024];
};

/// \brief Ensure an ArrowError is null-terminated by zeroing the first character.
///
/// If error is NULL, this function does nothing.
static inline void ArrowErrorInit(struct ArrowError* error) {
  if (error) {
    error->message[0] = '\0';
  }
}

/// \brief Set the contents of an error using printf syntax.
///
/// If error is NULL, this function does nothing and returns NANOARROW_OK.
ArrowErrorCode ArrowErrorSet(struct ArrowError* error, const char* fmt, ...);

/// \brief Get the contents of an error
///
/// If error is NULL, returns "", or returns the contents of the error message
/// otherwise.
const char* ArrowErrorMessage(struct ArrowError* error);

/// @}

/// \defgroup nanoarrow-utils Utility data structures
///
/// @{

/// \brief Return a version string in the form "major.minor.patch"
const char* ArrowNanoarrowVersion(void);

/// \brief Return an integer that can be used to compare versions sequentially
int ArrowNanoarrowVersionInt(void);

/// \brief Initialize a description of buffer arrangements from a storage type
void ArrowLayoutInit(struct ArrowLayout* layout, enum ArrowType storage_type);

/// \brief Create a string view from a null-terminated string
static inline struct ArrowStringView ArrowCharView(const char* value);

/// @}

/// \defgroup nanoarrow-schema Creating schemas
///
/// These functions allocate, copy, and destroy ArrowSchema structures
///
/// @{

/// \brief Initialize an ArrowSchema
///
/// Initializes the fields and release callback of schema_out. Caller
/// is responsible for calling the schema->release callback if
/// NANOARROW_OK is returned.
void ArrowSchemaInit(struct ArrowSchema* schema);

/// \brief Initialize an ArrowSchema from an ArrowType
///
/// A convenience constructor for that calls ArrowSchemaInit() and
/// ArrowSchemaSetType() for the common case of constructing an
/// unparameterized type. The caller is responsible for calling the schema->release
/// callback if NANOARROW_OK is returned.
ArrowErrorCode ArrowSchemaInitFromType(struct ArrowSchema* schema, enum ArrowType type);

/// \brief Get a human-readable summary of a Schema
///
/// Writes a summary of an ArrowSchema to out (up to n - 1 characters)
/// and returns the number of characters required for the output if
/// n were sufficiently large. If recursive is non-zero, the result will
/// also include children.
int64_t ArrowSchemaToString(const struct ArrowSchema* schema, char* out, int64_t n,
                            char recursive);

/// \brief Set the format field of a schema from an ArrowType
///
/// Initializes the fields and release callback of schema_out. For
/// NANOARROW_TYPE_LIST, NANOARROW_TYPE_LARGE_LIST, and
/// NANOARROW_TYPE_MAP, the appropriate number of children are
/// allocated, initialized, and named; however, the caller must
/// ArrowSchemaSetType() on the preinitialized children. Schema must have been initialized
/// using ArrowSchemaInit() or ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetType(struct ArrowSchema* schema, enum ArrowType type);

/// \brief Set the format field and initialize children of a struct schema
///
/// The specified number of children are initialized; however, the caller is responsible
/// for calling ArrowSchemaSetType() and ArrowSchemaSetName() on each child.
/// Schema must have been initialized using ArrowSchemaInit() or ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetTypeStruct(struct ArrowSchema* schema, int64_t n_children);

/// \brief Set the format field of a fixed-size schema
///
/// Returns EINVAL for fixed_size <= 0 or for type that is not
/// NANOARROW_TYPE_FIXED_SIZE_BINARY or NANOARROW_TYPE_FIXED_SIZE_LIST.
/// For NANOARROW_TYPE_FIXED_SIZE_LIST, the appropriate number of children are
/// allocated, initialized, and named; however, the caller must
/// ArrowSchemaSetType() the first child. Schema must have been initialized using
/// ArrowSchemaInit() or ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetTypeFixedSize(struct ArrowSchema* schema,
                                           enum ArrowType type, int32_t fixed_size);

/// \brief Set the format field of a decimal schema
///
/// Returns EINVAL for scale <= 0 or for type that is not
/// NANOARROW_TYPE_DECIMAL128 or NANOARROW_TYPE_DECIMAL256. Schema must have been
/// initialized using ArrowSchemaInit() or ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetTypeDecimal(struct ArrowSchema* schema, enum ArrowType type,
                                         int32_t decimal_precision,
                                         int32_t decimal_scale);

/// \brief Set the format field of a time, timestamp, or duration schema
///
/// Returns EINVAL for type that is not
/// NANOARROW_TYPE_TIME32, NANOARROW_TYPE_TIME64,
/// NANOARROW_TYPE_TIMESTAMP, or NANOARROW_TYPE_DURATION. The
/// timezone parameter must be NULL for a non-timestamp type. Schema must have been
/// initialized using ArrowSchemaInit() or ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetTypeDateTime(struct ArrowSchema* schema, enum ArrowType type,
                                          enum ArrowTimeUnit time_unit,
                                          const char* timezone);

/// \brief Seet the format field of a union schema
///
/// Returns EINVAL for a type that is not NANOARROW_TYPE_DENSE_UNION
/// or NANOARROW_TYPE_SPARSE_UNION. The specified number of children are
/// allocated, and initialized.
ArrowErrorCode ArrowSchemaSetTypeUnion(struct ArrowSchema* schema, enum ArrowType type,
                                       int64_t n_children);

/// \brief Make a (recursive) copy of a schema
///
/// Allocates and copies fields of schema into schema_out.
ArrowErrorCode ArrowSchemaDeepCopy(const struct ArrowSchema* schema,
                                   struct ArrowSchema* schema_out);

/// \brief Copy format into schema->format
///
/// schema must have been allocated using ArrowSchemaInitFromType() or
/// ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetFormat(struct ArrowSchema* schema, const char* format);

/// \brief Copy name into schema->name
///
/// schema must have been allocated using ArrowSchemaInitFromType() or
/// ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaSetName(struct ArrowSchema* schema, const char* name);

/// \brief Copy metadata into schema->metadata
///
/// schema must have been allocated using ArrowSchemaInitFromType() or
/// ArrowSchemaDeepCopy.
ArrowErrorCode ArrowSchemaSetMetadata(struct ArrowSchema* schema, const char* metadata);

/// \brief Allocate the schema->children array
///
/// Includes the memory for each child struct ArrowSchema.
/// schema must have been allocated using ArrowSchemaInitFromType() or
/// ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaAllocateChildren(struct ArrowSchema* schema,
                                           int64_t n_children);

/// \brief Allocate the schema->dictionary member
///
/// schema must have been allocated using ArrowSchemaInitFromType() or
/// ArrowSchemaDeepCopy().
ArrowErrorCode ArrowSchemaAllocateDictionary(struct ArrowSchema* schema);

/// @}

/// \defgroup nanoarrow-metadata Create, read, and modify schema metadata
///
/// @{

/// \brief Reader for key/value pairs in schema metadata
///
/// The ArrowMetadataReader does not own any data and is only valid
/// for the lifetime of the underlying metadata pointer.
struct ArrowMetadataReader {
  /// \brief A metadata string from a schema->metadata field.
  const char* metadata;

  /// \brief The current offset into the metadata string
  int64_t offset;

  /// \brief The number of remaining keys
  int32_t remaining_keys;
};

/// \brief Initialize an ArrowMetadataReader
ArrowErrorCode ArrowMetadataReaderInit(struct ArrowMetadataReader* reader,
                                       const char* metadata);

/// \brief Read the next key/value pair from an ArrowMetadataReader
ArrowErrorCode ArrowMetadataReaderRead(struct ArrowMetadataReader* reader,
                                       struct ArrowStringView* key_out,
                                       struct ArrowStringView* value_out);

/// \brief The number of bytes in in a key/value metadata string
int64_t ArrowMetadataSizeOf(const char* metadata);

/// \brief Check for a key in schema metadata
char ArrowMetadataHasKey(const char* metadata, struct ArrowStringView key);

/// \brief Extract a value from schema metadata
///
/// If key does not exist in metadata, value_out is unmodified
ArrowErrorCode ArrowMetadataGetValue(const char* metadata, struct ArrowStringView key,
                                     struct ArrowStringView* value_out);

/// \brief Initialize a builder for schema metadata from key/value pairs
///
/// metadata can be an existing metadata string or NULL to initialize
/// an empty metadata string.
ArrowErrorCode ArrowMetadataBuilderInit(struct ArrowBuffer* buffer, const char* metadata);

/// \brief Append a key/value pair to a buffer containing serialized metadata
ArrowErrorCode ArrowMetadataBuilderAppend(struct ArrowBuffer* buffer,
                                          struct ArrowStringView key,
                                          struct ArrowStringView value);

/// \brief Set a key/value pair to a buffer containing serialized metadata
///
/// Ensures that the only entry for key in the metadata is set to value.
/// This function maintains the existing position of (the first instance of)
/// key if present in the data.
ArrowErrorCode ArrowMetadataBuilderSet(struct ArrowBuffer* buffer,
                                       struct ArrowStringView key,
                                       struct ArrowStringView value);

/// \brief Remove a key from a buffer containing serialized metadata
ArrowErrorCode ArrowMetadataBuilderRemove(struct ArrowBuffer* buffer,
                                          struct ArrowStringView key);

/// @}

/// \defgroup nanoarrow-schema-view Reading schemas
///
/// @{

/// \brief A non-owning view of a parsed ArrowSchema
///
/// Contains more readily extractable values than a raw ArrowSchema.
/// Clients can stack or statically allocate this structure but are
/// encouraged to use the provided getters to ensure forward
/// compatibility.
struct ArrowSchemaView {
  /// \brief A pointer to the schema represented by this view
  const struct ArrowSchema* schema;

  /// \brief The data type represented by the schema
  ///
  /// This value may be NANOARROW_TYPE_DICTIONARY if the schema has a
  /// non-null dictionary member; datetime types are valid values.
  /// This value will never be NANOARROW_TYPE_EXTENSION (see
  /// extension_name and/or extension_metadata to check for
  /// an extension type).
  enum ArrowType type;

  /// \brief The storage data type represented by the schema
  ///
  /// This value will never be NANOARROW_TYPE_DICTIONARY, NANOARROW_TYPE_EXTENSION
  /// or any datetime type. This value represents only the type required to
  /// interpret the buffers in the array.
  enum ArrowType storage_type;

  /// \brief The storage layout represented by the schema
  struct ArrowLayout layout;

  /// \brief The extension type name if it exists
  ///
  /// If the ARROW:extension:name key is present in schema.metadata,
  /// extension_name.data will be non-NULL.
  struct ArrowStringView extension_name;

  /// \brief The extension type metadata if it exists
  ///
  /// If the ARROW:extension:metadata key is present in schema.metadata,
  /// extension_metadata.data will be non-NULL.
  struct ArrowStringView extension_metadata;

  /// \brief Format fixed size parameter
  ///
  /// This value is set when parsing a fixed-size binary or fixed-size
  /// list schema; this value is undefined for other types. For a
  /// fixed-size binary schema this value is in bytes; for a fixed-size
  /// list schema this value refers to the number of child elements for
  /// each element of the parent.
  int32_t fixed_size;

  /// \brief Decimal bitwidth
  ///
  /// This value is set when parsing a decimal type schema;
  /// this value is undefined for other types.
  int32_t decimal_bitwidth;

  /// \brief Decimal precision
  ///
  /// This value is set when parsing a decimal type schema;
  /// this value is undefined for other types.
  int32_t decimal_precision;

  /// \brief Decimal scale
  ///
  /// This value is set when parsing a decimal type schema;
  /// this value is undefined for other types.
  int32_t decimal_scale;

  /// \brief Format time unit parameter
  ///
  /// This value is set when parsing a date/time type. The value is
  /// undefined for other types.
  enum ArrowTimeUnit time_unit;

  /// \brief Format timezone parameter
  ///
  /// This value is set when parsing a timestamp type and represents
  /// the timezone format parameter. This value points to
  /// data within the schema and is undefined for other types.
  const char* timezone;

  /// \brief Union type ids parameter
  ///
  /// This value is set when parsing a union type and represents
  /// type ids parameter. This value points to
  /// data within the schema and is undefined for other types.
  const char* union_type_ids;
};

/// \brief Initialize an ArrowSchemaView
ArrowErrorCode ArrowSchemaViewInit(struct ArrowSchemaView* schema_view,
                                   const struct ArrowSchema* schema,
                                   struct ArrowError* error);

/// @}

/// \defgroup nanoarrow-buffer Owning, growable buffers
///
/// @{

/// \brief Initialize an ArrowBuffer
///
/// Initialize a buffer with a NULL, zero-size buffer using the default
/// buffer allocator.
static inline void ArrowBufferInit(struct ArrowBuffer* buffer);

/// \brief Set a newly-initialized buffer's allocator
///
/// Returns EINVAL if the buffer has already been allocated.
static inline ArrowErrorCode ArrowBufferSetAllocator(
    struct ArrowBuffer* buffer, struct ArrowBufferAllocator allocator);

/// \brief Reset an ArrowBuffer
///
/// Releases the buffer using the allocator's free method if
/// the buffer's data member is non-null, sets the data member
/// to NULL, and sets the buffer's size and capacity to 0.
static inline void ArrowBufferReset(struct ArrowBuffer* buffer);

/// \brief Move an ArrowBuffer
///
/// Transfers the buffer data and lifecycle management to another
/// address and resets buffer.
static inline void ArrowBufferMove(struct ArrowBuffer* src, struct ArrowBuffer* dst);

/// \brief Grow or shrink a buffer to a given capacity
///
/// When shrinking the capacity of the buffer, the buffer is only reallocated
/// if shrink_to_fit is non-zero. Calling ArrowBufferResize() does not
/// adjust the buffer's size member except to ensure that the invariant
/// capacity >= size remains true.
static inline ArrowErrorCode ArrowBufferResize(struct ArrowBuffer* buffer,
                                               int64_t new_capacity_bytes,
                                               char shrink_to_fit);

/// \brief Ensure a buffer has at least a given additional capacity
///
/// Ensures that the buffer has space to append at least
/// additional_size_bytes, overallocating when required.
static inline ArrowErrorCode ArrowBufferReserve(struct ArrowBuffer* buffer,
                                                int64_t additional_size_bytes);

/// \brief Write data to buffer and increment the buffer size
///
/// This function does not check that buffer has the required capacity
static inline void ArrowBufferAppendUnsafe(struct ArrowBuffer* buffer, const void* data,
                                           int64_t size_bytes);

/// \brief Write data to buffer and increment the buffer size
///
/// This function writes and ensures that the buffer has the required capacity,
/// possibly by reallocating the buffer. Like ArrowBufferReserve, this will
/// overallocate when reallocation is required.
static inline ArrowErrorCode ArrowBufferAppend(struct ArrowBuffer* buffer,
                                               const void* data, int64_t size_bytes);

/// \brief Write fill to buffer and increment the buffer size
///
/// This function writes the specified number of fill bytes and
/// ensures that the buffer has the required capacity,
static inline ArrowErrorCode ArrowBufferAppendFill(struct ArrowBuffer* buffer,
                                                   uint8_t value, int64_t size_bytes);

/// \brief Write an 8-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendInt8(struct ArrowBuffer* buffer,
                                                   int8_t value);

/// \brief Write an unsigned 8-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendUInt8(struct ArrowBuffer* buffer,
                                                    uint8_t value);

/// \brief Write a 16-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendInt16(struct ArrowBuffer* buffer,
                                                    int16_t value);

/// \brief Write an unsigned 16-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendUInt16(struct ArrowBuffer* buffer,
                                                     uint16_t value);

/// \brief Write a 32-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendInt32(struct ArrowBuffer* buffer,
                                                    int32_t value);

/// \brief Write an unsigned 32-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendUInt32(struct ArrowBuffer* buffer,
                                                     uint32_t value);

/// \brief Write a 64-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendInt64(struct ArrowBuffer* buffer,
                                                    int64_t value);

/// \brief Write an unsigned 64-bit integer to a buffer
static inline ArrowErrorCode ArrowBufferAppendUInt64(struct ArrowBuffer* buffer,
                                                     uint64_t value);

/// \brief Write a double to a buffer
static inline ArrowErrorCode ArrowBufferAppendDouble(struct ArrowBuffer* buffer,
                                                     double value);

/// \brief Write a float to a buffer
static inline ArrowErrorCode ArrowBufferAppendFloat(struct ArrowBuffer* buffer,
                                                    float value);

/// \brief Write an ArrowStringView to a buffer
static inline ArrowErrorCode ArrowBufferAppendStringView(struct ArrowBuffer* buffer,
                                                         struct ArrowStringView value);

/// \brief Write an ArrowBufferView to a buffer
static inline ArrowErrorCode ArrowBufferAppendBufferView(struct ArrowBuffer* buffer,
                                                         struct ArrowBufferView value);

/// @}

/// \defgroup nanoarrow-bitmap Bitmap utilities
///
/// @{

/// \brief Extract a boolean value from a bitmap
static inline int8_t ArrowBitGet(const uint8_t* bits, int64_t i);

/// \brief Set a boolean value to a bitmap to true
static inline void ArrowBitSet(uint8_t* bits, int64_t i);

/// \brief Set a boolean value to a bitmap to false
static inline void ArrowBitClear(uint8_t* bits, int64_t i);

/// \brief Set a boolean value to a bitmap
static inline void ArrowBitSetTo(uint8_t* bits, int64_t i, uint8_t value);

/// \brief Set a boolean value to a range in a bitmap
static inline void ArrowBitsSetTo(uint8_t* bits, int64_t start_offset, int64_t length,
                                  uint8_t bits_are_set);

/// \brief Count true values in a bitmap
static inline int64_t ArrowBitCountSet(const uint8_t* bits, int64_t i_from, int64_t i_to);

/// \brief Extract int8 boolean values from a range in a bitmap
static inline void ArrowBitsUnpackInt8(const uint8_t* bits, int64_t start_offset,
                                       int64_t length, int8_t* out);

/// \brief Extract int32 boolean values from a range in a bitmap
static inline void ArrowBitsUnpackInt32(const uint8_t* bits, int64_t start_offset,
                                        int64_t length, int32_t* out);

/// \brief Initialize an ArrowBitmap
///
/// Initialize the builder's buffer, empty its cache, and reset the size to zero
static inline void ArrowBitmapInit(struct ArrowBitmap* bitmap);

/// \brief Move an ArrowBitmap
///
/// Transfers the underlying buffer data and lifecycle management to another
/// address and resets the bitmap.
static inline void ArrowBitmapMove(struct ArrowBitmap* src, struct ArrowBitmap* dst);

/// \brief Ensure a bitmap builder has at least a given additional capacity
///
/// Ensures that the buffer has space to append at least
/// additional_size_bits, overallocating when required.
static inline ArrowErrorCode ArrowBitmapReserve(struct ArrowBitmap* bitmap,
                                                int64_t additional_size_bits);

/// \brief Grow or shrink a bitmap to a given capacity
///
/// When shrinking the capacity of the bitmap, the bitmap is only reallocated
/// if shrink_to_fit is non-zero. Calling ArrowBitmapResize() does not
/// adjust the buffer's size member except when shrinking new_capacity_bits
/// to a value less than the current number of bits in the bitmap.
static inline ArrowErrorCode ArrowBitmapResize(struct ArrowBitmap* bitmap,
                                               int64_t new_capacity_bits,
                                               char shrink_to_fit);

/// \brief Reserve space for and append zero or more of the same boolean value to a bitmap
static inline ArrowErrorCode ArrowBitmapAppend(struct ArrowBitmap* bitmap,
                                               uint8_t bits_are_set, int64_t length);

/// \brief Append zero or more of the same boolean value to a bitmap
static inline void ArrowBitmapAppendUnsafe(struct ArrowBitmap* bitmap,
                                           uint8_t bits_are_set, int64_t length);

/// \brief Append boolean values encoded as int8_t to a bitmap
///
/// The values must all be 0 or 1.
static inline void ArrowBitmapAppendInt8Unsafe(struct ArrowBitmap* bitmap,
                                               const int8_t* values, int64_t n_values);

/// \brief Append boolean values encoded as int32_t to a bitmap
///
/// The values must all be 0 or 1.
static inline void ArrowBitmapAppendInt32Unsafe(struct ArrowBitmap* bitmap,
                                                const int32_t* values, int64_t n_values);

/// \brief Reset a bitmap builder
///
/// Releases any memory held by buffer, empties the cache, and resets the size to zero
static inline void ArrowBitmapReset(struct ArrowBitmap* bitmap);

/// @}

/// \defgroup nanoarrow-array Creating arrays
///
/// These functions allocate, copy, and destroy ArrowArray structures.
/// Once an ArrowArray has been initialized via ArrowArrayInitFromType()
/// or ArrowArrayInitFromSchema(), the caller is responsible for releasing
/// it using the embedded release callback.
///
/// @{

/// \brief Initialize the fields of an array
///
/// Initializes the fields and release callback of array. Caller
/// is responsible for calling the array->release callback if
/// NANOARROW_OK is returned.
ArrowErrorCode ArrowArrayInitFromType(struct ArrowArray* array,
                                      enum ArrowType storage_type);

/// \brief Initialize the contents of an ArrowArray from an ArrowSchema
///
/// Caller is responsible for calling the array->release callback if
/// NANOARROW_OK is returned.
ArrowErrorCode ArrowArrayInitFromSchema(struct ArrowArray* array,
                                        const struct ArrowSchema* schema,
                                        struct ArrowError* error);

/// \brief Initialize the contents of an ArrowArray from an ArrowArrayView
///
/// Caller is responsible for calling the array->release callback if
/// NANOARROW_OK is returned.
ArrowErrorCode ArrowArrayInitFromArrayView(struct ArrowArray* array,
                                           const struct ArrowArrayView* array_view,
                                           struct ArrowError* error);

/// \brief Allocate the array->children array
///
/// Includes the memory for each child struct ArrowArray,
/// whose members are marked as released and may be subsequently initialized
/// with ArrowArrayInitFromType() or moved from an existing ArrowArray.
/// schema must have been allocated using ArrowArrayInitFromType().
ArrowErrorCode ArrowArrayAllocateChildren(struct ArrowArray* array, int64_t n_children);

/// \brief Allocate the array->dictionary member
///
/// Includes the memory for the struct ArrowArray, whose contents
/// is marked as released and may be subsequently initialized
/// with ArrowArrayInitFromType() or moved from an existing ArrowArray.
/// array must have been allocated using ArrowArrayInitFromType()
ArrowErrorCode ArrowArrayAllocateDictionary(struct ArrowArray* array);

/// \brief Set the validity bitmap of an ArrowArray
///
/// array must have been allocated using ArrowArrayInitFromType()
void ArrowArraySetValidityBitmap(struct ArrowArray* array, struct ArrowBitmap* bitmap);

/// \brief Set a buffer of an ArrowArray
///
/// array must have been allocated using ArrowArrayInitFromType()
ArrowErrorCode ArrowArraySetBuffer(struct ArrowArray* array, int64_t i,
                                   struct ArrowBuffer* buffer);

/// \brief Get the validity bitmap of an ArrowArray
///
/// array must have been allocated using ArrowArrayInitFromType()
static inline struct ArrowBitmap* ArrowArrayValidityBitmap(struct ArrowArray* array);

/// \brief Get a buffer of an ArrowArray
///
/// array must have been allocated using ArrowArrayInitFromType()
static inline struct ArrowBuffer* ArrowArrayBuffer(struct ArrowArray* array, int64_t i);

/// \brief Start element-wise appending to an ArrowArray
///
/// Initializes any values needed to use ArrowArrayAppend*() functions.
/// All element-wise appenders append by value and return EINVAL if the exact value
/// cannot be represented by the underlying storage type.
/// array must have been allocated using ArrowArrayInitFromType()
static inline ArrowErrorCode ArrowArrayStartAppending(struct ArrowArray* array);

/// \brief Reserve space for future appends
///
/// For buffer sizes that can be calculated (i.e., not string data buffers or
/// child array sizes for non-fixed-size arrays), recursively reserve space for
/// additional elements. This is useful for reducing the number of reallocations
/// that occur using the item-wise appenders.
ArrowErrorCode ArrowArrayReserve(struct ArrowArray* array,
                                 int64_t additional_size_elements);

/// \brief Append a null value to an array
static inline ArrowErrorCode ArrowArrayAppendNull(struct ArrowArray* array, int64_t n);

/// \brief Append an empty, non-null value to an array
static inline ArrowErrorCode ArrowArrayAppendEmpty(struct ArrowArray* array, int64_t n);

/// \brief Append a signed integer value to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type or EINVAL otherwise (e.g., value
/// is outside the valid array range).
static inline ArrowErrorCode ArrowArrayAppendInt(struct ArrowArray* array, int64_t value);

/// \brief Append an unsigned integer value to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type or EINVAL otherwise (e.g., value
/// is outside the valid array range).
static inline ArrowErrorCode ArrowArrayAppendUInt(struct ArrowArray* array,
                                                  uint64_t value);

/// \brief Append a double value to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type or EINVAL otherwise (e.g., value
/// is outside the valid array range or there is an attempt to append
/// a non-integer to an array with an integer storage type).
static inline ArrowErrorCode ArrowArrayAppendDouble(struct ArrowArray* array,
                                                    double value);

/// \brief Append a string of bytes to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type, EOVERFLOW if appending value would overflow
/// the offset type (e.g., if the data buffer would be larger than 2 GB for a
/// non-large string type), or EINVAL otherwise (e.g., the underlying array is not a
/// binary, string, large binary, large string, or fixed-size binary array, or value is
/// the wrong size for a fixed-size binary array).
static inline ArrowErrorCode ArrowArrayAppendBytes(struct ArrowArray* array,
                                                   struct ArrowBufferView value);

/// \brief Append a string value to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type, EOVERFLOW if appending value would overflow
/// the offset type (e.g., if the data buffer would be larger than 2 GB for a
/// non-large string type), or EINVAL otherwise (e.g., the underlying array is not a
/// string or large string array).
static inline ArrowErrorCode ArrowArrayAppendString(struct ArrowArray* array,
                                                    struct ArrowStringView value);

/// \brief Append a Interval to an array
///
/// Returns NANOARROW_OK if value can be exactly represented by
/// the underlying storage type or EINVAL otherwise.
static inline ArrowErrorCode ArrowArrayAppendInterval(struct ArrowArray* array,
                                                      const struct ArrowInterval* value);

/// \brief Append a decimal value to an array
///
/// Returns NANOARROW_OK if array is a decimal array with the appropriate
/// bitwidth or EINVAL otherwise.
static inline ArrowErrorCode ArrowArrayAppendDecimal(struct ArrowArray* array,
                                                     const struct ArrowDecimal* value);

/// \brief Finish a nested array element
///
/// Appends a non-null element to the array based on the first child's current
/// length. Returns NANOARROW_OK if the item was successfully added, EOVERFLOW
/// if the child of a list or map array would exceed INT_MAX elements, or EINVAL
/// if the underlying storage type is not a struct, list, large list, or fixed-size
/// list, or if there was an attempt to add a struct or fixed-size list element where the
/// length of the child array(s) did not match the expected length.
static inline ArrowErrorCode ArrowArrayFinishElement(struct ArrowArray* array);

/// \brief Finish a union array element
///
/// Appends an element to the union type ids buffer and increments array->length.
/// For sparse unions, up to one element is added to non type-id children. Returns
/// EINVAL if the underlying storage type is not a union, if type_id is not valid,
/// or if child sizes after appending are inconsistent.
static inline ArrowErrorCode ArrowArrayFinishUnionElement(struct ArrowArray* array,
                                                          int8_t type_id);

/// \brief Shrink buffer capacity to the size required
///
/// Also applies shrinking to any child arrays. array must have been allocated using
/// ArrowArrayInitFromType
static inline ArrowErrorCode ArrowArrayShrinkToFit(struct ArrowArray* array);

/// \brief Finish building an ArrowArray
///
/// Flushes any pointers from internal buffers that may have been reallocated
/// into array->buffers and checks the actual size of the buffers
/// against the expected size based on the final length.
/// array must have been allocated using ArrowArrayInitFromType()
ArrowErrorCode ArrowArrayFinishBuildingDefault(struct ArrowArray* array,
                                               struct ArrowError* error);

/// \brief Finish building an ArrowArray with explicit validation
///
/// Finish building with an explicit validation level. This could perform less validation
/// (i.e. NANOARROW_VALIDATION_LEVEL_NONE or NANOARROW_VALIDATION_LEVEL_MINIMAL) if CPU
/// buffer data access is not possible or more validation (i.e.,
/// NANOARROW_VALIDATION_LEVEL_FULL) if buffer content was obtained from an untrusted or
/// corruptible source.
ArrowErrorCode ArrowArrayFinishBuilding(struct ArrowArray* array,
                                        enum ArrowValidationLevel validation_level,
                                        struct ArrowError* error);

/// @}

/// \defgroup nanoarrow-array-view Reading arrays
///
/// These functions read and validate the contents ArrowArray structures.
///
/// @{

/// \brief Initialize the contents of an ArrowArrayView
void ArrowArrayViewInitFromType(struct ArrowArrayView* array_view,
                                enum ArrowType storage_type);

/// \brief Move an ArrowArrayView
///
/// Transfers the ArrowArrayView data and lifecycle management to another
/// address and resets the contents of src.
static inline void ArrowArrayViewMove(struct ArrowArrayView* src,
                                      struct ArrowArrayView* dst);

/// \brief Initialize the contents of an ArrowArrayView from an ArrowSchema
ArrowErrorCode ArrowArrayViewInitFromSchema(struct ArrowArrayView* array_view,
                                            const struct ArrowSchema* schema,
                                            struct ArrowError* error);

/// \brief Allocate the array_view->children array
///
/// Includes the memory for each child struct ArrowArrayView
ArrowErrorCode ArrowArrayViewAllocateChildren(struct ArrowArrayView* array_view,
                                              int64_t n_children);

/// \brief Allocate array_view->dictionary
ArrowErrorCode ArrowArrayViewAllocateDictionary(struct ArrowArrayView* array_view);

/// \brief Set data-independent buffer sizes from length
void ArrowArrayViewSetLength(struct ArrowArrayView* array_view, int64_t length);

/// \brief Set buffer sizes and data pointers from an ArrowArray
ArrowErrorCode ArrowArrayViewSetArray(struct ArrowArrayView* array_view,
                                      const struct ArrowArray* array,
                                      struct ArrowError* error);

/// \brief Set buffer sizes and data pointers from an ArrowArray except for those
/// that require dereferencing buffer content.
ArrowErrorCode ArrowArrayViewSetArrayMinimal(struct ArrowArrayView* array_view,
                                             const struct ArrowArray* array,
                                             struct ArrowError* error);

/// \brief Performs checks on the content of an ArrowArrayView
///
/// If using ArrowArrayViewSetArray() to back array_view with an ArrowArray,
/// the buffer sizes and some content (fist and last offset) have already
/// been validated at the "default" level. If setting the buffer pointers
/// and sizes otherwise, you may wish to perform checks at a different level. See
/// documentation for ArrowValidationLevel for the details of checks performed
/// at each level.
ArrowErrorCode ArrowArrayViewValidate(struct ArrowArrayView* array_view,
                                      enum ArrowValidationLevel validation_level,
                                      struct ArrowError* error);

/// \brief Reset the contents of an ArrowArrayView and frees resources
void ArrowArrayViewReset(struct ArrowArrayView* array_view);

/// \brief Check for a null element in an ArrowArrayView
static inline int8_t ArrowArrayViewIsNull(const struct ArrowArrayView* array_view,
                                          int64_t i);

/// \brief Get the type id of a union array element
static inline int8_t ArrowArrayViewUnionTypeId(const struct ArrowArrayView* array_view,
                                               int64_t i);

/// \brief Get the child index of a union array element
static inline int8_t ArrowArrayViewUnionChildIndex(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get the index to use into the relevant union child array
static inline int64_t ArrowArrayViewUnionChildOffset(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get an element in an ArrowArrayView as an integer
///
/// This function does not check for null values, that values are actually integers, or
/// that values are within a valid range for an int64.
static inline int64_t ArrowArrayViewGetIntUnsafe(const struct ArrowArrayView* array_view,
                                                 int64_t i);

/// \brief Get an element in an ArrowArrayView as an unsigned integer
///
/// This function does not check for null values, that values are actually integers, or
/// that values are within a valid range for a uint64.
static inline uint64_t ArrowArrayViewGetUIntUnsafe(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get an element in an ArrowArrayView as a double
///
/// This function does not check for null values, or
/// that values are within a valid range for a double.
static inline double ArrowArrayViewGetDoubleUnsafe(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get an element in an ArrowArrayView as an ArrowStringView
///
/// This function does not check for null values.
static inline struct ArrowStringView ArrowArrayViewGetStringUnsafe(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get an element in an ArrowArrayView as an ArrowBufferView
///
/// This function does not check for null values.
static inline struct ArrowBufferView ArrowArrayViewGetBytesUnsafe(
    const struct ArrowArrayView* array_view, int64_t i);

/// \brief Get an element in an ArrowArrayView as an ArrowDecimal
///
/// This function does not check for null values. The out parameter must
/// be initialized with ArrowDecimalInit() with the proper parameters for this
/// type before calling this for the first time.
static inline void ArrowArrayViewGetDecimalUnsafe(const struct ArrowArrayView* array_view,
                                                  int64_t i, struct ArrowDecimal* out);

/// @}

/// \defgroup nanoarrow-basic-array-stream Basic ArrowArrayStream implementation
///
/// An implementation of an ArrowArrayStream based on a collection of
/// zero or more previously-existing ArrowArray objects. Users should
/// initialize and/or validate the contents before transferring the
/// responsibility of the ArrowArrayStream elsewhere.
///
/// @{

/// \brief Initialize an ArrowArrayStream backed by this implementation
///
/// This function moves the ownership of schema to the array_stream. If
/// this function returns NANOARROW_OK, the caller is responsible for
/// releasing the ArrowArrayStream.
ArrowErrorCode ArrowBasicArrayStreamInit(struct ArrowArrayStream* array_stream,
                                         struct ArrowSchema* schema, int64_t n_arrays);

/// \brief Set the ith ArrowArray in this ArrowArrayStream.
///
/// array_stream must have been initialized with ArrowBasicArrayStreamInit().
/// This function move the ownership of array to the array_stream. i must
/// be greater than zero and less than the value of n_arrays passed in
/// ArrowBasicArrayStreamInit(). Callers are not required to fill all
/// n_arrays members (i.e., n_arrays is a maximum bound).
void ArrowBasicArrayStreamSetArray(struct ArrowArrayStream* array_stream, int64_t i,
                                   struct ArrowArray* array);

/// \brief Validate the contents of this ArrowArrayStream
///
/// array_stream must have been initialized with ArrowBasicArrayStreamInit().
/// This function uses ArrowArrayStreamInitFromSchema() and ArrowArrayStreamSetArray()
/// to validate the contents of the arrays.
ArrowErrorCode ArrowBasicArrayStreamValidate(const struct ArrowArrayStream* array_stream,
                                             struct ArrowError* error);

/// @}

// Inline function definitions

#ifdef __cplusplus
}
#endif

#endif
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_BUFFER_INLINE_H_INCLUDED
#define NANOARROW_BUFFER_INLINE_H_INCLUDED

#include <errno.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int64_t _ArrowGrowByFactor(int64_t current_capacity, int64_t new_capacity) {
  int64_t doubled_capacity = current_capacity * 2;
  if (doubled_capacity > new_capacity) {
    return doubled_capacity;
  } else {
    return new_capacity;
  }
}

static inline void ArrowBufferInit(struct ArrowBuffer* buffer) {
  buffer->data = NULL;
  buffer->size_bytes = 0;
  buffer->capacity_bytes = 0;
  buffer->allocator = ArrowBufferAllocatorDefault();
}

static inline ArrowErrorCode ArrowBufferSetAllocator(
    struct ArrowBuffer* buffer, struct ArrowBufferAllocator allocator) {
  if (buffer->data == NULL) {
    buffer->allocator = allocator;
    return NANOARROW_OK;
  } else {
    return EINVAL;
  }
}

static inline void ArrowBufferReset(struct ArrowBuffer* buffer) {
  if (buffer->data != NULL) {
    buffer->allocator.free(&buffer->allocator, (uint8_t*)buffer->data,
                           buffer->capacity_bytes);
    buffer->data = NULL;
  }

  buffer->capacity_bytes = 0;
  buffer->size_bytes = 0;
}

static inline void ArrowBufferMove(struct ArrowBuffer* src, struct ArrowBuffer* dst) {
  memcpy(dst, src, sizeof(struct ArrowBuffer));
  src->data = NULL;
  ArrowBufferReset(src);
}

static inline ArrowErrorCode ArrowBufferResize(struct ArrowBuffer* buffer,
                                               int64_t new_capacity_bytes,
                                               char shrink_to_fit) {
  if (new_capacity_bytes < 0) {
    return EINVAL;
  }

  if (new_capacity_bytes > buffer->capacity_bytes || shrink_to_fit) {
    buffer->data = buffer->allocator.reallocate(
        &buffer->allocator, buffer->data, buffer->capacity_bytes, new_capacity_bytes);
    if (buffer->data == NULL && new_capacity_bytes > 0) {
      buffer->capacity_bytes = 0;
      buffer->size_bytes = 0;
      return ENOMEM;
    }

    buffer->capacity_bytes = new_capacity_bytes;
  }

  // Ensures that when shrinking that size <= capacity
  if (new_capacity_bytes < buffer->size_bytes) {
    buffer->size_bytes = new_capacity_bytes;
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBufferReserve(struct ArrowBuffer* buffer,
                                                int64_t additional_size_bytes) {
  int64_t min_capacity_bytes = buffer->size_bytes + additional_size_bytes;
  if (min_capacity_bytes <= buffer->capacity_bytes) {
    return NANOARROW_OK;
  }

  return ArrowBufferResize(
      buffer, _ArrowGrowByFactor(buffer->capacity_bytes, min_capacity_bytes), 0);
}

static inline void ArrowBufferAppendUnsafe(struct ArrowBuffer* buffer, const void* data,
                                           int64_t size_bytes) {
  if (size_bytes > 0) {
    memcpy(buffer->data + buffer->size_bytes, data, size_bytes);
    buffer->size_bytes += size_bytes;
  }
}

static inline ArrowErrorCode ArrowBufferAppend(struct ArrowBuffer* buffer,
                                               const void* data, int64_t size_bytes) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer, size_bytes));

  ArrowBufferAppendUnsafe(buffer, data, size_bytes);
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBufferAppendInt8(struct ArrowBuffer* buffer,
                                                   int8_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(int8_t));
}

static inline ArrowErrorCode ArrowBufferAppendUInt8(struct ArrowBuffer* buffer,
                                                    uint8_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(uint8_t));
}

static inline ArrowErrorCode ArrowBufferAppendInt16(struct ArrowBuffer* buffer,
                                                    int16_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(int16_t));
}

static inline ArrowErrorCode ArrowBufferAppendUInt16(struct ArrowBuffer* buffer,
                                                     uint16_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(uint16_t));
}

static inline ArrowErrorCode ArrowBufferAppendInt32(struct ArrowBuffer* buffer,
                                                    int32_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(int32_t));
}

static inline ArrowErrorCode ArrowBufferAppendUInt32(struct ArrowBuffer* buffer,
                                                     uint32_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(uint32_t));
}

static inline ArrowErrorCode ArrowBufferAppendInt64(struct ArrowBuffer* buffer,
                                                    int64_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(int64_t));
}

static inline ArrowErrorCode ArrowBufferAppendUInt64(struct ArrowBuffer* buffer,
                                                     uint64_t value) {
  return ArrowBufferAppend(buffer, &value, sizeof(uint64_t));
}

static inline ArrowErrorCode ArrowBufferAppendDouble(struct ArrowBuffer* buffer,
                                                     double value) {
  return ArrowBufferAppend(buffer, &value, sizeof(double));
}

static inline ArrowErrorCode ArrowBufferAppendFloat(struct ArrowBuffer* buffer,
                                                    float value) {
  return ArrowBufferAppend(buffer, &value, sizeof(float));
}

static inline ArrowErrorCode ArrowBufferAppendStringView(struct ArrowBuffer* buffer,
                                                         struct ArrowStringView value) {
  return ArrowBufferAppend(buffer, value.data, value.size_bytes);
}

static inline ArrowErrorCode ArrowBufferAppendBufferView(struct ArrowBuffer* buffer,
                                                         struct ArrowBufferView value) {
  return ArrowBufferAppend(buffer, value.data.data, value.size_bytes);
}

static inline ArrowErrorCode ArrowBufferAppendFill(struct ArrowBuffer* buffer,
                                                   uint8_t value, int64_t size_bytes) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer, size_bytes));

  memset(buffer->data + buffer->size_bytes, value, size_bytes);
  buffer->size_bytes += size_bytes;
  return NANOARROW_OK;
}

static const uint8_t _ArrowkBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t _ArrowkFlippedBitmask[] = {254, 253, 251, 247, 239, 223, 191, 127};
static const uint8_t _ArrowkPrecedingBitmask[] = {0, 1, 3, 7, 15, 31, 63, 127};
static const uint8_t _ArrowkTrailingBitmask[] = {255, 254, 252, 248, 240, 224, 192, 128};

static const uint8_t _ArrowkBytePopcount[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
    4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
    5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
    4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
    3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
    5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static inline int64_t _ArrowRoundUpToMultipleOf8(int64_t value) {
  return (value + 7) & ~((int64_t)7);
}

static inline int64_t _ArrowRoundDownToMultipleOf8(int64_t value) {
  return (value / 8) * 8;
}

static inline int64_t _ArrowBytesForBits(int64_t bits) {
  return (bits >> 3) + ((bits & 7) != 0);
}

static inline void _ArrowBitsUnpackInt8(const uint8_t word, int8_t* out) {
  out[0] = (word & 0x1) != 0;
  out[1] = (word & 0x2) != 0;
  out[2] = (word & 0x4) != 0;
  out[3] = (word & 0x8) != 0;
  out[4] = (word & 0x10) != 0;
  out[5] = (word & 0x20) != 0;
  out[6] = (word & 0x40) != 0;
  out[7] = (word & 0x80) != 0;
}

static inline void _ArrowBitsUnpackInt32(const uint8_t word, int32_t* out) {
  out[0] = (word & 0x1) != 0;
  out[1] = (word & 0x2) != 0;
  out[2] = (word & 0x4) != 0;
  out[3] = (word & 0x8) != 0;
  out[4] = (word & 0x10) != 0;
  out[5] = (word & 0x20) != 0;
  out[6] = (word & 0x40) != 0;
  out[7] = (word & 0x80) != 0;
}

static inline void _ArrowBitmapPackInt8(const int8_t* values, uint8_t* out) {
  *out = (values[0] | ((values[1] + 0x1) & 0x2) | ((values[2] + 0x3) & 0x4) |
          ((values[3] + 0x7) & 0x8) | ((values[4] + 0xf) & 0x10) |
          ((values[5] + 0x1f) & 0x20) | ((values[6] + 0x3f) & 0x40) |
          ((values[7] + 0x7f) & 0x80));
}

static inline void _ArrowBitmapPackInt32(const int32_t* values, uint8_t* out) {
  *out = (values[0] | ((values[1] + 0x1) & 0x2) | ((values[2] + 0x3) & 0x4) |
          ((values[3] + 0x7) & 0x8) | ((values[4] + 0xf) & 0x10) |
          ((values[5] + 0x1f) & 0x20) | ((values[6] + 0x3f) & 0x40) |
          ((values[7] + 0x7f) & 0x80));
}

static inline int8_t ArrowBitGet(const uint8_t* bits, int64_t i) {
  return (bits[i >> 3] >> (i & 0x07)) & 1;
}

static inline void ArrowBitsUnpackInt8(const uint8_t* bits, int64_t start_offset,
                                       int64_t length, int8_t* out) {
  if (length == 0) {
    return;
  }

  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;
  const int64_t i_last_valid = i_end - 1;

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_last_valid = i_last_valid / 8;

  if (bytes_begin == bytes_last_valid) {
    for (int i = 0; i < length; i++) {
      out[i] = ArrowBitGet(&bits[bytes_begin], i + i_begin % 8);
    }

    return;
  }

  // first byte
  for (int i = 0; i < 8 - (i_begin % 8); i++) {
    *out++ = ArrowBitGet(&bits[bytes_begin], i + i_begin % 8);
  }

  // middle bytes
  for (int64_t i = bytes_begin + 1; i < bytes_last_valid; i++) {
    _ArrowBitsUnpackInt8(bits[i], out);
    out += 8;
  }

  // last byte
  const int bits_remaining = i_end % 8 == 0 ? 8 : i_end % 8;
  for (int i = 0; i < bits_remaining; i++) {
    *out++ = ArrowBitGet(&bits[bytes_last_valid], i);
  }
}

static inline void ArrowBitsUnpackInt32(const uint8_t* bits, int64_t start_offset,
                                        int64_t length, int32_t* out) {
  if (length == 0) {
    return;
  }

  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;
  const int64_t i_last_valid = i_end - 1;

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_last_valid = i_last_valid / 8;

  if (bytes_begin == bytes_last_valid) {
    for (int i = 0; i < length; i++) {
      out[i] = ArrowBitGet(&bits[bytes_begin], i + i_begin % 8);
    }

    return;
  }

  // first byte
  for (int i = 0; i < 8 - (i_begin % 8); i++) {
    *out++ = ArrowBitGet(&bits[bytes_begin], i + i_begin % 8);
  }

  // middle bytes
  for (int64_t i = bytes_begin + 1; i < bytes_last_valid; i++) {
    _ArrowBitsUnpackInt32(bits[i], out);
    out += 8;
  }

  // last byte
  const int bits_remaining = i_end % 8 == 0 ? 8 : i_end % 8;
  for (int i = 0; i < bits_remaining; i++) {
    *out++ = ArrowBitGet(&bits[bytes_last_valid], i);
  }
}

static inline void ArrowBitSet(uint8_t* bits, int64_t i) {
  bits[i / 8] |= _ArrowkBitmask[i % 8];
}

static inline void ArrowBitClear(uint8_t* bits, int64_t i) {
  bits[i / 8] &= _ArrowkFlippedBitmask[i % 8];
}

static inline void ArrowBitSetTo(uint8_t* bits, int64_t i, uint8_t bit_is_set) {
  bits[i / 8] ^=
      ((uint8_t)(-((uint8_t)(bit_is_set != 0)) ^ bits[i / 8])) & _ArrowkBitmask[i % 8];
}

static inline void ArrowBitsSetTo(uint8_t* bits, int64_t start_offset, int64_t length,
                                  uint8_t bits_are_set) {
  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;
  const uint8_t fill_byte = (uint8_t)(-bits_are_set);

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_end = i_end / 8 + 1;

  const uint8_t first_byte_mask = _ArrowkPrecedingBitmask[i_begin % 8];
  const uint8_t last_byte_mask = _ArrowkTrailingBitmask[i_end % 8];

  if (bytes_end == bytes_begin + 1) {
    // set bits within a single byte
    const uint8_t only_byte_mask =
        i_end % 8 == 0 ? first_byte_mask : (uint8_t)(first_byte_mask | last_byte_mask);
    bits[bytes_begin] &= only_byte_mask;
    bits[bytes_begin] |= (uint8_t)(fill_byte & ~only_byte_mask);
    return;
  }

  // set/clear trailing bits of first byte
  bits[bytes_begin] &= first_byte_mask;
  bits[bytes_begin] |= (uint8_t)(fill_byte & ~first_byte_mask);

  if (bytes_end - bytes_begin > 2) {
    // set/clear whole bytes
    memset(bits + bytes_begin + 1, fill_byte, (size_t)(bytes_end - bytes_begin - 2));
  }

  if (i_end % 8 == 0) {
    return;
  }

  // set/clear leading bits of last byte
  bits[bytes_end - 1] &= last_byte_mask;
  bits[bytes_end - 1] |= (uint8_t)(fill_byte & ~last_byte_mask);
}

static inline int64_t ArrowBitCountSet(const uint8_t* bits, int64_t start_offset,
                                       int64_t length) {
  if (length == 0) {
    return 0;
  }

  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;
  const int64_t i_last_valid = i_end - 1;

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_last_valid = i_last_valid / 8;

  if (bytes_begin == bytes_last_valid) {
    // count bits within a single byte
    const uint8_t first_byte_mask = _ArrowkPrecedingBitmask[i_end % 8];
    const uint8_t last_byte_mask = _ArrowkTrailingBitmask[i_begin % 8];

    const uint8_t only_byte_mask =
        i_end % 8 == 0 ? last_byte_mask : (uint8_t)(first_byte_mask & last_byte_mask);

    const uint8_t byte_masked = bits[bytes_begin] & only_byte_mask;
    return _ArrowkBytePopcount[byte_masked];
  }

  const uint8_t first_byte_mask = _ArrowkPrecedingBitmask[i_begin % 8];
  const uint8_t last_byte_mask = i_end % 8 == 0 ? 0 : _ArrowkTrailingBitmask[i_end % 8];
  int64_t count = 0;

  // first byte
  count += _ArrowkBytePopcount[bits[bytes_begin] & ~first_byte_mask];

  // middle bytes
  for (int64_t i = bytes_begin + 1; i < bytes_last_valid; i++) {
    count += _ArrowkBytePopcount[bits[i]];
  }

  // last byte
  count += _ArrowkBytePopcount[bits[bytes_last_valid] & ~last_byte_mask];

  return count;
}

static inline void ArrowBitmapInit(struct ArrowBitmap* bitmap) {
  ArrowBufferInit(&bitmap->buffer);
  bitmap->size_bits = 0;
}

static inline void ArrowBitmapMove(struct ArrowBitmap* src, struct ArrowBitmap* dst) {
  ArrowBufferMove(&src->buffer, &dst->buffer);
  dst->size_bits = src->size_bits;
  src->size_bits = 0;
}

static inline ArrowErrorCode ArrowBitmapReserve(struct ArrowBitmap* bitmap,
                                                int64_t additional_size_bits) {
  int64_t min_capacity_bits = bitmap->size_bits + additional_size_bits;
  if (min_capacity_bits <= (bitmap->buffer.capacity_bytes * 8)) {
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(
      ArrowBufferReserve(&bitmap->buffer, _ArrowBytesForBits(additional_size_bits)));

  bitmap->buffer.data[bitmap->buffer.capacity_bytes - 1] = 0;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapResize(struct ArrowBitmap* bitmap,
                                               int64_t new_capacity_bits,
                                               char shrink_to_fit) {
  if (new_capacity_bits < 0) {
    return EINVAL;
  }

  int64_t new_capacity_bytes = _ArrowBytesForBits(new_capacity_bits);
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferResize(&bitmap->buffer, new_capacity_bytes, shrink_to_fit));

  if (new_capacity_bits < bitmap->size_bits) {
    bitmap->size_bits = new_capacity_bits;
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapAppend(struct ArrowBitmap* bitmap,
                                               uint8_t bits_are_set, int64_t length) {
  NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(bitmap, length));

  ArrowBitmapAppendUnsafe(bitmap, bits_are_set, length);
  return NANOARROW_OK;
}

static inline void ArrowBitmapAppendUnsafe(struct ArrowBitmap* bitmap,
                                           uint8_t bits_are_set, int64_t length) {
  ArrowBitsSetTo(bitmap->buffer.data, bitmap->size_bits, length, bits_are_set);
  bitmap->size_bits += length;
  bitmap->buffer.size_bytes = _ArrowBytesForBits(bitmap->size_bits);
}

static inline void ArrowBitmapAppendInt8Unsafe(struct ArrowBitmap* bitmap,
                                               const int8_t* values, int64_t n_values) {
  if (n_values == 0) {
    return;
  }

  const int8_t* values_cursor = values;
  int64_t n_remaining = n_values;
  int64_t out_i_cursor = bitmap->size_bits;
  uint8_t* out_cursor = bitmap->buffer.data + bitmap->size_bits / 8;

  // First byte
  if ((out_i_cursor % 8) != 0) {
    int64_t n_partial_bits = _ArrowRoundUpToMultipleOf8(out_i_cursor) - out_i_cursor;
    for (int i = 0; i < n_partial_bits; i++) {
      ArrowBitSetTo(bitmap->buffer.data, out_i_cursor++, values[i]);
    }

    out_cursor++;
    values_cursor += n_partial_bits;
    n_remaining -= n_partial_bits;
  }

  // Middle bytes
  int64_t n_full_bytes = n_remaining / 8;
  for (int64_t i = 0; i < n_full_bytes; i++) {
    _ArrowBitmapPackInt8(values_cursor, out_cursor);
    values_cursor += 8;
    out_cursor++;
  }

  // Last byte
  out_i_cursor += n_full_bytes * 8;
  n_remaining -= n_full_bytes * 8;
  if (n_remaining > 0) {
    // Zero out the last byte
    *out_cursor = 0x00;
    for (int i = 0; i < n_remaining; i++) {
      ArrowBitSetTo(bitmap->buffer.data, out_i_cursor++, values_cursor[i]);
    }
    out_cursor++;
  }

  bitmap->size_bits += n_values;
  bitmap->buffer.size_bytes = out_cursor - bitmap->buffer.data;
}

static inline void ArrowBitmapAppendInt32Unsafe(struct ArrowBitmap* bitmap,
                                                const int32_t* values, int64_t n_values) {
  if (n_values == 0) {
    return;
  }

  const int32_t* values_cursor = values;
  int64_t n_remaining = n_values;
  int64_t out_i_cursor = bitmap->size_bits;
  uint8_t* out_cursor = bitmap->buffer.data + bitmap->size_bits / 8;

  // First byte
  if ((out_i_cursor % 8) != 0) {
    int64_t n_partial_bits = _ArrowRoundUpToMultipleOf8(out_i_cursor) - out_i_cursor;
    for (int i = 0; i < n_partial_bits; i++) {
      ArrowBitSetTo(bitmap->buffer.data, out_i_cursor++, values[i]);
    }

    out_cursor++;
    values_cursor += n_partial_bits;
    n_remaining -= n_partial_bits;
  }

  // Middle bytes
  int64_t n_full_bytes = n_remaining / 8;
  for (int64_t i = 0; i < n_full_bytes; i++) {
    _ArrowBitmapPackInt32(values_cursor, out_cursor);
    values_cursor += 8;
    out_cursor++;
  }

  // Last byte
  out_i_cursor += n_full_bytes * 8;
  n_remaining -= n_full_bytes * 8;
  if (n_remaining > 0) {
    // Zero out the last byte
    *out_cursor = 0x00;
    for (int i = 0; i < n_remaining; i++) {
      ArrowBitSetTo(bitmap->buffer.data, out_i_cursor++, values_cursor[i]);
    }
    out_cursor++;
  }

  bitmap->size_bits += n_values;
  bitmap->buffer.size_bytes = out_cursor - bitmap->buffer.data;
}

static inline void ArrowBitmapReset(struct ArrowBitmap* bitmap) {
  ArrowBufferReset(&bitmap->buffer);
  bitmap->size_bits = 0;
}

#ifdef __cplusplus
}
#endif

#endif
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_ARRAY_INLINE_H_INCLUDED
#define NANOARROW_ARRAY_INLINE_H_INCLUDED

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline struct ArrowBitmap* ArrowArrayValidityBitmap(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  return &private_data->bitmap;
}

static inline struct ArrowBuffer* ArrowArrayBuffer(struct ArrowArray* array, int64_t i) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  switch (i) {
    case 0:
      return &private_data->bitmap.buffer;
    default:
      return private_data->buffers + i - 1;
  }
}

// We don't currently support the case of unions where type_id != child_index;
// however, these functions are used to keep track of where that assumption
// is made.
static inline int8_t _ArrowArrayUnionChildIndex(struct ArrowArray* array,
                                                int8_t type_id) {
  return type_id;
}

static inline int8_t _ArrowArrayUnionTypeId(struct ArrowArray* array,
                                            int8_t child_index) {
  return child_index;
}

static inline int8_t _ArrowParseUnionTypeIds(const char* type_ids, int8_t* out) {
  if (*type_ids == '\0') {
    return 0;
  }

  int32_t i = 0;
  long type_id;
  char* end_ptr;
  do {
    type_id = strtol(type_ids, &end_ptr, 10);
    if (end_ptr == type_ids || type_id < 0 || type_id > 127) {
      return -1;
    }

    if (out != NULL) {
      out[i] = (int8_t)type_id;
    }

    i++;

    type_ids = end_ptr;
    if (*type_ids == '\0') {
      return i;
    } else if (*type_ids != ',') {
      return -1;
    } else {
      type_ids++;
    }
  } while (1);

  return -1;
}

static inline int8_t _ArrowParsedUnionTypeIdsWillEqualChildIndices(const int8_t* type_ids,
                                                                   int64_t n_type_ids,
                                                                   int64_t n_children) {
  if (n_type_ids != n_children) {
    return 0;
  }

  for (int8_t i = 0; i < n_type_ids; i++) {
    if (type_ids[i] != i) {
      return 0;
    }
  }

  return 1;
}

static inline int8_t _ArrowUnionTypeIdsWillEqualChildIndices(const char* type_id_str,
                                                             int64_t n_children) {
  int8_t type_ids[128];
  int8_t n_type_ids = _ArrowParseUnionTypeIds(type_id_str, type_ids);
  return _ArrowParsedUnionTypeIdsWillEqualChildIndices(type_ids, n_type_ids, n_children);
}

static inline ArrowErrorCode ArrowArrayStartAppending(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
      return EINVAL;
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      // Note that this value could be -1 if the type_ids string was invalid
      if (private_data->union_type_id_is_child_index != 1) {
        return EINVAL;
      } else {
        break;
      }
    default:
      break;
  }
  if (private_data->storage_type == NANOARROW_TYPE_UNINITIALIZED) {
    return EINVAL;
  }

  // Initialize any data offset buffer with a single zero
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
        private_data->layout.element_size_bits[i] == 64) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt64(ArrowArrayBuffer(array, i), 0));
    } else if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
               private_data->layout.element_size_bits[i] == 32) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(ArrowArrayBuffer(array, i), 0));
    }
  }

  // Start building any child arrays or dictionaries
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array->children[i]));
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array->dictionary));
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayShrinkToFit(struct ArrowArray* array) {
  for (int64_t i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(buffer, buffer->size_bytes, 1));
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayShrinkToFit(array->children[i]));
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayShrinkToFit(array->dictionary));
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode _ArrowArrayAppendBits(struct ArrowArray* array,
                                                   int64_t buffer_i, uint8_t value,
                                                   int64_t n) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  struct ArrowBuffer* buffer = ArrowArrayBuffer(array, buffer_i);
  int64_t bytes_required =
      _ArrowRoundUpToMultipleOf8(private_data->layout.element_size_bits[buffer_i] *
                                 (array->length + 1)) /
      8;
  if (bytes_required > buffer->size_bytes) {
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendFill(buffer, 0, bytes_required - buffer->size_bytes));
  }

  ArrowBitsSetTo(buffer->data, array->length, n, value);
  return NANOARROW_OK;
}

static inline ArrowErrorCode _ArrowArrayAppendEmptyInternal(struct ArrowArray* array,
                                                            int64_t n, uint8_t is_valid) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (n == 0) {
    return NANOARROW_OK;
  }

  // Some type-specific handling
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_NA:
      // (An empty value for a null array *is* a null)
      array->null_count += n;
      array->length += n;
      return NANOARROW_OK;

    case NANOARROW_TYPE_DENSE_UNION: {
      // Add one null to the first child and append n references to that child
      int8_t type_id = _ArrowArrayUnionTypeId(array, 0);
      NANOARROW_RETURN_NOT_OK(
          _ArrowArrayAppendEmptyInternal(array->children[0], 1, is_valid));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendFill(ArrowArrayBuffer(array, 0), type_id, n));
      for (int64_t i = 0; i < n; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
            ArrowArrayBuffer(array, 1), (int32_t)array->children[0]->length - 1));
      }
      // For the purposes of array->null_count, union elements are never considered "null"
      // even if some children contain nulls.
      array->length += n;
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_SPARSE_UNION: {
      // Add n nulls to the first child and append n references to that child
      int8_t type_id = _ArrowArrayUnionTypeId(array, 0);
      NANOARROW_RETURN_NOT_OK(
          _ArrowArrayAppendEmptyInternal(array->children[0], n, is_valid));
      for (int64_t i = 1; i < array->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], n));
      }

      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendFill(ArrowArrayBuffer(array, 0), type_id, n));
      // For the purposes of array->null_count, union elements are never considered "null"
      // even if some children contain nulls.
      array->length += n;
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(
          array->children[0], n * private_data->layout.child_size_elements));
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], n));
      }
      break;

    default:
      break;
  }

  // Append n is_valid bits to the validity bitmap. If we haven't allocated a bitmap yet
  // and we need to append nulls, do it now.
  if (!is_valid && private_data->bitmap.buffer.data == NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, array->length + n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 1, array->length);
    ArrowBitmapAppendUnsafe(&private_data->bitmap, is_valid, n);
  } else if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, is_valid, n);
  }

  // Add appropriate buffer fill
  struct ArrowBuffer* buffer;
  int64_t size_bytes;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    buffer = ArrowArrayBuffer(array, i);
    size_bytes = private_data->layout.element_size_bits[i] / 8;

    switch (private_data->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_NONE:
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Append the current value at the end of the offset buffer for each element
        NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer, size_bytes * n));

        for (int64_t j = 0; j < n; j++) {
          ArrowBufferAppendUnsafe(buffer, buffer->data + size_bytes * (array->length + j),
                                  size_bytes);
        }

        // Skip the data buffer
        i++;
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
        // Zero out the next bit of memory
        if (private_data->layout.element_size_bits[i] % 8 == 0) {
          NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFill(buffer, 0, size_bytes * n));
        } else {
          NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, i, 0, n));
        }
        continue;

      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        // These cases return above
        return EINVAL;
    }
  }

  array->length += n;
  array->null_count += n * !is_valid;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendNull(struct ArrowArray* array, int64_t n) {
  return _ArrowArrayAppendEmptyInternal(array, n, 0);
}

static inline ArrowErrorCode ArrowArrayAppendEmpty(struct ArrowArray* array, int64_t n) {
  return _ArrowArrayAppendEmptyInternal(array, n, 1);
}

static inline ArrowErrorCode ArrowArrayAppendInt(struct ArrowArray* array,
                                                 int64_t value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_INT64:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(int64_t)));
      break;
    case NANOARROW_TYPE_INT32:
      _NANOARROW_CHECK_RANGE(value, INT32_MIN, INT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, (int32_t)value));
      break;
    case NANOARROW_TYPE_INT16:
      _NANOARROW_CHECK_RANGE(value, INT16_MIN, INT16_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt16(data_buffer, (int16_t)value));
      break;
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_RANGE(value, INT8_MIN, INT8_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt8(data_buffer, (int8_t)value));
      break;
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_RANGE(value, 0, INT64_MAX);
      return ArrowArrayAppendUInt(array, value);
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, (double)value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    case NANOARROW_TYPE_BOOL:
      NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, 1, value != 0, 1));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendUInt(struct ArrowArray* array,
                                                  uint64_t value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_UINT64:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(uint64_t)));
      break;
    case NANOARROW_TYPE_UINT32:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(data_buffer, (uint32_t)value));
      break;
    case NANOARROW_TYPE_UINT16:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT16_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt16(data_buffer, (uint16_t)value));
      break;
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT8_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(data_buffer, (uint8_t)value));
      break;
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_UPPER_LIMIT(value, INT64_MAX);
      return ArrowArrayAppendInt(array, value);
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, (double)value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    case NANOARROW_TYPE_BOOL:
      NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, 1, value != 0, 1));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendDouble(struct ArrowArray* array,
                                                    double value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(double)));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendBytes(struct ArrowArray* array,
                                                   struct ArrowBufferView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* offset_buffer = ArrowArrayBuffer(array, 1);
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(
      array, 1 + (private_data->storage_type != NANOARROW_TYPE_FIXED_SIZE_BINARY));
  int32_t offset;
  int64_t large_offset;
  int64_t fixed_size_bytes = private_data->layout.element_size_bits[1] / 8;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      offset = ((int32_t*)offset_buffer->data)[array->length];
      if ((((int64_t)offset) + value.size_bytes) > INT32_MAX) {
        return EOVERFLOW;
      }

      offset += (int32_t)value.size_bytes;
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(offset_buffer, &offset, sizeof(int32_t)));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      large_offset = ((int64_t*)offset_buffer->data)[array->length];
      large_offset += value.size_bytes;
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(offset_buffer, &large_offset, sizeof(int64_t)));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
      break;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      if (value.size_bytes != fixed_size_bytes) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendString(struct ArrowArray* array,
                                                    struct ArrowStringView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = value.data;
  buffer_view.size_bytes = value.size_bytes;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      return ArrowArrayAppendBytes(array, buffer_view);
    default:
      return EINVAL;
  }
}

static inline ArrowErrorCode ArrowArrayAppendInterval(struct ArrowArray* array,
                                                      const struct ArrowInterval* value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_INTERVAL_MONTHS: {
      if (value->type != NANOARROW_TYPE_INTERVAL_MONTHS) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->months));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      if (value->type != NANOARROW_TYPE_INTERVAL_DAY_TIME) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->days));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->ms));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      if (value->type != NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->months));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->days));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt64(data_buffer, value->ns));
      break;
    }
    default:
      return EINVAL;
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendDecimal(struct ArrowArray* array,
                                                     const struct ArrowDecimal* value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DECIMAL128:
      if (value->n_words != 2) {
        return EINVAL;
      } else {
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value->words, 2 * sizeof(uint64_t)));
        break;
      }
    case NANOARROW_TYPE_DECIMAL256:
      if (value->n_words != 4) {
        return EINVAL;
      } else {
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value->words, 4 * sizeof(uint64_t)));
        break;
      }
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayFinishElement(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int64_t child_length;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      child_length = array->children[0]->length;
      if (child_length > INT32_MAX) {
        return EOVERFLOW;
      }
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendInt32(ArrowArrayBuffer(array, 1), (int32_t)child_length));
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      child_length = array->children[0]->length;
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendInt64(ArrowArrayBuffer(array, 1), child_length));
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      child_length = array->children[0]->length;
      if (child_length !=
          ((array->length + 1) * private_data->layout.child_size_elements)) {
        return EINVAL;
      }
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        child_length = array->children[i]->length;
        if (child_length != (array->length + 1)) {
          return EINVAL;
        }
      }
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayFinishUnionElement(struct ArrowArray* array,
                                                          int8_t type_id) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int64_t child_index = _ArrowArrayUnionChildIndex(array, type_id);
  if (child_index < 0 || child_index >= array->n_children) {
    return EINVAL;
  }

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
      // Append the target child length to the union offsets buffer
      _NANOARROW_CHECK_RANGE(array->children[child_index]->length, 0, INT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
          ArrowArrayBuffer(array, 1), (int32_t)array->children[child_index]->length - 1));
      break;
    case NANOARROW_TYPE_SPARSE_UNION:
      // Append one empty to any non-target column that isn't already the right length
      // or abort if appending a null will result in a column with invalid length
      for (int64_t i = 0; i < array->n_children; i++) {
        if (i == child_index || array->children[i]->length == (array->length + 1)) {
          continue;
        }

        if (array->children[i]->length != array->length) {
          return EINVAL;
        }

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], 1));
      }

      break;
    default:
      return EINVAL;
  }

  // Write to the type_ids buffer
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt8(ArrowArrayBuffer(array, 0), (int8_t)type_id));
  array->length++;
  return NANOARROW_OK;
}

static inline void ArrowArrayViewMove(struct ArrowArrayView* src,
                                      struct ArrowArrayView* dst) {
  memcpy(dst, src, sizeof(struct ArrowArrayView));
  ArrowArrayViewInitFromType(src, NANOARROW_TYPE_UNINITIALIZED);
}

static inline int8_t ArrowArrayViewIsNull(const struct ArrowArrayView* array_view,
                                          int64_t i) {
  const uint8_t* validity_buffer = array_view->buffer_views[0].data.as_uint8;
  i += array_view->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return 0x01;
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      // Unions are "never null" in Arrow land
      return 0x00;
    default:
      return validity_buffer != NULL && !ArrowBitGet(validity_buffer, i);
  }
}

static inline int8_t ArrowArrayViewUnionTypeId(const struct ArrowArrayView* array_view,
                                               int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      return array_view->buffer_views[0].data.as_int8[i];
    default:
      return -1;
  }
}

static inline int8_t ArrowArrayViewUnionChildIndex(
    const struct ArrowArrayView* array_view, int64_t i) {
  int8_t type_id = ArrowArrayViewUnionTypeId(array_view, i);
  if (array_view->union_type_id_map == NULL) {
    return type_id;
  } else {
    return array_view->union_type_id_map[type_id];
  }
}

static inline int64_t ArrowArrayViewUnionChildOffset(
    const struct ArrowArrayView* array_view, int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
      return array_view->buffer_views[1].data.as_int32[i];
    case NANOARROW_TYPE_SPARSE_UNION:
      return i;
    default:
      return -1;
  }
}

static inline int64_t ArrowArrayViewListChildOffset(
    const struct ArrowArrayView* array_view, int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_LIST:
      return array_view->buffer_views[1].data.as_int32[i];
    case NANOARROW_TYPE_LARGE_LIST:
      return array_view->buffer_views[1].data.as_int64[i];
    default:
      return -1;
  }
}

static inline int64_t ArrowArrayViewGetIntUnsafe(const struct ArrowArrayView* array_view,
                                                 int64_t i) {
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  i += array_view->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return (int64_t)data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return (int64_t)data_view->data.as_float[i];
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return INT64_MAX;
  }
}

static inline uint64_t ArrowArrayViewGetUIntUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return (uint64_t)data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return (uint64_t)data_view->data.as_float[i];
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return UINT64_MAX;
  }
}

static inline double ArrowArrayViewGetDoubleUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return (double)data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return (double)data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return data_view->data.as_float[i];
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return DBL_MAX;
  }
}

static inline struct ArrowStringView ArrowArrayViewGetStringUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const char* data_view = array_view->buffer_views[2].data.as_char;

  struct ArrowStringView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.data = data_view + offsets_view->data.as_int32[i];
      view.size_bytes =
          offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.data = data_view + offsets_view->data.as_int64[i];
      view.size_bytes =
          offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.size_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data = array_view->buffer_views[1].data.as_char + (i * view.size_bytes);
      break;
    default:
      view.data = NULL;
      view.size_bytes = 0;
      break;
  }

  return view;
}

static inline struct ArrowBufferView ArrowArrayViewGetBytesUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const uint8_t* data_view = array_view->buffer_views[2].data.as_uint8;

  struct ArrowBufferView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.size_bytes =
          offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.size_bytes =
          offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.size_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data.as_uint8 =
          array_view->buffer_views[1].data.as_uint8 + (i * view.size_bytes);
      break;
    default:
      view.data.data = NULL;
      view.size_bytes = 0;
      break;
  }

  return view;
}

static inline void ArrowArrayViewGetIntervalUnsafe(
    const struct ArrowArrayView* array_view, int64_t i, struct ArrowInterval* out) {
  const uint8_t* data_view = array_view->buffer_views[1].data.as_uint8;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INTERVAL_MONTHS: {
      const size_t size = sizeof(int32_t);
      memcpy(&out->months, data_view + i * size, sizeof(int32_t));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      const size_t size = sizeof(int32_t) + sizeof(int32_t);
      memcpy(&out->days, data_view + i * size, sizeof(int32_t));
      memcpy(&out->ms, data_view + i * size + 4, sizeof(int32_t));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      const size_t size = sizeof(int32_t) + sizeof(int32_t) + sizeof(int64_t);
      memcpy(&out->months, data_view + i * size, sizeof(int32_t));
      memcpy(&out->days, data_view + i * size + 4, sizeof(int32_t));
      memcpy(&out->ns, data_view + i * size + 8, sizeof(int64_t));
      break;
    }
    default:
      break;
  }
}

static inline void ArrowArrayViewGetDecimalUnsafe(const struct ArrowArrayView* array_view,
                                                  int64_t i, struct ArrowDecimal* out) {
  i += array_view->offset;
  const uint8_t* data_view = array_view->buffer_views[1].data.as_uint8;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DECIMAL128:
      ArrowDecimalSetBytes(out, data_view + (i * 16));
      break;
    case NANOARROW_TYPE_DECIMAL256:
      ArrowDecimalSetBytes(out, data_view + (i * 32));
      break;
    default:
      memset(out->words, 0, sizeof(out->words));
      break;
  }
}

#ifdef __cplusplus
}
#endif

#endif
#pragma GCC diagnostic push
#include <errno.h>
#include <string.h>





static GeoArrowErrorCode GeoArrowSchemaInitCoordFixedSizeList(struct ArrowSchema* schema,
                                                              const char* dims) {
  int64_t n_dims = strlen(dims);
  ArrowSchemaInit(schema);
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeFixedSize(
      schema, NANOARROW_TYPE_FIXED_SIZE_LIST, (int32_t)n_dims));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], dims));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_DOUBLE));

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaInitCoordStruct(struct ArrowSchema* schema,
                                                       const char* dims) {
  int64_t n_dims = strlen(dims);
  char dim_name[] = {'\0', '\0'};
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, n_dims));
  for (int64_t i = 0; i < n_dims; i++) {
    dim_name[0] = dims[i];
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaInitFromType(schema->children[i], NANOARROW_TYPE_DOUBLE));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[i], dim_name));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaInitListStruct(struct ArrowSchema* schema,
                                                      enum GeoArrowCoordType coord_type,
                                                      const char* dims, int n,
                                                      const char** child_names) {
  if (n == 0) {
    switch (coord_type) {
      case GEOARROW_COORD_TYPE_SEPARATE:
        return GeoArrowSchemaInitCoordStruct(schema, dims);
      case GEOARROW_COORD_TYPE_INTERLEAVED:
        return GeoArrowSchemaInitCoordFixedSizeList(schema, dims);
      default:
        return EINVAL;
    }
  } else {
    ArrowSchemaInit(schema);
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetFormat(schema, "+l"));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, 1));
    NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitListStruct(schema->children[0], coord_type,
                                                         dims, n - 1, child_names + 1));
    return ArrowSchemaSetName(schema->children[0], child_names[0]);
  }
}

#define CHILD_NAMES_LINESTRING \
  (const char*[]) { "vertices" }
#define CHILD_NAMES_POLYGON \
  (const char*[]) { "rings", "vertices" }
#define CHILD_NAMES_MULTIPOINT \
  (const char*[]) { "points" }
#define CHILD_NAMES_MULTILINESTRING \
  (const char*[]) { "linestrings", "vertices" }
#define CHILD_NAMES_MULTIPOLYGON \
  (const char*[]) { "polygons", "rings", "vertices" }

GeoArrowErrorCode GeoArrowSchemaInit(struct ArrowSchema* schema, enum GeoArrowType type) {
  schema->release = NULL;

  switch (type) {
    case GEOARROW_TYPE_WKB:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
    case GEOARROW_TYPE_LARGE_WKB:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_BINARY);

    case GEOARROW_TYPE_WKT:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
    case GEOARROW_TYPE_LARGE_WKT:
      return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);

    default:
      break;
  }

  enum GeoArrowDimensions dimensions = GeoArrowDimensionsFromType(type);
  enum GeoArrowCoordType coord_type = GeoArrowCoordTypeFromType(type);
  enum GeoArrowGeometryType geometry_type = GeoArrowGeometryTypeFromType(type);

  const char* dims;
  switch (dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      dims = "xy";
      break;
    case GEOARROW_DIMENSIONS_XYZ:
      dims = "xyz";
      break;
    case GEOARROW_DIMENSIONS_XYM:
      dims = "xym";
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      dims = "xyzm";
      break;
    default:
      return EINVAL;
  }

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      switch (coord_type) {
        case GEOARROW_COORD_TYPE_SEPARATE:
          return GeoArrowSchemaInitCoordStruct(schema, dims);
        case GEOARROW_COORD_TYPE_INTERLEAVED:
          return GeoArrowSchemaInitCoordFixedSizeList(schema, dims);
        default:
          return EINVAL;
      }

    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      return GeoArrowSchemaInitListStruct(schema, coord_type, dims, 1,
                                          CHILD_NAMES_LINESTRING);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      return GeoArrowSchemaInitListStruct(schema, coord_type, dims, 1,
                                          CHILD_NAMES_MULTIPOINT);
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      return GeoArrowSchemaInitListStruct(schema, coord_type, dims, 2,
                                          CHILD_NAMES_POLYGON);
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      return GeoArrowSchemaInitListStruct(schema, coord_type, dims, 2,
                                          CHILD_NAMES_MULTILINESTRING);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      return GeoArrowSchemaInitListStruct(schema, coord_type, dims, 3,
                                          CHILD_NAMES_MULTIPOLYGON);

    default:
      return ENOTSUP;
  }
}

GeoArrowErrorCode GeoArrowSchemaInitExtension(struct ArrowSchema* schema,
                                              enum GeoArrowType type) {
  const char* ext_type = GeoArrowExtensionNameFromType(type);
  if (ext_type == NULL) {
    return EINVAL;
  }

  struct ArrowBuffer metadata;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&metadata, NULL));
  int result = ArrowMetadataBuilderAppend(
      &metadata, ArrowCharView("ARROW:extension:name"), ArrowCharView(ext_type));
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&metadata);
    return result;
  }

  result = GeoArrowSchemaInit(schema, type);
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&metadata);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)metadata.data);
  ArrowBufferReset(&metadata);
  return result;
}

#include <errno.h>
#include <stddef.h>
#include <string.h>




static int GeoArrowParsePointFixedSizeList(const struct ArrowSchema* schema,
                                           struct GeoArrowSchemaView* schema_view,
                                           struct ArrowError* error,
                                           const char* ext_name) {
  if (schema->n_children != 1 || strcmp(schema->children[0]->format, "g") != 0) {
    ArrowErrorSet(
        error,
        "Expected fixed-size list coordinate child 0 to have storage type of double for "
        "extension '%s'",
        ext_name);
    return EINVAL;
  }

  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, error));
  const char* maybe_dims = schema->children[0]->name;
  if (maybe_dims == NULL) {
    maybe_dims = "<NULL>";
  }

  if (strcmp(maybe_dims, "xy") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
  } else if (strcmp(maybe_dims, "xyz") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (strcmp(maybe_dims, "xym") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYM;
  } else if (strcmp(maybe_dims, "xyzm") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else {
    switch (na_schema_view.fixed_size) {
      case 2:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
        break;
      case 3:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
        break;
      case 4:
        schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
        break;
      default:
        ArrowErrorSet(error,
                      "Can't guess dimensions for fixed size list coord array with child "
                      "name '%s' and fixed size %d for extension '%s'",
                      maybe_dims, na_schema_view.fixed_size, ext_name);
        return EINVAL;
    }
  }

  int expected_n_dims = _GeoArrowkNumDimensions[schema_view->dimensions];
  if (expected_n_dims != na_schema_view.fixed_size) {
    ArrowErrorSet(error,
                  "Expected fixed size list coord array with child name '%s' to have "
                  "fixed size %d but found fixed size %d for extension '%s'",
                  maybe_dims, expected_n_dims, na_schema_view.fixed_size, ext_name);
    return EINVAL;
  }

  schema_view->coord_type = GEOARROW_COORD_TYPE_INTERLEAVED;
  return NANOARROW_OK;
}

static int GeoArrowParsePointStruct(const struct ArrowSchema* schema,
                                    struct GeoArrowSchemaView* schema_view,
                                    struct ArrowError* error, const char* ext_name) {
  if (schema->n_children < 2 || schema->n_children > 4) {
    ArrowErrorSet(
        error,
        "Expected 2, 3, or 4 children for coord array for extension '%s' but got %d",
        ext_name, (int)schema->n_children);
    return EINVAL;
  }

  char dim[5];
  memset(dim, 0, sizeof(dim));
  for (int64_t i = 0; i < schema->n_children; i++) {
    const char* child_name = schema->children[i]->name;
    if (child_name == NULL || strlen(child_name) != 1) {
      ArrowErrorSet(error,
                    "Expected coordinate child %d to have single character name for "
                    "extension '%s'",
                    (int)i, ext_name);
      return EINVAL;
    }

    if (strcmp(schema->children[i]->format, "g") != 0) {
      ArrowErrorSet(error,
                    "Expected coordinate child %d to have storage type of double for "
                    "extension '%s'",
                    (int)i, ext_name);
      return EINVAL;
    }

    dim[i] = child_name[0];
  }

  if (strcmp(dim, "xy") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XY;
  } else if (strcmp(dim, "xyz") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (strcmp(dim, "xym") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYM;
  } else if (strcmp(dim, "xyzm") == 0) {
    schema_view->dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else {
    ArrowErrorSet(error,
                  "Expected dimensions 'xy', 'xyz', 'xym', or 'xyzm' for extension "
                  "'%s' but found '%s'",
                  ext_name, dim);
    return EINVAL;
  }

  schema_view->coord_type = GEOARROW_COORD_TYPE_SEPARATE;
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowParseNestedSchema(const struct ArrowSchema* schema,
                                                   int n,
                                                   struct GeoArrowSchemaView* schema_view,
                                                   struct ArrowError* error,
                                                   const char* ext_name) {
  if (n == 0) {
    if (strcmp(schema->format, "+s") == 0) {
      return GeoArrowParsePointStruct(schema, schema_view, error, ext_name);
    } else if (strncmp(schema->format, "+w:", 3) == 0) {
      return GeoArrowParsePointFixedSizeList(schema, schema_view, error, ext_name);
    } else {
      ArrowErrorSet(error,
                    "Expected storage type fixed-size list or struct for coord array for "
                    "extension '%s'",
                    ext_name);
      return EINVAL;
    }
  } else {
    if (strcmp(schema->format, "+l") != 0 || schema->n_children != 1) {
      ArrowErrorSet(error,
                    "Expected valid list type for coord parent %d for extension '%s'", n,
                    ext_name);
      return EINVAL;
    }

    return GeoArrowParseNestedSchema(schema->children[0], n - 1, schema_view, error,
                                     ext_name);
  }
}

static GeoArrowErrorCode GeoArrowSchemaViewInitInternal(
    struct GeoArrowSchemaView* schema_view, const struct ArrowSchema* schema,
    struct ArrowSchemaView* na_schema_view, struct ArrowError* na_error) {
  const char* ext_name = na_schema_view->extension_name.data;
  int64_t ext_len = na_schema_view->extension_name.size_bytes;

  if (ext_len >= 14 && strncmp(ext_name, "geoarrow.point", 14) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_POINT;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowParseNestedSchema(schema, 0, schema_view, na_error, "geoarrow.point"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 19 && strncmp(ext_name, "geoarrow.linestring", 19) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_LINESTRING;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 1, schema_view, na_error,
                                                      "geoarrow.linestring"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 16 && strncmp(ext_name, "geoarrow.polygon", 16) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_POLYGON;
    NANOARROW_RETURN_NOT_OK(
        GeoArrowParseNestedSchema(schema, 2, schema_view, na_error, "geoarrow.polygon"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 19 && strncmp(ext_name, "geoarrow.multipoint", 19) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 1, schema_view, na_error,
                                                      "geoarrow.multipoint"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 24 && strncmp(ext_name, "geoarrow.multilinestring", 24) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 2, schema_view, na_error,
                                                      "geoarrow.multilinestring"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 21 && strncmp(ext_name, "geoarrow.multipolygon", 21) == 0) {
    schema_view->geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
    NANOARROW_RETURN_NOT_OK(GeoArrowParseNestedSchema(schema, 3, schema_view, na_error,
                                                      "geoarrow.multipolygon"));
    schema_view->type = GeoArrowMakeType(
        schema_view->geometry_type, schema_view->dimensions, schema_view->coord_type);
  } else if (ext_len >= 12 && strncmp(ext_name, "geoarrow.wkt", 12) == 0) {
    switch (na_schema_view->type) {
      case NANOARROW_TYPE_STRING:
        schema_view->type = GEOARROW_TYPE_WKT;
        break;
      case NANOARROW_TYPE_LARGE_STRING:
        schema_view->type = GEOARROW_TYPE_LARGE_WKT;
        break;
      default:
        ArrowErrorSet(na_error,
                      "Expected storage type of string or large_string for extension "
                      "'geoarrow.wkt'");
        return EINVAL;
    }

    schema_view->geometry_type = GeoArrowGeometryTypeFromType(schema_view->type);
    schema_view->dimensions = GeoArrowDimensionsFromType(schema_view->type);
    schema_view->coord_type = GeoArrowCoordTypeFromType(schema_view->type);
  } else if (ext_len >= 12 && strncmp(ext_name, "geoarrow.wkb", 12) == 0) {
    switch (na_schema_view->type) {
      case NANOARROW_TYPE_BINARY:
        schema_view->type = GEOARROW_TYPE_WKB;
        break;
      case NANOARROW_TYPE_LARGE_BINARY:
        schema_view->type = GEOARROW_TYPE_LARGE_WKB;
        break;
      default:
        ArrowErrorSet(na_error,
                      "Expected storage type of binary or large_binary for extension "
                      "'geoarrow.wkb'");
        return EINVAL;
    }

    schema_view->geometry_type = GeoArrowGeometryTypeFromType(schema_view->type);
    schema_view->dimensions = GeoArrowDimensionsFromType(schema_view->type);
    schema_view->coord_type = GeoArrowCoordTypeFromType(schema_view->type);
  } else {
    ArrowErrorSet(na_error, "Unrecognized GeoArrow extension name: '%.*s'", (int)ext_len,
                  ext_name);
    return EINVAL;
  }

  schema_view->extension_name.data = na_schema_view->extension_name.data;
  schema_view->extension_name.size_bytes = na_schema_view->extension_name.size_bytes;
  schema_view->extension_metadata.data = na_schema_view->extension_metadata.data;
  schema_view->extension_metadata.size_bytes =
      na_schema_view->extension_metadata.size_bytes;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowSchemaViewInit(struct GeoArrowSchemaView* schema_view,
                                         const struct ArrowSchema* schema,
                                         struct GeoArrowError* error) {
  struct ArrowError* na_error = (struct ArrowError*)error;
  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, na_error));

  const char* ext_name = na_schema_view.extension_name.data;
  if (ext_name == NULL) {
    ArrowErrorSet(na_error, "Expected extension type");
    return EINVAL;
  }

  return GeoArrowSchemaViewInitInternal(schema_view, schema, &na_schema_view, na_error);
}

GeoArrowErrorCode GeoArrowSchemaViewInitFromStorage(
    struct GeoArrowSchemaView* schema_view, const struct ArrowSchema* schema,
    struct GeoArrowStringView extension_name, struct GeoArrowError* error) {
  struct ArrowError* na_error = (struct ArrowError*)error;
  struct ArrowSchemaView na_schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&na_schema_view, schema, na_error));
  na_schema_view.extension_name.data = extension_name.data;
  na_schema_view.extension_name.size_bytes = extension_name.size_bytes;
  return GeoArrowSchemaViewInitInternal(schema_view, schema, &na_schema_view, na_error);
}

GeoArrowErrorCode GeoArrowSchemaViewInitFromType(struct GeoArrowSchemaView* schema_view,
                                                 enum GeoArrowType type) {
  schema_view->schema = NULL;
  schema_view->extension_name.data = NULL;
  schema_view->extension_name.size_bytes = 0;
  schema_view->extension_metadata.data = NULL;
  schema_view->extension_metadata.size_bytes = 0;
  schema_view->type = type;
  schema_view->geometry_type = GeoArrowGeometryTypeFromType(type);
  schema_view->dimensions = GeoArrowDimensionsFromType(type);
  schema_view->coord_type = GeoArrowCoordTypeFromType(type);

  if (type == GEOARROW_TYPE_UNINITIALIZED) {
    return GEOARROW_OK;
  }

  const char* extension_name = GeoArrowExtensionNameFromType(type);
  if (extension_name == NULL) {
    return EINVAL;
  }

  schema_view->extension_name.data = extension_name;
  schema_view->extension_name.size_bytes = strlen(extension_name);

  return GEOARROW_OK;
}

#include <errno.h>
#include <stdio.h>





#define CHECK_POS(n)                               \
  if ((pos + (int32_t)(n)) > ((int32_t)pos_max)) { \
    return EINVAL;                                 \
  }

// A early draft implementation used something like the Arrow C Data interface
// metadata specification instead of JSON. To help with the transition, this
// bit of code parses the original metadata format.
static GeoArrowErrorCode GeoArrowMetadataViewInitDeprecated(
    struct GeoArrowMetadataView* metadata_view, struct GeoArrowError* error) {
  const char* metadata = metadata_view->metadata.data;
  int32_t pos_max = (int32_t)metadata_view->metadata.size_bytes;
  int32_t pos = 0;
  int32_t name_len;
  int32_t value_len;
  int32_t m;

  CHECK_POS(sizeof(int32_t));
  memcpy(&m, metadata + pos, sizeof(int32_t));
  pos += sizeof(int32_t);

  for (int j = 0; j < m; j++) {
    CHECK_POS(sizeof(int32_t));
    memcpy(&name_len, metadata + pos, sizeof(int32_t));
    pos += sizeof(int32_t);

    CHECK_POS(name_len)
    const char* name = metadata + pos;
    pos += name_len;

    CHECK_POS(sizeof(int32_t))
    memcpy(&value_len, metadata + pos, sizeof(int32_t));
    pos += sizeof(int32_t);

    CHECK_POS(value_len)
    const char* value = metadata + pos;
    pos += value_len;

    if (name_len == 0 || value_len == 0) {
      continue;
    }

    if (name_len == 3 && strncmp(name, "crs", 3) == 0) {
      metadata_view->crs.size_bytes = value_len;
      metadata_view->crs.data = value;
      metadata_view->crs_type = GEOARROW_CRS_TYPE_UNKNOWN;
    } else if (name_len == 5 && strncmp(name, "edges", 5) == 0) {
      if (value_len == 9 && strncmp(value, "spherical", 9) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_SPHERICAL;
      } else {
        // unuspported value for 'edges' key
      }
    } else {
      // unsupported metadata key
    }
  }

  return GEOARROW_OK;
}

static int ParseChar(struct ArrowStringView* s, char c) {
  if (s->size_bytes > 0 && s->data[0] == c) {
    s->size_bytes--;
    s->data++;
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static void SkipWhitespace(struct ArrowStringView* s) {
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      s->size_bytes--;
      s->data++;
    } else {
      break;
    }
  }
}

static int SkipUntil(struct ArrowStringView* s, const char* items) {
  int64_t n_items = strlen(items);
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == '\0') {
      return 0;
    }

    for (int64_t i = 0; i < n_items; i++) {
      if (c == items[i]) {
        return 1;
      }
    }

    s->size_bytes--;
    s->data++;
  }

  return 0;
}

static GeoArrowErrorCode FindString(struct ArrowStringView* s,
                                    struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '\"') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;

  int is_escape = 0;
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (!is_escape && c == '\\') {
      is_escape = 1;
      s->size_bytes--;
      s->data++;
      continue;
    }

    if (!is_escape && c == '\"') {
      s->size_bytes--;
      s->data++;
      out->size_bytes = s->data - out->data;
      return GEOARROW_OK;
    }

    s->size_bytes--;
    s->data++;
    is_escape = 0;
  }

  return EINVAL;
}

static GeoArrowErrorCode FindObject(struct ArrowStringView* s,
                                    struct ArrowStringView* out);

static GeoArrowErrorCode FindList(struct ArrowStringView* s,
                                  struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '[') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;
  struct ArrowStringView tmp_value;
  while (s->size_bytes > 0) {
    if (SkipUntil(s, "[{\"]")) {
      char c = *(s->data);
      switch (c) {
        case '\"':
          NANOARROW_RETURN_NOT_OK(FindString(s, &tmp_value));
          break;
        case '[':
          NANOARROW_RETURN_NOT_OK(FindList(s, &tmp_value));
          break;
        case '{':
          NANOARROW_RETURN_NOT_OK(FindObject(s, &tmp_value));
          break;
        case ']':
          s->size_bytes--;
          s->data++;
          out->size_bytes = s->data - out->data;
          return GEOARROW_OK;
        default:
          break;
      }
    }
  }

  return EINVAL;
}

static GeoArrowErrorCode FindObject(struct ArrowStringView* s,
                                    struct ArrowStringView* out) {
  out->data = s->data;
  if (s->data[0] != '{') {
    return EINVAL;
  }

  s->size_bytes--;
  s->data++;
  struct ArrowStringView tmp_value;
  while (s->size_bytes > 0) {
    if (SkipUntil(s, "{[\"}")) {
      char c = *(s->data);
      switch (c) {
        case '\"':
          NANOARROW_RETURN_NOT_OK(FindString(s, &tmp_value));
          break;
        case '[':
          NANOARROW_RETURN_NOT_OK(FindList(s, &tmp_value));
          break;
        case '{':
          NANOARROW_RETURN_NOT_OK(FindObject(s, &tmp_value));
          break;
        case '}':
          s->size_bytes--;
          s->data++;
          out->size_bytes = s->data - out->data;
          return GEOARROW_OK;
        default:
          break;
      }
    }
  }

  return EINVAL;
}

static GeoArrowErrorCode ParseJSONMetadata(struct GeoArrowMetadataView* metadata_view,
                                           struct ArrowStringView* s) {
  NANOARROW_RETURN_NOT_OK(ParseChar(s, '{'));
  SkipWhitespace(s);
  struct ArrowStringView k;
  struct ArrowStringView v;

  while (s->size_bytes > 0 && s->data[0] != '}') {
    SkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(FindString(s, &k));
    SkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(ParseChar(s, ':'));
    SkipWhitespace(s);

    switch (s->data[0]) {
      case '[':
        NANOARROW_RETURN_NOT_OK(FindList(s, &v));
        break;
      case '{':
        NANOARROW_RETURN_NOT_OK(FindObject(s, &v));
        break;
      case '\"':
        NANOARROW_RETURN_NOT_OK(FindString(s, &v));
        break;
      default:
        break;
    }

    if (k.size_bytes == 7 && strncmp(k.data, "\"edges\"", 7) == 0) {
      if (v.size_bytes == 11 && strncmp(v.data, "\"spherical\"", 11) == 0) {
        metadata_view->edge_type = GEOARROW_EDGE_TYPE_SPHERICAL;
      }
    } else if (k.size_bytes == 5 && strncmp(k.data, "\"crs\"", 5) == 0) {
      if (v.data[0] == '{') {
        metadata_view->crs_type = GEOARROW_CRS_TYPE_PROJJSON;
      } else if (v.data[0] == '\"') {
        metadata_view->crs_type = GEOARROW_CRS_TYPE_UNKNOWN;
      } else {
        return EINVAL;
      }

      metadata_view->crs.data = v.data;
      metadata_view->crs.size_bytes = v.size_bytes;
    }

    SkipUntil(s, ",}");
    if (s->data[0] == ',') {
      s->size_bytes--;
      s->data++;
    }
  }

  if (s->size_bytes > 0 && s->data[0] == '}') {
    s->size_bytes--;
    s->data++;
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static GeoArrowErrorCode GeoArrowMetadataViewInitJSON(
    struct GeoArrowMetadataView* metadata_view, struct GeoArrowError* error) {
  struct ArrowStringView metadata;
  metadata.data = metadata_view->metadata.data;
  metadata.size_bytes = metadata_view->metadata.size_bytes;

  struct ArrowStringView s = metadata;
  SkipWhitespace(&s);

  if (ParseJSONMetadata(metadata_view, &s) != GEOARROW_OK) {
    GeoArrowErrorSet(error, "Expected valid GeoArrow JSON metadata but got '%.*s'",
                     (int)metadata.size_bytes, metadata.data);
    return EINVAL;
  }

  SkipWhitespace(&s);
  if (s.data != (metadata.data + metadata.size_bytes)) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Expected JSON object with no trailing characters but found trailing '%.*s'",
        (int)s.size_bytes, s.data);
    return EINVAL;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowMetadataViewInit(struct GeoArrowMetadataView* metadata_view,
                                           struct GeoArrowStringView metadata,
                                           struct GeoArrowError* error) {
  metadata_view->metadata = metadata;
  metadata_view->edge_type = GEOARROW_EDGE_TYPE_PLANAR;
  metadata_view->crs_type = GEOARROW_CRS_TYPE_NONE;
  metadata_view->crs.data = NULL;
  metadata_view->crs.size_bytes = 0;

  if (metadata.size_bytes == 0) {
    return GEOARROW_OK;
  }

  if (metadata.size_bytes >= 4 && metadata.data[0] != '{') {
    if (GeoArrowMetadataViewInitDeprecated(metadata_view, error) == GEOARROW_OK) {
      return GEOARROW_OK;
    }
  }

  return GeoArrowMetadataViewInitJSON(metadata_view, error);
}

static GeoArrowErrorCode GeoArrowMetadataSerializeInternalDeprecated(
    const struct GeoArrowMetadataView* metadata_view, struct ArrowBuffer* buffer) {
  switch (metadata_view->edge_type) {
    case GEOARROW_EDGE_TYPE_SPHERICAL:
      NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderAppend(buffer, ArrowCharView("edges"),
                                                         ArrowCharView("spherical")));
      break;
    default:
      break;
  }

  struct ArrowStringView crs_value;
  if (metadata_view->crs.size_bytes > 0) {
    crs_value.data = metadata_view->crs.data;
    crs_value.size_bytes = metadata_view->crs.size_bytes;
    NANOARROW_RETURN_NOT_OK(
        ArrowMetadataBuilderAppend(buffer, ArrowCharView("crs"), crs_value));
  }

  return NANOARROW_OK;
}

static GeoArrowErrorCode GeoArrowMetadataSerializeInternal(
    const struct GeoArrowMetadataView* metadata_view, struct ArrowBuffer* buffer) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, "{", 1));

  int needs_leading_comma = 0;
  const char* spherical_edges_json = "\"edges\":\"spherical\"";
  switch (metadata_view->edge_type) {
    case GEOARROW_EDGE_TYPE_SPHERICAL:
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(buffer, spherical_edges_json, strlen(spherical_edges_json)));
      needs_leading_comma = 1;
      break;
    default:
      break;
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE && needs_leading_comma) {
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, ",", 1));
  }

  if (metadata_view->crs_type != GEOARROW_CRS_TYPE_NONE) {
    const char* crs_json_prefix = "\"crs\":";
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppend(buffer, crs_json_prefix, strlen(crs_json_prefix)));
  }

  if (metadata_view->crs_type == GEOARROW_CRS_TYPE_PROJJSON) {
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, metadata_view->crs.data,
                                              metadata_view->crs.size_bytes));
  } else if (metadata_view->crs_type == GEOARROW_CRS_TYPE_UNKNOWN) {
    // Escape quotes in the string if the string does not start with '"'
    if (metadata_view->crs.size_bytes > 0 && metadata_view->crs.data[0] == '\"') {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, metadata_view->crs.data,
                                                metadata_view->crs.size_bytes));
    } else {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, "\"", 1));
      for (int64_t i = 0; i < metadata_view->crs.size_bytes; i++) {
        char c = metadata_view->crs.data[i];
        if (c == '\"') {
          NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, "\\", 1));
        }
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt8(buffer, c));
      }
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, "\"", 1));
    }
  }

  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(buffer, "}", 1));
  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowSchemaSetMetadataInternal(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view,
    int use_deprecated) {
  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);

  int result = 0;
  if (use_deprecated) {
    result = GeoArrowMetadataSerializeInternalDeprecated(metadata_view, &buffer);
  } else {
    result = GeoArrowMetadataSerializeInternal(metadata_view, &buffer);
  }

  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  struct ArrowBuffer existing_buffer;
  result = ArrowMetadataBuilderInit(&existing_buffer, schema->metadata);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  struct ArrowStringView value;
  value.data = (const char*)buffer.data;
  value.size_bytes = buffer.size_bytes;
  result = ArrowMetadataBuilderSet(&existing_buffer,
                                   ArrowCharView("ARROW:extension:metadata"), value);
  ArrowBufferReset(&buffer);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&existing_buffer);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)existing_buffer.data);
  ArrowBufferReset(&existing_buffer);
  return result;
}

int64_t GeoArrowMetadataSerialize(const struct GeoArrowMetadataView* metadata_view,
                                  char* out, int64_t n) {
  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);
  int result = ArrowBufferReserve(&buffer, n);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return -1;
  }

  result = GeoArrowMetadataSerializeInternal(metadata_view, &buffer);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return -1;
  }

  int64_t size_needed = buffer.size_bytes;
  int64_t n_copy;
  if (n >= size_needed) {
    n_copy = size_needed;
  } else {
    n_copy = n;
  }

  if (n_copy > 0) {
    memcpy(out, buffer.data, n_copy);
  }

  if (n > size_needed) {
    out[size_needed] = '\0';
  }

  ArrowBufferReset(&buffer);
  return size_needed;
}

GeoArrowErrorCode GeoArrowSchemaSetMetadata(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view) {
  return GeoArrowSchemaSetMetadataInternal(schema, metadata_view, 0);
}

GeoArrowErrorCode GeoArrowSchemaSetMetadataDeprecated(
    struct ArrowSchema* schema, const struct GeoArrowMetadataView* metadata_view) {
  return GeoArrowSchemaSetMetadataInternal(schema, metadata_view, 1);
}

GeoArrowErrorCode GeoArrowSchemaSetMetadataFrom(struct ArrowSchema* schema,
                                                const struct ArrowSchema* schema_src) {
  struct ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema_src, NULL));

  struct ArrowBuffer buffer;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&buffer, schema->metadata));
  int result = ArrowMetadataBuilderSet(&buffer, ArrowCharView("ARROW:extension:metadata"),
                                       schema_view.extension_metadata);
  if (result != GEOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)buffer.data);
  ArrowBufferReset(&buffer);
  return result;
}

int64_t GeoArrowUnescapeCrs(struct GeoArrowStringView crs, char* out, int64_t n) {
  if (crs.size_bytes == 0) {
    if (n > 0) {
      out[0] = '\0';
    }
    return 0;
  }

  if (crs.data[0] != '\"') {
    if (n > crs.size_bytes) {
      memcpy(out, crs.data, crs.size_bytes);
      out[crs.size_bytes] = '\0';
    } else {
      memcpy(out, crs.data, n);
    }

    return crs.size_bytes;
  }

  int64_t out_i = 0;
  int is_escape = 0;
  for (int64_t i = 1; i < (crs.size_bytes - 1); i++) {
    if (!is_escape && crs.data[i] == '\\') {
      is_escape = 1;
      continue;
    } else {
      is_escape = 0;
    }

    if (out_i < n) {
      out[out_i] = crs.data[i];
    }

    out_i++;
  }

  if (out_i < n) {
    out[out_i] = '\0';
  }

  return out_i;
}

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>





static int kernel_start_void(struct GeoArrowKernel* kernel, struct ArrowSchema* schema,
                             const char* options, struct ArrowSchema* out,
                             struct GeoArrowError* error) {
  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_NA);
}

static int kernel_push_batch_void(struct GeoArrowKernel* kernel, struct ArrowArray* array,
                                  struct ArrowArray* out, struct GeoArrowError* error) {
  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_NA));
  tmp.length = array->length;
  tmp.null_count = array->length;
  ArrowArrayMove(&tmp, out);
  return NANOARROW_OK;
}

static int kernel_finish_void(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                              struct GeoArrowError* error) {
  if (out != NULL) {
    return EINVAL;
  }

  return NANOARROW_OK;
}

static void kernel_release_void(struct GeoArrowKernel* kernel) { kernel->release = NULL; }

static void GeoArrowKernelInitVoid(struct GeoArrowKernel* kernel) {
  kernel->start = &kernel_start_void;
  kernel->push_batch = &kernel_push_batch_void;
  kernel->finish = &kernel_finish_void;
  kernel->release = &kernel_release_void;
  kernel->private_data = NULL;
}

static int kernel_push_batch_void_agg(struct GeoArrowKernel* kernel,
                                      struct ArrowArray* array, struct ArrowArray* out,
                                      struct GeoArrowError* error) {
  if (out != NULL) {
    return EINVAL;
  }

  return NANOARROW_OK;
}

static int kernel_finish_void_agg(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                                  struct GeoArrowError* error) {
  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_NA));
  tmp.length = 1;
  tmp.null_count = 1;
  ArrowArrayMove(&tmp, out);
  return NANOARROW_OK;
}

static void GeoArrowKernelInitVoidAgg(struct GeoArrowKernel* kernel) {
  kernel->start = &kernel_start_void;
  kernel->push_batch = &kernel_push_batch_void_agg;
  kernel->finish = &kernel_finish_void_agg;
  kernel->release = &kernel_release_void;
  kernel->private_data = NULL;
}

// Visitor-based kernels
//
// These kernels implement generic operations by visiting each feature in
// the input (since all GeoArrow types including WKB/WKT can be visited).
// This for conversion to/from WKB and WKT whose readers and writers are
// visitor-based. Most other operations are probably faster phrased as
// "cast to GeoArrow in batches then do the thing" (but require these kernels to
// do the "cast to GeoArrow" step).

struct GeoArrowGeometryTypesVisitorPrivate {
  enum GeoArrowGeometryType geometry_type;
  enum GeoArrowDimensions dimensions;
  uint64_t geometry_types_mask;
};

struct GeoArrowBox2DPrivate {
  int feat_null;
  double min_values[2];
  double max_values[2];
  struct ArrowBitmap validity;
  struct ArrowBuffer values[4];
  int64_t null_count;
};

struct GeoArrowVisitorKernelPrivate {
  struct GeoArrowVisitor v;
  int visit_by_feature;
  struct GeoArrowArrayReader reader;
  struct GeoArrowArrayView array_view;
  struct GeoArrowArrayWriter writer;
  struct GeoArrowWKTWriter wkt_writer;
  struct GeoArrowGeometryTypesVisitorPrivate geometry_types_private;
  struct GeoArrowBox2DPrivate box2d_private;
  int (*finish_push_batch)(struct GeoArrowVisitorKernelPrivate* private_data,
                           struct ArrowArray* out, struct GeoArrowError* error);
  int (*finish_start)(struct GeoArrowVisitorKernelPrivate* private_data,
                      struct ArrowSchema* schema, const char* options,
                      struct ArrowSchema* out, struct GeoArrowError* error);
};

static int kernel_get_arg_long(const char* options, const char* key, long* out,
                               int required, struct GeoArrowError* error) {
  struct ArrowStringView type_str;
  type_str.data = NULL;
  type_str.size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataGetValue(options, ArrowCharView(key), &type_str));
  if (type_str.data == NULL && required) {
    GeoArrowErrorSet(error, "Missing required parameter '%s'", key);
    return EINVAL;
  } else if (type_str.data == NULL && !required) {
    return NANOARROW_OK;
  }

  char type_str0[16];
  memset(type_str0, 0, sizeof(type_str0));
  snprintf(type_str0, sizeof(type_str0), "%.*s", (int)type_str.size_bytes, type_str.data);
  *out = atoi(type_str0);
  return NANOARROW_OK;
}

static int finish_push_batch_do_nothing(struct GeoArrowVisitorKernelPrivate* private_data,
                                        struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  return NANOARROW_OK;
}

static void kernel_release_visitor(struct GeoArrowKernel* kernel) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;
  if (private_data->reader.private_data != NULL) {
    GeoArrowArrayReaderReset(&private_data->reader);
  }

  if (private_data->writer.private_data != NULL) {
    GeoArrowArrayWriterReset(&private_data->writer);
  }

  if (private_data->wkt_writer.private_data != NULL) {
    GeoArrowWKTWriterReset(&private_data->wkt_writer);
  }

  for (int i = 0; i < 4; i++) {
    ArrowBufferReset(&private_data->box2d_private.values[i]);
  }

  ArrowBitmapReset(&private_data->box2d_private.validity);

  ArrowFree(private_data);
  kernel->release = NULL;
}

static int kernel_push_batch(struct GeoArrowKernel* kernel, struct ArrowArray* array,
                             struct ArrowArray* out, struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayViewSetArray(&private_data->array_view, array, error));

  private_data->v.error = error;
  NANOARROW_RETURN_NOT_OK(GeoArrowArrayReaderVisit(&private_data->reader,
                                                   &private_data->array_view, 0,
                                                   array->length, &private_data->v));

  return private_data->finish_push_batch(private_data, out, error);
}

static int kernel_push_batch_by_feature(struct GeoArrowKernel* kernel,
                                        struct ArrowArray* array, struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayViewSetArray(&private_data->array_view, array, error));

  private_data->v.error = error;
  int result;
  for (int64_t i = 0; i < array->length; i++) {
    result = GeoArrowArrayReaderVisit(&private_data->reader, &private_data->array_view, i,
                                      1, &private_data->v);

    if (result == EAGAIN) {
      NANOARROW_RETURN_NOT_OK(private_data->v.feat_end(&private_data->v));
    } else if (result != NANOARROW_OK) {
      return result;
    }
  }

  return private_data->finish_push_batch(private_data, out, error);
}

static int kernel_visitor_start(struct GeoArrowKernel* kernel, struct ArrowSchema* schema,
                                const char* options, struct ArrowSchema* out,
                                struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, error));

  switch (schema_view.type) {
    case GEOARROW_TYPE_UNINITIALIZED:
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_LARGE_WKT:
      return EINVAL;
    default:
      NANOARROW_RETURN_NOT_OK(GeoArrowArrayReaderInit(&private_data->reader));
      if (private_data->visit_by_feature) {
        kernel->push_batch = &kernel_push_batch_by_feature;
      } else {
        kernel->push_batch = &kernel_push_batch;
      }
      NANOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewInitFromType(&private_data->array_view, schema_view.type));
      break;
  }

  return private_data->finish_start(private_data, schema, options, out, error);
}

// Kernel visit_void_agg
//
// This kernel visits every feature and returns a single null item at the end.
// This is useful for (1) testing and (2) validating well-known text or well-known
// binary.

static int finish_start_visit_void_agg(struct GeoArrowVisitorKernelPrivate* private_data,
                                       struct ArrowSchema* schema, const char* options,
                                       struct ArrowSchema* out,
                                       struct GeoArrowError* error) {
  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_NA);
}

// Kernel format_wkt
//
// Visits every feature in the input and writes the corresponding well-known text output,
// optionally specifying precision and max_element_size_bytes.

static int finish_start_format_wkt(struct GeoArrowVisitorKernelPrivate* private_data,
                                   struct ArrowSchema* schema, const char* options,
                                   struct ArrowSchema* out, struct GeoArrowError* error) {
  long precision = private_data->wkt_writer.precision;
  NANOARROW_RETURN_NOT_OK(
      kernel_get_arg_long(options, "precision", &precision, 0, error));
  private_data->wkt_writer.precision = (int)precision;

  long max_element_size_bytes = private_data->wkt_writer.max_element_size_bytes;
  NANOARROW_RETURN_NOT_OK(kernel_get_arg_long(options, "max_element_size_bytes",
                                              &max_element_size_bytes, 0, error));
  private_data->wkt_writer.max_element_size_bytes = max_element_size_bytes;

  GeoArrowWKTWriterInitVisitor(&private_data->wkt_writer, &private_data->v);

  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(out, NANOARROW_TYPE_STRING));
  return GEOARROW_OK;
}

static int finish_push_batch_format_wkt(struct GeoArrowVisitorKernelPrivate* private_data,
                                        struct ArrowArray* out,
                                        struct GeoArrowError* error) {
  return GeoArrowWKTWriterFinish(&private_data->wkt_writer, out, error);
}

// Kernel as_geoarrow
//
// Visits every feature in the input and writes an array of the specified type.
// Takes option 'type' as the desired integer enum GeoArrowType.

static int finish_start_as_geoarrow(struct GeoArrowVisitorKernelPrivate* private_data,
                                    struct ArrowSchema* schema, const char* options,
                                    struct ArrowSchema* out,
                                    struct GeoArrowError* error) {
  long out_type_long;
  NANOARROW_RETURN_NOT_OK(kernel_get_arg_long(options, "type", &out_type_long, 1, error));
  enum GeoArrowType out_type = (enum GeoArrowType)out_type_long;

  if (private_data->writer.private_data != NULL) {
    GeoArrowErrorSet(error, "Expected exactly one call to start(as_geoarrow)");
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayWriterInitFromType(&private_data->writer, out_type));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowArrayWriterInitVisitor(&private_data->writer, &private_data->v));

  struct ArrowSchema tmp;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaInitExtension(&tmp, out_type));

  int result = GeoArrowSchemaSetMetadataFrom(&tmp, schema);
  if (result != GEOARROW_OK) {
    GeoArrowErrorSet(error, "GeoArrowSchemaSetMetadataFrom() failed");
    tmp.release(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, out);
  return GEOARROW_OK;
}

static int finish_push_batch_as_geoarrow(
    struct GeoArrowVisitorKernelPrivate* private_data, struct ArrowArray* out,
    struct GeoArrowError* error) {
  return GeoArrowArrayWriterFinish(&private_data->writer, out, error);
}

// Kernel unique_geometry_types_agg
//
// This kernel collects all geometry type + dimension combinations in the
// input. EMPTY values are not counted as any particular geometry type;
// however, note that POINTs as represented in WKB or GeoArrow cannot be
// EMPTY and this kernel does not check for the convention of EMPTY as
// all coordinates == nan. This is mosty to facilitate choosing an appropriate destination
// type (e.g., point, linestring, etc.). This visitor is not exposed as a standalone
// visitor in the geoarrow.h header.
//
// The internals use GeoArrowDimensions * 8 + GeoArrowGeometryType as the
// "key" for a given combination. This gives an integer between 0 and 39.
// The types are accumulated in a uint64_t bitmask and translated into the
// corresponding ISO WKB type codes at the end.
static int32_t kGeoArrowGeometryTypeWkbValues[] = {
    -1000, -999, -998, -997, -996, -995, -994, -993, 0,    1,    2,    3,    4,    5,
    6,     7,    1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2000, 2001, 2002, 2003,
    2004,  2005, 2006, 2007, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007};

static int feat_start_geometry_types(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->geometry_types_private.geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private_data->geometry_types_private.dimensions = GEOARROW_DIMENSIONS_UNKNOWN;
  return GEOARROW_OK;
}

static int geom_start_geometry_types(struct GeoArrowVisitor* v,
                                     enum GeoArrowGeometryType geometry_type,
                                     enum GeoArrowDimensions dimensions) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  // Only record the first seen geometry type/dimension combination
  if (private_data->geometry_types_private.geometry_type ==
      GEOARROW_GEOMETRY_TYPE_GEOMETRY) {
    private_data->geometry_types_private.geometry_type = geometry_type;
    private_data->geometry_types_private.dimensions = dimensions;
  }

  return GEOARROW_OK;
}

static int coords_geometry_types(struct GeoArrowVisitor* v,
                                 const struct GeoArrowCoordView* coords) {
  if (coords->n_coords > 0) {
    struct GeoArrowVisitorKernelPrivate* private_data =
        (struct GeoArrowVisitorKernelPrivate*)v->private_data;

    // At the first coordinate, add the geometry type to the bitmask
    int bitshift = private_data->geometry_types_private.dimensions * 8 +
                   private_data->geometry_types_private.geometry_type;
    uint64_t bitmask = ((uint64_t)1) << bitshift;
    private_data->geometry_types_private.geometry_types_mask |= bitmask;
    return EAGAIN;
  } else {
    return GEOARROW_OK;
  }
}

static int finish_start_unique_geometry_types_agg(
    struct GeoArrowVisitorKernelPrivate* private_data, struct ArrowSchema* schema,
    const char* options, struct ArrowSchema* out, struct GeoArrowError* error) {
  private_data->v.feat_start = &feat_start_geometry_types;
  private_data->v.geom_start = &geom_start_geometry_types;
  private_data->v.coords = &coords_geometry_types;
  private_data->v.private_data = private_data;
  return ArrowSchemaInitFromType(out, NANOARROW_TYPE_INT32);
}

static int kernel_finish_unique_geometry_types_agg(struct GeoArrowKernel* kernel,
                                                   struct ArrowArray* out,
                                                   struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;
  uint64_t result_mask = private_data->geometry_types_private.geometry_types_mask;

  int n_types = 0;
  for (int i = 0; i < 40; i++) {
    uint64_t bitmask = ((uint64_t)1) << i;
    n_types += (result_mask & bitmask) != 0;
  }

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(&tmp, NANOARROW_TYPE_INT32));
  struct ArrowBuffer* data = ArrowArrayBuffer(&tmp, 1);
  int result = ArrowBufferReserve(data, n_types * sizeof(int32_t));
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  int result_i = 0;
  int32_t* data_int32 = (int32_t*)data->data;
  for (int i = 0; i < 40; i++) {
    uint64_t bitmask = ((uint64_t)1) << i;
    if (result_mask & bitmask) {
      data_int32[result_i++] = kGeoArrowGeometryTypeWkbValues[i];
    }
  }

  result = ArrowArrayFinishBuildingDefault(&tmp, NULL);
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  tmp.length = n_types;
  tmp.null_count = 0;
  ArrowArrayMove(&tmp, out);
  return GEOARROW_OK;
}

// Kernel box + box_agg
//
// Calculate bounding box values by feature or as an aggregate.
// This visitor is not exposed as a standalone visitor in the geoarrow.h header.

static ArrowErrorCode schema_box(struct ArrowSchema* schema) {
  ArrowSchemaInit(schema);
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 4));
  const char* names[] = {"xmin", "xmax", "ymin", "ymax"};
  for (int i = 0; i < 4; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaSetType(schema->children[i], NANOARROW_TYPE_DOUBLE));
    NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[i], names[i]));
  }

  return GEOARROW_OK;
}

static ArrowErrorCode array_box(struct ArrowArray* array) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(array, 4));
  for (int i = 0; i < 4; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayInitFromType(array->children[i], NANOARROW_TYPE_DOUBLE));
  }

  return GEOARROW_OK;
}

static ArrowErrorCode box_flush(struct GeoArrowVisitorKernelPrivate* private_data) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[0], private_data->box2d_private.min_values[0]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[1], private_data->box2d_private.max_values[0]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[2], private_data->box2d_private.min_values[1]));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(
      &private_data->box2d_private.values[3], private_data->box2d_private.max_values[1]));

  return NANOARROW_OK;
}

static ArrowErrorCode box_finish(struct GeoArrowVisitorKernelPrivate* private_data,
                                 struct ArrowArray* out, struct ArrowError* error) {
  struct ArrowArray tmp;
  tmp.release = NULL;
  int result = array_box(&tmp);
  if (result != GEOARROW_OK) {
    if (tmp.release != NULL) {
      tmp.release(&tmp);
    }
  }

  int64_t length = private_data->box2d_private.values[0].size_bytes / sizeof(double);

  for (int i = 0; i < 4; i++) {
    ArrowArraySetBuffer(tmp.children[i], 1, &private_data->box2d_private.values[i]);
    tmp.children[i]->length = length;
  }

  tmp.length = length;
  if (private_data->box2d_private.null_count > 0) {
    ArrowArraySetValidityBitmap(&tmp, &private_data->box2d_private.validity);
  } else {
    ArrowBitmapReset(&private_data->box2d_private.validity);
  }

  result = ArrowArrayFinishBuildingDefault(&tmp, ((struct ArrowError*)error));
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  tmp.null_count = private_data->box2d_private.null_count;
  private_data->box2d_private.null_count = 0;
  ArrowArrayMove(&tmp, out);
  return GEOARROW_OK;
}

static int feat_start_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->box2d_private.max_values[0] = -INFINITY;
  private_data->box2d_private.max_values[1] = -INFINITY;
  private_data->box2d_private.min_values[0] = INFINITY;
  private_data->box2d_private.min_values[1] = INFINITY;
  private_data->box2d_private.feat_null = 0;
  return GEOARROW_OK;
}

static int null_feat_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;
  private_data->box2d_private.feat_null = 1;
  return GEOARROW_OK;
}

static int coords_box(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  double value;
  for (int dim_i = 0; dim_i < 2; dim_i++) {
    for (int64_t i = 0; i < coords->n_coords; i++) {
      value = GEOARROW_COORD_VIEW_VALUE(coords, i, dim_i);
      if (value < private_data->box2d_private.min_values[dim_i]) {
        private_data->box2d_private.min_values[dim_i] = value;
      }

      if (value > private_data->box2d_private.max_values[dim_i]) {
        private_data->box2d_private.max_values[dim_i] = value;
      }
    }
  }

  return GEOARROW_OK;
}

static int feat_end_box(struct GeoArrowVisitor* v) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)v->private_data;

  if (private_data->box2d_private.feat_null) {
    if (private_data->box2d_private.validity.buffer.data == NULL) {
      int64_t length = private_data->box2d_private.values[0].size_bytes / sizeof(double);
      NANOARROW_RETURN_NOT_OK(
          ArrowBitmapAppend(&private_data->box2d_private.validity, 1, length));
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowBitmapAppend(&private_data->box2d_private.validity, 0, 1));
    private_data->box2d_private.null_count++;
  } else if (private_data->box2d_private.validity.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowBitmapAppend(&private_data->box2d_private.validity, 1, 1));
  }

  NANOARROW_RETURN_NOT_OK(box_flush(private_data));
  return GEOARROW_OK;
}

static int finish_start_box_agg(struct GeoArrowVisitorKernelPrivate* private_data,
                                struct ArrowSchema* schema, const char* options,
                                struct ArrowSchema* out, struct GeoArrowError* error) {
  private_data->v.coords = &coords_box;
  private_data->v.private_data = private_data;

  private_data->box2d_private.max_values[0] = -INFINITY;
  private_data->box2d_private.max_values[1] = -INFINITY;
  private_data->box2d_private.min_values[0] = INFINITY;
  private_data->box2d_private.min_values[1] = INFINITY;
  private_data->box2d_private.feat_null = 0;

  ArrowBitmapInit(&private_data->box2d_private.validity);
  for (int i = 0; i < 4; i++) {
    ArrowBufferInit(&private_data->box2d_private.values[i]);
  }

  struct ArrowSchema tmp;
  int result = schema_box(&tmp);
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, out);
  return GEOARROW_OK;
}

static int kernel_finish_box_agg(struct GeoArrowKernel* kernel, struct ArrowArray* out,
                                 struct GeoArrowError* error) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)kernel->private_data;

  NANOARROW_RETURN_NOT_OK(box_flush(private_data));
  NANOARROW_RETURN_NOT_OK(box_finish(private_data, out, (struct ArrowError*)error));
  return GEOARROW_OK;
}

static int finish_start_box(struct GeoArrowVisitorKernelPrivate* private_data,
                            struct ArrowSchema* schema, const char* options,
                            struct ArrowSchema* out, struct GeoArrowError* error) {
  private_data->v.feat_start = &feat_start_box;
  private_data->v.null_feat = &null_feat_box;
  private_data->v.coords = &coords_box;
  private_data->v.feat_end = &feat_end_box;
  private_data->v.private_data = private_data;

  ArrowBitmapInit(&private_data->box2d_private.validity);
  for (int i = 0; i < 4; i++) {
    ArrowBufferInit(&private_data->box2d_private.values[i]);
  }

  struct ArrowSchema tmp;
  int result = schema_box(&tmp);
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, out);
  return GEOARROW_OK;
}

static int finish_push_batch_box(struct GeoArrowVisitorKernelPrivate* private_data,
                                 struct ArrowArray* out, struct GeoArrowError* error) {
  NANOARROW_RETURN_NOT_OK(box_finish(private_data, out, (struct ArrowError*)error));
  return GEOARROW_OK;
}

static int GeoArrowInitVisitorKernelInternal(struct GeoArrowKernel* kernel,
                                             const char* name) {
  struct GeoArrowVisitorKernelPrivate* private_data =
      (struct GeoArrowVisitorKernelPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowVisitorKernelPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowVisitorKernelPrivate));
  private_data->finish_push_batch = &finish_push_batch_do_nothing;
  GeoArrowVisitorInitVoid(&private_data->v);
  private_data->visit_by_feature = 0;

  int result = GEOARROW_OK;

  if (strcmp(name, "visit_void_agg") == 0) {
    kernel->finish = &kernel_finish_void_agg;
    private_data->finish_start = &finish_start_visit_void_agg;
  } else if (strcmp(name, "format_wkt") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_format_wkt;
    private_data->finish_push_batch = &finish_push_batch_format_wkt;
    result = GeoArrowWKTWriterInit(&private_data->wkt_writer);
    private_data->visit_by_feature = 1;
  } else if (strcmp(name, "as_geoarrow") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_as_geoarrow;
    private_data->finish_push_batch = &finish_push_batch_as_geoarrow;
  } else if (strcmp(name, "unique_geometry_types_agg") == 0) {
    kernel->finish = &kernel_finish_unique_geometry_types_agg;
    private_data->finish_start = &finish_start_unique_geometry_types_agg;
    private_data->visit_by_feature = 1;
  } else if (strcmp(name, "box") == 0) {
    kernel->finish = &kernel_finish_void;
    private_data->finish_start = &finish_start_box;
    private_data->finish_push_batch = &finish_push_batch_box;
  } else if (strcmp(name, "box_agg") == 0) {
    kernel->finish = &kernel_finish_box_agg;
    private_data->finish_start = &finish_start_box_agg;
  }

  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  kernel->start = &kernel_visitor_start;
  kernel->push_batch = &kernel_push_batch_void_agg;
  kernel->release = &kernel_release_visitor;
  kernel->private_data = private_data;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowKernelInit(struct GeoArrowKernel* kernel, const char* name,
                                     const char* options) {
  if (strcmp(name, "void") == 0) {
    GeoArrowKernelInitVoid(kernel);
    return NANOARROW_OK;
  } else if (strcmp(name, "void_agg") == 0) {
    GeoArrowKernelInitVoidAgg(kernel);
    return NANOARROW_OK;
  } else if (strcmp(name, "visit_void_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "format_wkt") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "as_geoarrow") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "unique_geometry_types_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "box") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  } else if (strcmp(name, "box_agg") == 0) {
    return GeoArrowInitVisitorKernelInternal(kernel, name);
  }

  return ENOTSUP;
}

#include <string.h>





// Bytes for four quiet (little-endian) NANs
static uint8_t kEmptyPointCoords[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};

struct BuilderPrivate {
  // The ArrowSchema (without extension) for this builder
  struct ArrowSchema schema;

  // The ArrowArray responsible for owning the memory
  struct ArrowArray array;

  // Cached pointers pointing inside the array's private data
  // Depending on what exactly is being built, these pointers
  // might be NULL.
  struct ArrowBitmap* validity;
  struct ArrowBuffer* buffers[8];

  // Fields to keep track of state when using the visitor pattern
  int visitor_initialized;
  int feat_is_null;
  int nesting_multipoint;
  double empty_coord_values[4];
  struct GeoArrowCoordView empty_coord;
  enum GeoArrowDimensions last_dimensions;
  int64_t size[32];
  int32_t level;
  int64_t null_count;
};

static ArrowErrorCode GeoArrowBuilderInitArrayAndCachePointers(
    struct GeoArrowBuilder* builder) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  NANOARROW_RETURN_NOT_OK(
      ArrowArrayInitFromSchema(&private->array, &private->schema, NULL));

  private->validity = ArrowArrayValidityBitmap(&private->array);

  struct _GeoArrowFindBufferResult res;
  for (int64_t i = 0; i < builder->view.n_buffers; i++) {
    res.array = NULL;
    _GeoArrowArrayFindBuffer(&private->array, &res, i, 0, 0);
    if (res.array == NULL) {
      return EINVAL;
    }

    private->buffers[i] = ArrowArrayBuffer(res.array, res.i);
    builder->view.buffers[i].data.as_uint8 = NULL;
    builder->view.buffers[i].size_bytes = 0;
    builder->view.buffers[i].capacity_bytes = 0;
  }

  // Reset the coordinate counts and values
  builder->view.coords.size_coords = 0;
  builder->view.coords.capacity_coords = 0;
  for (int i = 0; i < 4; i++) {
    builder->view.coords.values[i] = NULL;
  }

  // Set the null_count to zero
  private->null_count = 0;

  // When we use the visitor pattern we initialize some things that need
  // to happen exactly once (e.g., append an initial zero to offset buffers)
  private->visitor_initialized = 0;

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowBuilderPrepareForVisiting(
    struct GeoArrowBuilder* builder) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  if (!private->visitor_initialized) {
    int32_t zero = 0;
    for (int i = 0; i < builder->view.n_offsets; i++) {
      NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, i, &zero, 1));
    }

    builder->view.coords.size_coords = 0;
    builder->view.coords.capacity_coords = 0;

    private->visitor_initialized = 1;
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowBuilderInitInternal(struct GeoArrowBuilder* builder) {
  enum GeoArrowType type = builder->view.schema_view.type;

  // Initialize an array view to help set some fields
  struct GeoArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowArrayViewInitFromType(&array_view, type));

  struct BuilderPrivate* private =
      (struct BuilderPrivate*)ArrowMalloc(sizeof(struct BuilderPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  memset(private, 0, sizeof(struct BuilderPrivate));
  builder->private_data = private;

  // Initialize our copy of the schema for the storage type
  int result = GeoArrowSchemaInit(&private->schema, type);
  if (result != GEOARROW_OK) {
    ArrowFree(private);
    builder->private_data = NULL;
    return result;
  }

  // Update a few things about the writable view from the regular view
  // that never change.
  builder->view.coords.n_values = array_view.coords.n_values;
  builder->view.coords.coords_stride = array_view.coords.coords_stride;
  builder->view.n_offsets = array_view.n_offsets;
  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      builder->view.n_buffers = 1 + array_view.n_offsets + array_view.coords.n_values;
      break;

    // interleaved + WKB + WKT
    default:
      builder->view.n_buffers = 1 + array_view.n_offsets + 1;
      break;
  }

  // Initialize an empty array; cache the ArrowBitmap and ArrowBuffer pointers we need
  result = GeoArrowBuilderInitArrayAndCachePointers(builder);
  if (result != GEOARROW_OK) {
    private->schema.release(&private->schema);
    ArrowFree(private);
    builder->private_data = NULL;
    return result;
  }

  // Initalize one empty coordinate for the visitor pattern
  memcpy(private->empty_coord_values, kEmptyPointCoords, 4 * sizeof(double));
  private->empty_coord.values[0] = private->empty_coord_values;
  private->empty_coord.values[1] = private->empty_coord_values + 1;
  private->empty_coord.values[2] = private->empty_coord_values + 2;
  private->empty_coord.values[3] = private->empty_coord_values + 3;
  private->empty_coord.n_coords = 1;
  private->empty_coord.n_values = 4;
  private->empty_coord.coords_stride = 1;

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowBuilderInitFromType(struct GeoArrowBuilder* builder,
                                              enum GeoArrowType type) {
  memset(builder, 0, sizeof(struct GeoArrowBuilder));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInitFromType(&builder->view.schema_view, type));
  return GeoArrowBuilderInitInternal(builder);
}

GeoArrowErrorCode GeoArrowBuilderInitFromSchema(struct GeoArrowBuilder* builder,
                                                const struct ArrowSchema* schema,
                                                struct GeoArrowError* error) {
  memset(builder, 0, sizeof(struct GeoArrowBuilder));
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInit(&builder->view.schema_view, schema, error));
  return GeoArrowBuilderInitInternal(builder);
}

GeoArrowErrorCode GeoArrowBuilderReserveBuffer(struct GeoArrowBuilder* builder, int64_t i,
                                               int64_t additional_size_bytes) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  struct ArrowBuffer* buffer_src = private->buffers[i];
  struct GeoArrowWritableBufferView* buffer_dst = builder->view.buffers + i;

  // Sync any changes from the builder's view of the buffer to nanoarrow's
  buffer_src->size_bytes = buffer_dst->size_bytes;

  // Use nanoarrow's reserve
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer_src, additional_size_bytes));

  // Sync any changes back to the builder's view
  builder->view.buffers[i].data.data = buffer_src->data;
  builder->view.buffers[i].capacity_bytes = buffer_src->capacity_bytes;
  return GEOARROW_OK;
}

struct GeoArrowBufferDeallocatorPrivate {
  void (*custom_free)(uint8_t* ptr, int64_t size, void* private_data);
  void* private_data;
};

static void GeoArrowBufferDeallocateWrapper(struct ArrowBufferAllocator* allocator,
                                            uint8_t* ptr, int64_t size) {
  struct GeoArrowBufferDeallocatorPrivate* private_data =
      (struct GeoArrowBufferDeallocatorPrivate*)allocator->private_data;
  private_data->custom_free(ptr, size, private_data->private_data);
  ArrowFree(private_data);
}

GeoArrowErrorCode GeoArrowBuilderSetOwnedBuffer(
    struct GeoArrowBuilder* builder, int64_t i, struct GeoArrowBufferView value,
    void (*custom_free)(uint8_t* ptr, int64_t size, void* private_data),
    void* private_data) {
  if (i < 0 || i >= builder->view.n_buffers) {
    return EINVAL;
  }

  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  struct ArrowBuffer* buffer_src = private->buffers[i];

  struct GeoArrowBufferDeallocatorPrivate* deallocator =
      (struct GeoArrowBufferDeallocatorPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowBufferDeallocatorPrivate));
  if (deallocator == NULL) {
    return ENOMEM;
  }

  deallocator->custom_free = custom_free;
  deallocator->private_data = private_data;

  ArrowBufferReset(buffer_src);
  buffer_src->allocator =
      ArrowBufferDeallocator(&GeoArrowBufferDeallocateWrapper, deallocator);
  buffer_src->data = (uint8_t*)value.data;
  buffer_src->size_bytes = value.size_bytes;
  buffer_src->capacity_bytes = value.size_bytes;

  // Sync this information to the writable view
  builder->view.buffers[i].data.data = buffer_src->data;
  builder->view.buffers[i].size_bytes = buffer_src->size_bytes;
  builder->view.buffers[i].capacity_bytes = buffer_src->capacity_bytes;

  return GEOARROW_OK;
}

static void GeoArrowSetArrayLengthFromBufferLength(struct GeoArrowSchemaView* schema_view,
                                                   struct _GeoArrowFindBufferResult* res,
                                                   int64_t size_bytes);

static void GeoArrowSetCoordContainerLength(struct GeoArrowBuilder* builder);

GeoArrowErrorCode GeoArrowBuilderFinish(struct GeoArrowBuilder* builder,
                                        struct ArrowArray* array,
                                        struct GeoArrowError* error) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If the coordinate appender was used, we may need to update the buffer sizes
  struct GeoArrowWritableCoordView* writable_view = &builder->view.coords;
  int64_t last_buffer = builder->view.n_buffers - 1;
  int n_values = writable_view->n_values;
  int64_t size_by_coords;

  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      size_by_coords = writable_view->size_coords * sizeof(double) * n_values;
      if (size_by_coords > builder->view.buffers[last_buffer].size_bytes) {
        builder->view.buffers[last_buffer].size_bytes = size_by_coords;
      }
      break;

    case GEOARROW_COORD_TYPE_SEPARATE:
      for (int64_t i = last_buffer - n_values + 1; i <= last_buffer; i++) {
        size_by_coords = writable_view->size_coords * sizeof(double);
        if (size_by_coords > builder->view.buffers[i].size_bytes) {
          builder->view.buffers[i].size_bytes = size_by_coords;
        }
      }
      break;

    default:
      return EINVAL;
  }

  // If the validity bitmap was used, we need to update the validity buffer size
  if (private->validity->buffer.data != NULL &&
      builder->view.buffers[0].data.data == NULL) {
    builder->view.buffers[0].data.as_uint8 = private->validity->buffer.data;
    builder->view.buffers[0].size_bytes = private->validity->buffer.size_bytes;
    builder->view.buffers[0].capacity_bytes = private->validity->buffer.capacity_bytes;
  }

  // Sync builder's buffers back to the array; set array lengths from buffer sizes
  struct _GeoArrowFindBufferResult res;
  for (int64_t i = 0; i < builder->view.n_buffers; i++) {
    private->buffers[i]->size_bytes = builder->view.buffers[i].size_bytes;

    res.array = NULL;
    _GeoArrowArrayFindBuffer(&private->array, &res, i, 0, 0);
    if (res.array == NULL) {
      return EINVAL;
    }

    GeoArrowSetArrayLengthFromBufferLength(&builder->view.schema_view, &res,
                                           private->buffers[i]->size_bytes);
  }

  // Set the struct or fixed-size list container length
  GeoArrowSetCoordContainerLength(builder);

  // Call finish building, which will flush the buffer pointers into the array
  // and validate sizes.
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayFinishBuildingDefault(&private->array, (struct ArrowError*)error));

  // If the null_count was incremented, we know what it is; if the first buffer
  // is non-null, we don't know what it is
  if (private->null_count > 0) {
    private->array.null_count = private->null_count;
  } else if (private->array.buffers[0] != NULL) {
    private->array.null_count = -1;
  }

  // Move the result out of private so we can maybe prepare for the next round
  struct ArrowArray tmp;
  ArrowArrayMove(&private->array, &tmp);

  // Prepare for another round of visiting (e.g., append zeroes to the offset arrays)
  int need_reinit_visitor = private->visitor_initialized;
  int result = GeoArrowBuilderInitArrayAndCachePointers(builder);
  if (result != GEOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  if (need_reinit_visitor) {
    result = GeoArrowBuilderPrepareForVisiting(builder);
    if (result != GEOARROW_OK) {
      tmp.release(&tmp);
      return result;
    }
  }

  // Move the result
  ArrowArrayMove(&tmp, array);
  return GEOARROW_OK;
}

void GeoArrowBuilderReset(struct GeoArrowBuilder* builder) {
  if (builder->private_data != NULL) {
    struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

    if (private->schema.release != NULL) {
      private->schema.release(&private->schema);
    }

    if (private->array.release != NULL) {
      private->array.release(&private->array);
    }

    ArrowFree(private);
    builder->private_data = NULL;
  }
}

static void GeoArrowSetArrayLengthFromBufferLength(struct GeoArrowSchemaView* schema_view,
                                                   struct _GeoArrowFindBufferResult* res,
                                                   int64_t size_bytes) {
  // By luck, buffer index 1 for every array is the one we use to infer the length;
  // however, this is a slightly different formula for each type/depth
  if (res->i != 1) {
    return;
  }

  // ...but in all cases, if the size is 0, the length is 0
  if (size_bytes == 0) {
    res->array->length = 0;
    return;
  }

  switch (schema_view->type) {
    case GEOARROW_TYPE_WKB:
    case GEOARROW_TYPE_WKT:
      res->array->length = (size_bytes / sizeof(int32_t)) - 1;
      return;
    case GEOARROW_TYPE_LARGE_WKB:
    case GEOARROW_TYPE_LARGE_WKT:
      res->array->length = (size_bytes / sizeof(int64_t)) - 1;
      return;
    default:
      break;
  }

  int coord_level;
  switch (schema_view->geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      coord_level = 0;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      coord_level = 1;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      coord_level = 2;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      coord_level = 3;
      break;
    default:
      return;
  }

  if (res->level < coord_level) {
    // This is an offset buffer
    res->array->length = (size_bytes / sizeof(int32_t)) - 1;
  } else {
    // This is a data buffer
    res->array->length = size_bytes / sizeof(double);
  }
}

static void GeoArrowSetCoordContainerLength(struct GeoArrowBuilder* builder) {
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // At this point all the array lengths should be set except for the
  // fixed-size list or struct parent to the coordinate array(s).
  int scale = -1;
  switch (builder->view.schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      scale = 1;
      break;
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      switch (builder->view.schema_view.dimensions) {
        case GEOARROW_DIMENSIONS_XY:
          scale = 2;
          break;
        case GEOARROW_DIMENSIONS_XYZ:
        case GEOARROW_DIMENSIONS_XYM:
          scale = 3;
          break;
        case GEOARROW_DIMENSIONS_XYZM:
          scale = 4;
          break;
        default:
          return;
      }
      break;
    default:
      // e.g., WKB
      break;
  }

  switch (builder->view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      private
      ->array.length = private->array.children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private
      ->array.children[0]->length =
          private->array.children[0]->children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      private
      ->array.children[0]->children[0]->length =
          private->array.children[0]->children[0]->children[0]->length / scale;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      private
      ->array.children[0]->children[0]->children[0]->length =
          private->array.children[0]->children[0]->children[0]->children[0]->length /
          scale;
      break;
    default:
      // e.g., WKB
      break;
  }
}

static int feat_start_point(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level = 0;
  private->size[0] = 0;
  private->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_point(struct GeoArrowVisitor* v,
                            enum GeoArrowGeometryType geometry_type,
                            enum GeoArrowDimensions dimensions) {
  // level++, geometry type, dimensions, reset size
  // validate dimensions, maybe against some options that indicate
  // error for mismatch, fill, or drop behaviour
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->last_dimensions = dimensions;
  return GEOARROW_OK;
}

static int ring_start_point(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int coords_point(struct GeoArrowVisitor* v,
                        const struct GeoArrowCoordView* coords) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->size[0] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(builder, coords, private->last_dimensions, 0,
                                     coords->n_coords);
}

static int ring_end_point(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int geom_end_point(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int null_feat_point(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_point(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If there weren't any coords (i.e., EMPTY), we need to write some NANs here
  // if there was >1 coords, we also need to error or we'll get misaligned output
  if (private->size[0] == 0) {
    int n_dim = _GeoArrowkNumDimensions[builder->view.schema_view.dimensions];
    private->empty_coord.n_values = n_dim;
    NANOARROW_RETURN_NOT_OK(coords_point(v, &private->empty_coord));
  } else if (private->size[0] != 1) {
    GeoArrowErrorSet(v->error, "Can't convert feature with >1 coordinate to POINT");
    return EINVAL;
  }

  if (private->feat_is_null) {
    int64_t current_length = builder->view.coords.size_coords;
    if (private->validity->buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(private->validity, current_length));
      ArrowBitmapAppendUnsafe(private->validity, 1, current_length - 1);
    }

    private->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 0, 1));
  } else if (private->validity->buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitPoint(struct GeoArrowBuilder* builder,
                                     struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_point;
  v->null_feat = &null_feat_point;
  v->geom_start = &geom_start_point;
  v->ring_start = &ring_start_point;
  v->coords = &coords_point;
  v->ring_end = &ring_end_point;
  v->geom_end = &geom_end_point;
  v->feat_end = &feat_end_point;
  v->private_data = builder;
}

static int feat_start_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level = 0;
  private->size[0] = 0;
  private->size[1] = 0;
  private->feat_is_null = 0;
  private->nesting_multipoint = 0;
  return GEOARROW_OK;
}

static int geom_start_multipoint(struct GeoArrowVisitor* v,
                                 enum GeoArrowGeometryType geometry_type,
                                 enum GeoArrowDimensions dimensions) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      private
      ->level++;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private
      ->nesting_multipoint = 1;
      private->level++;
      break;
    case GEOARROW_GEOMETRY_TYPE_POINT:
      if (private->nesting_multipoint) {
        private->nesting_multipoint++;
      }
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level++;
  return GEOARROW_OK;
}

static int coords_multipoint(struct GeoArrowVisitor* v,
                             const struct GeoArrowCoordView* coords) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->size[1] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(builder, coords, private->last_dimensions, 0,
                                     coords->n_coords);
}

static int ring_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  private->level--;
  private->size[0]++;
  if (builder->view.coords.size_coords > 2147483647) {
    return EOVERFLOW;
  }
  int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
  NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 0, &n_coord32, 1));

  return GEOARROW_OK;
}

static int geom_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // Ignore geom_end calls from the end of a POINT nested within a MULTIPOINT
  if (private->nesting_multipoint == 2) {
    private->nesting_multipoint--;
    return GEOARROW_OK;
  }

  if (private->level == 1) {
    private->size[0]++;
    private->level--;
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 0, &n_coord32, 1));
  }

  return GEOARROW_OK;
}

static int null_feat_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multipoint(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If we didn't finish any sequences, finish at least one. This is usually an
  // EMPTY but could also be a single point.
  if (private->size[0] == 0) {
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 0, &n_coord32, 1));
  } else if (private->size[0] != 1) {
    GeoArrowErrorSet(v->error, "Can't convert feature with >1 sequence to LINESTRING");
    return EINVAL;
  }

  if (private->feat_is_null) {
    int64_t current_length = builder->view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private->validity->buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(private->validity, current_length));
      ArrowBitmapAppendUnsafe(private->validity, 1, current_length - 1);
    }

    private->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 0, 1));
  } else if (private->validity->buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitLinestring(struct GeoArrowBuilder* builder,
                                          struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multipoint;
  v->null_feat = &null_feat_multipoint;
  v->geom_start = &geom_start_multipoint;
  v->ring_start = &ring_start_multipoint;
  v->coords = &coords_multipoint;
  v->ring_end = &ring_end_multipoint;
  v->geom_end = &geom_end_multipoint;
  v->feat_end = &feat_end_multipoint;
  v->private_data = builder;
}

static int feat_start_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level = 0;
  private->size[0] = 0;
  private->size[1] = 0;
  private->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_multilinestring(struct GeoArrowVisitor* v,
                                      enum GeoArrowGeometryType geometry_type,
                                      enum GeoArrowDimensions dimensions) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private
      ->level++;
      break;
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level++;
  return GEOARROW_OK;
}

static int coords_multilinestring(struct GeoArrowVisitor* v,
                                  const struct GeoArrowCoordView* coords) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->size[1] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(builder, coords, private->last_dimensions, 0,
                                     coords->n_coords);
}

static int ring_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  private->level--;
  if (private->size[1] > 0) {
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 1, &n_coord32, 1));
    private->size[0]++;
    private->size[1] = 0;
  }

  return GEOARROW_OK;
}

static int geom_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  if (private->level == 1) {
    private->level--;
    if (private->size[1] > 0) {
      if (builder->view.coords.size_coords > 2147483647) {
        return EOVERFLOW;
      }
      int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
      NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 1, &n_coord32, 1));
      private->size[0]++;
      private->size[1] = 0;
    }
  }

  return GEOARROW_OK;
}

static int null_feat_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multilinestring(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If we have an unfinished sequence left over, finish it now. This could have
  // occurred if the last geometry that was visited was a POINT.
  if (private->size[1] > 0) {
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 1, &n_coord32, 1));
  }

  // Finish off the sequence of sequences. This is a polygon or multilinestring
  // so it can any number of them.
  int32_t n_seq32 = (int32_t)(builder->view.buffers[2].size_bytes / sizeof(int32_t)) - 1;
  NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 0, &n_seq32, 1));

  if (private->feat_is_null) {
    int64_t current_length = builder->view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private->validity->buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(private->validity, current_length));
      ArrowBitmapAppendUnsafe(private->validity, 1, current_length - 1);
    }

    private->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 0, 1));
  } else if (private->validity->buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitMultiLinestring(struct GeoArrowBuilder* builder,
                                               struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multilinestring;
  v->null_feat = &null_feat_multilinestring;
  v->geom_start = &geom_start_multilinestring;
  v->ring_start = &ring_start_multilinestring;
  v->coords = &coords_multilinestring;
  v->ring_end = &ring_end_multilinestring;
  v->geom_end = &geom_end_multilinestring;
  v->feat_end = &feat_end_multilinestring;
  v->private_data = builder;
}

static int feat_start_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level = 0;
  private->size[0] = 0;
  private->size[1] = 0;
  private->size[2] = 0;
  private->feat_is_null = 0;
  return GEOARROW_OK;
}

static int geom_start_multipolygon(struct GeoArrowVisitor* v,
                                   enum GeoArrowGeometryType geometry_type,
                                   enum GeoArrowDimensions dimensions) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->last_dimensions = dimensions;

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      private
      ->level++;
      break;
    default:
      break;
  }

  return GEOARROW_OK;
}

static int ring_start_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->level++;
  return GEOARROW_OK;
}

static int coords_multipolygon(struct GeoArrowVisitor* v,
                               const struct GeoArrowCoordView* coords) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->size[2] += coords->n_coords;
  return GeoArrowBuilderCoordsAppend(builder, coords, private->last_dimensions, 0,
                                     coords->n_coords);
}

static int ring_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  private->level--;
  if (private->size[2] > 0) {
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 2, &n_coord32, 1));
    private->size[1]++;
    private->size[2] = 0;
  }

  return GEOARROW_OK;
}

static int geom_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  if (private->level == 2) {
    private->level--;
    if (private->size[2] > 0) {
      if (builder->view.coords.size_coords > 2147483647) {
        return EOVERFLOW;
      }
      int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
      NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 2, &n_coord32, 1));
      private->size[1]++;
      private->size[2] = 0;
    }
  } else if (private->level == 1) {
    private->level--;
    if (private->size[1] > 0) {
      int32_t n_seq32 =
          (int32_t)(builder->view.buffers[3].size_bytes / sizeof(int32_t)) - 1;
      NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 1, &n_seq32, 1));
      private->size[0]++;
      private->size[1] = 0;
    }
  }

  return GEOARROW_OK;
}

static int null_feat_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int feat_end_multipolygon(struct GeoArrowVisitor* v) {
  struct GeoArrowBuilder* builder = (struct GeoArrowBuilder*)v->private_data;
  struct BuilderPrivate* private = (struct BuilderPrivate*)builder->private_data;

  // If we have an unfinished sequence left over, finish it now. This could have
  // occurred if the last geometry that was visited was a POINT.
  if (private->size[2] > 0) {
    if (builder->view.coords.size_coords > 2147483647) {
      return EOVERFLOW;
    }
    int32_t n_coord32 = (int32_t)builder->view.coords.size_coords;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 2, &n_coord32, 1));
    private->size[1]++;
  }

  // If we have an unfinished sequence of sequences left over, finish it now.
  // This could have occurred if the last geometry that was visited was a POINT.
  if (private->size[1] > 0) {
    int32_t n_seq32 =
        (int32_t)(builder->view.buffers[3].size_bytes / sizeof(int32_t)) - 1;
    NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 1, &n_seq32, 1));
  }

  // Finish off the sequence of sequence of sequences. This is a multipolygon
  // so it can be any number of them.
  int32_t n_seq_seq32 =
      (int32_t)(builder->view.buffers[2].size_bytes / sizeof(int32_t)) - 1;
  NANOARROW_RETURN_NOT_OK(GeoArrowBuilderOffsetAppend(builder, 0, &n_seq_seq32, 1));

  if (private->feat_is_null) {
    int64_t current_length = builder->view.buffers[1].size_bytes / sizeof(int32_t) - 1;
    if (private->validity->buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(private->validity, current_length));
      ArrowBitmapAppendUnsafe(private->validity, 1, current_length - 1);
    }

    private->null_count++;
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 0, 1));
  } else if (private->validity->buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(private->validity, 1, 1));
  }

  return GEOARROW_OK;
}

static void GeoArrowVisitorInitMultiPolygon(struct GeoArrowBuilder* builder,
                                            struct GeoArrowVisitor* v) {
  struct GeoArrowError* previous_error = v->error;
  GeoArrowVisitorInitVoid(v);
  v->error = previous_error;

  v->feat_start = &feat_start_multipolygon;
  v->null_feat = &null_feat_multipolygon;
  v->geom_start = &geom_start_multipolygon;
  v->ring_start = &ring_start_multipolygon;
  v->coords = &coords_multipolygon;
  v->ring_end = &ring_end_multipolygon;
  v->geom_end = &geom_end_multipolygon;
  v->feat_end = &feat_end_multipolygon;
  v->private_data = builder;
}

GeoArrowErrorCode GeoArrowBuilderInitVisitor(struct GeoArrowBuilder* builder,
                                             struct GeoArrowVisitor* v) {
  switch (builder->view.schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      GeoArrowVisitorInitPoint(builder, v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      GeoArrowVisitorInitLinestring(builder, v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      GeoArrowVisitorInitMultiLinestring(builder, v);
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      GeoArrowVisitorInitMultiPolygon(builder, v);
      break;
    default:
      return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(GeoArrowBuilderPrepareForVisiting(builder));
  return GEOARROW_OK;
}

#include <errno.h>





static int32_t kZeroInt32 = 0;

static int GeoArrowArrayViewInitInternal(struct GeoArrowArrayView* array_view,
                                         struct GeoArrowError* error) {
  switch (array_view->schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      array_view->n_offsets = 0;
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      array_view->n_offsets = 1;
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      array_view->n_offsets = 2;
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      array_view->n_offsets = 3;
      break;
    default:
      // i.e., serialized type
      array_view->n_offsets = 1;
      break;
  }

  for (int i = 0; i < 4; i++) {
    array_view->length[i] = 0;
    array_view->offset[i] = 0;
  }

  array_view->validity_bitmap = NULL;
  for (int i = 0; i < 3; i++) {
    array_view->offsets[i] = NULL;
  }
  array_view->data = NULL;

  array_view->coords.n_coords = 0;
  switch (array_view->schema_view.dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      array_view->coords.n_values = 2;
      break;
    case GEOARROW_DIMENSIONS_XYZ:
    case GEOARROW_DIMENSIONS_XYM:
      array_view->coords.n_values = 3;
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      array_view->coords.n_values = 4;
      break;
    default:
      // i.e., serialized type
      array_view->coords.n_coords = 0;
      break;
  }

  switch (array_view->schema_view.coord_type) {
    case GEOARROW_COORD_TYPE_SEPARATE:
      array_view->coords.coords_stride = 1;
      break;
    case GEOARROW_COORD_TYPE_INTERLEAVED:
      array_view->coords.coords_stride = array_view->coords.n_values;
      break;
    default:
      // i.e., serialized type
      array_view->coords.coords_stride = 0;
      break;
  }

  for (int i = 0; i < 4; i++) {
    array_view->coords.values[i] = NULL;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewInitFromType(struct GeoArrowArrayView* array_view,
                                                enum GeoArrowType type) {
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInitFromType(&array_view->schema_view, type));
  return GeoArrowArrayViewInitInternal(array_view, NULL);
}

GeoArrowErrorCode GeoArrowArrayViewInitFromSchema(struct GeoArrowArrayView* array_view,
                                                  const struct ArrowSchema* schema,
                                                  struct GeoArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      GeoArrowSchemaViewInit(&array_view->schema_view, schema, error));
  return GeoArrowArrayViewInitInternal(array_view, error);
}

static int GeoArrowArrayViewSetArrayInternal(struct GeoArrowArrayView* array_view,
                                             const struct ArrowArray* array,
                                             struct GeoArrowError* error, int level) {
  // Set offset + length of the array
  array_view->offset[level] = array->offset;
  array_view->length[level] = array->length;

  if (level == array_view->n_offsets) {
    // We're at the coord array!

    // n_coords is last_offset[level - 1] or array->length if level == 0
    if (level > 0) {
      int32_t first_offset = array_view->first_offset[level - 1];
      array_view->coords.n_coords = array_view->last_offset[level - 1] - first_offset;
    } else {
      array_view->coords.n_coords = array->length;
    }

    switch (array_view->schema_view.coord_type) {
      case GEOARROW_COORD_TYPE_SEPARATE:
        if (array->n_children != array_view->coords.n_values) {
          GeoArrowErrorSet(error,
                           "Unexpected number of children for struct coordinate array "
                           "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        // Set the coord pointers to the data buffer of each child (applying
        // offset before assigning the pointer)
        for (int32_t i = 0; i < array_view->coords.n_values; i++) {
          if (array->children[i]->n_buffers != 2) {
            ArrowErrorSet(
                (struct ArrowError*)error,
                "Unexpected number of buffers for struct coordinate array child "
                "in GeoArrowArrayViewSetArray()");
            return EINVAL;
          }

          array_view->coords.values[i] = ((const double*)array->children[i]->buffers[1]) +
                                         array->children[i]->offset;
        }

        break;

      case GEOARROW_COORD_TYPE_INTERLEAVED:
        if (array->n_children != 1) {
          GeoArrowErrorSet(
              error,
              "Unexpected number of children for interleaved coordinate array "
              "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        if (array->children[0]->n_buffers != 2) {
          ArrowErrorSet(
              (struct ArrowError*)error,
              "Unexpected number of buffers for interleaved coordinate array child "
              "in GeoArrowArrayViewSetArray()");
          return EINVAL;
        }

        // Set the coord pointers to the first four doubles in the data buffers

        for (int32_t i = 0; i < array_view->coords.n_values; i++) {
          array_view->coords.values[i] = ((const double*)array->children[0]->buffers[1]) +
                                         array->children[0]->offset + i;
        }

        break;

      default:
        GeoArrowErrorSet(error, "Unexpected coordinate type GeoArrowArrayViewSetArray()");
        return EINVAL;
    }

    return GEOARROW_OK;
  }

  if (array->n_buffers != 2) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Unexpected number of buffers in list array in GeoArrowArrayViewSetArray()");
    return EINVAL;
  }

  if (array->n_children != 1) {
    ArrowErrorSet(
        (struct ArrowError*)error,
        "Unexpected number of children in list array in GeoArrowArrayViewSetArray()");
    return EINVAL;
  }

  // Set the offsets buffer and the last_offset value of level
  if (array->length > 0) {
    array_view->offsets[level] = (const int32_t*)array->buffers[1];
    array_view->first_offset[level] = array_view->offsets[level][array->offset];
    array_view->last_offset[level] =
        array_view->offsets[level][array->offset + array->length];
  } else {
    array_view->offsets[level] = &kZeroInt32;
    array_view->first_offset[level] = 0;
    array_view->last_offset[level] = 0;
  }

  return GeoArrowArrayViewSetArrayInternal(array_view, array->children[0], error,
                                           level + 1);
}

static GeoArrowErrorCode GeoArrowArrayViewSetArraySerialized(
    struct GeoArrowArrayView* array_view, const struct ArrowArray* array,
    struct GeoArrowError* error) {
  array_view->length[0] = array->length;
  array_view->offset[0] = array->offset;

  array_view->offsets[0] = (const int32_t*)array->buffers[1];
  array_view->data = (const uint8_t*)array->buffers[2];
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewSetArray(struct GeoArrowArrayView* array_view,
                                            const struct ArrowArray* array,
                                            struct GeoArrowError* error) {
  switch (array_view->schema_view.type) {
    case GEOARROW_TYPE_WKT:
    case GEOARROW_TYPE_WKB:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewSetArraySerialized(array_view, array, error));
      break;
    default:
      NANOARROW_RETURN_NOT_OK(
          GeoArrowArrayViewSetArrayInternal(array_view, array, error, 0));
      break;
  }

  array_view->validity_bitmap = array->buffers[0];
  return GEOARROW_OK;
}

static inline void GeoArrowCoordViewUpdate(const struct GeoArrowCoordView* src,
                                           struct GeoArrowCoordView* dst, int64_t offset,
                                           int64_t length) {
  for (int j = 0; j < dst->n_values; j++) {
    dst->values[j] = src->values[j] + (offset * src->coords_stride);
  }
  dst->n_coords = length;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitPoint(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT,
                                            array_view->schema_view.dimensions));
      GeoArrowCoordViewUpdate(&array_view->coords, &coords,
                              array_view->offset[0] + offset + i, 1);
      NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));

    for (int j = 0; j < coords.n_values; j++) {
      coords.values[j] += coords.coords_stride;
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitLinestring(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING,
                                            array_view->schema_view.dimensions));
      coord_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_coords =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - coord_offset;
      coord_offset += array_view->offset[1];
      GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
      NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitPolygon(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t ring_offset;
  int64_t n_rings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON,
                                            array_view->schema_view.dimensions));
      ring_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_rings =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - ring_offset;
      ring_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_rings; j++) {
        NANOARROW_RETURN_NOT_OK(v->ring_start(v));
        coord_offset = array_view->offsets[1][ring_offset + j];
        n_coords = array_view->offsets[1][ring_offset + j + 1] - coord_offset;
        coord_offset += array_view->offset[2];
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitMultipoint(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTIPOINT,
                                            array_view->schema_view.dimensions));
      coord_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_coords =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - coord_offset;
      coord_offset += array_view->offset[1];
      for (int64_t j = 0; j < n_coords; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT,
                                              array_view->schema_view.dimensions));
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset + j, 1);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitMultilinestring(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t linestring_offset;
  int64_t n_linestrings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTILINESTRING,
                                            array_view->schema_view.dimensions));
      linestring_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_linestrings = array_view->offsets[0][array_view->offset[0] + offset + i + 1] -
                      linestring_offset;
      linestring_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_linestrings; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING,
                                              array_view->schema_view.dimensions));
        coord_offset = array_view->offsets[1][linestring_offset + j];
        n_coords = array_view->offsets[1][linestring_offset + j + 1] - coord_offset;
        coord_offset += array_view->offset[2];
        GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
        NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitMultipolygon(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowVisitor* v) {
  struct GeoArrowCoordView coords = array_view->coords;

  int64_t polygon_offset;
  int64_t n_polygons;
  int64_t ring_offset;
  int64_t n_rings;
  int64_t coord_offset;
  int64_t n_coords;
  for (int64_t i = 0; i < length; i++) {
    NANOARROW_RETURN_NOT_OK(v->feat_start(v));
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON,
                                            array_view->schema_view.dimensions));

      polygon_offset = array_view->offsets[0][array_view->offset[0] + offset + i];
      n_polygons =
          array_view->offsets[0][array_view->offset[0] + offset + i + 1] - polygon_offset;
      polygon_offset += array_view->offset[1];

      for (int64_t j = 0; j < n_polygons; j++) {
        NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON,
                                              array_view->schema_view.dimensions));

        ring_offset = array_view->offsets[1][polygon_offset + j];
        n_rings = array_view->offsets[1][polygon_offset + j + 1] - ring_offset;
        ring_offset += array_view->offset[2];

        for (int64_t k = 0; k < n_rings; k++) {
          NANOARROW_RETURN_NOT_OK(v->ring_start(v));
          coord_offset = array_view->offsets[2][ring_offset + k];
          n_coords = array_view->offsets[2][ring_offset + k + 1] - coord_offset;
          coord_offset += array_view->offset[3];
          GeoArrowCoordViewUpdate(&array_view->coords, &coords, coord_offset, n_coords);
          NANOARROW_RETURN_NOT_OK(v->coords(v, &coords));
          NANOARROW_RETURN_NOT_OK(v->ring_end(v));
        }

        NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      }

      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
    }

    NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayViewVisit(const struct GeoArrowArrayView* array_view,
                                         int64_t offset, int64_t length,
                                         struct GeoArrowVisitor* v) {
  switch (array_view->schema_view.geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      return GeoArrowArrayViewVisitPoint(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      return GeoArrowArrayViewVisitLinestring(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      return GeoArrowArrayViewVisitPolygon(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      return GeoArrowArrayViewVisitMultipoint(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      return GeoArrowArrayViewVisitMultilinestring(array_view, offset, length, v);
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      return GeoArrowArrayViewVisitMultipolygon(array_view, offset, length, v);
    default:
      return ENOTSUP;
  }
}

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>



const char* GeoArrowVersion(void) { return GEOARROW_VERSION; }

int GeoArrowVersionInt(void) { return GEOARROW_VERSION_INT; }

GeoArrowErrorCode GeoArrowErrorSet(struct GeoArrowError* error, const char* fmt, ...) {
  if (error == NULL) {
    return GEOARROW_OK;
  }

  memset(error->message, 0, sizeof(error->message));

  va_list args;
  va_start(args, fmt);
  int chars_needed = vsnprintf(error->message, sizeof(error->message), fmt, args);
  va_end(args);

  if (chars_needed < 0) {
    return EINVAL;
  } else if (((size_t)chars_needed) >= sizeof(error->message)) {
    return ERANGE;
  } else {
    return GEOARROW_OK;
  }
}

const char* GeoArrowErrorMessage(struct GeoArrowError* error) {
  if (error == NULL) {
    return "";
  } else {
    return error->message;
  }
}

#include <stddef.h>



static int feat_start_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int null_feat_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int geom_start_void(struct GeoArrowVisitor* v,
                           enum GeoArrowGeometryType geometry_type,
                           enum GeoArrowDimensions dimensions) {
  return GEOARROW_OK;
}

static int ring_start_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int coords_void(struct GeoArrowVisitor* v,
                       const struct GeoArrowCoordView* coords) {
  return GEOARROW_OK;
}

static int ring_end_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int geom_end_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

static int feat_end_void(struct GeoArrowVisitor* v) { return GEOARROW_OK; }

void GeoArrowVisitorInitVoid(struct GeoArrowVisitor* v) {
  v->feat_start = &feat_start_void;
  v->null_feat = &null_feat_void;
  v->geom_start = &geom_start_void;
  v->ring_start = &ring_start_void;
  v->coords = &coords_void;
  v->ring_end = &ring_end_void;
  v->geom_end = &geom_end_void;
  v->feat_end = &feat_end_void;
  v->error = NULL;
  v->private_data = NULL;
}

#include <errno.h>
#include <stdint.h>





#define EWKB_Z_BIT 0x80000000
#define EWKB_M_BIT 0x40000000
#define EWKB_SRID_BIT 0x20000000

#ifndef GEOARROW_NATIVE_ENDIAN
#define GEOARROW_NATIVE_ENDIAN 0x01
#endif

#ifndef GEOARROW_BSWAP32
static inline uint32_t bswap_32(uint32_t x) {
  return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) |
          ((x & 0xFF000000) >> 24));
}
#define GEOARROW_BSWAP32(x) bswap_32(x)
#endif

#ifndef GEOARROW_BSWAP64
static inline uint64_t bswap_64(uint64_t x) {
  return (((x & 0xFFULL) << 56) | ((x & 0xFF00ULL) << 40) | ((x & 0xFF0000ULL) << 24) |
          ((x & 0xFF000000ULL) << 8) | ((x & 0xFF00000000ULL) >> 8) |
          ((x & 0xFF0000000000ULL) >> 24) | ((x & 0xFF000000000000ULL) >> 40) |
          ((x & 0xFF00000000000000ULL) >> 56));
}
#define GEOARROW_BSWAP64(x) bswap_64(x)
#endif

// This must be divisible by 2, 3, and 4
#define COORD_CACHE_SIZE_ELEMENTS 3072

struct WKBReaderPrivate {
  const uint8_t* data;
  int64_t size_bytes;
  const uint8_t* data0;
  int need_swapping;
  double coords[COORD_CACHE_SIZE_ELEMENTS];
  struct GeoArrowCoordView coord_view;
};

static inline int WKBReaderReadEndian(struct WKBReaderPrivate* s,
                                      struct GeoArrowError* error) {
  if (s->size_bytes > 0) {
    s->need_swapping = s->data[0] != GEOARROW_NATIVE_ENDIAN;
    s->data++;
    s->size_bytes--;
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected endian byte but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

static inline int WKBReaderReadUInt32(struct WKBReaderPrivate* s, uint32_t* out,
                                      struct GeoArrowError* error) {
  if (s->size_bytes >= 4) {
    memcpy(out, s->data, sizeof(uint32_t));
    s->data += sizeof(uint32_t);
    s->size_bytes -= sizeof(uint32_t);
    if (s->need_swapping) {
      *out = GEOARROW_BSWAP32(*out);
    }
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected uint32 but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

static inline void WKBReaderMaybeBswapCoords(struct WKBReaderPrivate* s, int64_t n) {
  if (s->need_swapping) {
    uint64_t* data64 = (uint64_t*)s->coords;
    for (int i = 0; i < n; i++) {
      data64[i] = GEOARROW_BSWAP64(data64[i]);
    }
  }
}

static int WKBReaderReadCoordinates(struct WKBReaderPrivate* s, int64_t n_coords,
                                    struct GeoArrowVisitor* v) {
  int64_t bytes_needed = n_coords * s->coord_view.n_values * sizeof(double);
  if (s->size_bytes < bytes_needed) {
    ArrowErrorSet(
        (struct ArrowError*)v->error,
        "Expected coordinate sequence of %ld coords (%ld bytes) but found %ld bytes "
        "remaining at byte %ld",
        (long)n_coords, (long)bytes_needed, (long)s->size_bytes,
        (long)(s->data - s->data0));
    return EINVAL;
  }

  int32_t chunk_size = COORD_CACHE_SIZE_ELEMENTS / s->coord_view.n_values;
  s->coord_view.n_coords = chunk_size;

  // Process full chunks
  while (n_coords > chunk_size) {
    memcpy(s->coords, s->data, COORD_CACHE_SIZE_ELEMENTS * sizeof(double));
    WKBReaderMaybeBswapCoords(s, COORD_CACHE_SIZE_ELEMENTS);
    NANOARROW_RETURN_NOT_OK(v->coords(v, &s->coord_view));
    s->data += COORD_CACHE_SIZE_ELEMENTS * sizeof(double);
    s->size_bytes -= COORD_CACHE_SIZE_ELEMENTS * sizeof(double);
    n_coords -= chunk_size;
  }

  // Process the last chunk
  int64_t remaining_bytes = n_coords * s->coord_view.n_values * sizeof(double);
  memcpy(s->coords, s->data, remaining_bytes);
  s->data += remaining_bytes;
  s->size_bytes -= remaining_bytes;
  s->coord_view.n_coords = n_coords;
  WKBReaderMaybeBswapCoords(s, n_coords * s->coord_view.n_values);
  return v->coords(v, &s->coord_view);
}

static int WKBReaderReadGeometry(struct WKBReaderPrivate* s, struct GeoArrowVisitor* v) {
  NANOARROW_RETURN_NOT_OK(WKBReaderReadEndian(s, v->error));
  uint32_t geometry_type;
  const uint8_t* data_at_geom_type = s->data;
  NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &geometry_type, v->error));

  int has_z = 0;
  int has_m = 0;

  // Handle EWKB high bits
  if (geometry_type & EWKB_Z_BIT) {
    has_z = 1;
  }

  if (geometry_type & EWKB_M_BIT) {
    has_m = 1;
  }

  if (geometry_type & EWKB_SRID_BIT) {
    // We ignore this because it's hard to work around if a user somehow
    // has embedded srid but still wants the data and doesn't have another way
    // to convert
    uint32_t embedded_srid;
    NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &embedded_srid, v->error));
  }

  geometry_type = geometry_type & 0x0000ffff;

  // Handle ISO X000 geometry types
  if (geometry_type >= 3000) {
    geometry_type = geometry_type - 3000;
    has_z = 1;
    has_m = 1;
  } else if (geometry_type >= 2000) {
    geometry_type = geometry_type - 2000;
    has_m = 1;
  } else if (geometry_type >= 1000) {
    geometry_type = geometry_type - 1000;
    has_z = 1;
  }

  // Read the number of coordinates/rings/parts
  uint32_t size;
  if (geometry_type != GEOARROW_GEOMETRY_TYPE_POINT) {
    NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &size, v->error));
  } else {
    size = 1;
  }

  // Set coord size
  s->coord_view.n_values = 2 + has_z + has_m;
  s->coord_view.coords_stride = s->coord_view.n_values;

  // Resolve dimensions
  enum GeoArrowDimensions dimensions;
  if (has_z && has_m) {
    dimensions = GEOARROW_DIMENSIONS_XYZM;
  } else if (has_z) {
    dimensions = GEOARROW_DIMENSIONS_XYZ;
  } else if (has_m) {
    dimensions = GEOARROW_DIMENSIONS_XYM;
  } else {
    dimensions = GEOARROW_DIMENSIONS_XY;
  }

  NANOARROW_RETURN_NOT_OK(v->geom_start(v, geometry_type, dimensions));

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      NANOARROW_RETURN_NOT_OK(WKBReaderReadCoordinates(s, size, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      for (uint32_t i = 0; i < size; i++) {
        uint32_t ring_size;
        NANOARROW_RETURN_NOT_OK(WKBReaderReadUInt32(s, &ring_size, v->error));
        NANOARROW_RETURN_NOT_OK(v->ring_start(v));
        NANOARROW_RETURN_NOT_OK(WKBReaderReadCoordinates(s, ring_size, v));
        NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      }
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      for (uint32_t i = 0; i < size; i++) {
        NANOARROW_RETURN_NOT_OK(WKBReaderReadGeometry(s, v));
      }
      break;
    default:
      GeoArrowErrorSet(v->error,
                       "Expected valid geometry type code but found %u at byte %ld",
                       (unsigned int)geometry_type, (long)(data_at_geom_type - s->data0));
      return EINVAL;
  }

  return v->geom_end(v);
}

GeoArrowErrorCode GeoArrowWKBReaderInit(struct GeoArrowWKBReader* reader) {
  struct WKBReaderPrivate* s =
      (struct WKBReaderPrivate*)ArrowMalloc(sizeof(struct WKBReaderPrivate));

  if (s == NULL) {
    return ENOMEM;
  }

  s->data0 = NULL;
  s->data = NULL;
  s->size_bytes = 0;
  s->need_swapping = 0;

  s->coord_view.coords_stride = 2;
  s->coord_view.n_values = 2;
  s->coord_view.n_coords = 0;
  s->coord_view.values[0] = s->coords + 0;
  s->coord_view.values[1] = s->coords + 1;
  s->coord_view.values[2] = s->coords + 2;
  s->coord_view.values[3] = s->coords + 3;

  reader->private_data = s;
  return GEOARROW_OK;
}

void GeoArrowWKBReaderReset(struct GeoArrowWKBReader* reader) {
  ArrowFree(reader->private_data);
}

GeoArrowErrorCode GeoArrowWKBReaderVisit(struct GeoArrowWKBReader* reader,
                                         struct GeoArrowBufferView src,
                                         struct GeoArrowVisitor* v) {
  struct WKBReaderPrivate* s = (struct WKBReaderPrivate*)reader->private_data;
  s->data0 = src.data;
  s->data = src.data;
  s->size_bytes = src.size_bytes;

  NANOARROW_RETURN_NOT_OK(v->feat_start(v));
  NANOARROW_RETURN_NOT_OK(WKBReaderReadGeometry(s, v));
  NANOARROW_RETURN_NOT_OK(v->feat_end(v));

  return GEOARROW_OK;
}

#include <string.h>





struct WKBWriterPrivate {
  enum ArrowType storage_type;
  struct ArrowBitmap validity;
  struct ArrowBuffer offsets;
  struct ArrowBuffer values;
  enum GeoArrowGeometryType geometry_type[32];
  enum GeoArrowDimensions dimensions[32];
  int64_t size_pos[32];
  uint32_t size[32];
  int32_t level;
  int64_t length;
  int64_t null_count;
  int feat_is_null;
};

#ifndef GEOARROW_NATIVE_ENDIAN
#define GEOARROW_NATIVE_ENDIAN 0x01
#endif

static uint8_t kWKBWriterEmptyPointCoords2[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                                0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00,
                                                0x00, 0x00, 0xf8, 0x7f};
static uint8_t kWKBWriterEmptyPointCoords3[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};
static uint8_t kWKBWriterEmptyPointCoords4[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f};

static inline int WKBWriterCheckLevel(struct WKBWriterPrivate* private) {
  if (private->level >= 0 && private->level <= 30) {
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static int feat_start_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  private->level = 0;
  private->size[private->level] = 0;
  private->length++;
  private->feat_is_null = 0;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  return ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes);
}

static int null_feat_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int geom_start_wkb(struct GeoArrowVisitor* v,
                          enum GeoArrowGeometryType geometry_type,
                          enum GeoArrowDimensions dimensions) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  private->size[private->level]++;
  private->level++;
  private->geometry_type[private->level] = geometry_type;
  private->dimensions[private->level] = dimensions;
  private->size[private->level] = 0;

  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendUInt8(&private->values, GEOARROW_NATIVE_ENDIAN));
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(
      &private->values, geometry_type + ((dimensions - 1) * 1000)));
  if (geometry_type != GEOARROW_GEOMETRY_TYPE_POINT) {
    private->size_pos[private->level] = private->values.size_bytes;
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(&private->values, 0));
  }

  return GEOARROW_OK;
}

static int ring_start_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  private->size[private->level]++;
  private->level++;
  private->geometry_type[private->level] = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private->size_pos[private->level] = private->values.size_bytes;
  private->size[private->level] = 0;
  return ArrowBufferAppendUInt32(&private->values, 0);
}

static int coords_wkb(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  private->size[private->level] += coords->n_coords;
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(
      &private->values, coords->n_values * coords->n_coords * sizeof(double)));
  for (int64_t i = 0; i < coords->n_coords; i++) {
    for (int32_t j = 0; j < coords->n_values; j++) {
      ArrowBufferAppendUnsafe(&private->values,
                              coords->values[j] + i * coords->coords_stride,
                              sizeof(double));
    }
  }

  return GEOARROW_OK;
}

static int ring_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  if (private->values.data == NULL) {
    return EINVAL;
  }
  memcpy(private->values.data + private->size_pos[private->level],
         private->size + private->level, sizeof(uint32_t));
  private->level--;
  return GEOARROW_OK;
}

static int geom_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKBWriterCheckLevel(private));
  if (private->values.data == NULL) {
    return EINVAL;
  }

  if (private->geometry_type[private->level] != GEOARROW_GEOMETRY_TYPE_POINT) {
    memcpy(private->values.data + private->size_pos[private->level],
           private->size + private->level, sizeof(uint32_t));
  } else if (private->size[private->level] == 0) {
    switch (private->dimensions[private->level]) {
      case GEOARROW_DIMENSIONS_XY:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords2,
                                                  sizeof(kWKBWriterEmptyPointCoords2)));
        break;
      case GEOARROW_DIMENSIONS_XYZ:
      case GEOARROW_DIMENSIONS_XYM:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords3,
                                                  sizeof(kWKBWriterEmptyPointCoords3)));
        break;
      case GEOARROW_DIMENSIONS_XYZM:
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&private->values,
                                                  kWKBWriterEmptyPointCoords4,
                                                  sizeof(kWKBWriterEmptyPointCoords4)));
        break;
      default:
        return EINVAL;
    }
  }

  private->level--;
  return GEOARROW_OK;
}

static int feat_end_wkb(struct GeoArrowVisitor* v) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)v->private_data;

  if (private->feat_is_null) {
    if (private->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private->validity, private->length));
      ArrowBitmapAppendUnsafe(&private->validity, 1, private->length - 1);
    }

    private->null_count++;
    return ArrowBitmapAppend(&private->validity, 0, 1);
  } else if (private->validity.buffer.data != NULL) {
    return ArrowBitmapAppend(&private->validity, 1, 1);
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKBWriterInit(struct GeoArrowWKBWriter* writer) {
  struct WKBWriterPrivate* private =
      (struct WKBWriterPrivate*)ArrowMalloc(sizeof(struct WKBWriterPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  private->storage_type = NANOARROW_TYPE_BINARY;
  private->length = 0;
  private->level = 0;
  private->null_count = 0;
  ArrowBitmapInit(&private->validity);
  ArrowBufferInit(&private->offsets);
  ArrowBufferInit(&private->values);
  writer->private_data = private;

  return GEOARROW_OK;
}

void GeoArrowWKBWriterInitVisitor(struct GeoArrowWKBWriter* writer,
                                  struct GeoArrowVisitor* v) {
  GeoArrowVisitorInitVoid(v);

  v->private_data = writer->private_data;
  v->feat_start = &feat_start_wkb;
  v->null_feat = &null_feat_wkb;
  v->geom_start = &geom_start_wkb;
  v->ring_start = &ring_start_wkb;
  v->coords = &coords_wkb;
  v->ring_end = &ring_end_wkb;
  v->geom_end = &geom_end_wkb;
  v->feat_end = &feat_end_wkb;
}

GeoArrowErrorCode GeoArrowWKBWriterFinish(struct GeoArrowWKBWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)writer->private_data;
  array->release = NULL;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }

  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, private->storage_type));
  ArrowArraySetValidityBitmap(array, &private->validity);
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 1, &private->offsets));
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 2, &private->values));
  array->length = private->length;
  array->null_count = private->null_count;
  private->length = 0;
  private->null_count = 0;
  return ArrowArrayFinishBuildingDefault(array, (struct ArrowError*)error);
}

void GeoArrowWKBWriterReset(struct GeoArrowWKBWriter* writer) {
  struct WKBWriterPrivate* private = (struct WKBWriterPrivate*)writer->private_data;
  ArrowBitmapReset(&private->validity);
  ArrowBufferReset(&private->offsets);
  ArrowBufferReset(&private->values);
  ArrowFree(private);
  writer->private_data = NULL;
}
#include <errno.h>
#include <stdlib.h>
#include <string.h>





#define COORD_CACHE_SIZE_COORDS 64

struct WKTReaderPrivate {
  const char* data;
  int64_t size_bytes;
  const char* data0;
  double coords[4 * COORD_CACHE_SIZE_COORDS];
  struct GeoArrowCoordView coord_view;
};

static inline void GeoArrowWKTAdvanceUnsafe(struct WKTReaderPrivate* s, int64_t n) {
  s->data += n;
  s->size_bytes -= n;
}

static inline void GeoArrowWKTSkipWhitespace(struct WKTReaderPrivate* s) {
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      s->size_bytes--;
      s->data++;
    } else {
      break;
    }
  }
}

static inline int GeoArrowWKTSkipUntil(struct WKTReaderPrivate* s, const char* items) {
  int64_t n_items = strlen(items);
  while (s->size_bytes > 0) {
    char c = *(s->data);
    if (c == '\0') {
      return 0;
    }

    for (int64_t i = 0; i < n_items; i++) {
      if (c == items[i]) {
        return 1;
      }
    }

    s->size_bytes--;
    s->data++;
  }

  return 0;
}

static inline void GeoArrowWKTSkipUntilSep(struct WKTReaderPrivate* s) {
  GeoArrowWKTSkipUntil(s, " \n\t\r,()");
}

static inline char PeekChar(struct WKTReaderPrivate* s) {
  if (s->size_bytes > 0) {
    return s->data[0];
  } else {
    return '\0';
  }
}

static inline struct ArrowStringView GeoArrowWKTPeekUntilSep(struct WKTReaderPrivate* s,
                                                             int max_chars) {
  struct WKTReaderPrivate tmp = *s;
  if (tmp.size_bytes > max_chars) {
    tmp.size_bytes = max_chars;
  }

  GeoArrowWKTSkipUntilSep(&tmp);
  struct ArrowStringView out = {s->data, tmp.data - s->data};
  return out;
}

static inline void SetParseErrorAuto(const char* expected, struct WKTReaderPrivate* s,
                                     struct GeoArrowError* error) {
  long pos = s->data - s->data0;
  // TODO: "but found ..." from s
  GeoArrowErrorSet(error, "Expected %s at byte %ld", expected, pos);
}

static inline int GeoArrowWKTAssertChar(struct WKTReaderPrivate* s, char c,
                                        struct GeoArrowError* error) {
  if (s->size_bytes > 0 && s->data[0] == c) {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  } else {
    char expected[4] = {'\'', c, '\'', '\0'};
    SetParseErrorAuto(expected, s, error);
    return EINVAL;
  }
}

static inline int GeoArrowWKTAssertWhitespace(struct WKTReaderPrivate* s,
                                              struct GeoArrowError* error) {
  if (s->size_bytes > 0 && (s->data[0] == ' ' || s->data[0] == '\t' ||
                            s->data[0] == '\r' || s->data[0] == '\n')) {
    GeoArrowWKTSkipWhitespace(s);
    return GEOARROW_OK;
  } else {
    SetParseErrorAuto("whitespace", s, error);
    return EINVAL;
  }
}

static inline int GeoArrowWKTAssertWordEmpty(struct WKTReaderPrivate* s,
                                             struct GeoArrowError* error) {
  struct ArrowStringView word = GeoArrowWKTPeekUntilSep(s, 6);
  if (word.size_bytes == 5 && strncmp(word.data, "EMPTY", 5) == 0) {
    GeoArrowWKTAdvanceUnsafe(s, 5);
    return GEOARROW_OK;
  }

  SetParseErrorAuto("'(' or 'EMPTY'", s, error);
  return EINVAL;
}

static inline int ReadOrdinate(struct WKTReaderPrivate* s, double* out,
                               struct GeoArrowError* error) {
  const char* start = s->data;
  GeoArrowWKTSkipUntilSep(s);
  int result = GeoArrowFromChars(start, s->data, out);
  if (result != GEOARROW_OK) {
    s->size_bytes += s->data - start;
    s->data = start;
    SetParseErrorAuto("number", s, error);
  }

  return result;
}

static inline void ResetCoordCache(struct WKTReaderPrivate* s) {
  s->coord_view.n_coords = 0;
}

static inline int FlushCoordCache(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  if (s->coord_view.n_coords > 0) {
    int result = v->coords(v, &s->coord_view);
    s->coord_view.n_coords = 0;
    return result;
  } else {
    return GEOARROW_OK;
  }
}

static inline int ReadCoordinate(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  if (s->coord_view.n_coords == COORD_CACHE_SIZE_COORDS) {
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
  }

  NANOARROW_RETURN_NOT_OK(ReadOrdinate(
      s, (double*)s->coord_view.values[0] + s->coord_view.n_coords, v->error));
  for (int i = 1; i < s->coord_view.n_values; i++) {
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertWhitespace(s, v->error));
    NANOARROW_RETURN_NOT_OK(ReadOrdinate(
        s, (double*)s->coord_view.values[i] + s->coord_view.n_coords, v->error));
  }

  s->coord_view.n_coords++;
  return NANOARROW_OK;
}

static inline int ReadEmptyOrCoordinates(struct WKTReaderPrivate* s,
                                         struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    ResetCoordCache(s);

    // Read the first coordinate (there must always be one)
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the coordinates
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
      GeoArrowWKTSkipWhitespace(s);
    }

    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipointFlat(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v,
                                     enum GeoArrowDimensions dimensions) {
  NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, '(', v->error));

  // Read the first coordinate (there must always be one)
  NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
  ResetCoordCache(s);
  NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
  NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
  NANOARROW_RETURN_NOT_OK(v->geom_end(v));
  GeoArrowWKTSkipWhitespace(s);

  // Read the rest of the coordinates
  while (PeekChar(s) != ')') {
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
    ResetCoordCache(s);
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);
  }

  GeoArrowWKTAdvanceUnsafe(s, 1);
  return GEOARROW_OK;
}

static inline int ReadEmptyOrPointCoordinate(struct WKTReaderPrivate* s,
                                             struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);

    GeoArrowWKTSkipWhitespace(s);
    ResetCoordCache(s);
    NANOARROW_RETURN_NOT_OK(ReadCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(FlushCoordCache(s, v));
    GeoArrowWKTSkipWhitespace(s);
    NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ')', v->error));
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadPolygon(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first ring (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->ring_start(v));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
    NANOARROW_RETURN_NOT_OK(v->ring_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the rings
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(v->ring_start(v));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      NANOARROW_RETURN_NOT_OK(v->ring_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipoint(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v,
                                 enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Both MULTIPOINT (1 2, 2 3) and MULTIPOINT ((1 2), (2 3)) have to parse here
    // if it doesn't look like the verbose version, try the flat version
    if (PeekChar(s) != '(' && PeekChar(s) != 'E') {
      s->data--;
      s->size_bytes++;
      return ReadMultipointFlat(s, v, dimensions);
    }

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POINT, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultilinestring(struct WKTReaderPrivate* s,
                                      struct GeoArrowVisitor* v,
                                      enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(
        v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(
          v->geom_start(v, GEOARROW_GEOMETRY_TYPE_LINESTRING, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadMultipolygon(struct WKTReaderPrivate* s, struct GeoArrowVisitor* v,
                                   enum GeoArrowDimensions dimensions) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON, dimensions));
    NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
    NANOARROW_RETURN_NOT_OK(v->geom_end(v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(
          v->geom_start(v, GEOARROW_GEOMETRY_TYPE_POLYGON, dimensions));
      NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
      NANOARROW_RETURN_NOT_OK(v->geom_end(v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadTaggedGeometry(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v);

static inline int ReadGeometryCollection(struct WKTReaderPrivate* s,
                                         struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) == '(') {
    GeoArrowWKTAdvanceUnsafe(s, 1);
    GeoArrowWKTSkipWhitespace(s);

    // Read the first geometry (there must always be one)
    NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
    GeoArrowWKTSkipWhitespace(s);

    // Read the rest of the geometries
    while (PeekChar(s) != ')') {
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTAssertChar(s, ',', v->error));
      GeoArrowWKTSkipWhitespace(s);
      NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
      GeoArrowWKTSkipWhitespace(s);
    }

    GeoArrowWKTAdvanceUnsafe(s, 1);
    return GEOARROW_OK;
  }

  return GeoArrowWKTAssertWordEmpty(s, v->error);
}

static inline int ReadTaggedGeometry(struct WKTReaderPrivate* s,
                                     struct GeoArrowVisitor* v) {
  GeoArrowWKTSkipWhitespace(s);

  struct ArrowStringView word = GeoArrowWKTPeekUntilSep(s, 19);
  enum GeoArrowGeometryType geometry_type;
  if (word.size_bytes == 5 && strncmp(word.data, "POINT", 5) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_POINT;
  } else if (word.size_bytes == 10 && strncmp(word.data, "LINESTRING", 10) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_LINESTRING;
  } else if (word.size_bytes == 7 && strncmp(word.data, "POLYGON", 7) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_POLYGON;
  } else if (word.size_bytes == 10 && strncmp(word.data, "MULTIPOINT", 10) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOINT;
  } else if (word.size_bytes == 15 && strncmp(word.data, "MULTILINESTRING", 15) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTILINESTRING;
  } else if (word.size_bytes == 12 && strncmp(word.data, "MULTIPOLYGON", 12) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON;
  } else if (word.size_bytes == 18 && strncmp(word.data, "GEOMETRYCOLLECTION", 18) == 0) {
    geometry_type = GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION;
  } else {
    SetParseErrorAuto("geometry type", s, v->error);
    return EINVAL;
  }

  GeoArrowWKTAdvanceUnsafe(s, word.size_bytes);
  GeoArrowWKTSkipWhitespace(s);

  enum GeoArrowDimensions dimensions = GEOARROW_DIMENSIONS_XY;
  s->coord_view.n_values = 2;
  word = GeoArrowWKTPeekUntilSep(s, 3);
  if (word.size_bytes == 1 && strncmp(word.data, "Z", 1) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYZ;
    s->coord_view.n_values = 3;
    GeoArrowWKTAdvanceUnsafe(s, 1);
  } else if (word.size_bytes == 1 && strncmp(word.data, "M", 1) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYM;
    s->coord_view.n_values = 3;
    GeoArrowWKTAdvanceUnsafe(s, 1);
  } else if (word.size_bytes == 2 && strncmp(word.data, "ZM", 2) == 0) {
    dimensions = GEOARROW_DIMENSIONS_XYZM;
    s->coord_view.n_values = 4;
    GeoArrowWKTAdvanceUnsafe(s, 2);
  }

  NANOARROW_RETURN_NOT_OK(v->geom_start(v, geometry_type, dimensions));

  switch (geometry_type) {
    case GEOARROW_GEOMETRY_TYPE_POINT:
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrPointCoordinate(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_LINESTRING:
      NANOARROW_RETURN_NOT_OK(ReadEmptyOrCoordinates(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_POLYGON:
      NANOARROW_RETURN_NOT_OK(ReadPolygon(s, v));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
      NANOARROW_RETURN_NOT_OK(ReadMultipoint(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
      NANOARROW_RETURN_NOT_OK(ReadMultilinestring(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
      NANOARROW_RETURN_NOT_OK(ReadMultipolygon(s, v, dimensions));
      break;
    case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
      NANOARROW_RETURN_NOT_OK(ReadGeometryCollection(s, v));
      break;
    default:
      GeoArrowErrorSet(v->error, "Internal error: unrecognized geometry type id");
      return EINVAL;
  }

  return v->geom_end(v);
}

GeoArrowErrorCode GeoArrowWKTReaderInit(struct GeoArrowWKTReader* reader) {
  struct WKTReaderPrivate* s =
      (struct WKTReaderPrivate*)ArrowMalloc(sizeof(struct WKTReaderPrivate));

  if (s == NULL) {
    return ENOMEM;
  }

  s->data0 = NULL;
  s->data = NULL;
  s->size_bytes = 0;

  s->coord_view.coords_stride = 1;
  s->coord_view.values[0] = s->coords;
  for (int i = 1; i < 4; i++) {
    s->coord_view.values[i] = s->coord_view.values[i - 1] + COORD_CACHE_SIZE_COORDS;
  }

  reader->private_data = s;
  return GEOARROW_OK;
}

void GeoArrowWKTReaderReset(struct GeoArrowWKTReader* reader) {
  ArrowFree(reader->private_data);
}

GeoArrowErrorCode GeoArrowWKTReaderVisit(struct GeoArrowWKTReader* reader,
                                         struct GeoArrowStringView src,
                                         struct GeoArrowVisitor* v) {
  struct WKTReaderPrivate* s = (struct WKTReaderPrivate*)reader->private_data;
  s->data0 = src.data;
  s->data = src.data;
  s->size_bytes = src.size_bytes;

  NANOARROW_RETURN_NOT_OK(v->feat_start(v));
  NANOARROW_RETURN_NOT_OK(ReadTaggedGeometry(s, v));
  NANOARROW_RETURN_NOT_OK(v->feat_end(v));
  GeoArrowWKTSkipWhitespace(s);
  if (PeekChar(s) != '\0') {
    SetParseErrorAuto("end of input", s, v->error);
    return EINVAL;
  }

  return GEOARROW_OK;
}

#include <stdio.h>
#include <string.h>





struct WKTWriterPrivate {
  enum ArrowType storage_type;
  struct ArrowBitmap validity;
  struct ArrowBuffer offsets;
  struct ArrowBuffer values;
  enum GeoArrowGeometryType geometry_type[32];
  int64_t i[32];
  int32_t level;
  int64_t length;
  int64_t null_count;
  int64_t values_feat_start;
  int precision;
  int use_flat_multipoint;
  int64_t max_element_size_bytes;
  int feat_is_null;
};

static inline int WKTWriterCheckLevel(struct WKTWriterPrivate* private) {
  if (private->level >= 0 && private->level <= 31) {
    return GEOARROW_OK;
  } else {
    return EINVAL;
  }
}

static inline int WKTWriterWrite(struct WKTWriterPrivate* private, const char* value) {
  return ArrowBufferAppend(&private->values, value, strlen(value));
}

static inline void WKTWriterWriteDoubleUnsafe(struct WKTWriterPrivate* private,
                                              double value) {
  private->values.size_bytes +=
      GeoArrowPrintDouble(value, private->precision,
                          ((char*)private->values.data) + private->values.size_bytes);
}

static int feat_start_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level = -1;
  private->length++;
  private->feat_is_null = 0;
  private->values_feat_start = private->values.size_bytes;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  return ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes);
}

static int null_feat_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->feat_is_null = 1;
  return GEOARROW_OK;
}

static int geom_start_wkt(struct GeoArrowVisitor* v,
                          enum GeoArrowGeometryType geometry_type,
                          enum GeoArrowDimensions dimensions) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level++;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->level > 0 && private->i[private->level - 1] > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, ", "));
  } else if (private->level > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, "("));
  }

  if (private->level == 0 || private->geometry_type[private->level - 1] ==
                                 GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
    const char* geometry_type_name = GeoArrowGeometryTypeString(geometry_type);
    if (geometry_type_name == NULL) {
      GeoArrowErrorSet(v->error, "WKTWriter::geom_start(): Unexpected `geometry_type`");
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, geometry_type_name));

    switch (dimensions) {
      case GEOARROW_DIMENSIONS_XY:
        break;
      case GEOARROW_DIMENSIONS_XYZ:
        NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, " Z"));
        break;
      case GEOARROW_DIMENSIONS_XYM:
        NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, " M"));
        break;
      case GEOARROW_DIMENSIONS_XYZM:
        NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, " ZM"));
        break;
      default:
        GeoArrowErrorSet(v->error, "WKTWriter::geom_start(): Unexpected `dimensions`");
        return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, " "));
  }

  if (private->level > 0) {
    private->i[private->level - 1]++;
  }

  private->geometry_type[private->level] = geometry_type;
  private->i[private->level] = 0;
  return GEOARROW_OK;
}

static int ring_start_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  private->level++;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->level > 0 && private->i[private->level - 1] > 0) {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, ", "));
  } else {
    NANOARROW_RETURN_NOT_OK(WKTWriterWrite(private, "("));
  }

  if (private->level > 0) {
    private->i[private->level - 1]++;
  }

  private->geometry_type[private->level] = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
  private->i[private->level] = 0;
  return GEOARROW_OK;
}

static int coords_wkt(struct GeoArrowVisitor* v, const struct GeoArrowCoordView* coords) {
  int64_t n_coords = coords->n_coords;
  int32_t n_dims = coords->n_values;
  if (n_coords == 0) {
    return GEOARROW_OK;
  }

  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  int64_t max_chars_needed = (n_coords * 2) +  // space + comma after coordinate
                             (n_coords * (n_dims - 1)) +  // spaces between ordinates
                             ((private->precision + 1 + 5) * n_coords *
                              n_dims);  // significant digits + decimal + exponent
  if (private->max_element_size_bytes >= 0 &&
      max_chars_needed > private->max_element_size_bytes) {
    // Because we write a coordinate before actually checking
    max_chars_needed = private->max_element_size_bytes + 1024;
  }

  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&private->values, max_chars_needed));

  // Write the first coordinate, possibly with a leading comma if there was
  // a previous call to coords, or the opening ( if it wasn't. Special case
  // for the flat multipoint output MULTIPOINT (1 2, 3 4, ...) which doesn't
  // have extra () for inner POINTs
  if (private->i[private->level] != 0) {
    ArrowBufferAppendUnsafe(&private->values, ", ", 2);
  } else if (private->level < 1 || !private->use_flat_multipoint ||
             private->geometry_type[private->level - 1] !=
                 GEOARROW_GEOMETRY_TYPE_MULTIPOINT) {
    ArrowBufferAppendUnsafe(&private->values, "(", 1);
  }

  WKTWriterWriteDoubleUnsafe(private, coords->values[0][0]);
  for (int32_t j = 1; j < n_dims; j++) {
    ArrowBufferAppendUnsafe(&private->values, " ", 1);
    WKTWriterWriteDoubleUnsafe(private, coords->values[j][0]);
  }

  // Write the remaining coordinates (which all have leading commas)
  for (int64_t i = 1; i < n_coords; i++) {
    if (private->max_element_size_bytes >= 0 &&
        (private->values.size_bytes - private->values_feat_start) >=
            private->max_element_size_bytes) {
      return EAGAIN;
    }

    ArrowBufferAppendUnsafe(&private->values, ", ", 2);
    WKTWriterWriteDoubleUnsafe(private, coords->values[0][i * coords->coords_stride]);
    for (int32_t j = 1; j < n_dims; j++) {
      ArrowBufferAppendUnsafe(&private->values, " ", 1);
      WKTWriterWriteDoubleUnsafe(private, coords->values[j][i * coords->coords_stride]);
    }
  }

  private->i[private->level] += n_coords;
  return GEOARROW_OK;
}

static int ring_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));
  if (private->i[private->level] == 0) {
    private->level--;
    return WKTWriterWrite(private, "EMPTY");
  } else {
    private->level--;
    return WKTWriterWrite(private, ")");
  }
}

static int geom_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;
  NANOARROW_RETURN_NOT_OK(WKTWriterCheckLevel(private));

  if (private->i[private->level] == 0) {
    private->level--;
    return WKTWriterWrite(private, "EMPTY");
  } else if (private->level < 1 || !private->use_flat_multipoint ||
             private->geometry_type[private->level - 1] !=
                 GEOARROW_GEOMETRY_TYPE_MULTIPOINT) {
    private->level--;
    return WKTWriterWrite(private, ")");
  } else {
    private->level--;
    return GEOARROW_OK;
  }
}

static int feat_end_wkt(struct GeoArrowVisitor* v) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)v->private_data;

  if (private->feat_is_null) {
    if (private->validity.buffer.data == NULL) {
      NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private->validity, private->length));
      ArrowBitmapAppendUnsafe(&private->validity, 1, private->length - 1);
    }

    private->null_count++;
    return ArrowBitmapAppend(&private->validity, 0, 1);
  } else if (private->validity.buffer.data != NULL) {
    return ArrowBitmapAppend(&private->validity, 1, 1);
  }

  if (private->max_element_size_bytes >= 0 &&
      (private->values.size_bytes - private->values_feat_start) >
          private->max_element_size_bytes) {
    private->values.size_bytes =
        private->values_feat_start + private->max_element_size_bytes;
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowWKTWriterInit(struct GeoArrowWKTWriter* writer) {
  struct WKTWriterPrivate* private =
      (struct WKTWriterPrivate*)ArrowMalloc(sizeof(struct WKTWriterPrivate));
  if (private == NULL) {
    return ENOMEM;
  }

  private->storage_type = NANOARROW_TYPE_STRING;
  private->length = 0;
  private->level = 0;
  private->null_count = 0;
  ArrowBitmapInit(&private->validity);
  ArrowBufferInit(&private->offsets);
  ArrowBufferInit(&private->values);
  writer->precision = 16;
  private->precision = 16;
  writer->use_flat_multipoint = 1;
  private->use_flat_multipoint = 1;
  writer->max_element_size_bytes = -1;
  private->max_element_size_bytes = -1;
  writer->private_data = private;

  return GEOARROW_OK;
}

void GeoArrowWKTWriterInitVisitor(struct GeoArrowWKTWriter* writer,
                                  struct GeoArrowVisitor* v) {
  GeoArrowVisitorInitVoid(v);

  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;
  private->precision = writer->precision;
  private->use_flat_multipoint = writer->use_flat_multipoint;
  private->max_element_size_bytes = writer->max_element_size_bytes;

  v->private_data = writer->private_data;
  v->feat_start = &feat_start_wkt;
  v->null_feat = &null_feat_wkt;
  v->geom_start = &geom_start_wkt;
  v->ring_start = &ring_start_wkt;
  v->coords = &coords_wkt;
  v->ring_end = &ring_end_wkt;
  v->geom_end = &geom_end_wkt;
  v->feat_end = &feat_end_wkt;
}

GeoArrowErrorCode GeoArrowWKTWriterFinish(struct GeoArrowWKTWriter* writer,
                                          struct ArrowArray* array,
                                          struct GeoArrowError* error) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;
  array->release = NULL;

  if (private->values.size_bytes > 2147483647) {
    return EOVERFLOW;
  }
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(&private->offsets, (int32_t) private->values.size_bytes));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, private->storage_type));
  ArrowArraySetValidityBitmap(array, &private->validity);
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 1, &private->offsets));
  NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(array, 2, &private->values));
  array->length = private->length;
  array->null_count = private->null_count;
  private->length = 0;
  private->null_count = 0;
  return ArrowArrayFinishBuildingDefault(array, (struct ArrowError*)error);
}

void GeoArrowWKTWriterReset(struct GeoArrowWKTWriter* writer) {
  struct WKTWriterPrivate* private = (struct WKTWriterPrivate*)writer->private_data;
  ArrowBitmapReset(&private->validity);
  ArrowBufferReset(&private->offsets);
  ArrowBufferReset(&private->values);
  ArrowFree(private);
  writer->private_data = NULL;
}





struct GeoArrowArrayReaderPrivate {
  struct GeoArrowWKTReader wkt_reader;
  struct GeoArrowWKBReader wkb_reader;
};

static GeoArrowErrorCode GeoArrowArrayViewVisitWKT(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKTReader* reader, struct GeoArrowVisitor* v) {
  struct GeoArrowStringView item;
  const int32_t* offset_begin = array_view->offsets[0] + array_view->offset[0] + offset;

  for (int64_t i = 0; i < length; i++) {
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      item.data = (const char*)(array_view->data + offset_begin[i]);
      item.size_bytes = offset_begin[i + 1] - offset_begin[i];
      NANOARROW_RETURN_NOT_OK(GeoArrowWKTReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

static GeoArrowErrorCode GeoArrowArrayViewVisitWKB(
    const struct GeoArrowArrayView* array_view, int64_t offset, int64_t length,
    struct GeoArrowWKBReader* reader, struct GeoArrowVisitor* v) {
  struct GeoArrowBufferView item;
  const int32_t* offset_begin = array_view->offsets[0] + array_view->offset[0] + offset;

  for (int64_t i = 0; i < length; i++) {
    if (!array_view->validity_bitmap ||
        ArrowBitGet(array_view->validity_bitmap, array_view->offset[0] + offset + i)) {
      item.data = array_view->data + offset_begin[i];
      item.size_bytes = offset_begin[i + 1] - offset_begin[i];
      NANOARROW_RETURN_NOT_OK(GeoArrowWKBReaderVisit(reader, item, v));
    } else {
      NANOARROW_RETURN_NOT_OK(v->feat_start(v));
      NANOARROW_RETURN_NOT_OK(v->null_feat(v));
      NANOARROW_RETURN_NOT_OK(v->feat_end(v));
    }
  }

  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayReaderInit(struct GeoArrowArrayReader* reader) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowArrayReaderPrivate));

  if (private_data == NULL) {
    return ENOMEM;
  }

  int result = GeoArrowWKTReaderInit(&private_data->wkt_reader);
  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  result = GeoArrowWKBReaderInit(&private_data->wkb_reader);
  if (result != GEOARROW_OK) {
    GeoArrowWKTReaderReset(&private_data->wkt_reader);
    ArrowFree(private_data);
    return result;
  }

  reader->private_data = private_data;
  return GEOARROW_OK;
}

void GeoArrowArrayReaderReset(struct GeoArrowArrayReader* reader) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;
  GeoArrowWKBReaderReset(&private_data->wkb_reader);
  GeoArrowWKTReaderReset(&private_data->wkt_reader);
  ArrowFree(reader->private_data);
}

GeoArrowErrorCode GeoArrowArrayReaderVisit(struct GeoArrowArrayReader* reader,
                                           const struct GeoArrowArrayView* array_view,
                                           int64_t offset, int64_t length,
                                           struct GeoArrowVisitor* v) {
  struct GeoArrowArrayReaderPrivate* private_data =
      (struct GeoArrowArrayReaderPrivate*)reader->private_data;

  switch (array_view->schema_view.type) {
    case GEOARROW_TYPE_WKT:
      return GeoArrowArrayViewVisitWKT(array_view, offset, length,
                                       &private_data->wkt_reader, v);
    case GEOARROW_TYPE_WKB:
      return GeoArrowArrayViewVisitWKB(array_view, offset, length,
                                       &private_data->wkb_reader, v);
    default:
      return GeoArrowArrayViewVisit(array_view, offset, length, v);
  }
}

#include <string.h>





struct GeoArrowArrayWriterPrivate {
  struct GeoArrowWKTWriter wkt_writer;
  struct GeoArrowWKBWriter wkb_writer;
  struct GeoArrowBuilder builder;
  enum GeoArrowType type;
};

GeoArrowErrorCode GeoArrowArrayWriterInitFromType(struct GeoArrowArrayWriter* writer,
                                                  enum GeoArrowType type) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)ArrowMalloc(
          sizeof(struct GeoArrowArrayWriterPrivate));

  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct GeoArrowArrayWriterPrivate));

  int result;
  switch (type) {
    case GEOARROW_TYPE_LARGE_WKT:
    case GEOARROW_TYPE_LARGE_WKB:
      return ENOTSUP;
    case GEOARROW_TYPE_WKT:
      result = GeoArrowWKTWriterInit(&private_data->wkt_writer);
      break;
    case GEOARROW_TYPE_WKB:
      result = GeoArrowWKBWriterInit(&private_data->wkb_writer);
      break;
    default:
      result = GeoArrowBuilderInitFromType(&private_data->builder, type);
      break;
  }

  if (result != GEOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  private_data->type = type;
  writer->private_data = private_data;
  return GEOARROW_OK;
}

GeoArrowErrorCode GeoArrowArrayWriterInitFromSchema(struct GeoArrowArrayWriter* writer,
                                                    const struct ArrowSchema* schema) {
  struct GeoArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(GeoArrowSchemaViewInit(&schema_view, schema, NULL));
  return GeoArrowArrayWriterInitFromType(writer, schema_view.type);
}

GeoArrowErrorCode GeoArrowArrayWriterInitVisitor(struct GeoArrowArrayWriter* writer,
                                                 struct GeoArrowVisitor* v) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  switch (private_data->type) {
    case GEOARROW_TYPE_WKT:
      GeoArrowWKTWriterInitVisitor(&private_data->wkt_writer, v);
      return GEOARROW_OK;
    case GEOARROW_TYPE_WKB:
      GeoArrowWKBWriterInitVisitor(&private_data->wkb_writer, v);
      return GEOARROW_OK;
    default:
      return GeoArrowBuilderInitVisitor(&private_data->builder, v);
  }
}

GeoArrowErrorCode GeoArrowArrayWriterFinish(struct GeoArrowArrayWriter* writer,
                                            struct ArrowArray* array,
                                            struct GeoArrowError* error) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  switch (private_data->type) {
    case GEOARROW_TYPE_WKT:
      return GeoArrowWKTWriterFinish(&private_data->wkt_writer, array, error);
    case GEOARROW_TYPE_WKB:
      return GeoArrowWKBWriterFinish(&private_data->wkb_writer, array, error);
    default:
      return GeoArrowBuilderFinish(&private_data->builder, array, error);
  }
}

void GeoArrowArrayWriterReset(struct GeoArrowArrayWriter* writer) {
  struct GeoArrowArrayWriterPrivate* private_data =
      (struct GeoArrowArrayWriterPrivate*)writer->private_data;

  if (private_data->wkt_writer.private_data != NULL) {
    GeoArrowWKTWriterReset(&private_data->wkt_writer);
  }

  if (private_data->wkb_writer.private_data != NULL) {
    GeoArrowWKBWriterReset(&private_data->wkb_writer);
  }

  if (private_data->builder.private_data != NULL) {
    GeoArrowBuilderReset(&private_data->builder);
  }

  ArrowFree(private_data);
}

#include <errno.h>
#include <stdlib.h>
#include <string.h>



#if !defined(GEOARROW_USE_FAST_FLOAT) || !GEOARROW_USE_FAST_FLOAT

GeoArrowErrorCode GeoArrowFromChars(const char* first, const char* last, double* out) {
  if (first == last) {
    return EINVAL;
  }

  int64_t size_bytes = last - first;

  // There is no guarantee that src.data is null-terminated. The maximum size of
  // a double is 24 characters, but if we can't fit all of src for some reason, error.
  char src_copy[64];
  if (size_bytes >= ((int64_t)sizeof(src_copy))) {
    return EINVAL;
  }

  memcpy(src_copy, first, size_bytes);
  char* last_copy = src_copy + size_bytes;
  *last_copy = '\0';

  char* end_ptr;
  double result = strtod(src_copy, &end_ptr);
  if (end_ptr != last_copy) {
    return EINVAL;
  } else {
    *out = result;
    return GEOARROW_OK;
  }
}

#endif



#if defined(GEOARROW_USE_RYU) && GEOARROW_USE_RYU

#include "ryu/ryu.h"

int64_t GeoArrowPrintDouble(double f, uint32_t precision, char* result) {
  return GeoArrowd2sfixed_buffered_n(f, precision, result);
}

#else

#include <stdio.h>

int64_t GeoArrowPrintDouble(double f, uint32_t precision, char* result) {
  int64_t n_chars = snprintf(result, 128, "%0.*f", precision, f);

  // Strip trailing zeroes + decimal
  for (int64_t i = n_chars - 1; i >= 0; i--) {
    if (result[i] == '0') {
      n_chars--;
    } else if (result[i] == '.') {
      n_chars--;
      break;
    } else {
      break;
    }
  }

  return n_chars;
}

#endif
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



const char* ArrowNanoarrowVersion(void) { return NANOARROW_VERSION; }

int ArrowNanoarrowVersionInt(void) { return NANOARROW_VERSION_INT; }

int ArrowErrorSet(struct ArrowError* error, const char* fmt, ...) {
  if (error == NULL) {
    return NANOARROW_OK;
  }

  memset(error->message, 0, sizeof(error->message));

  va_list args;
  va_start(args, fmt);
  int chars_needed = vsnprintf(error->message, sizeof(error->message), fmt, args);
  va_end(args);

  if (chars_needed < 0) {
    return EINVAL;
  } else if (((size_t)chars_needed) >= sizeof(error->message)) {
    return ERANGE;
  } else {
    return NANOARROW_OK;
  }
}

const char* ArrowErrorMessage(struct ArrowError* error) {
  if (error == NULL) {
    return "";
  } else {
    return error->message;
  }
}

void ArrowLayoutInit(struct ArrowLayout* layout, enum ArrowType storage_type) {
  layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_VALIDITY;
  layout->buffer_data_type[0] = NANOARROW_TYPE_BOOL;
  layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA;
  layout->buffer_data_type[1] = storage_type;
  layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_NONE;
  layout->buffer_data_type[2] = NANOARROW_TYPE_UNINITIALIZED;

  layout->element_size_bits[0] = 1;
  layout->element_size_bits[1] = 0;
  layout->element_size_bits[2] = 0;

  layout->child_size_elements = 0;

  switch (storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
    case NANOARROW_TYPE_NA:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[0] = NANOARROW_TYPE_UNINITIALIZED;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      layout->element_size_bits[0] = 0;
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_LARGE_LIST:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      break;

    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      break;

    case NANOARROW_TYPE_BOOL:
      layout->element_size_bits[1] = 1;
      break;

    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
      layout->element_size_bits[1] = 8;
      break;

    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_HALF_FLOAT:
      layout->element_size_bits[1] = 16;
      break;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_FLOAT:
      layout->element_size_bits[1] = 32;
      break;
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      layout->element_size_bits[1] = 64;
      break;

    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      layout->element_size_bits[1] = 128;
      break;

    case NANOARROW_TYPE_DECIMAL256:
      layout->element_size_bits[1] = 256;
      break;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      layout->buffer_data_type[1] = NANOARROW_TYPE_BINARY;
      break;

    case NANOARROW_TYPE_DENSE_UNION:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_TYPE_ID;
      layout->buffer_data_type[0] = NANOARROW_TYPE_INT8;
      layout->element_size_bits[0] = 8;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_UNION_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_SPARSE_UNION:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_TYPE_ID;
      layout->buffer_data_type[0] = NANOARROW_TYPE_INT8;
      layout->element_size_bits[0] = 8;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      break;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = storage_type;
      break;

    case NANOARROW_TYPE_LARGE_STRING:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = NANOARROW_TYPE_STRING;
      break;
    case NANOARROW_TYPE_LARGE_BINARY:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = NANOARROW_TYPE_BINARY;
      break;

    default:
      break;
  }
}

void* ArrowMalloc(int64_t size) { return malloc(size); }

void* ArrowRealloc(void* ptr, int64_t size) { return realloc(ptr, size); }

void ArrowFree(void* ptr) { free(ptr); }

static uint8_t* ArrowBufferAllocatorMallocReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  return (uint8_t*)ArrowRealloc(ptr, new_size);
}

static void ArrowBufferAllocatorMallocFree(struct ArrowBufferAllocator* allocator,
                                           uint8_t* ptr, int64_t size) {
  ArrowFree(ptr);
}

static struct ArrowBufferAllocator ArrowBufferAllocatorMalloc = {
    &ArrowBufferAllocatorMallocReallocate, &ArrowBufferAllocatorMallocFree, NULL};

struct ArrowBufferAllocator ArrowBufferAllocatorDefault(void) {
  return ArrowBufferAllocatorMalloc;
}

static uint8_t* ArrowBufferAllocatorNeverReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  return NULL;
}

struct ArrowBufferAllocator ArrowBufferDeallocator(
    void (*custom_free)(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                        int64_t size),
    void* private_data) {
  struct ArrowBufferAllocator allocator;
  allocator.reallocate = &ArrowBufferAllocatorNeverReallocate;
  allocator.free = custom_free;
  allocator.private_data = private_data;
  return allocator;
}
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



static void ArrowSchemaRelease(struct ArrowSchema* schema) {
  if (schema->format != NULL) ArrowFree((void*)schema->format);
  if (schema->name != NULL) ArrowFree((void*)schema->name);
  if (schema->metadata != NULL) ArrowFree((void*)schema->metadata);

  // This object owns the memory for all the children, but those
  // children may have been generated elsewhere and might have
  // their own release() callback.
  if (schema->children != NULL) {
    for (int64_t i = 0; i < schema->n_children; i++) {
      if (schema->children[i] != NULL) {
        if (schema->children[i]->release != NULL) {
          schema->children[i]->release(schema->children[i]);
        }

        ArrowFree(schema->children[i]);
      }
    }

    ArrowFree(schema->children);
  }

  // This object owns the memory for the dictionary but it
  // may have been generated somewhere else and have its own
  // release() callback.
  if (schema->dictionary != NULL) {
    if (schema->dictionary->release != NULL) {
      schema->dictionary->release(schema->dictionary);
    }

    ArrowFree(schema->dictionary);
  }

  // private data not currently used
  if (schema->private_data != NULL) {
    ArrowFree(schema->private_data);
  }

  schema->release = NULL;
}

static const char* ArrowSchemaFormatTemplate(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_UNINITIALIZED:
      return NULL;
    case NANOARROW_TYPE_NA:
      return "n";
    case NANOARROW_TYPE_BOOL:
      return "b";

    case NANOARROW_TYPE_UINT8:
      return "C";
    case NANOARROW_TYPE_INT8:
      return "c";
    case NANOARROW_TYPE_UINT16:
      return "S";
    case NANOARROW_TYPE_INT16:
      return "s";
    case NANOARROW_TYPE_UINT32:
      return "I";
    case NANOARROW_TYPE_INT32:
      return "i";
    case NANOARROW_TYPE_UINT64:
      return "L";
    case NANOARROW_TYPE_INT64:
      return "l";

    case NANOARROW_TYPE_HALF_FLOAT:
      return "e";
    case NANOARROW_TYPE_FLOAT:
      return "f";
    case NANOARROW_TYPE_DOUBLE:
      return "g";

    case NANOARROW_TYPE_STRING:
      return "u";
    case NANOARROW_TYPE_LARGE_STRING:
      return "U";
    case NANOARROW_TYPE_BINARY:
      return "z";
    case NANOARROW_TYPE_LARGE_BINARY:
      return "Z";

    case NANOARROW_TYPE_DATE32:
      return "tdD";
    case NANOARROW_TYPE_DATE64:
      return "tdm";
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      return "tiM";
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      return "tiD";
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      return "tin";

    case NANOARROW_TYPE_LIST:
      return "+l";
    case NANOARROW_TYPE_LARGE_LIST:
      return "+L";
    case NANOARROW_TYPE_STRUCT:
      return "+s";
    case NANOARROW_TYPE_MAP:
      return "+m";

    default:
      return NULL;
  }
}

static int ArrowSchemaInitChildrenIfNeeded(struct ArrowSchema* schema,
                                           enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, 1));
      ArrowSchemaInit(schema->children[0]);
      NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], "item"));
      break;
    case NANOARROW_TYPE_MAP:
      NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, 1));
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_STRUCT));
      NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], "entries"));
      schema->children[0]->flags &= ~ARROW_FLAG_NULLABLE;
      NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema->children[0], 2));
      ArrowSchemaInit(schema->children[0]->children[0]);
      ArrowSchemaInit(schema->children[0]->children[1]);
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaSetName(schema->children[0]->children[0], "key"));
      schema->children[0]->children[0]->flags &= ~ARROW_FLAG_NULLABLE;
      NANOARROW_RETURN_NOT_OK(
          ArrowSchemaSetName(schema->children[0]->children[1], "value"));
      break;
    default:
      break;
  }

  return NANOARROW_OK;
}

void ArrowSchemaInit(struct ArrowSchema* schema) {
  schema->format = NULL;
  schema->name = NULL;
  schema->metadata = NULL;
  schema->flags = ARROW_FLAG_NULLABLE;
  schema->n_children = 0;
  schema->children = NULL;
  schema->dictionary = NULL;
  schema->private_data = NULL;
  schema->release = &ArrowSchemaRelease;
}

ArrowErrorCode ArrowSchemaSetType(struct ArrowSchema* schema, enum ArrowType type) {
  // We don't allocate the dictionary because it has to be nullptr
  // for non-dictionary-encoded arrays.

  // Set the format to a valid format string for type
  const char* template_format = ArrowSchemaFormatTemplate(type);

  // If type isn't recognized and not explicitly unset
  if (template_format == NULL && type != NANOARROW_TYPE_UNINITIALIZED) {
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetFormat(schema, template_format));

  // For types with an umabiguous child structure, allocate children
  return ArrowSchemaInitChildrenIfNeeded(schema, type);
}

ArrowErrorCode ArrowSchemaSetTypeStruct(struct ArrowSchema* schema, int64_t n_children) {
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_STRUCT));
  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, n_children));
  for (int64_t i = 0; i < n_children; i++) {
    ArrowSchemaInit(schema->children[i]);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaInitFromType(struct ArrowSchema* schema, enum ArrowType type) {
  ArrowSchemaInit(schema);

  int result = ArrowSchemaSetType(schema, type);
  if (result != NANOARROW_OK) {
    schema->release(schema);
    return result;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaSetTypeFixedSize(struct ArrowSchema* schema,
                                           enum ArrowType type, int32_t fixed_size) {
  if (fixed_size <= 0) {
    return EINVAL;
  }

  char buffer[64];
  int n_chars;
  switch (type) {
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      n_chars = snprintf(buffer, sizeof(buffer), "w:%d", (int)fixed_size);
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      n_chars = snprintf(buffer, sizeof(buffer), "+w:%d", (int)fixed_size);
      break;
    default:
      return EINVAL;
  }

  buffer[n_chars] = '\0';
  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetFormat(schema, buffer));

  if (type == NANOARROW_TYPE_FIXED_SIZE_LIST) {
    NANOARROW_RETURN_NOT_OK(ArrowSchemaInitChildrenIfNeeded(schema, type));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaSetTypeDecimal(struct ArrowSchema* schema, enum ArrowType type,
                                         int32_t decimal_precision,
                                         int32_t decimal_scale) {
  if (decimal_precision <= 0) {
    return EINVAL;
  }

  char buffer[64];
  int n_chars;
  switch (type) {
    case NANOARROW_TYPE_DECIMAL128:
      n_chars =
          snprintf(buffer, sizeof(buffer), "d:%d,%d", decimal_precision, decimal_scale);
      break;
    case NANOARROW_TYPE_DECIMAL256:
      n_chars = snprintf(buffer, sizeof(buffer), "d:%d,%d,256", decimal_precision,
                         decimal_scale);
      break;
    default:
      return EINVAL;
  }

  buffer[n_chars] = '\0';
  return ArrowSchemaSetFormat(schema, buffer);
}

static const char* ArrowTimeUnitFormatString(enum ArrowTimeUnit time_unit) {
  switch (time_unit) {
    case NANOARROW_TIME_UNIT_SECOND:
      return "s";
    case NANOARROW_TIME_UNIT_MILLI:
      return "m";
    case NANOARROW_TIME_UNIT_MICRO:
      return "u";
    case NANOARROW_TIME_UNIT_NANO:
      return "n";
    default:
      return NULL;
  }
}

ArrowErrorCode ArrowSchemaSetTypeDateTime(struct ArrowSchema* schema, enum ArrowType type,
                                          enum ArrowTimeUnit time_unit,
                                          const char* timezone) {
  const char* time_unit_str = ArrowTimeUnitFormatString(time_unit);
  if (time_unit_str == NULL) {
    return EINVAL;
  }

  char buffer[128];
  int n_chars;
  switch (type) {
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_TIME64:
      if (timezone != NULL) {
        return EINVAL;
      }
      n_chars = snprintf(buffer, sizeof(buffer), "tt%s", time_unit_str);
      break;
    case NANOARROW_TYPE_TIMESTAMP:
      if (timezone == NULL) {
        timezone = "";
      }
      n_chars = snprintf(buffer, sizeof(buffer), "ts%s:%s", time_unit_str, timezone);
      break;
    case NANOARROW_TYPE_DURATION:
      if (timezone != NULL) {
        return EINVAL;
      }
      n_chars = snprintf(buffer, sizeof(buffer), "tD%s", time_unit_str);
      break;
    default:
      return EINVAL;
  }

  if (((size_t)n_chars) >= sizeof(buffer)) {
    return ERANGE;
  }

  buffer[n_chars] = '\0';

  return ArrowSchemaSetFormat(schema, buffer);
}

ArrowErrorCode ArrowSchemaSetTypeUnion(struct ArrowSchema* schema, enum ArrowType type,
                                       int64_t n_children) {
  if (n_children < 0 || n_children > 127) {
    return EINVAL;
  }

  // Max valid size would be +ud:0,1,...126 = 401 characters + null terminator
  char format_out[512];
  int64_t format_out_size = 512;
  memset(format_out, 0, format_out_size);
  int n_chars;
  char* format_cursor = format_out;

  switch (type) {
    case NANOARROW_TYPE_SPARSE_UNION:
      n_chars = snprintf(format_cursor, format_out_size, "+us:");
      format_cursor += n_chars;
      format_out_size -= n_chars;
      break;
    case NANOARROW_TYPE_DENSE_UNION:
      n_chars = snprintf(format_cursor, format_out_size, "+ud:");
      format_cursor += n_chars;
      format_out_size -= n_chars;
      break;
    default:
      return EINVAL;
  }

  if (n_children > 0) {
    n_chars = snprintf(format_cursor, format_out_size, "0");
    format_cursor += n_chars;
    format_out_size -= n_chars;

    for (int64_t i = 1; i < n_children; i++) {
      n_chars = snprintf(format_cursor, format_out_size, ",%d", (int)i);
      format_cursor += n_chars;
      format_out_size -= n_chars;
    }
  }

  NANOARROW_RETURN_NOT_OK(ArrowSchemaSetFormat(schema, format_out));

  NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateChildren(schema, n_children));
  for (int64_t i = 0; i < n_children; i++) {
    ArrowSchemaInit(schema->children[i]);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaSetFormat(struct ArrowSchema* schema, const char* format) {
  if (schema->format != NULL) {
    ArrowFree((void*)schema->format);
  }

  if (format != NULL) {
    size_t format_size = strlen(format) + 1;
    schema->format = (const char*)ArrowMalloc(format_size);
    if (schema->format == NULL) {
      return ENOMEM;
    }

    memcpy((void*)schema->format, format, format_size);
  } else {
    schema->format = NULL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaSetName(struct ArrowSchema* schema, const char* name) {
  if (schema->name != NULL) {
    ArrowFree((void*)schema->name);
  }

  if (name != NULL) {
    size_t name_size = strlen(name) + 1;
    schema->name = (const char*)ArrowMalloc(name_size);
    if (schema->name == NULL) {
      return ENOMEM;
    }

    memcpy((void*)schema->name, name, name_size);
  } else {
    schema->name = NULL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaSetMetadata(struct ArrowSchema* schema, const char* metadata) {
  if (schema->metadata != NULL) {
    ArrowFree((void*)schema->metadata);
  }

  if (metadata != NULL) {
    size_t metadata_size = ArrowMetadataSizeOf(metadata);
    schema->metadata = (const char*)ArrowMalloc(metadata_size);
    if (schema->metadata == NULL) {
      return ENOMEM;
    }

    memcpy((void*)schema->metadata, metadata, metadata_size);
  } else {
    schema->metadata = NULL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaAllocateChildren(struct ArrowSchema* schema,
                                           int64_t n_children) {
  if (schema->children != NULL) {
    return EEXIST;
  }

  if (n_children > 0) {
    schema->children =
        (struct ArrowSchema**)ArrowMalloc(n_children * sizeof(struct ArrowSchema*));

    if (schema->children == NULL) {
      return ENOMEM;
    }

    schema->n_children = n_children;

    memset(schema->children, 0, n_children * sizeof(struct ArrowSchema*));

    for (int64_t i = 0; i < n_children; i++) {
      schema->children[i] = (struct ArrowSchema*)ArrowMalloc(sizeof(struct ArrowSchema));

      if (schema->children[i] == NULL) {
        return ENOMEM;
      }

      schema->children[i]->release = NULL;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaAllocateDictionary(struct ArrowSchema* schema) {
  if (schema->dictionary != NULL) {
    return EEXIST;
  }

  schema->dictionary = (struct ArrowSchema*)ArrowMalloc(sizeof(struct ArrowSchema));
  if (schema->dictionary == NULL) {
    return ENOMEM;
  }

  schema->dictionary->release = NULL;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaDeepCopy(const struct ArrowSchema* schema,
                                   struct ArrowSchema* schema_out) {
  ArrowSchemaInit(schema_out);

  int result = ArrowSchemaSetFormat(schema_out, schema->format);
  if (result != NANOARROW_OK) {
    schema_out->release(schema_out);
    return result;
  }

  schema_out->flags = schema->flags;

  result = ArrowSchemaSetName(schema_out, schema->name);
  if (result != NANOARROW_OK) {
    schema_out->release(schema_out);
    return result;
  }

  result = ArrowSchemaSetMetadata(schema_out, schema->metadata);
  if (result != NANOARROW_OK) {
    schema_out->release(schema_out);
    return result;
  }

  result = ArrowSchemaAllocateChildren(schema_out, schema->n_children);
  if (result != NANOARROW_OK) {
    schema_out->release(schema_out);
    return result;
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    result = ArrowSchemaDeepCopy(schema->children[i], schema_out->children[i]);
    if (result != NANOARROW_OK) {
      schema_out->release(schema_out);
      return result;
    }
  }

  if (schema->dictionary != NULL) {
    result = ArrowSchemaAllocateDictionary(schema_out);
    if (result != NANOARROW_OK) {
      schema_out->release(schema_out);
      return result;
    }

    result = ArrowSchemaDeepCopy(schema->dictionary, schema_out->dictionary);
    if (result != NANOARROW_OK) {
      schema_out->release(schema_out);
      return result;
    }
  }

  return NANOARROW_OK;
}

static void ArrowSchemaViewSetPrimitive(struct ArrowSchemaView* schema_view,
                                        enum ArrowType type) {
  schema_view->type = type;
  schema_view->storage_type = type;
}

static ArrowErrorCode ArrowSchemaViewParse(struct ArrowSchemaView* schema_view,
                                           const char* format,
                                           const char** format_end_out,
                                           struct ArrowError* error) {
  *format_end_out = format;

  // needed for decimal parsing
  const char* parse_start;
  char* parse_end;

  switch (format[0]) {
    case 'n':
      schema_view->type = NANOARROW_TYPE_NA;
      schema_view->storage_type = NANOARROW_TYPE_NA;
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'b':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_BOOL);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'c':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT8);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'C':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_UINT8);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 's':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT16);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'S':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_UINT16);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'i':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT32);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'I':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_UINT32);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'l':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'L':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_UINT64);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'e':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_HALF_FLOAT);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'f':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_FLOAT);
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'g':
      ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_DOUBLE);
      *format_end_out = format + 1;
      return NANOARROW_OK;

    // decimal
    case 'd':
      if (format[1] != ':' || format[2] == '\0') {
        ArrowErrorSet(error, "Expected ':precision,scale[,bitwidth]' following 'd'",
                      format + 3);
        return EINVAL;
      }

      parse_start = format + 2;
      schema_view->decimal_precision = (int32_t)strtol(parse_start, &parse_end, 10);
      if (parse_end == parse_start || parse_end[0] != ',') {
        ArrowErrorSet(error, "Expected 'precision,scale[,bitwidth]' following 'd:'");
        return EINVAL;
      }

      parse_start = parse_end + 1;
      schema_view->decimal_scale = (int32_t)strtol(parse_start, &parse_end, 10);
      if (parse_end == parse_start) {
        ArrowErrorSet(error, "Expected 'scale[,bitwidth]' following 'd:precision,'");
        return EINVAL;
      } else if (parse_end[0] != ',') {
        schema_view->decimal_bitwidth = 128;
      } else {
        parse_start = parse_end + 1;
        schema_view->decimal_bitwidth = (int32_t)strtol(parse_start, &parse_end, 10);
        if (parse_start == parse_end) {
          ArrowErrorSet(error, "Expected precision following 'd:precision,scale,'");
          return EINVAL;
        }
      }

      *format_end_out = parse_end;

      switch (schema_view->decimal_bitwidth) {
        case 128:
          ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_DECIMAL128);
          return NANOARROW_OK;
        case 256:
          ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_DECIMAL256);
          return NANOARROW_OK;
        default:
          ArrowErrorSet(error, "Expected decimal bitwidth of 128 or 256 but found %d",
                        (int)schema_view->decimal_bitwidth);
          return EINVAL;
      }

    // validity + data
    case 'w':
      schema_view->type = NANOARROW_TYPE_FIXED_SIZE_BINARY;
      schema_view->storage_type = NANOARROW_TYPE_FIXED_SIZE_BINARY;
      if (format[1] != ':' || format[2] == '\0') {
        ArrowErrorSet(error, "Expected ':<width>' following 'w'");
        return EINVAL;
      }

      schema_view->fixed_size = (int32_t)strtol(format + 2, (char**)format_end_out, 10);
      return NANOARROW_OK;

    // validity + offset + data
    case 'z':
      schema_view->type = NANOARROW_TYPE_BINARY;
      schema_view->storage_type = NANOARROW_TYPE_BINARY;
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'u':
      schema_view->type = NANOARROW_TYPE_STRING;
      schema_view->storage_type = NANOARROW_TYPE_STRING;
      *format_end_out = format + 1;
      return NANOARROW_OK;

    // validity + large_offset + data
    case 'Z':
      schema_view->type = NANOARROW_TYPE_LARGE_BINARY;
      schema_view->storage_type = NANOARROW_TYPE_LARGE_BINARY;
      *format_end_out = format + 1;
      return NANOARROW_OK;
    case 'U':
      schema_view->type = NANOARROW_TYPE_LARGE_STRING;
      schema_view->storage_type = NANOARROW_TYPE_LARGE_STRING;
      *format_end_out = format + 1;
      return NANOARROW_OK;

    // nested types
    case '+':
      switch (format[1]) {
        // list has validity + offset or offset
        case 'l':
          schema_view->storage_type = NANOARROW_TYPE_LIST;
          schema_view->type = NANOARROW_TYPE_LIST;
          *format_end_out = format + 2;
          return NANOARROW_OK;

        // large list has validity + large_offset or large_offset
        case 'L':
          schema_view->storage_type = NANOARROW_TYPE_LARGE_LIST;
          schema_view->type = NANOARROW_TYPE_LARGE_LIST;
          *format_end_out = format + 2;
          return NANOARROW_OK;

        // just validity buffer
        case 'w':
          if (format[2] != ':' || format[3] == '\0') {
            ArrowErrorSet(error, "Expected ':<width>' following '+w'");
            return EINVAL;
          }

          schema_view->storage_type = NANOARROW_TYPE_FIXED_SIZE_LIST;
          schema_view->type = NANOARROW_TYPE_FIXED_SIZE_LIST;
          schema_view->fixed_size =
              (int32_t)strtol(format + 3, (char**)format_end_out, 10);
          return NANOARROW_OK;
        case 's':
          schema_view->storage_type = NANOARROW_TYPE_STRUCT;
          schema_view->type = NANOARROW_TYPE_STRUCT;
          *format_end_out = format + 2;
          return NANOARROW_OK;
        case 'm':
          schema_view->storage_type = NANOARROW_TYPE_MAP;
          schema_view->type = NANOARROW_TYPE_MAP;
          *format_end_out = format + 2;
          return NANOARROW_OK;

        // unions
        case 'u':
          switch (format[2]) {
            case 'd':
              schema_view->storage_type = NANOARROW_TYPE_DENSE_UNION;
              schema_view->type = NANOARROW_TYPE_DENSE_UNION;
              break;
            case 's':
              schema_view->storage_type = NANOARROW_TYPE_SPARSE_UNION;
              schema_view->type = NANOARROW_TYPE_SPARSE_UNION;
              break;
            default:
              ArrowErrorSet(error,
                            "Expected union format string +us:<type_ids> or "
                            "+ud:<type_ids> but found '%s'",
                            format);
              return EINVAL;
          }

          if (format[3] == ':') {
            schema_view->union_type_ids = format + 4;
            int64_t n_type_ids =
                _ArrowParseUnionTypeIds(schema_view->union_type_ids, NULL);
            if (n_type_ids != schema_view->schema->n_children) {
              ArrowErrorSet(
                  error,
                  "Expected union type_ids parameter to be a comma-separated list of %ld "
                  "values between 0 and 127 but found '%s'",
                  (long)schema_view->schema->n_children, schema_view->union_type_ids);
              return EINVAL;
            }
            *format_end_out = format + strlen(format);
            return NANOARROW_OK;
          } else {
            ArrowErrorSet(error,
                          "Expected union format string +us:<type_ids> or +ud:<type_ids> "
                          "but found '%s'",
                          format);
            return EINVAL;
          }

        default:
          ArrowErrorSet(error, "Expected nested type format string but found '%s'",
                        format);
          return EINVAL;
      }

    // date/time types
    case 't':
      switch (format[1]) {
        // date
        case 'd':
          switch (format[2]) {
            case 'D':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT32);
              schema_view->type = NANOARROW_TYPE_DATE32;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'm':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_DATE64;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            default:
              ArrowErrorSet(error, "Expected 'D' or 'm' following 'td' but found '%s'",
                            format + 2);
              return EINVAL;
          }

        // time of day
        case 't':
          switch (format[2]) {
            case 's':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT32);
              schema_view->type = NANOARROW_TYPE_TIME32;
              schema_view->time_unit = NANOARROW_TIME_UNIT_SECOND;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'm':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT32);
              schema_view->type = NANOARROW_TYPE_TIME32;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MILLI;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'u':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIME64;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MICRO;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'n':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIME64;
              schema_view->time_unit = NANOARROW_TIME_UNIT_NANO;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            default:
              ArrowErrorSet(
                  error, "Expected 's', 'm', 'u', or 'n' following 'tt' but found '%s'",
                  format + 2);
              return EINVAL;
          }

        // timestamp
        case 's':
          switch (format[2]) {
            case 's':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIMESTAMP;
              schema_view->time_unit = NANOARROW_TIME_UNIT_SECOND;
              break;
            case 'm':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIMESTAMP;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MILLI;
              break;
            case 'u':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIMESTAMP;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MICRO;
              break;
            case 'n':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_TIMESTAMP;
              schema_view->time_unit = NANOARROW_TIME_UNIT_NANO;
              break;
            default:
              ArrowErrorSet(
                  error, "Expected 's', 'm', 'u', or 'n' following 'ts' but found '%s'",
                  format + 2);
              return EINVAL;
          }

          if (format[3] != ':') {
            ArrowErrorSet(error, "Expected ':' following '%.3s' but found '%s'", format,
                          format + 3);
            return EINVAL;
          }

          schema_view->timezone = format + 4;
          *format_end_out = format + strlen(format);
          return NANOARROW_OK;

        // duration
        case 'D':
          switch (format[2]) {
            case 's':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_DURATION;
              schema_view->time_unit = NANOARROW_TIME_UNIT_SECOND;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'm':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_DURATION;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MILLI;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'u':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_DURATION;
              schema_view->time_unit = NANOARROW_TIME_UNIT_MICRO;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'n':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INT64);
              schema_view->type = NANOARROW_TYPE_DURATION;
              schema_view->time_unit = NANOARROW_TIME_UNIT_NANO;
              *format_end_out = format + 3;
              return NANOARROW_OK;
            default:
              ArrowErrorSet(error,
                            "Expected 's', 'm', u', or 'n' following 'tD' but found '%s'",
                            format + 2);
              return EINVAL;
          }

        // interval
        case 'i':
          switch (format[2]) {
            case 'M':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INTERVAL_MONTHS);
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'D':
              ArrowSchemaViewSetPrimitive(schema_view, NANOARROW_TYPE_INTERVAL_DAY_TIME);
              *format_end_out = format + 3;
              return NANOARROW_OK;
            case 'n':
              ArrowSchemaViewSetPrimitive(schema_view,
                                          NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO);
              *format_end_out = format + 3;
              return NANOARROW_OK;
            default:
              ArrowErrorSet(error,
                            "Expected 'M', 'D', or 'n' following 'ti' but found '%s'",
                            format + 2);
              return EINVAL;
          }

        default:
          ArrowErrorSet(
              error, "Expected 'd', 't', 's', 'D', or 'i' following 't' but found '%s'",
              format + 1);
          return EINVAL;
      }

    default:
      ArrowErrorSet(error, "Unknown format: '%s'", format);
      return EINVAL;
  }
}

static ArrowErrorCode ArrowSchemaViewValidateNChildren(
    struct ArrowSchemaView* schema_view, int64_t n_children, struct ArrowError* error) {
  if (n_children != -1 && schema_view->schema->n_children != n_children) {
    ArrowErrorSet(error, "Expected schema with %d children but found %d children",
                  (int)n_children, (int)schema_view->schema->n_children);
    return EINVAL;
  }

  // Don't do a full validation of children but do check that they won't
  // segfault if inspected
  struct ArrowSchema* child;
  for (int64_t i = 0; i < schema_view->schema->n_children; i++) {
    child = schema_view->schema->children[i];
    if (child == NULL) {
      ArrowErrorSet(error, "Expected valid schema at schema->children[%d] but found NULL",
                    i);
      return EINVAL;
    } else if (child->release == NULL) {
      ArrowErrorSet(
          error,
          "Expected valid schema at schema->children[%d] but found a released schema", i);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowSchemaViewValidateUnion(struct ArrowSchemaView* schema_view,
                                                   struct ArrowError* error) {
  return ArrowSchemaViewValidateNChildren(schema_view, -1, error);
}

static ArrowErrorCode ArrowSchemaViewValidateMap(struct ArrowSchemaView* schema_view,
                                                 struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewValidateNChildren(schema_view, 1, error));

  if (schema_view->schema->children[0]->n_children != 2) {
    ArrowErrorSet(error, "Expected child of map type to have 2 children but found %d",
                  (int)schema_view->schema->children[0]->n_children);
    return EINVAL;
  }

  if (strcmp(schema_view->schema->children[0]->format, "+s") != 0) {
    ArrowErrorSet(error, "Expected format of child of map type to be '+s' but found '%s'",
                  schema_view->schema->children[0]->format);
    return EINVAL;
  }

  if (schema_view->schema->children[0]->flags & ARROW_FLAG_NULLABLE) {
    ArrowErrorSet(error,
                  "Expected child of map type to be non-nullable but was nullable");
    return EINVAL;
  }

  if (schema_view->schema->children[0]->children[0]->flags & ARROW_FLAG_NULLABLE) {
    ArrowErrorSet(error, "Expected key of map type to be non-nullable but was nullable");
    return EINVAL;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowSchemaViewValidateDictionary(
    struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  // check for valid index type
  switch (schema_view->storage_type) {
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
      break;
    default:
      ArrowErrorSet(
          error,
          "Expected dictionary schema index type to be an integral type but found '%s'",
          schema_view->schema->format);
      return EINVAL;
  }

  struct ArrowSchemaView dictionary_schema_view;
  return ArrowSchemaViewInit(&dictionary_schema_view, schema_view->schema->dictionary,
                             error);
}

static ArrowErrorCode ArrowSchemaViewValidate(struct ArrowSchemaView* schema_view,
                                              enum ArrowType type,
                                              struct ArrowError* error) {
  switch (type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_DATE32:
    case NANOARROW_TYPE_DATE64:
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
    case NANOARROW_TYPE_TIMESTAMP:
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_TIME64:
    case NANOARROW_TYPE_DURATION:
      return ArrowSchemaViewValidateNChildren(schema_view, 0, error);

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      if (schema_view->fixed_size <= 0) {
        ArrowErrorSet(error, "Expected size > 0 for fixed size binary but found size %d",
                      schema_view->fixed_size);
        return EINVAL;
      }
      return ArrowSchemaViewValidateNChildren(schema_view, 0, error);

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      return ArrowSchemaViewValidateNChildren(schema_view, 1, error);

    case NANOARROW_TYPE_STRUCT:
      return ArrowSchemaViewValidateNChildren(schema_view, -1, error);

    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      return ArrowSchemaViewValidateUnion(schema_view, error);

    case NANOARROW_TYPE_MAP:
      return ArrowSchemaViewValidateMap(schema_view, error);

    case NANOARROW_TYPE_DICTIONARY:
      return ArrowSchemaViewValidateDictionary(schema_view, error);

    default:
      ArrowErrorSet(error, "Expected a valid enum ArrowType value but found %d",
                    (int)schema_view->type);
      return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowSchemaViewInit(struct ArrowSchemaView* schema_view,
                                   const struct ArrowSchema* schema,
                                   struct ArrowError* error) {
  if (schema == NULL) {
    ArrowErrorSet(error, "Expected non-NULL schema");
    return EINVAL;
  }

  if (schema->release == NULL) {
    ArrowErrorSet(error, "Expected non-released schema");
    return EINVAL;
  }

  schema_view->schema = schema;

  const char* format = schema->format;
  if (format == NULL) {
    ArrowErrorSet(
        error,
        "Error parsing schema->format: Expected a null-terminated string but found NULL");
    return EINVAL;
  }

  size_t format_len = strlen(format);
  if (format_len == 0) {
    ArrowErrorSet(error, "Error parsing schema->format: Expected a string with size > 0");
    return EINVAL;
  }

  const char* format_end_out;
  ArrowErrorCode result =
      ArrowSchemaViewParse(schema_view, format, &format_end_out, error);

  if (result != NANOARROW_OK) {
    if (error != NULL) {
      char child_error[1024];
      memcpy(child_error, ArrowErrorMessage(error), 1024);
      ArrowErrorSet(error, "Error parsing schema->format: %s", child_error);
    }

    return result;
  }

  if ((format + format_len) != format_end_out) {
    ArrowErrorSet(error, "Error parsing schema->format '%s': parsed %d/%d characters",
                  format, (int)(format_end_out - format), (int)(format_len));
    return EINVAL;
  }

  if (schema->dictionary != NULL) {
    schema_view->type = NANOARROW_TYPE_DICTIONARY;
  }

  result = ArrowSchemaViewValidate(schema_view, schema_view->storage_type, error);
  if (result != NANOARROW_OK) {
    return result;
  }

  if (schema_view->storage_type != schema_view->type) {
    result = ArrowSchemaViewValidate(schema_view, schema_view->type, error);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  ArrowLayoutInit(&schema_view->layout, schema_view->storage_type);
  if (schema_view->storage_type == NANOARROW_TYPE_FIXED_SIZE_BINARY) {
    schema_view->layout.element_size_bits[1] = schema_view->fixed_size * 8;
  } else if (schema_view->storage_type == NANOARROW_TYPE_FIXED_SIZE_LIST) {
    schema_view->layout.child_size_elements = schema_view->fixed_size;
  }

  schema_view->extension_name = ArrowCharView(NULL);
  schema_view->extension_metadata = ArrowCharView(NULL);
  ArrowMetadataGetValue(schema->metadata, ArrowCharView("ARROW:extension:name"),
                        &schema_view->extension_name);
  ArrowMetadataGetValue(schema->metadata, ArrowCharView("ARROW:extension:metadata"),
                        &schema_view->extension_metadata);

  return NANOARROW_OK;
}

static int64_t ArrowSchemaTypeToStringInternal(struct ArrowSchemaView* schema_view,
                                               char* out, int64_t n) {
  const char* type_string = ArrowTypeString(schema_view->type);
  switch (schema_view->type) {
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      return snprintf(out, n, "%s(%d, %d)", type_string,
                      (int)schema_view->decimal_precision,
                      (int)schema_view->decimal_scale);
    case NANOARROW_TYPE_TIMESTAMP:
      return snprintf(out, n, "%s('%s', '%s')", type_string,
                      ArrowTimeUnitString(schema_view->time_unit), schema_view->timezone);
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_TIME64:
    case NANOARROW_TYPE_DURATION:
      return snprintf(out, n, "%s('%s')", type_string,
                      ArrowTimeUnitString(schema_view->time_unit));
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      return snprintf(out, n, "%s(%ld)", type_string, (long)schema_view->fixed_size);
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      return snprintf(out, n, "%s([%s])", type_string, schema_view->union_type_ids);
    default:
      return snprintf(out, n, "%s", type_string);
  }
}

// Helper for bookkeeping to emulate sprintf()-like behaviour spread
// among multiple sprintf calls.
static inline void ArrowToStringLogChars(char** out, int64_t n_chars_last,
                                         int64_t* n_remaining, int64_t* n_chars) {
  *n_chars += n_chars_last;
  *n_remaining -= n_chars_last;

  // n_remaining is never less than 0
  if (*n_remaining < 0) {
    *n_remaining = 0;
  }

  // Can't do math on a NULL pointer
  if (*out != NULL) {
    *out += n_chars_last;
  }
}

int64_t ArrowSchemaToString(const struct ArrowSchema* schema, char* out, int64_t n,
                            char recursive) {
  if (schema == NULL) {
    return snprintf(out, n, "[invalid: pointer is null]");
  }

  if (schema->release == NULL) {
    return snprintf(out, n, "[invalid: schema is released]");
  }

  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  if (ArrowSchemaViewInit(&schema_view, schema, &error) != NANOARROW_OK) {
    return snprintf(out, n, "[invalid: %s]", ArrowErrorMessage(&error));
  }

  // Extension type and dictionary should include both the top-level type
  // and the storage type.
  int is_extension = schema_view.extension_name.size_bytes > 0;
  int is_dictionary = schema->dictionary != NULL;
  int64_t n_chars = 0;
  int64_t n_chars_last = 0;

  // Uncommon but not technically impossible that both are true
  if (is_extension && is_dictionary) {
    n_chars_last = snprintf(
        out, n, "%.*s{dictionary(%s)<", (int)schema_view.extension_name.size_bytes,
        schema_view.extension_name.data, ArrowTypeString(schema_view.storage_type));
  } else if (is_extension) {
    n_chars_last = snprintf(out, n, "%.*s{", (int)schema_view.extension_name.size_bytes,
                            schema_view.extension_name.data);
  } else if (is_dictionary) {
    n_chars_last =
        snprintf(out, n, "dictionary(%s)<", ArrowTypeString(schema_view.storage_type));
  }

  ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);

  if (!is_dictionary) {
    n_chars_last = ArrowSchemaTypeToStringInternal(&schema_view, out, n);
  } else {
    n_chars_last = ArrowSchemaToString(schema->dictionary, out, n, recursive);
  }

  ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);

  if (recursive && schema->format[0] == '+') {
    n_chars_last = snprintf(out, n, "<");
    ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);

    for (int64_t i = 0; i < schema->n_children; i++) {
      if (i > 0) {
        n_chars_last = snprintf(out, n, ", ");
        ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);
      }

      // ArrowSchemaToStringInternal() will validate the child and print the error,
      // but we need the name first
      if (schema->children[i] != NULL && schema->children[i]->release != NULL &&
          schema->children[i]->name != NULL) {
        n_chars_last = snprintf(out, n, "%s: ", schema->children[i]->name);
        ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);
      }

      n_chars_last = ArrowSchemaToString(schema->children[i], out, n, recursive);
      ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);
    }

    n_chars_last = snprintf(out, n, ">");
    ArrowToStringLogChars(&out, n_chars_last, &n, &n_chars);
  }

  if (is_extension && is_dictionary) {
    n_chars += snprintf(out, n, ">}");
  } else if (is_extension) {
    n_chars += snprintf(out, n, "}");
  } else if (is_dictionary) {
    n_chars += snprintf(out, n, ">");
  }

  return n_chars;
}

ArrowErrorCode ArrowMetadataReaderInit(struct ArrowMetadataReader* reader,
                                       const char* metadata) {
  reader->metadata = metadata;

  if (reader->metadata == NULL) {
    reader->offset = 0;
    reader->remaining_keys = 0;
  } else {
    memcpy(&reader->remaining_keys, reader->metadata, sizeof(int32_t));
    reader->offset = sizeof(int32_t);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataReaderRead(struct ArrowMetadataReader* reader,
                                       struct ArrowStringView* key_out,
                                       struct ArrowStringView* value_out) {
  if (reader->remaining_keys <= 0) {
    return EINVAL;
  }

  int64_t pos = 0;

  int32_t key_size;
  memcpy(&key_size, reader->metadata + reader->offset + pos, sizeof(int32_t));
  pos += sizeof(int32_t);

  key_out->data = reader->metadata + reader->offset + pos;
  key_out->size_bytes = key_size;
  pos += key_size;

  int32_t value_size;
  memcpy(&value_size, reader->metadata + reader->offset + pos, sizeof(int32_t));
  pos += sizeof(int32_t);

  value_out->data = reader->metadata + reader->offset + pos;
  value_out->size_bytes = value_size;
  pos += value_size;

  reader->offset += pos;
  reader->remaining_keys--;
  return NANOARROW_OK;
}

int64_t ArrowMetadataSizeOf(const char* metadata) {
  if (metadata == NULL) {
    return 0;
  }

  struct ArrowMetadataReader reader;
  struct ArrowStringView key;
  struct ArrowStringView value;
  ArrowMetadataReaderInit(&reader, metadata);

  int64_t size = sizeof(int32_t);
  while (ArrowMetadataReaderRead(&reader, &key, &value) == NANOARROW_OK) {
    size += sizeof(int32_t) + key.size_bytes + sizeof(int32_t) + value.size_bytes;
  }

  return size;
}

static ArrowErrorCode ArrowMetadataGetValueInternal(const char* metadata,
                                                    struct ArrowStringView* key,
                                                    struct ArrowStringView* value_out) {
  struct ArrowMetadataReader reader;
  struct ArrowStringView existing_key;
  struct ArrowStringView existing_value;
  ArrowMetadataReaderInit(&reader, metadata);

  while (ArrowMetadataReaderRead(&reader, &existing_key, &existing_value) ==
         NANOARROW_OK) {
    int key_equal = key->size_bytes == existing_key.size_bytes &&
                    strncmp(key->data, existing_key.data, existing_key.size_bytes) == 0;
    if (key_equal) {
      value_out->data = existing_value.data;
      value_out->size_bytes = existing_value.size_bytes;
      break;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataGetValue(const char* metadata, struct ArrowStringView key,
                                     struct ArrowStringView* value_out) {
  if (value_out == NULL) {
    return EINVAL;
  }

  return ArrowMetadataGetValueInternal(metadata, &key, value_out);
}

char ArrowMetadataHasKey(const char* metadata, struct ArrowStringView key) {
  struct ArrowStringView value = ArrowCharView(NULL);
  ArrowMetadataGetValue(metadata, key, &value);
  return value.data != NULL;
}

ArrowErrorCode ArrowMetadataBuilderInit(struct ArrowBuffer* buffer,
                                        const char* metadata) {
  ArrowBufferInit(buffer);
  return ArrowBufferAppend(buffer, metadata, ArrowMetadataSizeOf(metadata));
}

static ArrowErrorCode ArrowMetadataBuilderAppendInternal(struct ArrowBuffer* buffer,
                                                         struct ArrowStringView* key,
                                                         struct ArrowStringView* value) {
  if (value == NULL) {
    return NANOARROW_OK;
  }

  if (buffer->capacity_bytes == 0) {
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(buffer, 0));
  }

  if (((size_t)buffer->capacity_bytes) < sizeof(int32_t)) {
    return EINVAL;
  }

  int32_t n_keys;
  memcpy(&n_keys, buffer->data, sizeof(int32_t));

  int32_t key_size = (int32_t)key->size_bytes;
  int32_t value_size = (int32_t)value->size_bytes;
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(
      buffer, sizeof(int32_t) + key_size + sizeof(int32_t) + value_size));

  ArrowBufferAppendUnsafe(buffer, &key_size, sizeof(int32_t));
  ArrowBufferAppendUnsafe(buffer, key->data, key_size);
  ArrowBufferAppendUnsafe(buffer, &value_size, sizeof(int32_t));
  ArrowBufferAppendUnsafe(buffer, value->data, value_size);

  n_keys++;
  memcpy(buffer->data, &n_keys, sizeof(int32_t));

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowMetadataBuilderSetInternal(struct ArrowBuffer* buffer,
                                                      struct ArrowStringView* key,
                                                      struct ArrowStringView* value) {
  // Inspect the current value to see if we can avoid copying the buffer
  struct ArrowStringView current_value = ArrowCharView(NULL);
  NANOARROW_RETURN_NOT_OK(
      ArrowMetadataGetValueInternal((const char*)buffer->data, key, &current_value));

  // The key should be removed but no key exists
  if (value == NULL && current_value.data == NULL) {
    return NANOARROW_OK;
  }

  // The key/value can be appended because no key exists
  if (value != NULL && current_value.data == NULL) {
    return ArrowMetadataBuilderAppendInternal(buffer, key, value);
  }

  struct ArrowMetadataReader reader;
  struct ArrowStringView existing_key;
  struct ArrowStringView existing_value;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderInit(&reader, (const char*)buffer->data));

  struct ArrowBuffer new_buffer;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(&new_buffer, NULL));

  while (reader.remaining_keys > 0) {
    int result = ArrowMetadataReaderRead(&reader, &existing_key, &existing_value);
    if (result != NANOARROW_OK) {
      ArrowBufferReset(&new_buffer);
      return result;
    }

    if (key->size_bytes == existing_key.size_bytes &&
        strncmp((const char*)key->data, (const char*)existing_key.data,
                existing_key.size_bytes) == 0) {
      result = ArrowMetadataBuilderAppendInternal(&new_buffer, key, value);
      value = NULL;
    } else {
      result =
          ArrowMetadataBuilderAppendInternal(&new_buffer, &existing_key, &existing_value);
    }

    if (result != NANOARROW_OK) {
      ArrowBufferReset(&new_buffer);
      return result;
    }
  }

  ArrowBufferReset(buffer);
  ArrowBufferMove(&new_buffer, buffer);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowMetadataBuilderAppend(struct ArrowBuffer* buffer,
                                          struct ArrowStringView key,
                                          struct ArrowStringView value) {
  return ArrowMetadataBuilderAppendInternal(buffer, &key, &value);
}

ArrowErrorCode ArrowMetadataBuilderSet(struct ArrowBuffer* buffer,
                                       struct ArrowStringView key,
                                       struct ArrowStringView value) {
  return ArrowMetadataBuilderSetInternal(buffer, &key, &value);
}

ArrowErrorCode ArrowMetadataBuilderRemove(struct ArrowBuffer* buffer,
                                          struct ArrowStringView key) {
  return ArrowMetadataBuilderSetInternal(buffer, &key, NULL);
}
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <stdlib.h>
#include <string.h>



static void ArrowArrayRelease(struct ArrowArray* array) {
  // Release buffers held by this array
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  if (private_data != NULL) {
    ArrowBitmapReset(&private_data->bitmap);
    ArrowBufferReset(&private_data->buffers[0]);
    ArrowBufferReset(&private_data->buffers[1]);
    ArrowFree(private_data);
  }

  // This object owns the memory for all the children, but those
  // children may have been generated elsewhere and might have
  // their own release() callback.
  if (array->children != NULL) {
    for (int64_t i = 0; i < array->n_children; i++) {
      if (array->children[i] != NULL) {
        if (array->children[i]->release != NULL) {
          array->children[i]->release(array->children[i]);
        }

        ArrowFree(array->children[i]);
      }
    }

    ArrowFree(array->children);
  }

  // This object owns the memory for the dictionary but it
  // may have been generated somewhere else and have its own
  // release() callback.
  if (array->dictionary != NULL) {
    if (array->dictionary->release != NULL) {
      array->dictionary->release(array->dictionary);
    }

    ArrowFree(array->dictionary);
  }

  // Mark released
  array->release = NULL;
}

static ArrowErrorCode ArrowArraySetStorageType(struct ArrowArray* array,
                                               enum ArrowType storage_type) {
  switch (storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
    case NANOARROW_TYPE_NA:
      array->n_buffers = 0;
      break;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      array->n_buffers = 1;
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_DENSE_UNION:
      array->n_buffers = 2;
      break;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      array->n_buffers = 3;
      break;

    default:
      return EINVAL;

      return NANOARROW_OK;
  }

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  private_data->storage_type = storage_type;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromType(struct ArrowArray* array,
                                      enum ArrowType storage_type) {
  array->length = 0;
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 0;
  array->n_children = 0;
  array->buffers = NULL;
  array->children = NULL;
  array->dictionary = NULL;
  array->release = &ArrowArrayRelease;
  array->private_data = NULL;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)ArrowMalloc(sizeof(struct ArrowArrayPrivateData));
  if (private_data == NULL) {
    array->release = NULL;
    return ENOMEM;
  }

  ArrowBitmapInit(&private_data->bitmap);
  ArrowBufferInit(&private_data->buffers[0]);
  ArrowBufferInit(&private_data->buffers[1]);
  private_data->buffer_data[0] = NULL;
  private_data->buffer_data[1] = NULL;
  private_data->buffer_data[2] = NULL;

  array->private_data = private_data;
  array->buffers = (const void**)(&private_data->buffer_data);

  int result = ArrowArraySetStorageType(array, storage_type);
  if (result != NANOARROW_OK) {
    array->release(array);
    return result;
  }

  ArrowLayoutInit(&private_data->layout, storage_type);
  // We can only know this not to be true when initializing based on a schema
  // so assume this to be true.
  private_data->union_type_id_is_child_index = 1;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromArrayView(struct ArrowArray* array,
                                           const struct ArrowArrayView* array_view,
                                           struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowArrayInitFromType(array, array_view->storage_type), error);
  int result;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  private_data->layout = array_view->layout;

  if (array_view->n_children > 0) {
    result = ArrowArrayAllocateChildren(array, array_view->n_children);
    if (result != NANOARROW_OK) {
      array->release(array);
      return result;
    }

    for (int64_t i = 0; i < array_view->n_children; i++) {
      result =
          ArrowArrayInitFromArrayView(array->children[i], array_view->children[i], error);
      if (result != NANOARROW_OK) {
        array->release(array);
        return result;
      }
    }
  }

  if (array_view->dictionary != NULL) {
    result = ArrowArrayAllocateDictionary(array);
    if (result != NANOARROW_OK) {
      array->release(array);
      return result;
    }

    result =
        ArrowArrayInitFromArrayView(array->dictionary, array_view->dictionary, error);
    if (result != NANOARROW_OK) {
      array->release(array);
      return result;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromSchema(struct ArrowArray* array,
                                        const struct ArrowSchema* schema,
                                        struct ArrowError* error) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(&array_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(array, &array_view, error));
  if (array_view.storage_type == NANOARROW_TYPE_DENSE_UNION ||
      array_view.storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    struct ArrowArrayPrivateData* private_data =
        (struct ArrowArrayPrivateData*)array->private_data;
    // We can still build arrays if this isn't true; however, the append
    // functions won't work. Instead, we store this value and error only
    // when StartAppending is called.
    private_data->union_type_id_is_child_index =
        _ArrowUnionTypeIdsWillEqualChildIndices(schema->format + 4, schema->n_children);
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateChildren(struct ArrowArray* array, int64_t n_children) {
  if (array->children != NULL) {
    return EINVAL;
  }

  if (n_children == 0) {
    return NANOARROW_OK;
  }

  array->children =
      (struct ArrowArray**)ArrowMalloc(n_children * sizeof(struct ArrowArray*));
  if (array->children == NULL) {
    return ENOMEM;
  }

  memset(array->children, 0, n_children * sizeof(struct ArrowArray*));

  for (int64_t i = 0; i < n_children; i++) {
    array->children[i] = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
    if (array->children[i] == NULL) {
      return ENOMEM;
    }
    array->children[i]->release = NULL;
  }

  array->n_children = n_children;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateDictionary(struct ArrowArray* array) {
  if (array->dictionary != NULL) {
    return EINVAL;
  }

  array->dictionary = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
  if (array->dictionary == NULL) {
    return ENOMEM;
  }

  array->dictionary->release = NULL;
  return NANOARROW_OK;
}

void ArrowArraySetValidityBitmap(struct ArrowArray* array, struct ArrowBitmap* bitmap) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  ArrowBufferMove(&bitmap->buffer, &private_data->bitmap.buffer);
  private_data->bitmap.size_bits = bitmap->size_bits;
  bitmap->size_bits = 0;
  private_data->buffer_data[0] = private_data->bitmap.buffer.data;
  array->null_count = -1;
}

ArrowErrorCode ArrowArraySetBuffer(struct ArrowArray* array, int64_t i,
                                   struct ArrowBuffer* buffer) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  switch (i) {
    case 0:
      ArrowBufferMove(buffer, &private_data->bitmap.buffer);
      private_data->buffer_data[i] = private_data->bitmap.buffer.data;
      break;
    case 1:
    case 2:
      ArrowBufferMove(buffer, &private_data->buffers[i - 1]);
      private_data->buffer_data[i] = private_data->buffers[i - 1].data;
      break;
    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayViewInitFromArray(struct ArrowArrayView* array_view,
                                                  struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  ArrowArrayViewInitFromType(array_view, private_data->storage_type);
  array_view->layout = private_data->layout;
  array_view->array = array;
  array_view->length = array->length;
  array_view->offset = array->offset;
  array_view->null_count = array->null_count;

  array_view->buffer_views[0].data.as_uint8 = private_data->bitmap.buffer.data;
  array_view->buffer_views[0].size_bytes = private_data->bitmap.buffer.size_bytes;
  array_view->buffer_views[1].data.as_uint8 = private_data->buffers[0].data;
  array_view->buffer_views[1].size_bytes = private_data->buffers[0].size_bytes;
  array_view->buffer_views[2].data.as_uint8 = private_data->buffers[1].data;
  array_view->buffer_views[2].size_bytes = private_data->buffers[1].size_bytes;

  int result = ArrowArrayViewAllocateChildren(array_view, array->n_children);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    result = ArrowArrayViewInitFromArray(array_view->children[i], array->children[i]);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (array->dictionary != NULL) {
    result = ArrowArrayViewAllocateDictionary(array_view);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }

    result = ArrowArrayViewInitFromArray(array_view->dictionary, array->dictionary);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayReserveInternal(struct ArrowArray* array,
                                                struct ArrowArrayView* array_view) {
  // Loop through buffers and reserve the extra space that we know about
  for (int64_t i = 0; i < array->n_buffers; i++) {
    // Don't reserve on a validity buffer that hasn't been allocated yet
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        ArrowArrayBuffer(array, i)->data == NULL) {
      continue;
    }

    int64_t additional_size_bytes =
        array_view->buffer_views[i].size_bytes - ArrowArrayBuffer(array, i)->size_bytes;

    if (additional_size_bytes > 0) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferReserve(ArrowArrayBuffer(array, i), additional_size_bytes));
    }
  }

  // Recursively reserve children
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayReserveInternal(array->children[i], array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayReserve(struct ArrowArray* array,
                                 int64_t additional_size_elements) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromArray(&array_view, array));

  // Calculate theoretical buffer sizes (recursively)
  ArrowArrayViewSetLength(&array_view, array->length + additional_size_elements);

  // Walk the structure (recursively)
  int result = ArrowArrayReserveInternal(array, &array_view);
  ArrowArrayViewReset(&array_view);
  if (result != NANOARROW_OK) {
    return result;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayFinalizeBuffers(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  // The only buffer finalizing this currently does is make sure the data
  // buffer for (Large)String|Binary is never NULL
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
      if (ArrowArrayBuffer(array, 2)->data == NULL) {
        ArrowBufferAppendUInt8(ArrowArrayBuffer(array, 2), 0);
      }
      break;
    default:
      break;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array->children[i]));
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array->dictionary));
  }

  return NANOARROW_OK;
}

static void ArrowArrayFlushInternalPointers(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  for (int64_t i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    private_data->buffer_data[i] = ArrowArrayBuffer(array, i)->data;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    ArrowArrayFlushInternalPointers(array->children[i]);
  }

  if (array->dictionary != NULL) {
    ArrowArrayFlushInternalPointers(array->dictionary);
  }
}

ArrowErrorCode ArrowArrayFinishBuilding(struct ArrowArray* array,
                                        enum ArrowValidationLevel validation_level,
                                        struct ArrowError* error) {
  // Even if the data buffer is size zero, the pointer value needed to be non-null
  // in some implementations (at least one version of Arrow C++ at the time this
  // was added). Only do this fix if we can assume CPU data access.
  if (validation_level >= NANOARROW_VALIDATION_LEVEL_DEFAULT) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinalizeBuffers(array), error);
  }

  // Make sure the value we get with array->buffers[i] is set to the actual
  // pointer (which may have changed from the original due to reallocation)
  ArrowArrayFlushInternalPointers(array);

  if (validation_level == NANOARROW_VALIDATION_LEVEL_NONE) {
    return NANOARROW_OK;
  }

  // For validation, initialize an ArrowArrayView with our known buffer sizes
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayViewInitFromArray(&array_view, array),
                                     error);
  int result = ArrowArrayViewValidate(&array_view, validation_level, error);
  ArrowArrayViewReset(&array_view);
  return result;
}

ArrowErrorCode ArrowArrayFinishBuildingDefault(struct ArrowArray* array,
                                               struct ArrowError* error) {
  return ArrowArrayFinishBuilding(array, NANOARROW_VALIDATION_LEVEL_DEFAULT, error);
}

void ArrowArrayViewInitFromType(struct ArrowArrayView* array_view,
                                enum ArrowType storage_type) {
  memset(array_view, 0, sizeof(struct ArrowArrayView));
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);
}

ArrowErrorCode ArrowArrayViewAllocateChildren(struct ArrowArrayView* array_view,
                                              int64_t n_children) {
  if (array_view->children != NULL) {
    return EINVAL;
  }

  array_view->children =
      (struct ArrowArrayView**)ArrowMalloc(n_children * sizeof(struct ArrowArrayView*));
  if (array_view->children == NULL) {
    return ENOMEM;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] = NULL;
  }

  array_view->n_children = n_children;

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] =
        (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
    if (array_view->children[i] == NULL) {
      return ENOMEM;
    }
    ArrowArrayViewInitFromType(array_view->children[i], NANOARROW_TYPE_UNINITIALIZED);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewAllocateDictionary(struct ArrowArrayView* array_view) {
  if (array_view->dictionary != NULL) {
    return EINVAL;
  }

  array_view->dictionary =
      (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
  if (array_view->dictionary == NULL) {
    return ENOMEM;
  }

  ArrowArrayViewInitFromType(array_view->dictionary, NANOARROW_TYPE_UNINITIALIZED);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewInitFromSchema(struct ArrowArrayView* array_view,
                                            const struct ArrowSchema* schema,
                                            struct ArrowError* error) {
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowArrayViewInitFromType(array_view, schema_view.storage_type);
  array_view->layout = schema_view.layout;

  result = ArrowArrayViewAllocateChildren(array_view, schema->n_children);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowArrayViewAllocateChildren() failed");
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    result =
        ArrowArrayViewInitFromSchema(array_view->children[i], schema->children[i], error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (schema->dictionary != NULL) {
    result = ArrowArrayViewAllocateDictionary(array_view);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }

    result =
        ArrowArrayViewInitFromSchema(array_view->dictionary, schema->dictionary, error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_SPARSE_UNION ||
      array_view->storage_type == NANOARROW_TYPE_DENSE_UNION) {
    array_view->union_type_id_map = (int8_t*)ArrowMalloc(256 * sizeof(int8_t));
    if (array_view->union_type_id_map == NULL) {
      return ENOMEM;
    }

    memset(array_view->union_type_id_map, -1, 256);
    int8_t n_type_ids = _ArrowParseUnionTypeIds(schema_view.union_type_ids,
                                                array_view->union_type_id_map + 128);
    for (int8_t child_index = 0; child_index < n_type_ids; child_index++) {
      int8_t type_id = array_view->union_type_id_map[128 + child_index];
      array_view->union_type_id_map[type_id] = child_index;
    }
  }

  return NANOARROW_OK;
}

void ArrowArrayViewReset(struct ArrowArrayView* array_view) {
  if (array_view->children != NULL) {
    for (int64_t i = 0; i < array_view->n_children; i++) {
      if (array_view->children[i] != NULL) {
        ArrowArrayViewReset(array_view->children[i]);
        ArrowFree(array_view->children[i]);
      }
    }

    ArrowFree(array_view->children);
  }

  if (array_view->dictionary != NULL) {
    ArrowArrayViewReset(array_view->dictionary);
    ArrowFree(array_view->dictionary);
  }

  if (array_view->union_type_id_map != NULL) {
    ArrowFree(array_view->union_type_id_map);
  }

  ArrowArrayViewInitFromType(array_view, NANOARROW_TYPE_UNINITIALIZED);
}

void ArrowArrayViewSetLength(struct ArrowArrayView* array_view, int64_t length) {
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;

    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        array_view->buffer_views[i].size_bytes = _ArrowBytesForBits(length);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        array_view->buffer_views[i].size_bytes =
            (length != 0) * element_size_bytes * (length + 1);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
        array_view->buffer_views[i].size_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] * length) /
            8;
        continue;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        array_view->buffer_views[i].size_bytes = element_size_bytes * length;
        continue;
      case NANOARROW_BUFFER_TYPE_NONE:
        array_view->buffer_views[i].size_bytes = 0;
        continue;
    }
  }

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        ArrowArrayViewSetLength(array_view->children[i], length);
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      if (array_view->n_children >= 1) {
        ArrowArrayViewSetLength(array_view->children[0],
                                length * array_view->layout.child_size_elements);
      }
    default:
      break;
  }
}

// This version recursively extracts information from the array and stores it
// in the array view, performing any checks that require the original array.
static int ArrowArrayViewSetArrayInternal(struct ArrowArrayView* array_view,
                                          const struct ArrowArray* array,
                                          struct ArrowError* error) {
  array_view->array = array;
  array_view->offset = array->offset;
  array_view->length = array->length;
  array_view->null_count = array->null_count;

  int64_t buffers_required = 0;
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    buffers_required++;

    // Set buffer pointer
    array_view->buffer_views[i].data.data = array->buffers[i];

    // If non-null, set buffer size to unknown.
    if (array->buffers[i] == NULL) {
      array_view->buffer_views[i].size_bytes = 0;
    } else {
      array_view->buffer_views[i].size_bytes = -1;
    }
  }

  // Check the number of buffers
  if (buffers_required != array->n_buffers) {
    ArrowErrorSet(error, "Expected array with %d buffer(s) but found %d buffer(s)",
                  (int)buffers_required, (int)array->n_buffers);
    return EINVAL;
  }

  // Check number of children
  if (array_view->n_children != array->n_children) {
    ArrowErrorSet(error, "Expected %ld children but found %ld children",
                  (long)array_view->n_children, (long)array->n_children);
    return EINVAL;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view->children[i],
                                                           array->children[i], error));
  }

  // Check dictionary
  if (array->dictionary == NULL && array_view->dictionary != NULL) {
    ArrowErrorSet(error, "Expected dictionary but found NULL");
    return EINVAL;
  }

  if (array->dictionary != NULL && array_view->dictionary == NULL) {
    ArrowErrorSet(error, "Expected NULL dictionary but found dictionary member");
    return EINVAL;
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArrayInternal(array_view->dictionary, array->dictionary, error));
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateMinimal(struct ArrowArrayView* array_view,
                                         struct ArrowError* error) {
  if (array_view->length < 0) {
    ArrowErrorSet(error, "Expected length >= 0 but found length %ld",
                  (long)array_view->length);
    return EINVAL;
  }

  if (array_view->offset < 0) {
    ArrowErrorSet(error, "Expected offset >= 0 but found offset %ld",
                  (long)array_view->offset);
    return EINVAL;
  }

  // Calculate buffer sizes that do not require buffer access. If marked as
  // unknown, assign the buffer size; otherwise, validate it.
  int64_t offset_plus_length = array_view->offset + array_view->length;

  // Only loop over the first two buffers because the size of the third buffer
  // is always data dependent for all current Arrow types.
  for (int i = 0; i < 2; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;
    // Initialize with a value that will cause an error if accidentally used uninitialized
    int64_t min_buffer_size_bytes = array_view->buffer_views[i].size_bytes + 1;

    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        if (array_view->null_count == 0 && array_view->buffer_views[i].size_bytes == 0) {
          continue;
        }

        min_buffer_size_bytes = _ArrowBytesForBits(offset_plus_length);
        break;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        min_buffer_size_bytes =
            (offset_plus_length != 0) * element_size_bytes * (offset_plus_length + 1);
        break;
      case NANOARROW_BUFFER_TYPE_DATA:
        min_buffer_size_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] *
                                       offset_plus_length) /
            8;
        break;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        min_buffer_size_bytes = element_size_bytes * offset_plus_length;
        break;
      case NANOARROW_BUFFER_TYPE_NONE:
        continue;
    }

    // Assign or validate buffer size
    if (array_view->buffer_views[i].size_bytes == -1) {
      array_view->buffer_views[i].size_bytes = min_buffer_size_bytes;
    } else if (array_view->buffer_views[i].size_bytes < min_buffer_size_bytes) {
      ArrowErrorSet(error,
                    "Expected %s array buffer %d to have size >= %ld bytes but found "
                    "buffer with %ld bytes",
                    ArrowTypeString(array_view->storage_type), (int)i,
                    (long)min_buffer_size_bytes,
                    (long)array_view->buffer_views[i].size_bytes);
      return EINVAL;
    }
  }

  // For list, fixed-size list and map views, we can validate the number of children
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_MAP:
      if (array_view->n_children != 1) {
        ArrowErrorSet(error, "Expected 1 child of %s array but found %ld child arrays",
                      ArrowTypeString(array_view->storage_type),
                      (long)array_view->n_children);
        return EINVAL;
      }
    default:
      break;
  }

  // For struct, the sparse union, and the fixed-size list views, we can validate child
  // lengths.
  int64_t child_min_length;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_STRUCT:
      child_min_length = (array_view->offset + array_view->length);
      for (int64_t i = 0; i < array_view->n_children; i++) {
        if (array_view->children[i]->length < child_min_length) {
          ArrowErrorSet(
              error,
              "Expected struct child %d to have length >= %ld but found child with "
              "length %ld",
              (int)(i + 1), (long)(child_min_length),
              (long)array_view->children[i]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      child_min_length = (array_view->offset + array_view->length) *
                         array_view->layout.child_size_elements;
      if (array_view->children[0]->length < child_min_length) {
        ArrowErrorSet(error,
                      "Expected child of fixed_size_list array to have length >= %ld but "
                      "found array with length %ld",
                      (long)child_min_length, (long)array_view->children[0]->length);
        return EINVAL;
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewValidateMinimal(array_view->children[i], error));
  }

  // Recurse for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view->dictionary, error));
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateDefault(struct ArrowArrayView* array_view,
                                         struct ArrowError* error) {
  // Perform minimal validation. This will validate or assign
  // buffer sizes as long as buffer access is not required.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view, error));

  // Calculate buffer sizes or child lengths that require accessing the offsets
  // buffer. Where appropriate, validate that the first offset is >= 0.
  // If a buffer size is marked as unknown, assign it; otherwise, validate it.
  int64_t offset_plus_length = array_view->offset + array_view->length;

  int64_t first_offset;
  int64_t last_offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int32[0];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %ld",
                        (long)first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int32[offset_plus_length];

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset;
        } else if (array_view->buffer_views[2].size_bytes < last_offset) {
          ArrowErrorSet(error,
                        "Expected %s array buffer 2 to have size >= %ld bytes but found "
                        "buffer with %ld bytes",
                        ArrowTypeString(array_view->storage_type), (long)last_offset,
                        (long)array_view->buffer_views[2].size_bytes);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int64[0];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %ld",
                        (long)first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int64[offset_plus_length];

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset;
        } else if (array_view->buffer_views[2].size_bytes < last_offset) {
          ArrowErrorSet(error,
                        "Expected %s array buffer 2 to have size >= %ld bytes but found "
                        "buffer with %ld bytes",
                        ArrowTypeString(array_view->storage_type), (long)last_offset,
                        (long)array_view->buffer_views[2].size_bytes);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        if (array_view->children[i]->length < offset_plus_length) {
          ArrowErrorSet(
              error,
              "Expected struct child %d to have length >= %ld but found child with "
              "length %ld",
              (int)(i + 1), (long)offset_plus_length,
              (long)array_view->children[i]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int32[0];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %ld",
                        (long)first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int32[offset_plus_length];
        if (array_view->children[0]->length < last_offset) {
          ArrowErrorSet(
              error,
              "Expected child of %s array to have length >= %ld but found array with "
              "length %ld",
              ArrowTypeString(array_view->storage_type), (long)last_offset,
              (long)array_view->children[0]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LARGE_LIST:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int64[0];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %ld",
                        (long)first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int64[offset_plus_length];
        if (array_view->children[0]->length < last_offset) {
          ArrowErrorSet(
              error,
              "Expected child of large list array to have length >= %ld but found array "
              "with length %ld",
              (long)last_offset, (long)array_view->children[0]->length);
          return EINVAL;
        }
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewValidateDefault(array_view->children[i], error));
  }

  // Recurse for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view->dictionary, error));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewSetArray(struct ArrowArrayView* array_view,
                                      const struct ArrowArray* array,
                                      struct ArrowError* error) {
  // Extract information from the array into the array view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view, array, error));

  // Run default validation. Because we've marked all non-NULL buffers as having unknown
  // size, validation will also update the buffer sizes as it goes.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view, error));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewSetArrayMinimal(struct ArrowArrayView* array_view,
                                             const struct ArrowArray* array,
                                             struct ArrowError* error) {
  // Extract information from the array into the array view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view, array, error));

  // Run default validation. Because we've marked all non-NULL buffers as having unknown
  // size, validation will also update the buffer sizes as it goes.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view, error));

  return NANOARROW_OK;
}

static int ArrowAssertIncreasingInt32(struct ArrowBufferView view,
                                      struct ArrowError* error) {
  if (view.size_bytes <= (int64_t)sizeof(int32_t)) {
    return NANOARROW_OK;
  }

  for (int64_t i = 1; i < view.size_bytes / (int64_t)sizeof(int32_t); i++) {
    if (view.data.as_int32[i] < view.data.as_int32[i - 1]) {
      ArrowErrorSet(error, "[%ld] Expected element size >= 0", (long)i);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertIncreasingInt64(struct ArrowBufferView view,
                                      struct ArrowError* error) {
  if (view.size_bytes <= (int64_t)sizeof(int64_t)) {
    return NANOARROW_OK;
  }

  for (int64_t i = 1; i < view.size_bytes / (int64_t)sizeof(int64_t); i++) {
    if (view.data.as_int64[i] < view.data.as_int64[i - 1]) {
      ArrowErrorSet(error, "[%ld] Expected element size >= 0", (long)i);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertRangeInt8(struct ArrowBufferView view, int8_t min_value,
                                int8_t max_value, struct ArrowError* error) {
  for (int64_t i = 0; i < view.size_bytes; i++) {
    if (view.data.as_int8[i] < min_value || view.data.as_int8[i] > max_value) {
      ArrowErrorSet(error,
                    "[%ld] Expected buffer value between %d and %d but found value %d",
                    (long)i, (int)min_value, (int)max_value, (int)view.data.as_int8[i]);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertInt8In(struct ArrowBufferView view, const int8_t* values,
                             int64_t n_values, struct ArrowError* error) {
  for (int64_t i = 0; i < view.size_bytes; i++) {
    int item_found = 0;
    for (int64_t j = 0; j < n_values; j++) {
      if (view.data.as_int8[i] == values[j]) {
        item_found = 1;
        break;
      }
    }

    if (!item_found) {
      ArrowErrorSet(error, "[%ld] Unexpected buffer value %d", (long)i,
                    (int)view.data.as_int8[i]);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateFull(struct ArrowArrayView* array_view,
                                      struct ArrowError* error) {
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        if (array_view->layout.element_size_bits[i] == 32) {
          NANOARROW_RETURN_NOT_OK(
              ArrowAssertIncreasingInt32(array_view->buffer_views[i], error));
        } else {
          NANOARROW_RETURN_NOT_OK(
              ArrowAssertIncreasingInt64(array_view->buffer_views[i], error));
        }
        break;
      default:
        break;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_DENSE_UNION ||
      array_view->storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    if (array_view->union_type_id_map == NULL) {
      // If the union_type_id map is NULL (e.g., when using ArrowArrayInitFromType() +
      // ArrowArrayAllocateChildren() + ArrowArrayFinishBuilding()), we don't have enough
      // information to validate this buffer.
      ArrowErrorSet(error,
                    "Insufficient information provided for validation of union array");
      return EINVAL;
    } else if (_ArrowParsedUnionTypeIdsWillEqualChildIndices(
                   array_view->union_type_id_map, array_view->n_children,
                   array_view->n_children)) {
      NANOARROW_RETURN_NOT_OK(ArrowAssertRangeInt8(
          array_view->buffer_views[0], 0, (int8_t)(array_view->n_children - 1), error));
    } else {
      NANOARROW_RETURN_NOT_OK(ArrowAssertInt8In(array_view->buffer_views[0],
                                                array_view->union_type_id_map + 128,
                                                array_view->n_children, error));
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_DENSE_UNION &&
      array_view->union_type_id_map != NULL) {
    // Check that offsets refer to child elements that actually exist
    for (int64_t i = 0; i < array_view->length; i++) {
      int8_t child_id = ArrowArrayViewUnionChildIndex(array_view, i);
      int64_t offset = ArrowArrayViewUnionChildOffset(array_view, i);
      int64_t child_length = array_view->children[child_id]->length;
      if (offset < 0 || offset > child_length) {
        ArrowErrorSet(
            error,
            "[%ld] Expected union offset for child id %d to be between 0 and %ld but "
            "found offset value %ld",
            (long)i, (int)child_id, (long)child_length, offset);
        return EINVAL;
      }
    }
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateFull(array_view->children[i], error));
  }

  // Dictionary valiation not implemented
  if (array_view->dictionary != NULL) {
    ArrowErrorSet(error, "Validation for dictionary-encoded arrays is not implemented");
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewValidate(struct ArrowArrayView* array_view,
                                      enum ArrowValidationLevel validation_level,
                                      struct ArrowError* error) {
  switch (validation_level) {
    case NANOARROW_VALIDATION_LEVEL_NONE:
      return NANOARROW_OK;
    case NANOARROW_VALIDATION_LEVEL_MINIMAL:
      return ArrowArrayViewValidateMinimal(array_view, error);
    case NANOARROW_VALIDATION_LEVEL_DEFAULT:
      return ArrowArrayViewValidateDefault(array_view, error);
    case NANOARROW_VALIDATION_LEVEL_FULL:
      NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view, error));
      return ArrowArrayViewValidateFull(array_view, error);
  }

  ArrowErrorSet(error, "validation_level not recognized");
  return EINVAL;
}
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>



struct BasicArrayStreamPrivate {
  struct ArrowSchema schema;
  int64_t n_arrays;
  struct ArrowArray* arrays;
  int64_t arrays_i;
};

static int ArrowBasicArrayStreamGetSchema(struct ArrowArrayStream* array_stream,
                                          struct ArrowSchema* schema) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return EINVAL;
  }

  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;
  return ArrowSchemaDeepCopy(&private_data->schema, schema);
}

static int ArrowBasicArrayStreamGetNext(struct ArrowArrayStream* array_stream,
                                        struct ArrowArray* array) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return EINVAL;
  }

  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  if (private_data->arrays_i == private_data->n_arrays) {
    array->release = NULL;
    return NANOARROW_OK;
  }

  ArrowArrayMove(&private_data->arrays[private_data->arrays_i++], array);
  return NANOARROW_OK;
}

static const char* ArrowBasicArrayStreamGetLastError(
    struct ArrowArrayStream* array_stream) {
  return NULL;
}

static void ArrowBasicArrayStreamRelease(struct ArrowArrayStream* array_stream) {
  if (array_stream == NULL || array_stream->release == NULL) {
    return;
  }

  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  if (private_data->schema.release != NULL) {
    private_data->schema.release(&private_data->schema);
  }

  for (int64_t i = 0; i < private_data->n_arrays; i++) {
    if (private_data->arrays[i].release != NULL) {
      private_data->arrays[i].release(&private_data->arrays[i]);
    }
  }

  if (private_data->arrays != NULL) {
    ArrowFree(private_data->arrays);
  }

  ArrowFree(private_data);
  array_stream->release = NULL;
}

ArrowErrorCode ArrowBasicArrayStreamInit(struct ArrowArrayStream* array_stream,
                                         struct ArrowSchema* schema, int64_t n_arrays) {
  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)ArrowMalloc(
          sizeof(struct BasicArrayStreamPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  ArrowSchemaMove(schema, &private_data->schema);

  private_data->n_arrays = n_arrays;
  private_data->arrays = NULL;
  private_data->arrays_i = 0;

  if (n_arrays > 0) {
    private_data->arrays =
        (struct ArrowArray*)ArrowMalloc(n_arrays * sizeof(struct ArrowArray));
    if (private_data->arrays == NULL) {
      ArrowBasicArrayStreamRelease(array_stream);
      return ENOMEM;
    }
  }

  for (int64_t i = 0; i < private_data->n_arrays; i++) {
    private_data->arrays[i].release = NULL;
  }

  array_stream->get_schema = &ArrowBasicArrayStreamGetSchema;
  array_stream->get_next = &ArrowBasicArrayStreamGetNext;
  array_stream->get_last_error = ArrowBasicArrayStreamGetLastError;
  array_stream->release = ArrowBasicArrayStreamRelease;
  array_stream->private_data = private_data;
  return NANOARROW_OK;
}

void ArrowBasicArrayStreamSetArray(struct ArrowArrayStream* array_stream, int64_t i,
                                   struct ArrowArray* array) {
  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;
  ArrowArrayMove(array, &private_data->arrays[i]);
}

ArrowErrorCode ArrowBasicArrayStreamValidate(const struct ArrowArrayStream* array_stream,
                                             struct ArrowError* error) {
  struct BasicArrayStreamPrivate* private_data =
      (struct BasicArrayStreamPrivate*)array_stream->private_data;

  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&array_view, &private_data->schema, error));

  for (int64_t i = 0; i < private_data->n_arrays; i++) {
    if (private_data->arrays[i].release != NULL) {
      int result = ArrowArrayViewSetArray(&array_view, &private_data->arrays[i], error);
      if (result != NANOARROW_OK) {
        ArrowArrayViewReset(&array_view);
        return result;
      }
    }
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}
