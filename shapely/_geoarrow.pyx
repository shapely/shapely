
from cpython cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdint cimport int32_t, int64_t, uintptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memset

from shapely._geos cimport (
    GEOS_init_r,
    GEOS_finish_r,
    GEOSContextHandle_t,
    GEOSGeom_destroy_r,
    GEOSGeometry,
    get_geos_handle,
)
from shapely._pygeos_api cimport import_shapely_c_api, PyGEOS_CreateGeometry, PyGEOS_GetGEOSGeometry

import numpy as np

# initialize Shapely C API
import_shapely_c_api()

cdef extern from "geoarrow_geos.h" nogil:
    struct ArrowSchema:
        void (*release)(ArrowSchema*)

    struct ArrowArray:
        int64_t length
        void (*release)(ArrowArray*)

    struct GeoArrowGEOSArrayReader

    ctypedef int GeoArrowGEOSErrorCode
    cdef int GEOARROW_GEOS_OK

    GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderCreate(GEOSContextHandle_t handle,
                                                        ArrowSchema* schema,
                                                        GeoArrowGEOSArrayReader** out)

    const char* GeoArrowGEOSArrayReaderGetLastError(GeoArrowGEOSArrayReader* reader)

    GeoArrowGEOSErrorCode GeoArrowGEOSArrayReaderRead(GeoArrowGEOSArrayReader* reader,
                                                      ArrowArray* array, size_t offset,
                                                      size_t length, GEOSGeometry** out,
                                                      size_t* n_out)

    void GeoArrowGEOSArrayReaderDestroy(GeoArrowGEOSArrayReader* reader)


    struct GeoArrowGEOSArrayBuilder

    GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderCreate(
        GEOSContextHandle_t handle, ArrowSchema* schema,
        GeoArrowGEOSArrayBuilder** out)

    void GeoArrowGEOSArrayBuilderDestroy(GeoArrowGEOSArrayBuilder* builder)

    const char* GeoArrowGEOSArrayBuilderGetLastError(GeoArrowGEOSArrayBuilder* builder)

    GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderAppend(
        GeoArrowGEOSArrayBuilder* builder, GEOSGeometry** geom, size_t geom_size,
        size_t* n_appended)

    GeoArrowGEOSErrorCode GeoArrowGEOSArrayBuilderFinish(GeoArrowGEOSArrayBuilder* builder,
        ArrowArray* out)

    enum GeoArrowGEOSEncoding:
        GEOARROW_GEOS_ENCODING_UNKNOWN
        GEOARROW_GEOS_ENCODING_WKT
        GEOARROW_GEOS_ENCODING_WKB
        GEOARROW_GEOS_ENCODING_GEOARROW
        GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED

    GeoArrowGEOSErrorCode GeoArrowGEOSMakeSchema(int32_t encoding, int32_t wkb_type,
        ArrowSchema* out)

    struct GeoArrowGEOSSchemaCalculator

    GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorCreate(
        GeoArrowGEOSSchemaCalculator** out)

    void GeoArrowGEOSSchemaCalculatorIngest(GeoArrowGEOSSchemaCalculator* calc,
                                            const int32_t* wkb_type, size_t n)

    GeoArrowGEOSErrorCode GeoArrowGEOSSchemaCalculatorFinish(
        GeoArrowGEOSSchemaCalculator* calc, GeoArrowGEOSEncoding encoding,
        ArrowSchema* out)

    void GeoArrowGEOSSchemaCalculatorDestroy(GeoArrowGEOSSchemaCalculator* calc)

    int32_t GeoArrowGEOSWKBType(GEOSContextHandle_t handle,
                                const GEOSGeometry* geom)


cdef void pycapsule_schema_deleter(object schema_capsule) noexcept:
    cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    if schema.release != NULL:
        schema.release(schema)

    free(schema)


cdef object alloc_c_schema(ArrowSchema** c_schema) noexcept:
    c_schema[0] = <ArrowSchema*> malloc(sizeof(ArrowSchema))
    c_schema[0].release = NULL
    return PyCapsule_New(c_schema[0], 'arrow_schema', &pycapsule_schema_deleter)


cdef void pycapsule_array_deleter(object array_capsule) noexcept:
    cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(
        array_capsule, 'arrow_array'
    )
    if array.release != NULL:
        array.release(array)

    free(array)


cdef object alloc_c_array(ArrowArray** c_array) noexcept:
    c_array[0] = <ArrowArray*> malloc(sizeof(ArrowArray))
    c_array[0].release = NULL
    return PyCapsule_New(c_array[0], 'arrow_array', &pycapsule_array_deleter)


class GeoArrowGEOSException(Exception):

    def __init__(self, what, code, msg):
        super().__init__(f"{what} failed with code {code}: {msg}")


cdef class GEOSGeometryArray:
    cdef GEOSGeometry** _ptr
    cdef size_t _n

    def __cinit__(self, size_t n):
        self._n = 0
        self._ptr = <GEOSGeometry**>malloc(n * sizeof(GEOSGeometry*))
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        for i in range(n):
            self._ptr[i] = NULL

    cdef assign_into(self, GEOSContextHandle_t handle, out, int64_t out_offset, size_t n):
        for i in range(n):
            if self._ptr[i] != NULL:
                geom = PyGEOS_CreateGeometry(self._ptr[i], handle)
                self._ptr[i] = NULL
                out[out_offset + i] = geom
            else:
                out[out_offset + i] = None

    def __dealloc__(self):
        with get_geos_handle() as handle:
            for i in range(self._n):
                if self._ptr[i] != NULL:
                    GEOSGeom_destroy_r(handle, self._ptr[i])

        if self._ptr != NULL:
            free(self._ptr)


cdef class GeometryArrayIterator:
    cdef object _obj
    cdef int64_t _target_chunk_size
    cdef const GEOSGeometry* geometries[1024]

    def __cinit__(self, obj):
        self._obj = obj
        memset(self.geometries, 0, sizeof(self.geometries))
        self._target_chunk_size = 1024

    def __iter__(self):
        cdef int64_t start = 0
        cdef int64_t end
        cdef int64_t length
        cdef GEOSGeometry* geom

        while start < len(self._obj):
            end = start + self._target_chunk_size
            if end > len(self._obj):
                end = len(self._obj)
            length = end - start

            for i in range(length):
                if not PyGEOS_GetGEOSGeometry(<PyObject*>(self._obj[start + i]), &geom):
                    raise RuntimeError(
                        f"PyGEOS_GetGEOSGeometry(obj[{start + i}]) failed")
                self.geometries[i] = geom

            yield length
            start += self._target_chunk_size


cdef class SchemaCalculator:
    cdef GeoArrowGEOSSchemaCalculator* _ptr

    ENCODING_UNKNOWN = GeoArrowGEOSEncoding.GEOARROW_GEOS_ENCODING_UNKNOWN
    ENCODING_WKB = GeoArrowGEOSEncoding.GEOARROW_GEOS_ENCODING_WKB
    ENCODING_WKT = GeoArrowGEOSEncoding.GEOARROW_GEOS_ENCODING_WKT
    ENCODING_GEOARROW = GeoArrowGEOSEncoding.GEOARROW_GEOS_ENCODING_GEOARROW
    ENCODING_GEOARROW_INTERLEAVED = GeoArrowGEOSEncoding.GEOARROW_GEOS_ENCODING_GEOARROW_INTERLEAVED

    def __cinit__(self):
        self._ptr = NULL
        cdef int rc = GeoArrowGEOSSchemaCalculatorCreate(&self._ptr)
        if rc != GEOARROW_GEOS_OK:
            raise GeoArrowGEOSException("GeoArrowGEOSSchemaCalculatorCreate()", rc, "<none>")

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSSchemaCalculatorDestroy(self._ptr)

    def ingest_wkb_type(self, int32_t[:] wkb_types):
        GeoArrowGEOSSchemaCalculatorIngest(self._ptr, &(wkb_types[0]), len(wkb_types))

    def ingest_geometry(self, object geometries):
        cdef int32_t wkb_types[1024]
        cdef GeometryArrayIterator array_iterator = GeometryArrayIterator(geometries)

        with get_geos_handle() as handle:
            for chunk_size in array_iterator:
                for i in range(chunk_size):
                    wkb_types[i] = GeoArrowGEOSWKBType(handle, array_iterator.geometries[i])
                GeoArrowGEOSSchemaCalculatorIngest(self._ptr, wkb_types, chunk_size)

    def finish(self, int32_t encoding):
        cdef ArrowSchema* schema
        schema_capsule = alloc_c_schema(&schema)

        cdef int rc = GeoArrowGEOSSchemaCalculatorFinish(
            self._ptr, <GeoArrowGEOSEncoding>encoding, schema)
        if rc != GEOARROW_GEOS_OK:
            raise GeoArrowGEOSException("GeoArrowGEOSSchemaCalculatorFinish()", rc, "<none>")
        return schema_capsule

    @staticmethod
    def from_wkb_type(int32_t encoding, int32_t wkb_type = 0):
        cdef ArrowSchema* schema
        schema_capsule = alloc_c_schema(&schema)

        cdef int rc = GeoArrowGEOSMakeSchema(encoding, wkb_type, schema)
        if rc != GEOARROW_GEOS_OK:
            raise GeoArrowGEOSException("GeoArrowGEOSMakeSchema()", rc, "<none>")
        return schema_capsule


cdef class ArrayBuilder:
    cdef GEOSContextHandle_t _geos_handle
    cdef GeoArrowGEOSArrayBuilder* _ptr
    cdef list _chunks_out

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._geos_handle = GEOS_init_r()
        cdef int rc = GeoArrowGEOSArrayBuilderCreate(self._geos_handle, schema, &self._ptr)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error("GeoArrowGEOSArrayReaderCreate()", rc)

        self._chunks_out = []

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSArrayBuilderDestroy(self._ptr)
        GEOS_finish_r(self._geos_handle)

    def _finish_chunk(self, append_empty=False):
        cdef ArrowArray* array
        chunk = alloc_c_array(&array)

        cdef int rc = GeoArrowGEOSArrayBuilderFinish(self._ptr, array)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error("GeoArrowGEOSArrayBuilderFinish()", rc)

        if array.length > 0 or append_empty:
            self._chunks_out.append(chunk)

    def append(self, object geometries):
        if len(geometries) == 0:
            return

        self._assert_valid()

        cdef int rc
        cdef size_t n_appended = 0
        cdef int64_t n_appended_total = 0
        cdef int64_t c_chunk_size

        cdef GeometryArrayIterator array_iterator = GeometryArrayIterator(geometries)
        for chunk_size in array_iterator:
            c_chunk_size = chunk_size
            with nogil:
                rc = GeoArrowGEOSArrayBuilderAppend(
                    self._ptr,
                    array_iterator.geometries,
                    c_chunk_size,
                    &n_appended
                )

            # For EOVERFLOW (e.g., WKB >2GB reached) we could add a chunk and attempt
            # appending the rest.
            if rc != GEOARROW_GEOS_OK:
                self._raise_last_error("GeoArrowGEOSArrayBuilderAppend()", rc)

            n_appended_total += n_appended

        self._finish_chunk()
        return n_appended_total

    def finish(self, ensure_non_empty=False):
        self._assert_valid()
        if len(self._chunks_out) == 0 and ensure_non_empty:
            self._finish_chunk(append_empty=True)

        out = self._chunks_out
        self._chunks_out = []
        return out

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("GeoArrowGEOSArrayBuilder not valid")

    def _last_error(self):
        cdef const char* msg = NULL
        if self._ptr == NULL:
            msg = NULL
        else:
            msg = GeoArrowGEOSArrayBuilderGetLastError(self._ptr)

        if msg == NULL:
            msg = "<NULL>"

        return msg.decode("UTF-8")


cdef class ArrayReader:
    cdef GEOSContextHandle_t _geos_handle
    cdef GeoArrowGEOSArrayReader* _ptr
    cdef object _out
    cdef int64_t _out_size

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._geos_handle = GEOS_init_r()
        cdef int rc = GeoArrowGEOSArrayReaderCreate(self._geos_handle, schema, &self._ptr)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error("GeoArrowGEOSArrayReaderCreate()", rc)

        self._out = np.array([], dtype="O")

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSArrayReaderDestroy(self._ptr)
        GEOS_finish_r(self._geos_handle)

    def reserve(self, int64_t additional_size):
        if (self._out_size + additional_size) <= len(self._out):
            return

        out = np.repeat([None], self._out_size + additional_size)
        out[:self._out_size] = self._out
        self._out = out

    def read(self, object array_capsule):
        self._assert_valid()

        cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(array_capsule, "arrow_array")
        cdef size_t n_out = 0
        cdef GEOSGeometryArray out = GEOSGeometryArray(array.length)
        cdef int rc
        with nogil:
            rc = GeoArrowGEOSArrayReaderRead(self._ptr, array, 0, array.length, out._ptr, &n_out)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error("GeoArrowGEOSArrayReaderRead()", rc)

        if n_out != (<size_t>array.length):
            raise RuntimeError(f"Expected {array.length} values but got {n_out}: {self._last_error()}")

        self.reserve(n_out)
        out.assign_into(self._geos_handle, self._out, self._out_size, n_out)
        self._out_size += n_out
        return n_out

    def finish(self):
        if self._out_size == len(self._out):
            out = self._out
        else:
            out = self._out[:self._out_size]

        self._out = np.array([], dtype="O")
        self._out_size = 0
        return out

    def _assert_valid(self):
        if self._ptr == NULL:
           raise RuntimeError("GeoArrowGEOSArrayReader not valid")

    def _last_error(self):
        cdef const char* msg = NULL
        if self._ptr == NULL:
            msg = NULL
        else:
           msg = GeoArrowGEOSArrayReaderGetLastError(self._ptr)

        if msg == NULL:
            msg = "<NULL>"

        return msg.decode("UTF-8")

    def _raise_last_error(self, what, code):
        raise GeoArrowGEOSException(what, code, self._last_error())
