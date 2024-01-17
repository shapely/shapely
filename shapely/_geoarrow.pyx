
from cpython cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdint cimport int32_t, int64_t, uintptr_t
from libc.stdlib cimport free, malloc

from shapely._geos cimport (
    GEOSContextHandle_t,
    GEOSGeom_destroy_r,
    GEOSGeometry,
    get_geos_handle,
)
from shapely._pygeos_api cimport import_shapely_c_api, PyGEOS_CreateGeometry, PyGEOS_GetGEOSGeometry

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


class GeoArrowGEOSException(Exception):

    def __init__(self, what, code, msg):
        super().__init__(f"{what} failed with code {code}: {msg}")


cdef class GEOSGeometryArray:
    cdef GEOSGeometry** _ptr
    cdef size_t _n

    def __cinit__(self, size_t n):
        self._n = 0
        self._ptr = <GEOSGeometry**>malloc(n)
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        for i in range(n):
            self._ptr[i] = NULL

    cdef to_pylist(self, GEOSContextHandle_t handle, size_t n):
        out = []
        for i in range(n):
            if self._ptr[i] != NULL:
                geom = PyGEOS_CreateGeometry(self._ptr[i], handle)
                self._ptr[i] = NULL
                out.append(geom)
            else:
                out.append(None)

        return out

    def __dealloc__(self):
        with get_geos_handle() as handle:
            for i in range(self._n):
                if self._ptr[i] != NULL:
                    GEOSGeom_destroy_r(handle, self._ptr[i])

        if self._ptr != NULL:
            free(self._ptr)


cdef class ArrowArrayHolder:
    cdef ArrowArray _array

    def __cinit__(self):
        self._array.release = NULL

    def __dealloc__(self):
        if self._array.release != NULL:
            self._array.release(&self._array)

    def _addr(self):
        return <uintptr_t>(&self._array)


cdef class ArrowSchemaHolder:
    cdef ArrowSchema _schema

    def __cinit__(self):
        self._schema.release = NULL

    def __dealloc__(self):
        if self._schema.release != NULL:
            self._schema.release(&self._schema)

    def _addr(self):
        return <uintptr_t>(&self._schema)

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
        cdef int64_t n_wkb_types = 0
        cdef GEOSGeometry* geos_geom

        n_geometries = len(geometries)
        n_chunks = n_geometries // 1024
        cdef int64_t i = 0

        with get_geos_handle() as handle:
            for chunk_in_i in range(n_chunks):
                for chunk_i in range(1024):
                    if not PyGEOS_GetGEOSGeometry(<PyObject*>(geometries[i]), &geos_geom):
                        raise RuntimeError(
                            f"PyGEOS_GetGEOSGeometry(geometries[{i}]) failed")
                    wkb_types[chunk_i] = GeoArrowGEOSWKBType(handle, geos_geom)
                    i += 1
                n_wkb_types = 1024
                GeoArrowGEOSSchemaCalculatorIngest(self._ptr, wkb_types, n_wkb_types)

            for chunk_i in range(n_geometries % 1024):
                if not PyGEOS_GetGEOSGeometry(<PyObject*>(geometries[i]), &geos_geom):
                    raise RuntimeError(
                        f"PyGEOS_GetGEOSGeometry(geometries[{i}]) failed")
                wkb_types[chunk_i] = GeoArrowGEOSWKBType(handle, geos_geom)
                i += 1

            n_wkb_types = n_geometries % 1024
            GeoArrowGEOSSchemaCalculatorIngest(self._ptr, wkb_types, n_wkb_types)

    def finish(self, int32_t encoding):
        cdef ArrowSchemaHolder out = ArrowSchemaHolder()
        cdef int rc = GeoArrowGEOSSchemaCalculatorFinish(
            self._ptr, <GeoArrowGEOSEncoding>encoding, &(out._schema))
        if rc != GEOARROW_GEOS_OK:
            raise GeoArrowGEOSException("GeoArrowGEOSSchemaCalculatorFinish()", rc, "<none>")
        return out

    @staticmethod
    def from_wkb_type(int32_t encoding, int32_t wkb_type = 0):
        cdef ArrowSchemaHolder out = ArrowSchemaHolder()
        cdef int rc = GeoArrowGEOSMakeSchema(encoding, wkb_type, &(out._schema))
        if rc != GEOARROW_GEOS_OK:
            raise GeoArrowGEOSException("GeoArrowGEOSMakeSchema()", rc, "<none>")
        return out


cdef class ArrayBuilder:
    cdef get_geos_handle _handle
    cdef GEOSContextHandle_t _geos_handle
    cdef GeoArrowGEOSArrayBuilder* _ptr
    cdef list _chunks_out

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._handle = get_geos_handle()
        self._geos_handle = self._handle.__enter__()
        cdef int rc = GeoArrowGEOSArrayBuilderCreate(self._geos_handle, schema, &self._ptr)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc, "GeoArrowGEOSArrayReaderCreate()")

        self._chunks_out = []

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSArrayBuilderDestroy(self._ptr)
        self._handle.__exit__(None, None, None)

    def _finish_chunk(self):
        cdef ArrowArrayHolder chunk = ArrowArrayHolder()
        cdef int rc = GeoArrowGEOSArrayBuilderFinish(self._ptr, &chunk._array)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc, "GeoArrowGEOSArrayBuilderFinish()")

        if chunk._array.length > 0:
            self._chunks_out.append(chunk)

    def append(self, object geometries):
        if len(geometries) == 0:
            return

        self._assert_valid()

        cdef GEOSGeometry* input_chunk[1024]
        cdef int rc
        cdef size_t n_appended = 0
        cdef int64_t n_appended_total = 0
        cdef int64_t i = 0

        for chunk_in_i in range(len(geometries) // 1024):
            for chunk_i in range(1024):
                if not PyGEOS_GetGEOSGeometry(<PyObject*>(geometries[i]), &(input_chunk[chunk_i])):
                    raise RuntimeError(f"PyGEOS_GetGEOSGeometry(geometries[{i}])")
                i += 1

            with nogil:
                rc = GeoArrowGEOSArrayBuilderAppend(self._ptr, input_chunk, 1024, &n_appended)
            # TODO: check EOVERFLOW, e.g., WKB >2GB reached so we add a chunk and try again
            if rc != GEOARROW_GEOS_OK:
                self._raise_last_error(rc, "GeoArrowGEOSArrayBuilderAppend()")

            n_appended_total += n_appended

        cdef int64_t n_remaining = len(geometries) % 1024
        for chunk_i in range(n_remaining):
            if not PyGEOS_GetGEOSGeometry(<PyObject*>(geometries[i]), &(input_chunk[chunk_i])):
                raise RuntimeError(f"PyGEOS_GetGEOSGeometry() failed at last chunk[{chunk_i}]")
            i += 1

        with nogil:
                rc = GeoArrowGEOSArrayBuilderAppend(self._ptr, input_chunk, n_remaining, &n_appended)
        # TODO: check EOVERFLOW, e.g., WKB >2GB reached so we add a chunk and try again
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc, "GeoArrowGEOSArrayBuilderAppend()")

            n_appended_total += n_appended

        self._finish_chunk()
        return n_appended_total


    def finish(self):
        self._assert_valid()
        return self._chunks_out

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
    cdef get_geos_handle _handle
    cdef GEOSContextHandle_t _geos_handle
    cdef GeoArrowGEOSArrayReader* _ptr

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._handle = get_geos_handle()
        self._geos_handle = self._handle.__enter__()
        cdef int rc = GeoArrowGEOSArrayReaderCreate(self._geos_handle, schema, &self._ptr)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc, "GeoArrowGEOSArrayReaderCreate()")

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSArrayReaderDestroy(self._ptr)
        self._handle.__exit__(None, None, None)

    def read(self, object array_capsule):
        self._assert_valid()

        cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(array_capsule, "arrow_array")
        cdef size_t n_out = 0
        cdef GEOSGeometryArray out = GEOSGeometryArray(array.length)
        cdef int rc
        with nogil:
            rc = GeoArrowGEOSArrayReaderRead(self._ptr, array, 0, array.length, out._ptr, &n_out)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc, "GeoArrowGEOSArrayReaderRead()")

        if n_out != array.length:
            raise RuntimeError(f"Expected {array.length} values but got {n_out}: {self._last_error()}")

        return out.to_pylist(self._geos_handle, n_out)

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
