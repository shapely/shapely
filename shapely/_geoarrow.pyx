
from cpython cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdint cimport int64_t, uintptr_t
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
    struct ArrowSchema
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
        n_appended_total = 0
        cdef size_t n_full_chunks_in = len(geometries) // 1024

        for chunk_in_i in range(n_full_chunks_in):
            i_begin = (chunk_in_i * 1024)
            i_end = i_begin + 1024
            for geom, chunk_i in zip(geometries[i_begin:i_end], range(1024)):
                if not PyGEOS_GetGEOSGeometry(<PyObject*>geom, &(input_chunk[chunk_i])):
                    raise RuntimeError(f"PyGEOS_GetGEOSGeometry() failed at chunk {chunk_in_i}[{chunk_i}]")

            with nogil:
                rc = GeoArrowGEOSArrayBuilderAppend(self._ptr, input_chunk, 1024, &n_appended)
            # TODO: check EOVERFLOW, e.g., WKB >2GB reached so we add a chunk and try again
            if rc != GEOARROW_GEOS_OK:
                self._raise_last_error(rc, "GeoArrowGEOSArrayBuilderAppend()")

            n_appended_total += n_appended

        # Last chunk
        i_begin = n_full_chunks_in * 1024
        i_end = len(geometries)
        cdef size_t n_remaining = i_end - i_begin

        if n_remaining > 0:
            for geom, chunk_i in zip(geometries[i_begin:i_end], range(n_remaining)):
                if not PyGEOS_GetGEOSGeometry(<PyObject*>geom, &(input_chunk[chunk_i])):
                        raise RuntimeError(f"PyGEOS_GetGEOSGeometry() failed at last chunk[{chunk_i}]")
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
