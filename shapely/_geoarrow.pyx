
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdint cimport int64_t
from libc.stdlib cimport free, malloc

from shapely._geos cimport (
    GEOSContextHandle_t,
    GEOSGeom_destroy_r,
    GEOSGeometry,
    get_geos_handle,
)
from shapely._pygeos_api cimport import_shapely_c_api, PyGEOS_CreateGeometry

# initialize Shapely C API
import_shapely_c_api()

cdef extern from "geoarrow_geos.h" nogil:
    struct ArrowSchema
    struct ArrowArray:
        int64_t length
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


class GeoArrowGEOSException(Exception):

    def __init__(self, what, code, msg):
        super().__init__(f"{what} failed with code {code}: {msg}")


cdef class GEOSGeometryArray:
    cdef GEOSGeometry** _ptr
    cdef size_t _n

    def __cinit__(self,  size_t n):
        self._n = 0
        self._ptr = <GEOSGeometry**>malloc(n)
        if self._ptr == NULL:
            raise MemoryError()
        self._n = n
        for i in range(n):
            self._ptr[i] = NULL

    def to_pylist(self, size_t n):
        out = []
        with get_geos_handle() as handle:
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


cdef class ArrayReader:
    cdef get_geos_handle _handle
    cdef GeoArrowGEOSArrayReader* _ptr

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._handle = get_geos_handle()
        cdef int rc = GeoArrowGEOSArrayReaderCreate(self._handle.__enter__(), schema, &self._ptr)
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
            self._raise_last_error(rc, "GeoArrowGEOSArrayReaderCreate()")

        if n_out != array.length:
            raise RuntimeError(f"Expected {array.length} values but got {n_out}: {self._last_error()}")

        return out.to_pylist(n_out)

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
