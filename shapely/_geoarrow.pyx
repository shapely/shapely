
from libc.stdint cimport uintptr_t
from cpython.pycapsule cimport PyCapsule_GetPointer

from shapely._geos cimport GEOSContextHandle_t, GEOSGeometry, get_geos_handle


from shapely._pygeos_api cimport (
    import_shapely_c_api,
    PyGEOS_CreateGeometry,
)

# initialize Shapely C API
import_shapely_c_api()

cdef extern from "geoarrow_geos.h":
    struct ArrowSchema
    struct ArrowArray
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


cdef class ArrayReader:
    cdef get_geos_handle _handle
    cdef GeoArrowGEOSArrayReader* _ptr

    def __cinit__(self, object schema_capsule):
        cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        self._ptr = NULL
        self._handle = get_geos_handle()
        cdef int rc = GeoArrowGEOSArrayReaderCreate(self._handle.__enter__(), schema, &self._ptr)
        if rc != GEOARROW_GEOS_OK:
            self._raise_last_error(rc)

    def __dealloc__(self):
        if self._ptr != NULL:
            GeoArrowGEOSArrayReaderDestroy(self._ptr)
        self._handle.__exit__(None, None, None)

    def _raise_last_error(self, what, code):
        cdef const char* msg = NULL
        if self._ptr == NULL:
            msg = NULL
        else:
           msg = GeoArrowGEOSArrayReaderGetLastError(self._ptr)

        if msg == NULL:
            msg = "<NULL>"

        raise GeoArrowGEOSException(what, code, msg.decode("UTF-8"))
