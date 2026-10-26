from shapely import GEOSException
from shapely._pygeos_api cimport import_shapely_c_api, PyGEOS_InitGEOSContext, PyGEOS_InitGEOSErrorBuffer

# initialize Shapely C API
import_shapely_c_api()

cdef class get_geos_handle:
    '''This class provides a context manager that wraps the GEOS context handle.

    Example
    -------
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)
    '''
    cdef GEOSContextHandle_t __enter__(self):
        self.handle = PyGEOS_InitGEOSContext()
        self.last_error = PyGEOS_InitGEOSErrorBuffer()
        return self.handle

    def __exit__(self, type, value, traceback):
        # Check for GEOS errors in the threadlocal error buffer
        if self.last_error[0] != 0:
            raise GEOSException(self.last_error)
