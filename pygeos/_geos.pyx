cdef class get_geos_handle:
    '''This class provides a context manager that wraps the GEOS context handle.

    Example
    -------
    with get_geos_handle() as geos_handle:
        SomeGEOSFunc(geos_handle, ...<other params>)
    '''
    cdef GEOSContextHandle_t __enter__(self):
        self.handle = GEOS_init_r()
        return self.handle

    def __exit__(self, type, value, traceback):
        GEOS_finish_r(self.handle)


