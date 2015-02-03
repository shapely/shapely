"""
Proxies for the GEOS shared library, versions, and configure options
TODO: find a flexible way to load a dynamic library from a specified path
"""

__all__ = [
    'lgeos', 'geos_version_string', 'geos_version', 'geos_capi_version'
]
import os
import re
import sys
from ctypes import CDLL, cdll, c_void_p, c_char_p
from ctypes.util import find_library


def load_dll(libname, fallbacks=[]):
    '''Load GEOS dynamic library'''
    lib = find_library(libname)
    if lib is not None:
        try:
            return CDLL(lib)
        except OSError:
            pass
    for name in fallbacks:
        try:
            return CDLL(name)
        except OSError:
            # move on to the next fallback
            pass
    raise OSError(
        "Could not find library %s or load any of its variants %s" % (
            libname, fallbacks))


# Load dynamic library into a 'lgeos' object, which is system dependant

if sys.platform.startswith('linux'):
    lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = load_dll('c').free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == 'darwin':
    # First test to see if this is a delocated wheel with a GEOS dylib.
    geos_whl_dylib = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '.dylibs/libgeos_c.1.dylib'))
    if os.path.exists(geos_whl_dylib):
        lgeos = CDLL(geos_whl_dylib)
    else:
        if hasattr(sys, 'frozen'):
            # .app file from py2app
            alt_paths = [os.path.join(os.environ['RESOURCEPATH'],
                         '..', 'Frameworks', 'libgeos_c.dylib')]
        else:
            alt_paths = [
                # The Framework build from Kyng Chaos
                "/Library/Frameworks/GEOS.framework/Versions/Current/GEOS",
                # macports
                '/opt/local/lib/libgeos_c.dylib',
            ]
        lgeos = load_dll('geos_c', fallbacks=alt_paths)

    free = load_dll('c').free
    free.argtypes = [c_void_p]
    free.restype = None

elif sys.platform == 'win32':
    try:
        egg_dlls = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   "DLLs"))
        wininst_dlls = os.path.abspath(os.__file__ + "../../../DLLs")
        original_path = os.environ['PATH']
        os.environ['PATH'] = "%s;%s;%s" % \
            (egg_dlls, wininst_dlls, original_path)
        lgeos = CDLL("geos.dll")
    except (ImportError, WindowsError, OSError):
        raise

    def free(m):
        try:
            cdll.msvcrt.free(m)
        except WindowsError:
            # TODO: http://web.archive.org/web/20070810024932/
            #     + http://trac.gispython.org/projects/PCL/ticket/149
            pass

elif sys.platform == 'sunos5':
    lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = CDLL('libc.so.1').free
    free.argtypes = [c_void_p]
    free.restype = None
else:  # other *nix systems
    lgeos = load_dll('geos_c', fallbacks=['libgeos_c.so.1', 'libgeos_c.so'])
    free = load_dll('c', fallbacks=['libc.so.6']).free
    free.argtypes = [c_void_p]
    free.restype = None

# TODO: what to do with 'free'? It isn't used.


def _geos_version():
    # extern const char GEOS_DLL *GEOSversion();
    GEOSversion = lgeos.GEOSversion
    GEOSversion.restype = c_char_p
    GEOSversion.argtypes = []
    # #define GEOS_CAPI_VERSION "@VERSION@-CAPI-@CAPI_VERSION@"
    geos_version_string = GEOSversion()
    if sys.version_info[0] >= 3:
        geos_version_string = geos_version_string.decode('ascii')
    res = re.findall(r'(\d+)\.(\d+)\.(\d+)', geos_version_string)
    assert len(res) == 2, res
    geos_version = tuple(int(x) for x in res[0])
    capi_version = tuple(int(x) for x in res[1])
    return geos_version_string, geos_version, capi_version

geos_version_string, geos_version, geos_capi_version = _geos_version()
