#!/usr/bin/env python

# One environment variable influence this script.
#
# GEOS_CONFIG: the path to a geos-config program that points to GEOS version,
# headers, and libraries.

import errno
import glob
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
try:
    # If possible, use setuptools
    from setuptools import setup
    from setuptools.extension import Extension
    from setuptools.command.build_ext import build_ext as distutils_build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext as distutils_build_ext
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
from distutils.sysconfig import get_config_var

logging.basicConfig()
log = logging.getLogger(__file__)

# python -W all setup.py ...
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG

# Get the version from the shapely module
version = None
with open('shapely/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("\"'")
            break
if not version:
    raise ValueError("Could not determine Shapely's version")
shapely_version = tuple(int(x) for x in version.split('.'))

# Handle UTF-8 encoding of certain text files.
open_kwds = {}
if sys.version_info[0] > 3:
    open_kwds['encoding'] = 'utf-8'

with open('VERSION.txt', 'w', **open_kwds) as fp:
    fp.write(version)

with open('README.rst', 'r', **open_kwds) as fp:
    readme = fp.read()

with open('CREDITS.txt', 'r', **open_kwds) as fp:
    credits = fp.read()

with open('CHANGES.txt', 'r', **open_kwds) as fp:
    changes = fp.read()

long_description = readme + '\n\n' + credits + '\n\n' + changes

setup_args = dict(
    name                = 'Shapely',
    version             = version,
    requires            = ['Python (>=2.6)', 'libgeos_c (>=3.3)'],
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'https://github.com/Toblerity/Shapely',
    long_description    = long_description,
    packages            = [
        'shapely',
        'shapely.geometry',
        'shapely.algorithms',
        'shapely.examples',
        'shapely.speedups',
        'shapely.vectorized',
    ],
    classifiers         = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    data_files         = [('shapely', ['shapely/_geos.pxi'])],
    cmdclass           = {},
)

# Add DLLs to Windows packages.
if sys.platform == 'win32':
    try:
        os.mkdir('shapely/DLLs')
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
    if '(AMD64)' in sys.version:
        for dll in glob.glob('DLLs_AMD64_VC9/*.dll'):
            shutil.copy(dll, 'shapely/DLLs')
    elif sys.version_info[0:2] == (2, 5):
        for dll in glob.glob('DLLs_x86_VC7/*.dll'):
            shutil.copy(dll, 'shapely/DLLs')
    else:
        for dll in glob.glob('DLLs_x86_VC9/*.dll'):
            shutil.copy(dll, 'shapely/DLLs')
    setup_args.update(
        package_data={'shapely': ['shapely/DLLs/*.dll']},
        include_package_data=True,
    )

# Get configuartion information for GEOS library using command line tool
geos_config = os.environ.get('GEOS_CONFIG', 'geos-config')
log.debug('geos_config: %s', geos_config)


def get_geos_config(option):
    try:
        stdout, stderr = subprocess.Popen(
            [geos_config, option],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    except OSError as ex:
        # e.g., [Errno 2] No such file or directory
        raise OSError(
            'Could not find geos-config %r: %s' % (geos_config, ex))
    if stderr and not stdout:
        raise ValueError(stderr.strip())
    result = stdout.strip()
    log.debug('%s %s: %s', geos_config, option, result)
    return result

try:
    geos_version_string = get_geos_config('--version')
    res = re.findall(r'(\d+)\.(\d+)\.(\d+)', geos_version_string)
    geos_version = tuple(int(x) for x in res[0])
except OSError as ex:
    log.error(ex)
    log.error('Cannot determine GEOS library version or location')
    log.error('If available, specify a path to geos-config with a '
              'GEOS_CONFIG environment variable')
    geos_version = None
    geos_config = None

# Fail installation if we can't find a GEOS shared library with the right
# version. We ship it with Shapely for Windows, so no need to check on
# that platform.
if ('install' in sys.argv and geos_version and
        shapely_version >= (1, 3) and geos_version < (3, 3)):
    log.critical(
        "Shapely >= 1.3 requires GEOS >= 3.3. "
        "Install GEOS 3.3+ and reinstall Shapely.")
    sys.exit(1)

include_dirs = [get_config_var('INCLUDEDIR')]
library_dirs = []
libraries = []
extra_link_args = []

if geos_config:
    # Collect other options from GEOS
    for item in get_geos_config('--cflags').split():
        if item.startswith("-I"):
            include_dirs.extend(item[2:].split(":"))
    for item in get_geos_config('--clibs').split():
        if item.startswith("-L"):
            library_dirs.extend(item[2:].split(":"))
        elif item.startswith("-l"):
            libraries.append(item[2:])
        else:
            # e.g. -framework GEOS
            extra_link_args.append(item)


# Optional compilation of speedups
# setuptools stuff from Bob Ippolito's simplejson project
if sys.platform == 'win32' and sys.version_info > (2, 6):
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                  IOError)
else:
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)


class BuildFailed(Exception):
    pass


def construct_build_ext(build_ext):
    class WrappedBuildExt(build_ext):
        # This class allows C extension building to fail.

        def run(self):
            try:
                build_ext.run(self)
            except DistutilsPlatformError as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)
    return WrappedBuildExt


if (hasattr(platform, 'python_implementation')
        and platform.python_implementation() == 'PyPy'):
    # python_implementation is only available since 2.6
    ext_modules = []
    libraries = []


if os.path.exists("MANIFEST.in"):
    pyx_file = "shapely/speedups/_speedups.pyx"
    c_file = "shapely/speedups/_speedups.c"

    force_cython = False
    if 'sdist' in sys.argv:
        force_cython = True

    try:
        if (force_cython or not os.path.exists(c_file)
                or os.path.getmtime(pyx_file) > os.path.getmtime(c_file)):
            log.info("Updating C extension with Cython.")
            subprocess.check_call(["cython", "shapely/speedups/_speedups.pyx"])
    except (subprocess.CalledProcessError, OSError):
        log.warn("Could not (re)create C extension with Cython.")
        if force_cython:
            raise
    if not os.path.exists(c_file):
        log.warn("speedup extension not found")

ext_modules = [
    Extension(
        "shapely.speedups._speedups",
        ["shapely/speedups/_speedups.c"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ),
]

try:
    import numpy
    from Cython.Distutils import build_ext as cython_build_ext
    from distutils.extension import Extension as DistutilsExtension

    if 'build_ext' in setup_args['cmdclass']:
        raise ValueError('We need to put the Cython build_ext in '
                         'cmd_classes, but it is already defined.')
    setup_args['cmdclass']['build_ext'] = cython_build_ext

    include_dirs.append(numpy.get_include())
    libraries.append(numpy.get_include())

    ext_modules.append(DistutilsExtension(
        "shapely.vectorized._vectorized",
        sources=["shapely/vectorized/_vectorized.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    ))
except ImportError:
    log.info("Numpy or Cython not available, shapely.vectorized submodule "
             "not being built.")


try:
    # try building with speedups
    existing_build_ext = setup_args['cmdclass'].\
        get('build_ext', distutils_build_ext)
    setup_args['cmdclass']['build_ext'] = \
        construct_build_ext(existing_build_ext)
    setup(ext_modules=ext_modules, **setup_args)
except BuildFailed as ex:
    BUILD_EXT_WARNING = "The C extension could not be compiled, " \
                        "speedups are not enabled."
    log.warn(ex)
    log.warn(BUILD_EXT_WARNING)
    log.warn("Failure information, if any, is above.")
    log.warn("I'm retrying the build without the C extension now.")

    # Remove any previously defined build_ext command class.
    if 'build_ext' in setup_args['cmdclass']:
        del setup_args['cmdclass']['build_ext']

    setup(**setup_args)

    log.warn(BUILD_EXT_WARNING)
    log.info("Plain-Python installation succeeded.")
