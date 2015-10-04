#!/usr/bin/env python

# Two environment variables influence this script.
#
# GEOS_LIBRARY_PATH: a path to a GEOS C shared library.
#
# GEOS_CONFIG: the path to a geos-config program that points to GEOS version,
# headers, and libraries.
#
# NB: within this setup scripts, software versions are evaluated according
# to https://www.python.org/dev/peps/pep-0440/.

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

from distutils.version import StrictVersion as Version

# Get geos_version from GEOS dynamic library, which depends on
# GEOS_LIBRARY_PATH and/or GEOS_CONFIG environment variables
from shapely._buildcfg import geos_version_string, geos_version, \
        geos_config, get_geos_config

logging.basicConfig()
log = logging.getLogger(__file__)

# python -W all setup.py ...
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG

# Get the version from the shapely module
shapely_version = None
with open('shapely/__init__.py', 'r') as fp:
    for line in fp:
        if line.startswith("__version__"):
            shapely_version = Version(
                line.split("=")[1].strip().strip("\"'"))
            break

if not shapely_version:
    raise ValueError("Could not determine Shapely's version")

# Fail installation if the GEOS shared library does not meet the minimum
# version. We ship it with Shapely for Windows, so no need to check on
# that platform.
log.debug('GEOS shared library: %s %s', geos_version_string, geos_version)
if (set(sys.argv).intersection(['install', 'build', 'build_ext']) and
        shapely_version >= Version('1.3') and
        geos_version < (3, 3)):
    log.critical(
        "Shapely >= 1.3 requires GEOS >= 3.3. "
        "Install GEOS 3.3+ and reinstall Shapely.")
    sys.exit(1)

# Handle UTF-8 encoding of certain text files.
open_kwds = {}
if sys.version_info >= (3,):
    open_kwds['encoding'] = 'utf-8'

with open('VERSION.txt', 'w', **open_kwds) as fp:
    fp.write(str(shapely_version))

with open('README.rst', 'r', **open_kwds) as fp:
    readme = fp.read()

with open('CREDITS.txt', 'r', **open_kwds) as fp:
    credits = fp.read()

with open('CHANGES.txt', 'r', **open_kwds) as fp:
    changes = fp.read()

long_description = readme + '\n\n' + credits + '\n\n' + changes

setup_args = dict(
    name                = 'Shapely',
    version             = str(shapely_version),
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

# Add DLLs for Windows
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


# Prepare build opts and args for the speedups extension module.
include_dirs = []
library_dirs = []
libraries = []
extra_link_args = []

try:
    # Get the version from geos-config. Show error if this version tuple is
    # different to the GEOS version loaded from the dynamic library.
    geos_config_version_string = get_geos_config('--version')
    res = re.findall(r'(\d+)\.(\d+)\.(\d+)', geos_config_version_string)
    geos_config_version = tuple(int(x) for x in res[0])

    if geos_config_version != geos_version:
        log.error("The GEOS dynamic library version is %s %s,",
                  geos_version_string, geos_version)
        log.error("but the version reported by %s is %s %s.", geos_config,
                  geos_config_version_string, geos_config_version)
        sys.exit(1)
except OSError as ex:
    log.error(ex)
    log.error('Cannot find geos-config to get headers and check version.')
    log.error('If available, specify a path to geos-config with a '
              'GEOS_CONFIG environment variable')
    geos_config = None

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

cmd_classes = setup_args.setdefault('cmdclass', {})

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

    if 'build_ext' in cmd_classes:
        del cmd_classes['build_ext']

    setup(**setup_args)

    log.warn(BUILD_EXT_WARNING)
    log.info("Plain-Python installation succeeded.")
