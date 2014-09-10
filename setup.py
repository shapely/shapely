#!/usr/bin/env python

from __future__ import print_function

try:
    # If possible, use setuptools
    from setuptools import setup
    from setuptools.extension import Extension
    from setuptools.command.build_ext import build_ext as distutils_build_ext
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    from distutils.command.build_ext import build_ext as distutils_build_ext
from distutils.cmd import Command
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
from distutils.sysconfig import get_config_var
import errno
import glob
import os
import platform
import shutil
import subprocess
import sys


# Get the version from the shapely module
version = None
with open('shapely/__init__.py', 'r') as fp:
    for line in fp:
        if "__version__" in line:
            exec(line.replace('_', ''))
            break
if version is None:
    raise ValueError("Could not determine Shapely's version")

# Handle UTF-8 encoding of certain text files.
open_kwds = {}
if sys.version_info > (3,):
    open_kwds['encoding'] = 'utf-8'

with open('VERSION.txt', 'w', **open_kwds) as fp:
    fp.write(version)

with open('README.rst', 'r', **open_kwds) as fp:
    readme_text = fp.read()
readme_text = readme_text.replace(".. include:: CREDITS.txt", "")

with open('CREDITS.txt', 'r', **open_kwds) as fp:
    credits = fp.read()

with open('CHANGES.txt', 'r', **open_kwds) as fp:
    changes_text = fp.read()

setup_args = dict(
    name                = 'Shapely',
    version             = version,
    requires            = ['Python (>=2.6)', 'libgeos_c (>=3.1)'],
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'https://github.com/Toblerity/Shapely',
    long_description    = readme_text + "\n" + credits + "\n" + changes_text,
    packages            = [
        'shapely',
        'shapely.geometry',
        'shapely.algorithms',
        'shapely.examples',
        'shapely.speedups',
        'shapely.vectorized',
    ],
    cmdclass = {},
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
    data_files         = [('shapely', ['shapely/_geos.pxi'])]
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
elif sys.platform == 'win32':
    libraries = ['geos']
else:
    libraries = ['geos_c']


if os.path.exists("MANIFEST.in"):
    pyx_file = "shapely/speedups/_speedups.pyx"
    c_file = "shapely/speedups/_speedups.c"

    force_cython = False
    if 'sdist' in sys.argv:
        force_cython = True

    try:
        if (force_cython or not os.path.exists(c_file)
                or os.path.getmtime(pyx_file) > os.path.getmtime(c_file)):
            print("Updating C extension with Cython.", file=sys.stderr)
            subprocess.check_call(["cython", "shapely/speedups/_speedups.pyx"])
    except (subprocess.CalledProcessError, OSError):
        print("Warning: Could not (re)create C extension with Cython.",
              file=sys.stderr)
        if force_cython:
            raise
    if not os.path.exists("shapely/speedups/_speedups.c"):
        print("Warning: speedup extension not found", file=sys.stderr)

ext_modules = [
    Extension(
        "shapely.speedups._speedups",
        ["shapely/speedups/_speedups.c"],
        libraries=libraries,
        include_dirs=[get_config_var('INCLUDEDIR')],),
]

try:
    import numpy as np
    from Cython.Distutils import build_ext as cython_build_ext
    from distutils.extension import Extension as DistutilsExtension

    cmd_classes = setup_args.setdefault('cmdclass', {})
    if 'build_ext' in cmd_classes:
        raise ValueError('We need to put the Cython build_ext in '
                         'cmd_classes, but it is already defined.')
    cmd_classes['build_ext'] = cython_build_ext

    ext_modules.append(DistutilsExtension("shapely.vectorized._vectorized",
                                 sources=["shapely/vectorized/_vectorized.pyx"],
                                 libraries=libraries + [np.get_include()],
                                 include_dirs=[get_config_var('INCLUDEDIR'),
                                               np.get_include()],
                                 ))
except ImportError:
    print("Numpy or Cython not available, shapely.vectorized submodule not "
          "being built.")


try:
    # try building with speedups
    existing_build_ext = setup_args['cmdclass'].get('build_ext', distutils_build_ext)
    setup_args['cmdclass']['build_ext'] = construct_build_ext(existing_build_ext)
    setup(
        ext_modules=ext_modules,
        **setup_args
    )
except BuildFailed as ex:
    BUILD_EXT_WARNING = "Warning: The C extension could not be compiled, " \
                        "speedups are not enabled."
    print(ex)
    print(BUILD_EXT_WARNING)
    print("Failure information, if any, is above.")
    print("I'm retrying the build without the C extension now.")

    setup(**setup_args)

    print(BUILD_EXT_WARNING)
    print("Plain-Python installation succeeded.")
