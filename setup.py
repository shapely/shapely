#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import glob
import errno
import shutil
import platform
import subprocess
from distutils.core import setup
from distutils.cmd import Command
from distutils.extension import Extension
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
from distutils.command.build_ext import build_ext as distutils_build_ext
from unittest import TextTestRunner, TestLoader


class test(Command):
    """Run unit tests after in-place build"""
    description = __doc__
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import shapely.tests
        except ImportError:
            self.run_command('build_ext')
            import shapely.tests
        tests = TestLoader().loadTestsFromName('test_suite', shapely.tests)
        runner = TextTestRunner(verbosity=2)
        result = runner.run(tests)

        if result.wasSuccessful():
            sys.exit(0)
        else:
            sys.exit(1)

# Parse the version from the shapely module
version = None
for line in open('shapely/__init__.py', 'r'):
    if "__version__" in line:
        exec(line.replace('_', ''))
        break

assert version is not None

open('VERSION.txt', 'w').write(version)

readme_text = open('README.rst', 'r').read()
readme_text = readme_text.replace(".. include:: CREDITS.txt", "")

f = open('CREDITS.txt', 'r')
credits = f.read()

f = open('CHANGES.txt', 'r')
changes_text = f.read()

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
        'shapely.tests',
    ],
    cmdclass            = {'test': test},
    classifiers         = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: GIS',
    ],
)

# Add DLLs for Windows
if sys.platform == 'win32':
    try:
        os.mkdir('shapely/DLLs')
    except OSError as ex:
        if ex.errno != errno.EEXIST: raise
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

class build_ext(distutils_build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            distutils_build_ext.run(self)
        except DistutilsPlatformError as x:
            raise BuildFailed(x)

    def build_extension(self, ext):
        try:
            distutils_build_ext.build_extension(self, ext)
        except ext_errors as x:
            raise BuildFailed(x)

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
        libraries=libraries )]

try:
    # try building with speedups
    setup_args['cmdclass']['build_ext'] = build_ext
    setup(
        ext_modules=ext_modules,
        **setup_args
    )
except BuildFailed as ex:
    BUILD_EXT_WARNING = "Warning: The C extension could not be compiled, speedups are not enabled."
    print(ex)
    print(BUILD_EXT_WARNING)
    print("Failure information, if any, is above.")
    print("I'm retrying the build without the C extension now.")

    setup(**setup_args)

    print(BUILD_EXT_WARNING)
    print("Plain-Python installation succeeded.")

