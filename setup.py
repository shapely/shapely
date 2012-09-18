import warnings

try:
    from distribute_setup import use_setuptools
    use_setuptools()
except:
    warnings.warn(
    "Failed to import distribute_setup, continuing without distribute.",
    Warning)

from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
import glob
import os
import platform
from setuptools.extension import Extension
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as distutils_build_ext
import subprocess
import sys

# Parse the version from the shapely module
for line in open('shapely/__init__.py', 'rb'):
    if line.find("__version__") >= 0:
        version = line.split("=")[1].strip()
        version = version.strip('"')
        version = version.strip("'")
        continue

open('VERSION.txt', 'wb').write(version)

readme_text = open('README.rst', 'rb').read()
readme_text = readme_text.replace(".. include:: CREDITS.txt", "")

f = open('CREDITS.txt', 'rb')
credits = f.read()

f = open('CHANGES.txt', 'rb')
changes_text = f.read()

setup_args = dict(
    metadata_version    = '1.2',
    name                = 'Shapely',
    version             = version,
    requires_python     = '>=2.5,<3',
    requires_external   = 'libgeos_c (>=3.1)',
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'https://github.com/Toblerity/Shapely',
    long_description    = readme_text + "\n" + credits + "\n" + changes_text,
    packages            = find_packages(),
    test_suite          = 'shapely.tests.test_suite',
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
    if '(AMD64)' in sys.version:
        setup_args.update(
            data_files=[('DLLs', glob.glob('DLLs_AMD64_VC9/*.dll'))]
            )
    elif platform.python_version().startswith('2.5.'):
        setup_args.update(
            data_files=[('DLLs', glob.glob('DLLs_x86_VC7/*.dll'))]
            )
    else:
        setup_args.update(
            data_files=[('DLLs', glob.glob('DLLs_x86_VC9/*.dll'))]
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
        except DistutilsPlatformError, x:
            raise BuildFailed(x)

    def build_extension(self, ext):
        try:
            distutils_build_ext.build_extension(self, ext)
        except ext_errors, x:
            raise BuildFailed(x)

if (hasattr(platform, 'python_implementation')
    and platform.python_implementation() == 'PyPy'):
    # python_implementation is only available since 2.6
    ext_modules = []

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
            print >>sys.stderr, "Updating C extension with Cython."
            subprocess.check_call(["cython", "shapely/speedups/_speedups.pyx"])
    except (subprocess.CalledProcessError, OSError):
        print >>sys.stderr, "Warning: Could not (re)create C extension with Cython."
        if force_cython:
            raise
    if not os.path.exists("shapely/speedups/_speedups.c"):
        print >>sys.stderr, "Warning: speedup extension not found"

ext_modules = [
    Extension(
        "shapely.speedups._speedups",
        ["shapely/speedups/_speedups.c"],
        libraries=libraries )]

try:
    # try building with speedups
    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        **setup_args
    )
except BuildFailed, ex:
    BUILD_EXT_WARNING = "Warning: The C extension could not be compiled, speedups are not enabled."
    print ex
    print BUILD_EXT_WARNING
    print "Failure information, if any, is above."
    print "I'm retrying the build without the C extension now."

    setup(**setup_args)

    print BUILD_EXT_WARNING
    print "Plain-Python installation succeeded."

