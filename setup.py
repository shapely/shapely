import glob
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
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages
import sys
import platform

readme_text = open('README.rst', 'rb').read()

# Skip the first line of the changes file to get the right header level
f = open('CHANGES.txt', 'rb')
f.readline()
changes_text = f.read()

setup_args = dict(
    metadata_version    = '1.2',
    name                = 'Shapely',
    version             = '1.2.14',
    requires_python     = '>=2.5,<3',
    requires_external   = 'libgeos_c (>=3.1)', 
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'https://github.com/sgillies/shapely',
    long_description    = readme_text + "\n" + changes_text,
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

class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError, x:
            raise BuildFailed(x)

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors, x:
            raise BuildFailed(x)

if sys.platform == 'win32':
    # geos DLL is geos.dll instead of geos_c.dll on Windows
    ext_modules = [
        Extension("shapely.speedups._speedups",
              ["shapely/speedups/_speedups.c"], libraries=['geos']),
    ]
elif (hasattr(platform, 'python_implementation') 
    and platform.python_implementation() == 'PyPy'):
    # python_implementation >= 2.6
    ext_modules = []
else:
    ext_modules = [
        Extension("shapely.speedups._speedups", 
              ["shapely/speedups/_speedups.c"], libraries=['geos_c']),
    ]

try:
    # try building with speedups
    setup(
        cmdclass={'build_ext': ve_build_ext},
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

