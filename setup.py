import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import geosconfig

# https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py/21621689#21621689
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


module_ufuncs = Extension(
    "pygeos.ufuncs",
    sources=["src/ufuncs.c"],
    include_dirs=geosconfig.include_dirs,
    library_dirs=geosconfig.library_dirs,
    libraries=geosconfig.libraries,
    extra_link_args=geosconfig.extra_link_args,
)

try:
    descr = open(os.path.join(os.path.dirname(__file__), "README.rst")).read()
except IOError:
    descr = ""


setup(
    name="pygeos",
    version="0.3.dev0",
    description="GEOS wrapped in numpy ufuncs",
    long_description=descr,
    url="https://github.com/caspervdw/pygeos",
    author="Casper van der Wel",
    license="BSD 3-Clause",
    packages=["pygeos"],
    setup_requires=["numpy"],
    install_requires=["numpy>=1.10"],
    extras_require={"test": ["pytest"]},
    python_requires=">=3",
    include_package_data=True,
    ext_modules=[module_ufuncs],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 1 - Planning",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: GIS",
        "Operating System :: Unix",
    ],
    cmdclass={"build_ext": build_ext},
)
