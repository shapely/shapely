from distutils.core import setup, Extension
import geosconfig

module_geos_ufuncs = Extension(
    'pygeos.geos_ufuncs',
    sources=['src/geos_ufuncs.c'],
    include_dirs=geosconfig.include_dirs,
    library_dirs=geosconfig.library_dirs,
    libraries=geosconfig.libraries,
    extra_link_args=geosconfig.extra_link_args
)


setup(
    name='pygeos',
    version='0.1',
    description='GEOS wrapped in numpy ufuncs',
    url='https://github.com/caspervdw/pygeos',
    author='Casper van der Wel',
    license='BSD',
    packages=['pygeos'],
    install_requires=['numpy', 'shapely'],
    test_requires=['pytest'],
    python_requires='>=3',
    ext_modules=[module_geos_ufuncs]
)
