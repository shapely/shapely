import os
from setuptools import setup, Extension
import geosconfig

module_geos_ufuncs = Extension(
    'pygeos.geos_ufuncs',
    sources=['src/geos_ufuncs.c'],
    include_dirs=geosconfig.include_dirs,
    library_dirs=geosconfig.library_dirs,
    libraries=geosconfig.libraries,
    extra_link_args=geosconfig.extra_link_args
)

try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.rst')).read()
except IOError:
    descr = ''


setup(
    name='pygeos',
    version='0.1',
    description='GEOS wrapped in numpy ufuncs',
    long_description=descr,
    url='https://github.com/caspervdw/pygeos',
    author='Casper van der Wel',
    license='BSD 3-Clause',
    packages=['pygeos'],
    install_requires=['numpy', 'shapely'],
    test_requires=['pytest'],
    python_requires='>=3',
    ext_modules=[module_geos_ufuncs],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: GIS',
        'Operating System :: Unix',
    ]
)
