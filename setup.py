import glob
import warnings

try:
    from distribute_setup import use_setuptools
    use_setuptools()
except:
    warnings.warn(
    "Failed to import distribute_setup, continuing without distribute.", 
    Warning)

from setuptools import setup, find_packages
import sys

readme_text = file('README.txt', 'rb').read()

setup_args = dict(
    metadata_version    = '1.2',
    name                = 'Shapely',
    version             = '1.2.8',
    requires_python     = '>=2.5,<3',
    requires_external   = 'libgeos_c (>=3.1)', 
    description         = 'Geometric objects, predicates, and operations',
    license             = 'BSD',
    keywords            = 'geometry topology gis',
    author              = 'Sean Gillies',
    author_email        = 'sean.gillies@gmail.com',
    maintainer          = 'Sean Gillies',
    maintainer_email    = 'sean.gillies@gmail.com',
    url                 = 'http://trac.gispython.org/lab/wiki/Shapely',
    long_description    = readme_text,
    packages            = ['shapely', 'shapely.geometry'],
    scripts             = ['examples/dissolve.py', 'examples/intersect.py'],
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
    import glob
    if '(AMD64)' in sys.version:
        setup_args.update(
            data_files=[('DLLs', glob.glob('DLLs_AMD64/*.dll'))]
            )
    else:
        setup_args.update(
            data_files=[('DLLs', glob.glob('DLLs_x86/*.dll'))]
            )

setup(**setup_args)
