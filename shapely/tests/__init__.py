import sys
from shapely.geos import geos_version_string, lgeos, WKTWriter
from shapely import speedups

# Show some diagnostic information; handy for Travis CI
print('Python version: ' + sys.version)
print('GEOS version: ' + geos_version_string)
print('Cython speedups: ' + str(speedups.available))


if lgeos.geos_version >= (3, 3, 0):
    # Redefine WKT writer defaults to pass tests without modification
    #lgeos.wkt_writer.trim = False
    #lgeos.wkt_writer.output_dimension = 2
    WKTWriter.defaults = {}

if sys.version_info[0:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from . import test_doctests, test_prepared, test_equality, test_geomseq, \
    test_xy, test_collection, test_emptiness, test_singularity, \
    test_validation, test_mapping, test_delegated, test_dlls, \
    test_linear_referencing, test_products_z, test_box, test_speedups, \
    test_cga, test_getitem, test_ndarrays, test_unary_union, test_pickle, \
    test_affinity, test_transform, test_invalid_geometries, test_styles


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(test_doctests.test_suite())
    suite.addTest(test_prepared.test_suite())
    suite.addTest(test_emptiness.test_suite())
    suite.addTest(test_equality.test_suite())
    suite.addTest(test_geomseq.test_suite())
    suite.addTest(test_xy.test_suite())
    suite.addTest(test_collection.test_suite())
    suite.addTest(test_singularity.test_suite())
    suite.addTest(test_validation.test_suite())
    suite.addTest(test_mapping.test_suite())
    suite.addTest(test_delegated.test_suite())
    suite.addTest(test_dlls.test_suite())
    suite.addTest(test_linear_referencing.test_suite())
    suite.addTest(test_products_z.test_suite())
    suite.addTest(test_box.test_suite())
    suite.addTest(test_speedups.test_suite())
    suite.addTest(test_cga.test_suite())
    suite.addTest(test_getitem.test_suite())
    suite.addTest(test_ndarrays.test_suite())
    suite.addTest(test_unary_union.test_suite())
    suite.addTest(test_pickle.test_suite())
    suite.addTest(test_affinity.test_suite())
    suite.addTest(test_transform.test_suite())
    suite.addTest(test_invalid_geometries.test_suite())
    suite.addTest(test_styles.test_suite())
    return suite
