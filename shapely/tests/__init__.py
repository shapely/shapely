from unittest import TestSuite

import test_doctests, test_prepared, test_equality, test_geomseq, test_xy
import test_collection

def test_suite():
    suite = TestSuite()
    suite.addTest(test_doctests.test_suite())
    suite.addTest(test_prepared.test_suite())
    suite.addTest(test_equality.test_suite())
    suite.addTest(test_geomseq.test_suite())
    suite.addTest(test_xy.test_suite())
    suite.addTest(test_collection.test_suite())
    return suite

