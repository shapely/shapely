from unittest import TestSuite

import test_doctests, test_prepared

def test_suite():
    suite = TestSuite()
    suite.addTest(test_doctests.test_suite())
    suite.addTest(test_prepared.test_suite())
    return suite

