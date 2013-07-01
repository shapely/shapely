import unittest
from shapely.geometry import *
from shapely.validation import explain_validity

class ValidationTestCase(unittest.TestCase):
    def test_valid(self):
        self.failUnlessEqual(explain_validity(Point(0, 0)), 'Valid Geometry')

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ValidationTestCase)
