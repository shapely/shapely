import os
import sys
import unittest

from shapely.geos import load_libc

class LoadingTestCase(unittest.TestCase):
    def test_load(self):
        self.assertRaises(OSError, load_libc, 'geosh_c')
    def test_fallbacks(self):
        a, b = load_libc('geosh_c', fallbacks=['/opt/local/lib/libgeos_c.dylib'])

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(LoadingTestCase)
