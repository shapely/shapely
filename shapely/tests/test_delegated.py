import unittest
from shapely.geometry import Point
from shapely.impl import BaseImpl
from shapely.geometry.base import delegated

class Geometry(object):
    impl = BaseImpl({})
    @property
    @delegated
    def foo(self):
        return self.impl['foo']()

class WrapperTestCase(unittest.TestCase):
    """When the backend has no support for a method, we get an AttributeError"""
    def test_delegated(self):
        self.assertRaises(AttributeError, getattr, Geometry(), 'foo')
    def test_defaultimpl(self):
        del Point.impl.map['project']
        self.assertRaises(AttributeError, Point(0, 0).project, 1.0)

def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(WrapperTestCase)
