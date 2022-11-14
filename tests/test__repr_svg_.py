from xml.dom.minidom import parseString as parse_xml_string
from xml.etree import ElementTree as ET

from . import unittest
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString,\
    Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection


class _Repr_Svg_TestCase(unittest.TestCase):

    def assert_repr_svg_(self, geom, expected):
        """Helper function to check SVG representation for iPython notebook"""
        # get rid of order and spacing affecting comparison
        # if we compared these as ET.Element, we would miss the xmlns and
        # xmlns:xlink properties
        self.assertEqual(ET.tostring(ET.fromstring(geom._repr_svg_())),
                         ET.tostring(ET.fromstring(expected)))

    def assertCloseStr(self, test_str, expected, ndigits=5):
        self.assertEqual(round(float(test_str), ndigits), expected)

    def assertCloseStrList(self, test_str, expected, ndigits=5):
        tests_float = test_str.split()
        for t, e in zip(tests_float, expected):
            self.assertCloseStr(t, e, ndigits=ndigits)


    def test_empty(self):
        self.assertEqual(
            Point()._repr_svg_(),
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" />')

    def test_point(self):
        test_ET = ET.fromstring(Point(6, 7)._repr_svg_())
        self.assertCloseStr(test_ET.attrib['width'], 100)
        self.assertCloseStr(test_ET.attrib['height'], 100)
        self.assertCloseStrList(test_ET.attrib['viewBox'], [5, 6, 2, 2])

        self.assertEqual(test_ET[0].attrib['transform'], 'matrix(1,0,0,-1,0,14.0)')

    def test_non_point(self):
        """The logic for any non-point feature is the same, so this covers
        polygons, multipolygons, etc"""
        test_ET = ET.fromstring(MultiPoint([(6, 7), (3, 4)])._repr_svg_())
        self.assertCloseStr(test_ET.attrib['width'], 100)
        self.assertCloseStr(test_ET.attrib['height'], 100)
        self.assertCloseStrList(test_ET.attrib['viewBox'], [2.88, 3.88, 3.24, 3.24])

        self.assertEqual(test_ET[0].attrib['transform'], 'matrix(1,0,0,-1,0,11.0)')
