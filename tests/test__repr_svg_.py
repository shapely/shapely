import xml.etree.ElementTree as ET

from . import unittest
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString,\
    Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection


class _Repr_Svg_TestCase(unittest.TestCase):

    def assert_repr_svg_(self, geom, expected):
        """Helper function to check SVG representation for iPython notebook"""
        # avoid order-related issues in string comparison by running through
        # Element string parsing
        self.assertEqual(ET.tostring(ET.fromstring(geom._repr_svg_())),
                         ET.tostring(ET.fromstring(expected)))

    def test_empty(self):
        self.assert_repr_svg_(
            Point(), 
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" />')

    def test_point(self):
        self.assert_repr_svg_(
            Point(6, 7),
            '<svg width="100.0" height="100.0" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'xmlns="http://www.w3.org/2000/svg"'
            'viewBox="5.0 6.0 2.0 2.0" preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,14.0)">'
            '<circle cx="6.0" cy="7.0" r="0.06" stroke="#555555" '
            'stroke-width="0.02" fill="#66cc99" opacity="0.6" />'
            '</g></svg>')

    def test_multipoint(self):
        self.assert_repr_svg_(
            MultiPoint([(6, 7), (3, 4)]),
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="100.0" height="100.0" '
            'viewBox="2.88 3.88 3.24 3.24" preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,11.0)"><g>'
            '<circle cx="6.0" cy="7.0" r="0.09720000000000001" '
            'stroke="#555555" stroke-width="0.032400000000000005" '
            'fill="#66cc99" opacity="0.6" /><circle cx="3.0" cy="4.0" '
            'r="0.09720000000000001" stroke="#555555" '
            'stroke-width="0.032400000000000005" '
            'fill="#66cc99" opacity="0.6" />'
            '</g></g></svg>')

    def test_linestring(self):
        self.assert_repr_svg_(
            LineString([(5, 8), (496, -6), (530, 20)]), 
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="300" height="100.0" '
            'viewBox="-16.0 -27.0 567.0 68.0" '
            'preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,14.0)">'
            '<polyline fill="none" stroke="#66cc99" stroke-width="3.78" '
            'points="5.0,8.0 496.0,-6.0 530.0,20.0" opacity="0.8" />'
            '</g></svg>') 

    def test_multilinestring(self):
        self.assert_repr_svg_(
            MultiLineString([[(6, 7), (3, 4)], [(2, 8), (9, 1)]]), 
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="100.0" height="100.0" '
            'viewBox="1.72 0.72 7.56 7.56" preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,9.0)"><g>'
            '<polyline fill="none" stroke="#66cc99" stroke-width="0.1512" '
            'points="6.0,7.0 3.0,4.0" opacity="0.8" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="0.1512" '
            'points="2.0,8.0 9.0,1.0" opacity="0.8" />'
            '</g></g></svg>')


    def test_polygon(self):
        self.assert_repr_svg_(
            Polygon([(35, 10), (45, 45), (15, 40), (10, 20), (35, 10)],
                    [[(20, 30), (35, 35), (30, 20), (20, 30)]]),
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="100.0" height="100.0" '
            'viewBox="8.6 8.6 37.8 37.8" preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,55.0)">'
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="0.7559999999999999" opacity="0.6" '
            'd="M 35.0,10.0 L 45.0,45.0 L 15.0,40.0 L 10.0,20.0 L 35.0,10.0 z '
            'M 20.0,30.0 L 35.0,35.0 L 30.0,20.0 L 20.0,30.0 z" />'
            '</g></svg>')

    def test_multipolygon(self):
        self.assert_repr_svg_(
            MultiPolygon([
                Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
                Polygon([(20, 35), (10, 30), (10, 10), (30, 5), (45, 20),
                         (20, 35)],
                        [[(30, 20), (20, 15), (20, 25), (30, 20)]])
                ]), 
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="100.0" height="100.0" '
            'viewBox="8.4 3.4 38.2 43.2" preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,50.0)"><g>'
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="0.8640000000000001" opacity="0.6" '
            'd="M 40.0,40.0 L 20.0,45.0 L 45.0,30.0 L 40.0,40.0 z" />'
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="0.8640000000000001" opacity="0.6" '
            'd="M 20.0,35.0 L 10.0,30.0 L 10.0,10.0 L 30.0,5.0 L 45.0,20.0 L '
            '20.0,35.0 z M 30.0,20.0 L 20.0,15.0 L 20.0,25.0 L 30.0,20.0 z" />'
            '</g></g></svg>')

    def test_collection(self):
        self.assert_repr_svg_(
            GeometryCollection(
                [Point(7, 3), LineString([(4, 2), (8, 4)])]),
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
            'width="100.0" height="100.0" '
            'viewBox="3.84 1.84 4.32 2.3200000000000003" '
            'preserveAspectRatio="xMinYMin meet">'
            '<g transform="matrix(1,0,0,-1,0,6.0)"><g>'
            '<circle cx="7.0" cy="3.0" r="0.1296" stroke="#555555" '
            'stroke-width="0.0432" fill="#66cc99" opacity="0.6" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="0.0864" '
            'points="4.0,2.0 8.0,4.0" opacity="0.8" />'
            '</g></g></svg>')
