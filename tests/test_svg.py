# Tests SVG output and validity

import xml.etree.ElementTree as ET

from . import unittest
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString,\
    Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection

def is_valid_xml(x):
    try:
        ET.fromstring(x)
        return True
    except:
        return False


class SvgTestCase(unittest.TestCase):

    def test_point(self):
        # Empty
        g = Point()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = Point(6, 7)
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<circle cx="6.0" cy="7.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />')
        s = g.svg(5)
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<circle cx="6.0" cy="7.0" r="15.0" stroke="#555555" '
            'stroke-width="5.0" fill="#66cc99" opacity="0.6" />')

    def test_multipoint(self):
        # Empty
        g = MultiPoint()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = MultiPoint([(6, 7), (3, 4)])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<g><circle cx="6.0" cy="7.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />'
            '<circle cx="3.0" cy="4.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" /></g>')

    def test_linestring(self):
        # Empty
        g = LineString()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = LineString([(6, 7), (3, 4)])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="6.0,7.0 3.0,4.0" opacity="0.8" />')
        s = g.svg(5)
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<polyline fill="none" stroke="#66cc99" stroke-width="10.0" '
            'points="6.0,7.0 3.0,4.0" opacity="0.8" />')
        # Invalid
        g = LineString([(0, 0), (0, 0)])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="0.0,0.0 0.0,0.0" opacity="0.8" />')

    def test_multilinestring(self):
        # Empty
        g = MultiLineString()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = MultiLineString([[(6, 7), (3, 4)], [(2, 8), (9, 1)]])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<g><polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="6.0,7.0 3.0,4.0" opacity="0.8" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="2.0,8.0 9.0,1.0" opacity="0.8" /></g>')
        # Invalid
        g = MultiLineString([[(2, 3), (2, 3)], [(2, 8), (9, 1)]])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<g><polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="2.0,3.0 2.0,3.0" opacity="0.8" />'
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="2.0,8.0 9.0,1.0" opacity="0.8" /></g>')

    def test_polygon(self):
        # Empty
        g = Polygon()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = Polygon([(35, 10), (45, 45), (15, 40), (10, 20), (35, 10)],
                    [[(20, 30), (35, 35), (30, 20), (20, 30)]])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertEqual(
            s,
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 35.0,10.0 L 45.0,45.0 L '
            '15.0,40.0 L 10.0,20.0 L 35.0,10.0 z M 20.0,30.0 L 35.0,35.0 L '
            '30.0,20.0 L 20.0,30.0 z" />')
        s = g.svg(5)
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="10.0" opacity="0.6" d="M 35.0,10.0 L 45.0,45.0 L '
            '15.0,40.0 L 10.0,20.0 L 35.0,10.0 z M 20.0,30.0 L 35.0,35.0 L '
            '30.0,20.0 L 20.0,30.0 z" />')
        # Invalid
        g = Polygon([(0, 40), (0, 0), (40, 40), (40, 0), (0, 40)])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 0.0,40.0 L 0.0,0.0 L '
            '40.0,40.0 L 40.0,0.0 L 0.0,40.0 z" />')

    def test_multipolygon(self):
        # Empty
        g = MultiPolygon()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = MultiPolygon([
            Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
            Polygon(
                [(20, 35), (10, 30), (10, 10), (30, 5), (45, 20), (20, 35)],
                [[(30, 20), (20, 15), (20, 25), (30, 20)]])
        ])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<g><path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 40.0,40.0 L 20.0,45.0 L '
            '45.0,30.0 L 40.0,40.0 z" />'
            '<path fill-rule="evenodd" fill="#66cc99" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 20.0,35.0 L 10.0,30.0 L '
            '10.0,10.0 L 30.0,5.0 L 45.0,20.0 L 20.0,35.0 z M 30.0,20.0 L '
            '20.0,15.0 L 20.0,25.0 L 30.0,20.0 z" /></g>')
        # Invalid
        g = MultiPolygon([
            Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
            Polygon([(0, 40), (0, 0), (40, 40), (40, 0), (0, 40)])
        ])
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<g><path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 40.0,40.0 L 20.0,45.0 L '
            '45.0,30.0 L 40.0,40.0 z" />'
            '<path fill-rule="evenodd" fill="#ff3333" stroke="#555555" '
            'stroke-width="2.0" opacity="0.6" d="M 0.0,40.0 L 0.0,0.0 L '
            '40.0,40.0 L 40.0,0.0 L 0.0,40.0 z" /></g>')

    def test_collection(self):
        # Empty
        g = GeometryCollection()
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(s, '<g />')
        # Valid
        g = Point(7, 3).union(LineString([(4, 2), (8, 4)]))
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<g><circle cx="7.0" cy="3.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#66cc99" opacity="0.6" />'
            '<polyline fill="none" stroke="#66cc99" stroke-width="2.0" '
            'points="4.0,2.0 8.0,4.0" opacity="0.8" /></g>')
        # Invalid
        g = Point(7, 3).union(LineString([(4, 2), (4, 2)]))
        s = g.svg()
        self.assertTrue(is_valid_xml(s))
        self.assertTrue(is_valid_xml(g._repr_svg_()))
        self.assertEqual(
            s,
            '<g><circle cx="7.0" cy="3.0" r="3.0" stroke="#555555" '
            'stroke-width="1.0" fill="#ff3333" opacity="0.6" />'
            '<polyline fill="none" stroke="#ff3333" stroke-width="2.0" '
            'points="4.0,2.0 4.0,2.0" opacity="0.8" /></g>')


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(SvgTestCase)
