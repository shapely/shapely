from xml.etree import ElementTree as ET

from . import unittest

from shapely.geometry import Point, MultiPoint, LineString, MultiLineString,\
    Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection


class SvgPathElementTestCase(unittest.TestCase):

    def assertElement(self, test, expected):
        """Helper function to compare Element objects"""
        assert ET.tostring(test) == ET.tostring(expected)

    def test_point_noscale(self):
        expected = ET.Element('circle')
        expected.set('cx', '6.0')
        expected.set('cy', '7.0')
        expected.set('r', '3.0')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '1.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')
        expected.set('transform', 'matrix(1,0,0,-1,0,14.0)')
        self.assertElement(Point(6, 7).svg_path_element(), expected)

    def test_point_scale(self):
        expected = ET.Element('circle')
        expected.set('cx', '6.0')
        expected.set('cy', '7.0')
        expected.set('r', '15.0')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '5.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')
        expected.set('transform', 'matrix(1,0,0,-1,0,14.0)')
        self.assertElement(Point(6, 7).svg_path_element(5), expected)


    def test_multipoint_noscale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,11.0)')

        expected_circle1 = ET.SubElement(expected, 'circle')
        expected_circle1.set('cx', '6.0')
        expected_circle1.set('cy', '7.0')
        expected_circle1.set('r', '3.0')
        expected_circle1.set('stroke', '#555555')
        expected_circle1.set('stroke-width', '1.0')
        expected_circle1.set('fill', '#66cc99')
        expected_circle1.set('opacity', '0.6')

        expected_circle2 = ET.SubElement(expected, 'circle')
        expected_circle2.set('cx', '3.0')
        expected_circle2.set('cy', '4.0')
        expected_circle2.set('r', '3.0')
        expected_circle2.set('stroke', '#555555')
        expected_circle2.set('stroke-width', '1.0')
        expected_circle2.set('fill', '#66cc99')
        expected_circle2.set('opacity', '0.6')

        test_geom = MultiPoint([(6, 7), (3, 4)])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_multipoint_scale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,11.0)')

        expected_circle1 = ET.SubElement(expected, 'circle')
        expected_circle1.set('cx', '6.0')
        expected_circle1.set('cy', '7.0')
        expected_circle1.set('r', '15.0')
        expected_circle1.set('stroke', '#555555')
        expected_circle1.set('stroke-width', '5.0')
        expected_circle1.set('fill', '#66cc99')
        expected_circle1.set('opacity', '0.6')

        expected_circle2 = ET.SubElement(expected, 'circle')
        expected_circle2.set('cx', '3.0')
        expected_circle2.set('cy', '4.0')
        expected_circle2.set('r', '15.0')
        expected_circle2.set('stroke', '#555555')
        expected_circle2.set('stroke-width', '5.0')
        expected_circle2.set('fill', '#66cc99')
        expected_circle2.set('opacity', '0.6')

        test_geom = MultiPoint([(6, 7), (3, 4)])
        self.assertElement(test_geom.svg_path_element(5), expected)

    def test_linestring_noscale(self):
        expected = ET.Element('polyline')
        expected.set('transform', 'matrix(1,0,0,-1,0,14.0)')
        expected.set('points', '5.0,8.0 496.0,-6.0 530.0,20.0')
        expected.set('stroke', '#66cc99')
        expected.set('stroke-width', '2.0')
        expected.set('fill', 'none')
        expected.set('opacity', '0.8')

        test_geom = LineString([(5, 8), (496, -6), (530, 20)])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_linestring_scale(self):
        expected = ET.Element('polyline')
        expected.set('transform', 'matrix(1,0,0,-1,0,14.0)')
        expected.set('points', '5.0,8.0 496.0,-6.0 530.0,20.0')
        expected.set('stroke', '#66cc99')
        expected.set('stroke-width', '10.0')
        expected.set('fill', 'none')
        expected.set('opacity', '0.8')

        test_geom = LineString([(5, 8), (496, -6), (530, 20)])
        self.assertElement(test_geom.svg_path_element(5), expected)

    def test_multilinestring_noscale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,9.0)')
        expected_sub1 = ET.SubElement(expected, 'polyline')
        expected_sub1.set('points', '6.0,7.0 3.0,4.0')
        expected_sub1.set('stroke', '#66cc99')
        expected_sub1.set('stroke-width', '2.0')
        expected_sub1.set('fill', 'none')
        expected_sub1.set('opacity', '0.8')

        expected_sub2 = ET.SubElement(expected, 'polyline')
        expected_sub2.set('points', '2.0,8.0 9.0,1.0')
        expected_sub2.set('stroke', '#66cc99')
        expected_sub2.set('stroke-width', '2.0')
        expected_sub2.set('fill', 'none')
        expected_sub2.set('opacity', '0.8')

        test_geom = MultiLineString([[(6, 7), (3, 4)], [(2, 8), (9, 1)]])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_multilinestring_scale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,9.0)')
        expected_sub1 = ET.SubElement(expected, 'polyline')
        expected_sub1.set('points', '6.0,7.0 3.0,4.0')
        expected_sub1.set('stroke', '#66cc99')
        expected_sub1.set('stroke-width', '10.0')
        expected_sub1.set('fill', 'none')
        expected_sub1.set('opacity', '0.8')

        expected_sub2 = ET.SubElement(expected, 'polyline')
        expected_sub2.set('points', '2.0,8.0 9.0,1.0')
        expected_sub2.set('stroke', '#66cc99')
        expected_sub2.set('stroke-width', '10.0')
        expected_sub2.set('fill', 'none')
        expected_sub2.set('opacity', '0.8')

        test_geom = MultiLineString([[(6, 7), (3, 4)], [(2, 8), (9, 1)]])
        self.assertElement(test_geom.svg_path_element(5), expected)

    def test_polygon_noscale(self):
        expected = ET.Element('path')
        expected.set('transform', 'matrix(1,0,0,-1,0,5.0)')
        expected.set('d', 'M 2.0,2.0 L 3.0,3.0 L 4.0,2.0 L 2.0,2.0 z')
        expected.set('fill-rule', 'evenodd')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '2.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')

        test_geom = Polygon([[2, 2], [3, 3], [4, 2], [2, 2]])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_polygon_scale(self):
        expected = ET.Element('path')
        expected.set('transform', 'matrix(1,0,0,-1,0,5.0)')
        expected.set('d', 'M 2.0,2.0 L 3.0,3.0 L 4.0,2.0 L 2.0,2.0 z')
        expected.set('fill-rule', 'evenodd')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '10.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')

        test_geom = Polygon([[2, 2], [3, 3], [4, 2], [2, 2]])
        self.assertElement(test_geom.svg_path_element(5), expected)

    def test_polygon_empty(self):
        # The empty case is the same for every geometry
        self.assertElement(Polygon().svg_path_element(), ET.Element('g'))

    def test_polygon_yflip_None(self):
        # The yflip=scale case is the same for every geometry
        expected = ET.Element('path')
        expected.set('d', 'M 2.0,2.0 L 3.0,3.0 L 4.0,2.0 L 2.0,2.0 z')
        expected.set('fill-rule', 'evenodd')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '2.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')

        test_geom = Polygon([[2, 2], [3, 3], [4, 2], [2, 2]])
        self.assertElement(test_geom.svg_path_element(yflip=None), expected)

    def test_polygon_yflip_scale(self):
        # The yflip=scale case is the same for every geometry
        expected = ET.Element('path')
        expected.set('d', 'M 2.0,3.0 L 3.0,2.0 L 4.0,3.0 L 2.0,3.0 z')
        expected.set('fill-rule', 'evenodd')
        expected.set('stroke', '#555555')
        expected.set('stroke-width', '2.0')
        expected.set('fill', '#66cc99')
        expected.set('opacity', '0.6')

        test_geom = Polygon([[2, 2], [3, 3], [4, 2], [2, 2]])
        self.assertElement(test_geom.svg_path_element(yflip='scale'), expected)

    def test_multipolygon_noscale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,50.0)')
        expected_sub1 = ET.SubElement(expected, 'path')
        expected_sub1.set('d', 'M 40.0,40.0 L 20.0,45.0 L 45.0,30.0 L 40.0,40.0 z')
        expected_sub1.set('fill-rule', 'evenodd')
        expected_sub1.set('stroke', '#555555')
        expected_sub1.set('stroke-width', '2.0')
        expected_sub1.set('fill', '#66cc99')
        expected_sub1.set('opacity', '0.6')

        expected_sub2 = ET.SubElement(expected, 'path')
        expected_sub2.set('d', 'M 20.0,35.0 L 10.0,30.0 L 10.0,10.0 L 30.0,5.0 '
                          'L 45.0,20.0 L 20.0,35.0 z M 30.0,20.0 L 20.0,15.0 L '
                          '20.0,25.0 L 30.0,20.0 z')
        expected_sub2.set('fill-rule', 'evenodd')
        expected_sub2.set('stroke', '#555555')
        expected_sub2.set('stroke-width', '2.0')
        expected_sub2.set('fill', '#66cc99')
        expected_sub2.set('opacity', '0.6')

        test_geom = MultiPolygon([
                Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
                Polygon([(20, 35), (10, 30), (10, 10), (30, 5), (45, 20),
                         (20, 35)],
                        [[(30, 20), (20, 15), (20, 25), (30, 20)]])
                ])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_multipolygon_scale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,50.0)')
        expected_sub1 = ET.SubElement(expected, 'path')
        expected_sub1.set('d', 'M 40.0,40.0 L 20.0,45.0 L 45.0,30.0 L 40.0,40.0 z')
        expected_sub1.set('fill-rule', 'evenodd')
        expected_sub1.set('stroke', '#555555')
        expected_sub1.set('stroke-width', '10.0')
        expected_sub1.set('fill', '#66cc99')
        expected_sub1.set('opacity', '0.6')

        expected_sub2 = ET.SubElement(expected, 'path')
        expected_sub2.set('d', 'M 20.0,35.0 L 10.0,30.0 L 10.0,10.0 L 30.0,5.0 '
                          'L 45.0,20.0 L 20.0,35.0 z M 30.0,20.0 L 20.0,15.0 L '
                          '20.0,25.0 L 30.0,20.0 z')
        expected_sub2.set('fill-rule', 'evenodd')
        expected_sub2.set('stroke', '#555555')
        expected_sub2.set('stroke-width', '10.0')
        expected_sub2.set('fill', '#66cc99')
        expected_sub2.set('opacity', '0.6')

        test_geom = MultiPolygon([
                Polygon([(40, 40), (20, 45), (45, 30), (40, 40)]),
                Polygon([(20, 35), (10, 30), (10, 10), (30, 5), (45, 20),
                         (20, 35)],
                        [[(30, 20), (20, 15), (20, 25), (30, 20)]])
                ])
        self.assertElement(test_geom.svg_path_element(5), expected)

    def test_collection_noscale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,6.0)')
        expected_sub1 = ET.SubElement(expected, 'circle')
        expected_sub1.set('cx', '7.0')
        expected_sub1.set('cy', '3.0')
        expected_sub1.set('r', '3.0')
        expected_sub1.set('stroke', '#555555')
        expected_sub1.set('stroke-width', '1.0')
        expected_sub1.set('fill', '#66cc99')
        expected_sub1.set('opacity', '0.6')

        expected_sub2 = ET.SubElement(expected, 'polyline')
        expected_sub2.set('points', '4.0,2.0 8.0,4.0')
        expected_sub2.set('stroke', '#66cc99')
        expected_sub2.set('stroke-width', '2.0')
        expected_sub2.set('fill', 'none')
        expected_sub2.set('opacity', '0.8')

        test_geom = GeometryCollection([Point(7, 3), LineString([(4, 2), (8, 4)])])
        self.assertElement(test_geom.svg_path_element(), expected)

    def test_collection_scale(self):
        expected = ET.Element('g')
        expected.set('transform', 'matrix(1,0,0,-1,0,6.0)')
        expected_sub1 = ET.SubElement(expected, 'circle')
        expected_sub1.set('cx', '7.0')
        expected_sub1.set('cy', '3.0')
        expected_sub1.set('r', '15.0')
        expected_sub1.set('stroke', '#555555')
        expected_sub1.set('stroke-width', '5.0')
        expected_sub1.set('fill', '#66cc99')
        expected_sub1.set('opacity', '0.6')

        expected_sub2 = ET.SubElement(expected, 'polyline')
        expected_sub2.set('points', '4.0,2.0 8.0,4.0')
        expected_sub2.set('stroke', '#66cc99')
        expected_sub2.set('stroke-width', '10.0')
        expected_sub2.set('fill', 'none')
        expected_sub2.set('opacity', '0.8')

        test_geom = GeometryCollection([Point(7, 3), LineString([(4, 2), (8, 4)])])
        self.assertElement(test_geom.svg_path_element(5), expected)
