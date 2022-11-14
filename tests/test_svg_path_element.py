from xml.etree import ElementTree as ET

from shapely.geometry import Point, Polygon

from . import unittest
from unittest.mock import Mock


class SVGPathElementTestCase(unittest.TestCase):
    def setUp(self):
        # triangle that is lower-left half of a unit square
        # with lower-left corner offset from origin by +1 in x and y dir.
        self.coords = ((1,1),(1,2),(2,1),(1,1))

    def assertElementsEqual(self, test, expected):
        """Test equivalence of two Element Trees"""
        # Ref: https://stackoverflow.com/a/24349916
        self.assertEqual(test.tag, expected.tag)
        self.assertEqual(test.text, expected.text)
        self.assertEqual(test.tail, expected.tail)
        self.assertEqual(test.attrib, expected.attrib)
        self.assertEqual(len(test), len(expected))

        # run recursively for all children
        for c1, c2 in zip(test, expected):
            self.assertElementsEqual(c1, c2) 

    def test_kwargs(self):
        """Ensure that kwargs are passed on to svg method"""
        mocked_svg = Mock()
        mocked_svg.return_value = (
            '<path fill="#000000" stroke-width="200.0" opacity="0.0" '
            'd="M 1.0,1.0 L 1.0,2.0 L 2.0,1.0 L 1.0,1.0 z" />'
            )

        with unittest.mock.patch.object(Polygon, 'svg', mocked_svg):
            polygon = Polygon(self.coords)

            test = polygon.svg_path_element(
                yflip=None, scale_factor=100, fill_color="#000000", opacity=0)

            mocked_svg.assert_called_once_with(
                scale_factor=100, fill_color="#000000", opacity=0)

        expected = ET.Element(
                'path',
                {
                    "fill": "#000000",
                    "stroke-width": "200.0",
                    "opacity": "0.0",
                    "d": "M 1.0,1.0 L 1.0,2.0 L 2.0,1.0 L 1.0,1.0 z"
                })

    
        self.assertElementsEqual(test, expected)

    def test_yflip_invalid(self):
        with self.assertRaises(ValueError):
            Polygon(self.coords).svg_path_element(yflip='invalid')


    def test_yflip_transform(self):
        polygon = Polygon(self.coords)

        test = polygon.svg_path_element(yflip='transform')

        # test that actual path is not changed
        original_d = "M 1.0,1.0 L 1.0,2.0 L 2.0,1.0 L 1.0,1.0 z"
        assert test.attrib['d'] == original_d

        # test that transform entry is added and is accurate
        # ref: https://drafts.csswg.org/css-transforms/#transformation-matrix-computation
        # matrix format: matrix(1,0,tx,-1,0,3.0)
        expected_transform = 'matrix(1,0,0,-1,0,3.0)'
        assert test.attrib['transform'] == expected_transform

        # test that the geometry was not changed by the operation 
        assert polygon == Polygon(self.coords)

