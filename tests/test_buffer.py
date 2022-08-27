from . import unittest
from shapely import geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.constructive import BufferCapStyles, BufferJoinStyles


class BufferSingleSidedCase(unittest.TestCase):
    """Test Buffer Point/Line/Polygon with and without single_sided params"""

    def test_empty(self):
        g = geometry.Point(0, 0)
        h = g.buffer(0)
        assert h.is_empty

    def test_point(self):
        g = geometry.Point(0, 0)
        h = g.buffer(1, resolution=1)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [(1.0, 0.0), (0, -1.0), (-1.0, 0), (0, 1.0), (1.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_point_single_sidedd(self):
        g = geometry.Point(0, 0)
        h = g.buffer(1, resolution=1, single_sided=True)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [(1.0, 0.0), (0, -1.0), (-1.0, 0), (0, 1.0), (1.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_line(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(1, resolution=1)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [
            (-1.0, 1.0),
            (0, 2.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_line_single_sideded_left(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(1, resolution=1, single_sided=True)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [(0.0, 1.0), (0.0, 0.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_line_single_sideded_right(self):
        g = geometry.LineString([[0, 0], [0, 1]])
        h = g.buffer(-1, resolution=1, single_sided=True)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_polygon(self):
        g = geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        h = g.buffer(1, resolution=1)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.0),
            (-1.0, 0.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_polygon_single_sideded(self):
        g = geometry.Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        h = g.buffer(1, resolution=1, single_sided=True)
        self.assertEqual(h.geom_type, "Polygon")
        expected_coord = [
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 2.0),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.0, 0.0),
            (1.0, -1.0),
            (0.0, -1.0),
            (-1.0, 0.0),
        ]
        for index, coord in enumerate(h.exterior.coords):
            self.assertAlmostEqual(coord[0], expected_coord[index][0])
            self.assertAlmostEqual(coord[1], expected_coord[index][1])

    def test_styles(self):
        g = geometry.LineString([[0, 0], [1, 0]])
        h = g.buffer(1, cap_style=CAP_STYLE.round)
        self.assertEqual(h, g.buffer(1, cap_style="round"))
        self.assertEqual(h, g.buffer(1, cap_style=BufferCapStyles.round))

        h = g.buffer(1, cap_style=CAP_STYLE.flat)
        self.assertEqual(h, g.buffer(1, cap_style="flat"))
        self.assertEqual(h, g.buffer(1, cap_style=BufferCapStyles.flat))

        h = g.buffer(1, cap_style=CAP_STYLE.square)
        self.assertEqual(h, g.buffer(1, cap_style="square"))
        self.assertEqual(h, g.buffer(1, cap_style=BufferCapStyles.square))

        h = g.buffer(1, join_style=JOIN_STYLE.round)
        self.assertEqual(h, g.buffer(1, join_style="round"))
        self.assertEqual(h, g.buffer(1, join_style=BufferJoinStyles.round))

        h = g.buffer(1, join_style=JOIN_STYLE.mitre)
        self.assertEqual(h, g.buffer(1, join_style="mitre"))
        self.assertEqual(h, g.buffer(1, join_style=BufferJoinStyles.mitre))

        h = g.buffer(1, join_style=JOIN_STYLE.bevel)
        self.assertEqual(h, g.buffer(1, join_style="bevel"))
        self.assertEqual(h, g.buffer(1, join_style=BufferJoinStyles.bevel))
