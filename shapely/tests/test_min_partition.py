import unittest
from shapely.algorithms.min_partition import RectilinearPolygon
from shapely.geometry import Polygon, Point

class TestRectilinearPolygon(unittest.TestCase):
    def test_is_rectilinear(self):
        # Create a rectilinear polygon
        rect_coords = [(0, 0), (0, 4), (4, 4), (4, 0)]
        rect_polygon = Polygon(rect_coords)
        rectilinear_polygon = RectilinearPolygon(rect_polygon)
        self.assertTrue(rectilinear_polygon.is_rectilinear())

        # Create a non-rectilinear polygon
        non_rect_coords = [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0)]
        non_rect_polygon = Polygon(non_rect_coords)
        non_rectilinear_polygon = RectilinearPolygon(non_rect_polygon)
        self.assertFalse(non_rectilinear_polygon.is_rectilinear())


    def test_get_grid_points(self):
        # Create a rectilinear polygon
        rect_coords = [(0, 0), (0, 4), (4, 4), (4, 0)]
        rect_polygon = Polygon(rect_coords)
        rectilinear_polygon = RectilinearPolygon(rect_polygon)
        
        # Get the grid points
        grid_points = rectilinear_polygon.get_grid_points()
        
        # Check if the grid points are correct
        expected_grid_points = [
            Point(0, 0),
            Point(0, 1),
            Point(0, 2),
            Point(0, 3),
            Point(0, 4),
            Point(1, 0),
            Point(1, 4),
            Point(2, 0),
            Point(2, 4),
            Point(3, 0),
            Point(3, 4),
            Point(4, 0),
            Point(4, 1),
            Point(4, 2),
            Point(4, 3),
            Point(4, 4)
        ]
        self.assertEqual(grid_points, expected_grid_points)

if __name__ == '__main__':
    unittest.main()