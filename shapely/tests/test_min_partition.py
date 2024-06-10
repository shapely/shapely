'''
test the min_partition module
Programmer: Dvir Borochov
Date: 10/6/24 
'''

import unittest
from shapely.algorithms.min_partition import RectilinearPolygon
from shapely.geometry import Polygon, Point

class TestRectilinearPolygon(unittest.TestCase):
    # Define the polygons as class attributes shapely\algorithms\min_partition.py
    polygon1 = RectilinearPolygon(
        Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
    )
    polygon2 = RectilinearPolygon(
        Polygon([(1, 5), (1, 4), (3, 4), (3, 2), (5, 2), (5, 1), (8, 1), (8, 5)])
    )
    polygon3 = RectilinearPolygon(
        Polygon([(0, 4), (2, 4), (2, 0), (5, 0), (5, 4), (7, 4), (7, 5), (0, 5)])
    )

    # intialize the grid points
    grid_points = polygon1.get_grid_points()
    grid_points = polygon2.get_grid_points()
    grid_points = polygon3.get_grid_points()

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

        self.assertEqual(len(self.polygon1.grid_points), 10)

        self.assertEqual(len(self.polygon2.grid_points), 13)

    def test_find_matching_point(self):
        self.polygon1.grid_points = self.polygon1.get_grid_points()
        matching_point = self.polygon1.find_matching_point(Point(6, 0), [])
        self.assertEqual(matching_point, [Point(2, 4), Point(2, 6)])

        self.polygon2.grid_points = self.polygon2.get_grid_points()
        matching_point = self.polygon2.find_matching_point(Point(1, 4), [])
        self.assertEqual(matching_point, [Point(5, 5), Point(8, 5), Point(3, 5)])

    def test_partition(self):
        par1 = self.polygon1.partition()
        self.assertEqual(self.polygon1.min_partition_length, 4.0)
        par2 = self.polygon2.partition()
        self.assertEqual(self.polygon2.min_partition_length, 4.0)
        par3 = self.polygon3.partition()
        self.assertEqual(self.polygon3.min_partition_length, 2.0)


if __name__ == "__main__":
    unittest.main()
