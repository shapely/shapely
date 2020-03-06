import numpy as np
import pygeos


class PointPolygonTimeSuite:
    """Benchmarks running on 100000 points and one polygon"""
    def setup(self):
        self.points = pygeos.points(np.random.random((100000, 2)))
        self.polygon = pygeos.polygons(np.random.random((3, 2)))

    def time_contains(self):
        pygeos.contains(self.points, self.polygon)

    def time_distance(self):
        pygeos.distance(self.points, self.polygon)

    def time_intersection(self):
        pygeos.intersection(self.points, self.polygon)
