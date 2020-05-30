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


class IOSuite:
    """Benchmarks I/O operations (WKT and WKB) on a set of 10000 polygons"""
    def setup(self):
        self.to_write = pygeos.polygons(np.random.random((10000, 100, 2)))
        self.to_read_wkt = pygeos.to_wkt(self.to_write)
        self.to_read_wkb = pygeos.to_wkb(self.to_write)

    def time_write_to_wkt(self):
        pygeos.to_wkt(self.to_write)

    def time_write_to_wkb(self):
        pygeos.to_wkb(self.to_write)

    def time_read_from_wkt(self):
        pygeos.from_wkt(self.to_read_wkt)

    def time_read_from_wkb(self):
        pygeos.from_wkb(self.to_read_wkb)


class ConstructiveSuite:
    """Benchmarks constructive functions on a set of 10,000 points"""
    def setup(self):
        self.points = pygeos.points(np.random.random((10000, 2)))

    def time_voronoi_polygons(self):
        pygeos.voronoi_polygons(self.points)

    def time_envelope(self):
        pygeos.envelope(self.points)

    def time_convex_hull(self):
        pygeos.convex_hull(self.points)

    def time_delaunay_triangles(self):
        pygeos.delaunay_triangles(self.points)
