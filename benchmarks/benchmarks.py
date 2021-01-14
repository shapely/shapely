import numpy as np
import pygeos


# Seed the numpy random generator for more reproducible benchmarks
np.random.seed(0)


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


class ClipSuite:
    """Benchmarks for different methods of clipping geometries by boxes"""

    def setup(self):
        # create irregular polygons by merging overlapping point buffers
        self.polygon = pygeos.union_all(
                pygeos.buffer(pygeos.points(np.random.random((1000, 2)) * 500), 10)
            )
        xmin = np.random.random(100) * 100
        xmax = xmin + 100
        ymin = np.random.random(100) * 100
        ymax = ymin + 100
        self.bounds = np.array([xmin, ymin, xmax, ymax]).T
        self.boxes = pygeos.box(xmin, ymin, xmax, ymax)


    def time_clip_by_box(self):
        pygeos.intersection(self.polygon, self.boxes)

    def time_clip_by_rect(self):
        for bounds in self.bounds:
            pygeos.clip_by_rect(self.polygon, *bounds)


class GetParts:
    """Benchmarks for getting individual parts from 100 multipolygons of 100 polygons each"""

    def setup(self):
        self.multipolygons = np.array([pygeos.multipolygons(pygeos.polygons(np.random.random((2, 100, 2)))) for i in range(10000)], dtype=object)

    def time_get_parts(self):
        """Cython implementation of get_parts"""
        pygeos.get_parts(self.multipolygons)

    def time_get_parts_python(self):
        """Python / ufuncs version of get_parts"""

        parts = []
        for i in range(len(self.multipolygons)):
            num_parts = pygeos.get_num_geometries(self.multipolygons[i])
            parts.append(pygeos.get_geometry(self.multipolygons[i], range(num_parts)))

        parts = np.concatenate(parts)


class STRtree:
    """Benchmarks queries against STRtree"""

    def setup(self):
        # create irregular polygons my merging overlapping point buffers
        self.polygons = pygeos.get_parts(
            pygeos.union_all(
                pygeos.buffer(pygeos.points(np.random.random((2000, 2)) * 500), 5)
            )
        )
        self.tree = pygeos.STRtree(self.polygons)
        # initialize the tree by making a tiny query first
        self.tree.query(pygeos.points(0, 0))

    def time_tree_create(self):
        tree = pygeos.STRtree(self.polygons)
        tree.query(pygeos.points(0, 0))

    def time_tree_query_bulk(self):
        self.tree.query_bulk(self.polygons)

    def time_tree_query_bulk_intersects(self):
        self.tree.query_bulk(self.polygons, predicate="intersects")

    def time_tree_query_bulk_within(self):
        self.tree.query_bulk(self.polygons, predicate="within")

    def time_tree_query_bulk_contains(self):
        self.tree.query_bulk(self.polygons, predicate="contains")

    def time_tree_query_bulk_overlaps(self):
        self.tree.query_bulk(self.polygons, predicate="overlaps")

    def time_tree_query_bulk_crosses(self):
        self.tree.query_bulk(self.polygons, predicate="crosses")

    def time_tree_query_bulk_touches(self):
        self.tree.query_bulk(self.polygons, predicate="touches")

    def time_tree_query_bulk_covers(self):
        self.tree.query_bulk(self.polygons, predicate="covers")

    def time_tree_query_bulk_covered_by(self):
        self.tree.query_bulk(self.polygons, predicate="covered_by")

    def time_tree_query_bulk_contains_properly(self):
        self.tree.query_bulk(self.polygons, predicate="contains_properly")
