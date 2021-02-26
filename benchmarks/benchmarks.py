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
        self.multipolygons = np.array(
            [
                pygeos.multipolygons(pygeos.polygons(np.random.random((2, 100, 2))))
                for i in range(10000)
            ],
            dtype=object,
        )

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


class OverlaySuite:
    """Benchmarks for different methods of overlaying geometries"""

    def setup(self):
        # create irregular polygons by merging overlapping point buffers
        self.left = pygeos.union_all(
            pygeos.buffer(pygeos.points(np.random.random((500, 2)) * 500), 15)
        )
        # shift this up and right
        self.right = pygeos.apply(self.left, lambda x: x + 50)

    def time_difference(self):
        pygeos.difference(self.left, self.right)

    def time_difference_prec1(self):
        pygeos.difference(self.left, self.right, grid_size=1)

    def time_difference_prec2(self):
        pygeos.difference(self.left, self.right, grid_size=2)

    def time_intersection(self):
        pygeos.intersection(self.left, self.right)

    def time_intersection_prec1(self):
        pygeos.intersection(self.left, self.right, grid_size=1)

    def time_intersection_prec2(self):
        pygeos.intersection(self.left, self.right, grid_size=2)

    def time_symmetric_difference(self):
        pygeos.symmetric_difference(self.left, self.right)

    def time_symmetric_difference_prec1(self):
        pygeos.symmetric_difference(self.left, self.right, grid_size=1)

    def time_symmetric_difference_prec2(self):
        pygeos.symmetric_difference(self.left, self.right, grid_size=2)

    def time_union(self):
        pygeos.union(self.left, self.right)

    def time_union_prec1(self):
        pygeos.union(self.left, self.right, grid_size=1)

    def time_union_prec2(self):
        pygeos.union(self.left, self.right, grid_size=2)

    def time_union_all(self):
        pygeos.union_all([self.left, self.right])

    def time_union_all_prec1(self):
        pygeos.union_all([self.left, self.right], grid_size=1)

    def time_union_all_prec2(self):
        pygeos.union_all([self.left, self.right], grid_size=2)


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

        # create points that extend beyond the domain of the above polygons to ensure
        # some don't overlap
        self.points = pygeos.points((np.random.random((2000, 2)) * 750) - 125)
        self.point_tree = pygeos.STRtree(
            pygeos.points(np.random.random((2000, 2)) * 750)
        )
        self.point_tree.query(pygeos.points(0, 0))

        # create points on a grid for testing equidistant nearest neighbors
        # creates 2025 points
        grid_coords = np.mgrid[:45, :45].T.reshape(-1, 2)
        self.grid_point_tree = pygeos.STRtree(pygeos.points(grid_coords))
        self.grid_points = pygeos.points(grid_coords + 0.5)

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

    def time_tree_nearest_points(self):
        self.point_tree.nearest(self.points)

    def time_tree_nearest_points_equidistant(self):
        self.grid_point_tree.nearest(self.grid_points)

    def time_tree_nearest_points_equidistant_manual_all(self):
        # This benchmark approximates nearest_all for equidistant results
        # starting from singular nearest neighbors and searching for more
        # within same distance.

        # try to find all equidistant neighbors ourselves given single nearest
        # result
        l, r = self.grid_point_tree.nearest(self.grid_points)
        # calculate distance to nearest neighbor
        dist = pygeos.distance(self.grid_points.take(l), self.grid_point_tree.geometries.take(r))
        # include a slight epsilon to ensure nearest are within this radius
        b = pygeos.buffer(self.grid_points, dist + 1e-8)

        # query the tree for others in the same buffer distance
        left, right = self.grid_point_tree.query_bulk(b, predicate='intersects')
        dist = pygeos.distance(
            self.grid_points.take(left), self.grid_point_tree.geometries.take(right)
        )

        # sort by left, distance
        ix = np.lexsort((right, dist, left))
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

        run_start = np.r_[True, left[:-1] != left[1:]]
        run_counts = np.diff(np.r_[np.nonzero(run_start)[0], left.shape[0]])

        mins = dist[run_start]

        # spread to rest of array so we can extract out all within each group that match
        all_mins = np.repeat(mins, run_counts)
        ix = dist == all_mins
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

    def time_tree_nearest_all_points(self):
        self.point_tree.nearest_all(self.points)

    def time_tree_nearest_all_points_equidistant(self):
        self.grid_point_tree.nearest_all(self.grid_points)

    def time_tree_nearest_all_points_small_max_distance(self):
        # returns >300 results
        self.point_tree.nearest_all(self.points, max_distance=5)

    def time_tree_nearest_all_points_large_max_distance(self):
        # measures the overhead of using a distance that would encompass all tree points
        self.point_tree.nearest_all(self.points, max_distance=1000)

    def time_tree_nearest_poly(self):
        self.tree.nearest(self.points)

    def time_tree_nearest_all_poly(self):
        self.tree.nearest_all(self.points)

    def time_tree_nearest_all_poly_small_max_distance(self):
        # returns >300 results
        self.tree.nearest_all(self.points, max_distance=5)

    def time_tree_nearest_all_poly_python(self):
        # returns all input points

        # use an arbitrary search tolerance that seems appropriate for the density of
        # geometries
        tolerance = 200
        b = pygeos.buffer(self.points, tolerance, quadsegs=1)
        left, right = self.tree.query_bulk(b)
        dist = pygeos.distance(self.points.take(left), self.polygons.take(right))

        # sort by left, distance
        ix = np.lexsort((right, dist, left))
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

        run_start = np.r_[True, left[:-1] != left[1:]]
        run_counts = np.diff(np.r_[np.nonzero(run_start)[0], left.shape[0]])

        mins = dist[run_start]

        # spread to rest of array so we can extract out all within each group that match
        all_mins = np.repeat(mins, run_counts)
        ix = dist == all_mins
        left = left[ix]
        right = right[ix]
        dist = dist[ix]

        # arrays are now roughly representative of what tree.nearest_all would provide, though
        # some nearest_all neighbors may be missed if they are outside tolerance
