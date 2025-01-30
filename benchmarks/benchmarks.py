"""
Shapely benchmarks

These are run using asv: "pip install asv" or "conda install -c conda-forge asv"

To run a specific test within the existing environment, e.g., PointPolygonTimeSuite:
$ asv run -b PointPolygonTimeSuite -E 'existing'
"""

import numpy as np
import shapely


# Seed the numpy random generator for more reproducible benchmarks
np.random.seed(0)


class PointPolygonTimeSuite:
    """Benchmarks running on 100000 points and one polygon"""

    def setup(self):
        self.points = shapely.points(np.random.random((100000, 2)))
        self.polygon = shapely.polygons(np.random.random((3, 2)))

    def time_contains(self):
        shapely.contains(self.points, self.polygon)

    def time_distance(self):
        shapely.distance(self.points, self.polygon)

    def time_intersection(self):
        shapely.intersection(self.points, self.polygon)


class IOSuite:
    """Benchmarks I/O operations (WKT and WKB) on a set of 10000 polygons"""

    def setup(self):
        self.to_write = shapely.polygons(np.random.random((10000, 100, 2)))
        self.to_read_wkt = shapely.to_wkt(self.to_write)
        self.to_read_wkb = shapely.to_wkb(self.to_write)

    def time_write_to_wkt(self):
        shapely.to_wkt(self.to_write)

    def time_write_to_wkb(self):
        shapely.to_wkb(self.to_write)

    def time_read_from_wkt(self):
        shapely.from_wkt(self.to_read_wkt)

    def time_read_from_wkb(self):
        shapely.from_wkb(self.to_read_wkb)


class ConstructorsSuite:
    """Microbenchmarks for the Geometry class constructors"""

    def setup(self):
        self.coords = np.random.random((1000, 2))

    def time_point(self):
        shapely.Point(1.0, 2.0)

    def time_linestring_from_numpy(self):
        shapely.LineString(self.coords)

    def time_linearring_from_numpy(self):
        shapely.LinearRing(self.coords)


class ConstructiveSuite:
    """Benchmarks constructive functions on a set of 10,000 points"""

    def setup(self):
        self.coords = np.random.random((10000, 2))
        self.points = shapely.points(self.coords)

    def time_voronoi_polygons(self):
        shapely.voronoi_polygons(self.points)

    def time_envelope(self):
        shapely.envelope(self.points)

    def time_convex_hull(self):
        shapely.convex_hull(self.points)

    def time_concave_hull(self):
        shapely.concave_hull(self.points, ratio=0.2, allow_holes=False)

    def time_concave_hull_with_holes(self):
        shapely.concave_hull(self.points, ratio=0.2, allow_holes=True)

    def time_delaunay_triangles(self):
        shapely.delaunay_triangles(self.points)

    def time_box(self):
        shapely.box(*np.hstack([self.coords, self.coords + 100]).T)


class ClipSuite:
    """Benchmarks for different methods of clipping geometries by boxes"""

    def setup(self):
        # create irregular polygons by merging overlapping point buffers
        self.polygon = shapely.union_all(
            shapely.buffer(shapely.points(np.random.random((1000, 2)) * 500), 10)
        )
        xmin = np.random.random(100) * 100
        xmax = xmin + 100
        ymin = np.random.random(100) * 100
        ymax = ymin + 100
        self.bounds = np.array([xmin, ymin, xmax, ymax]).T
        self.boxes = shapely.box(xmin, ymin, xmax, ymax)

    def time_clip_by_box(self):
        shapely.intersection(self.polygon, self.boxes)

    def time_clip_by_rect(self):
        for bounds in self.bounds:
            shapely.clip_by_rect(self.polygon, *bounds)


class GetParts:
    """Benchmarks for getting individual parts from 100 multipolygons of 100 polygons each"""

    def setup(self):
        self.multipolygons = np.array(
            [
                shapely.multipolygons(shapely.polygons(np.random.random((2, 100, 2))))
                for i in range(10000)
            ],
            dtype=object,
        )

    def time_get_parts(self):
        """Cython implementation of get_parts"""
        shapely.get_parts(self.multipolygons)

    def time_get_parts_python(self):
        """Python / ufuncs version of get_parts"""

        parts = []
        for i in range(len(self.multipolygons)):
            num_parts = shapely.get_num_geometries(self.multipolygons[i])
            parts.append(shapely.get_geometry(self.multipolygons[i], range(num_parts)))

        parts = np.concatenate(parts)


class OverlaySuite:
    """Benchmarks for different methods of overlaying geometries"""

    def setup(self):
        # create irregular polygons by merging overlapping point buffers
        self.left = shapely.union_all(
            shapely.buffer(shapely.points(np.random.random((500, 2)) * 500), 15)
        )
        # shift this up and right
        self.right = shapely.transform(self.left, lambda x: x + 50)

    def time_difference(self):
        shapely.difference(self.left, self.right)

    def time_difference_prec1(self):
        shapely.difference(self.left, self.right, grid_size=1)

    def time_difference_prec2(self):
        shapely.difference(self.left, self.right, grid_size=2)

    def time_intersection(self):
        shapely.intersection(self.left, self.right)

    def time_intersection_prec1(self):
        shapely.intersection(self.left, self.right, grid_size=1)

    def time_intersection_prec2(self):
        shapely.intersection(self.left, self.right, grid_size=2)

    def time_symmetric_difference(self):
        shapely.symmetric_difference(self.left, self.right)

    def time_symmetric_difference_prec1(self):
        shapely.symmetric_difference(self.left, self.right, grid_size=1)

    def time_symmetric_difference_prec2(self):
        shapely.symmetric_difference(self.left, self.right, grid_size=2)

    def time_union(self):
        shapely.union(self.left, self.right)

    def time_union_prec1(self):
        shapely.union(self.left, self.right, grid_size=1)

    def time_union_prec2(self):
        shapely.union(self.left, self.right, grid_size=2)

    def time_union_all(self):
        shapely.union_all([self.left, self.right])

    def time_union_all_prec1(self):
        shapely.union_all([self.left, self.right], grid_size=1)

    def time_union_all_prec2(self):
        shapely.union_all([self.left, self.right], grid_size=2)


class STRtree:
    """Benchmarks queries against STRtree"""

    def setup(self):
        # create irregular polygons my merging overlapping point buffers
        self.polygons = shapely.get_parts(
            shapely.union_all(
                shapely.buffer(shapely.points(np.random.random((2000, 2)) * 500), 5)
            )
        )
        self.tree = shapely.STRtree(self.polygons)
        # initialize the tree by making a tiny query first
        self.tree.query(shapely.points(0, 0))

        # create points that extend beyond the domain of the above polygons to ensure
        # some don't overlap
        self.points = shapely.points((np.random.random((2000, 2)) * 750) - 125)
        self.point_tree = shapely.STRtree(
            shapely.points(np.random.random((2000, 2)) * 750)
        )
        self.point_tree.query(shapely.points(0, 0))

        # create points on a grid for testing equidistant nearest neighbors
        # creates 2025 points
        grid_coords = np.mgrid[:45, :45].T.reshape(-1, 2)
        self.grid_point_tree = shapely.STRtree(shapely.points(grid_coords))
        self.grid_points = shapely.points(grid_coords + 0.5)

    def time_tree_create(self):
        tree = shapely.STRtree(self.polygons)
        tree.query(shapely.points(0, 0))

    def time_tree_query(self):
        self.tree.query(self.polygons)

    def time_tree_query_intersects(self):
        self.tree.query(self.polygons, predicate="intersects")

    def time_tree_query_within(self):
        self.tree.query(self.polygons, predicate="within")

    def time_tree_query_contains(self):
        self.tree.query(self.polygons, predicate="contains")

    def time_tree_query_overlaps(self):
        self.tree.query(self.polygons, predicate="overlaps")

    def time_tree_query_crosses(self):
        self.tree.query(self.polygons, predicate="crosses")

    def time_tree_query_touches(self):
        self.tree.query(self.polygons, predicate="touches")

    def time_tree_query_covers(self):
        self.tree.query(self.polygons, predicate="covers")

    def time_tree_query_covered_by(self):
        self.tree.query(self.polygons, predicate="covered_by")

    def time_tree_query_contains_properly(self):
        self.tree.query(self.polygons, predicate="contains_properly")

    def time_tree_nearest_points(self):
        self.point_tree.nearest(self.points)

    def time_tree_nearest_points_equidistant(self):
        self.grid_point_tree.nearest(self.grid_points)

    def time_tree_nearest_points_equidistant_manual_all(self):
        # This benchmark approximates query_nearest for equidistant results
        # starting from singular nearest neighbors and searching for more
        # within same distance.

        # try to find all equidistant neighbors ourselves given single nearest
        # result
        l, r = self.grid_point_tree.nearest(self.grid_points)
        # calculate distance to nearest neighbor
        dist = shapely.distance(
            self.grid_points.take(l), self.grid_point_tree.geometries.take(r)
        )
        # include a slight epsilon to ensure nearest are within this radius
        b = shapely.buffer(self.grid_points, dist + 1e-8)

        # query the tree for others in the same buffer distance
        left, right = self.grid_point_tree.query(b, predicate="intersects")
        dist = shapely.distance(
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

    def time_tree_query_nearest_points(self):
        self.point_tree.query_nearest(self.points)

    def time_tree_query_nearest_points_equidistant(self):
        self.grid_point_tree.query_nearest(self.grid_points)

    def time_tree_query_nearest_points_small_max_distance(self):
        # returns >300 results
        self.point_tree.query_nearest(self.points, max_distance=5)

    def time_tree_query_nearest_points_large_max_distance(self):
        # measures the overhead of using a distance that would encompass all tree points
        self.point_tree.query_nearest(self.points, max_distance=1000)

    def time_tree_nearest_poly(self):
        self.tree.nearest(self.points)

    def time_tree_query_nearest_poly(self):
        self.tree.query_nearest(self.points)

    def time_tree_query_nearest_poly_small_max_distance(self):
        # returns >300 results
        self.tree.query_nearest(self.points, max_distance=5)

    def time_tree_query_nearest_poly_python(self):
        # returns all input points

        # use an arbitrary search tolerance that seems appropriate for the density of
        # geometries
        tolerance = 200
        b = shapely.buffer(self.points, tolerance, quad_segs=1)
        left, right = self.tree.query(b)
        dist = shapely.distance(self.points.take(left), self.polygons.take(right))

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

        # arrays are now roughly representative of what tree.query_nearest would provide, though
        # some query_nearest neighbors may be missed if they are outside tolerance
