import logging
import heapq
from shapely.ops import split
from shapely.geometry import (
    Polygon,
    LineString,
    Point,
    GeometryCollection,
    MultiPolygon,
)
from classes import PriorityQueueItem, ComparableLineString, Comperablepolygon


logger = logging.getLogger("polygon_partitioning")


class RectilinearPolygon:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        self.min_partition_length = float("inf")
        self.best_partition = []
        self.grid_points = []

    def is_rectilinear(self) -> bool:
        coords = list(self.polygon.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            if not (x1 == x2 or y1 == y2):
                return False
        return True

    def find_convex_points(self):
        """
        Finds the convex points in the given rectilinear polygon.

        Returns:
            list: List of Points representing the convex points of the polygon.

        >>> polygon = Polygon([(0, 0), (0, 2), (0, 4), (2, 4), (2, 2), (2, 0)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> convex_points = rect_polygon.find_convex_points()
        >>> convex_points == [Point(0, 0), Point(0, 4), Point(2, 4), Point(2, 0)]
        True
        """
        coords = list(
            self.polygon.exterior.coords[:-1]
        )  # Exclude the last point which is the same as the first

        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[
                (i + 1) % len(coords)
            ]  # Get the next point, wrapping around to the first point at the end
            x3, y3 = coords[(i + 2) % len(coords)]  # Get the point after the next one

            # Calculate the cross product of vectors (x2 - x1, y2 - y1) and (x3 - x2, y3 - y2)
            cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)

            # Check if the cross product is positive (indicating a left turn)
            if cross_product > 0:
                return Point(x2, y2)

    def is_concave_vertex(self, i, coords):
        # Calculate the cross product to determine if the vertex is concave
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        x3, y3 = coords[(i + 1) % len(coords)]

        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        return cross_product < 0

    def extend_lines(self, point):
        x, y = point.x, point.y
        min_x, min_y, max_x, max_y = self.polygon.bounds

        # Create extended horizontal and vertical lines
        horizontal_line = LineString([(min_x, y), (max_x, y)])
        vertical_line = LineString([(x, min_y), (x, max_y)])

        return horizontal_line, vertical_line

    def get_grid_points(self):
        coords = list(self.polygon.exterior.coords)
        grid_points = set(coords)

        # Collect extended lines from concave vertices
        extended_lines = []
        for i in range(len(coords) - 1):
            if self.is_concave_vertex(i, coords):
                concave_vertex = Point(coords[i])
                horizontal_line, vertical_line = self.extend_lines(concave_vertex)
                extended_lines.append(horizontal_line)
                extended_lines.append(vertical_line)

        # Find intersection points of extended lines with polygon sides
        for line in extended_lines:
            for j in range(len(coords) - 1):
                polygon_side = LineString([coords[j], coords[j + 1]])
                intersection = line.intersection(polygon_side)
                if isinstance(intersection, Point):
                    grid_points.add((intersection.x, intersection.y))

        # Add any remaining intersections of extended lines with each other
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                intersection = extended_lines[i].intersection(extended_lines[j])
                if isinstance(intersection, Point) and intersection.within(
                    self.polygon
                ):
                    grid_points.add((intersection.x, intersection.y))

        return [Point(x, y) for x, y in grid_points]

    def find_matching_point(self, candidate: Point, partial_figures):  # -> list[Point]:
        """
        Finds matching points on the grid inside the polygon and kitty-corner to the candidate point
        within a blocked rectangle inside the polygon.

        Args:
            candidate (Point): The candidate point.

        Returns:
            list: List of Points representing the matching points.
        """
        matching_points = []
        relevant_grid_points = []
        """
        Find the matching points on the grid that are not inside or on the partial_figures (that is a [Polygon]).
        We want to find the matching points that not in other partial figures (to make a new partial figure)
        """
        for point in self.grid_points:
            if not any(
                point.within(polygon) or point.touches(polygon)
                for polygon in partial_figures
            ):
                relevant_grid_points.append(point)

        for point in relevant_grid_points:
            if point != candidate:  # Exclude candidate point
                # Check if the point is kitty-corner to the candidate within a blocked rectangle
                min_x = min(candidate.x, point.x)
                max_x = max(candidate.x, point.x)
                min_y = min(candidate.y, point.y)
                max_y = max(candidate.y, point.y)
                blocked_rect = Polygon(
                    [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
                )

                if blocked_rect.within(self.polygon):
                    matching_points.append(point)

        return matching_points

    def find_blocked_rectangle(self, candidate: Point, matching: Point):
        # Create the four edges of the potential blocked rectangle
        edge1 = LineString([candidate, Point(candidate.x, matching.y)])
        edge2 = LineString([Point(candidate.x, matching.y), matching])
        edge3 = LineString([matching, Point(matching.x, candidate.y)])
        edge4 = LineString([Point(matching.x, candidate.y), candidate])

        # Collect all edges
        all_edges = [edge1, edge2, edge3, edge4]

        # Convert each LineString to a list of coordinate tuples
        coords_list = [list(edge.coords) for edge in all_edges]

        # Concatenate the coordinate lists, removing duplicates
        coords = list(dict.fromkeys(sum(coords_list, [])))

        # Construct the partial polygon
        polygon = Polygon(coords)

        # Find the segments of each edge that are not part of the polygon boundary
        internal_edges = []
        for edge in all_edges:
            difference = edge.difference(self.polygon.boundary)
            if not difference.is_empty:
                if difference.geom_type == "MultiLineString":
                    for (
                        line
                    ) in difference.geoms:  # Use .geoms to iterate over MultiLineString
                        internal_edges.append(line)
                else:
                    internal_edges.append(difference)

        # Return only the new internal edges that are not part of the polygon boundary
        return polygon, internal_edges if internal_edges else None

    def split_polygon(self, lines):
        """
        Splits the polygon using the given lines.
        Args:
            lines (list[LineString]): A list of LineString objects to split the polygon.
        Returns:
            list: List of Polygon objects resulting from the split operation.
        """
        if not lines:
            return [self.polygon]

        polygons = [self.polygon]

        for line in lines:
            new_polygons = []
            for polygon in polygons:
                split_result = split(polygon, line)
                if isinstance(split_result, GeometryCollection):
                    new_polygons.extend(
                        [
                            geom
                            for geom in split_result.geoms
                            if isinstance(geom, Polygon)
                        ]
                    )
                elif isinstance(split_result, (Polygon, MultiPolygon)):
                    if isinstance(split_result, MultiPolygon):
                        new_polygons.extend(list(split_result.geoms))
                    else:
                        new_polygons.append(split_result)
            polygons = new_polygons

        return polygons

    def find_candidate_point(self, constructed_lines: list[LineString]) -> list[Point]:
        """
        Find candidate point based on the constructed lines.

        Args:
            constructed_lines (list[LineString]): A list of constructed lines.

        Returns:
            list[Point]: A list of candidate points.

        Raises:
            None

        """
        constructed_lines = [normalize_line(line) for line in constructed_lines]

        if len(constructed_lines) == 1:
            return Point(constructed_lines[0].coords[1])
        elif len(constructed_lines) == 2:
            line1 = constructed_lines[0]
            line2 = constructed_lines[1]
            common_point = line1.intersection(line2)
            if common_point:
                return Point(common_point)
        else:
            return self.find_unsplit_subpolygon_point(constructed_lines)
            
    def find_unsplit_subpolygon_point(self, constructed_lines: list[LineString]) -> Point:
        """
        Finds the sub-polygon that isn't split into rectangles and returns one of the points
        of the constructed_lines to continue splitting the rectilinear polygon.

        Args:
            constructed_lines (list[LineString]): A list of constructed lines.

        Returns:
            Point: One of the points of the constructed_lines.
        """
        polygons = self.split_polygon(constructed_lines)

        for poly in polygons:
            if not self.is_rectangle(poly):
                # Find a point from the constructed_lines that is within the non-rectangular sub-polygon
                for line in constructed_lines:
                    for point in line.coords:
                        point_obj = Point(point)
                        if point_obj.within(poly):
                            return point_obj

        # If all sub-polygons are rectangles, return None
        return None



    def is_rectangle(self, poly: Polygon) -> bool:
        """
        Check if a given polygon is a rectangle.
        source: https://stackoverflow.com/questions/62467829/python-check-if-shapely-polygon-is-a-rectangle
        Args:
            poly (Polygon): The polygon to be checked.

        Returns:
            bool: True if the polygon is a rectangle, False otherwise.
        """
        return poly.area == poly.minimum_rotated_rectangle.area

    def is_partitioned_into_rectangles(self, partitions: list[LineString]) -> bool:
        """
        Checks if the polygon is partitioned into rectangles.

        Args:
            partitions (list[LineString]): A list of LineString objects representing the partitions.

        Returns:
            bool: True if the polygon is partitioned into rectangles, False otherwise.

        >>> polygon = Polygon([(0, 0), (0, 6), (6, 6), (6, 0)])
        >>> partitions = [LineString([(0, 3), (6, 3)]), LineString([(3, 0), (3, 6)])]
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_partitioned_into_rectangles(partitions)
        True


        >>> polygon = Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
        >>> partitions = [LineString([(0, 2), (4, 3)])]  # Non-rectangular partition
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_partitioned_into_rectangles(partitions)
        False

        """
        if not partitions:
            return self.is_rectangle(self.polygon)
        polygons = self.split_polygon(partitions)
        return all(self.is_rectangle(poly) for poly in polygons)
             

    def partition(self):
        if not self.is_rectilinear():
            return None

        initial_convex_point = self.find_convex_points()
        self.grid_points = self.get_grid_points()
        self.iterative_partition(initial_convex_point, [])

        return self.best_partition
    
    def iterative_partition(self, candidate_point, partition_list):
        """
        Iteratively partitions the given candidate points and updates the best partition.

        Args:
            initial_candidate_points (list): The initial list of candidate points to consider for partitioning.
            initial_partition_list (list): The initial partition list.

        Returns:
            None (just update the best partition).
        """
        total_length = sum(line.length for line in partition_list)
        initial_priority = total_length if total_length != 0 else float("inf")
        pq = []  # Priority queue
        partial_figures = []
        initial_item = PriorityQueueItem(
            initial_priority,
            tuple(candidate_point.coords[0]),
            [ComparableLineString(line) for line in partition_list],
            [Comperablepolygon(figure) for figure in partial_figures],
        )
        heapq.heappush(pq, initial_item)

        # Transform list into a heap
        heapq.heapify(pq)
        while pq:
            priority_item = heapq.heappop(pq)
            candidate_point_tuple = priority_item.candidate_point_tuple
            partition_list = priority_item.partition_list
            partial_figures = priority_item.partial_figures
            
            # Convert back to Point object
            candidate_point = Point(candidate_point_tuple)  
            matching_points = self.find_matching_point(candidate_point, partial_figures)
            if not matching_points:
                continue

            for matching_point in matching_points:
                new_partial_figure, new_lines = self.find_blocked_rectangle(
                    candidate_point, matching_point
                )
                if new_lines is None:
                    continue
                new_figures = partial_figures + [Comperablepolygon(new_partial_figure)]

                # Normalize lines before adding to the set to avoid duplication lines
                normalized_partition_list = {
                    normalize_line(ComparableLineString(line)) for line in new_lines
                }
                new_partition_list = [ComparableLineString(line) for line in normalized_partition_list.union(partition_list)]

                # Calculate the priority for the new state
                new_total_length = sum(line.length for line in new_partition_list)
                new_priority = new_total_length if new_total_length != 0 else float("inf")
                #cutting the search if the new partition is longer than the best partition
                if new_priority >= self.min_partition_length:
                    continue

                if self.is_partitioned_into_rectangles(new_partition_list):
                    if new_total_length < self.min_partition_length:
                        self.min_partition_length = new_total_length
                        self.best_partition = [line for line in new_partition_list]
                        logger.warning(f"New best partition found: {self.best_partition}")
                    else:
                       # logger.warning(f"Not the best : {new_partition_list}")
                        continue

                new_candidate_point = self.find_candidate_point(new_partition_list)
                if new_candidate_point is None or new_candidate_point.is_empty:
                    continue

                # Convert the candidate point to a tuple of coordinates
                new_candidate_point_tuple = tuple(new_candidate_point.coords[0])

                new_item = PriorityQueueItem(
                    new_priority,
                    new_candidate_point_tuple,
                    new_partition_list,
                    new_figures,
                )
                heapq.heappush(pq, new_item)



@staticmethod
def normalize_line(line: LineString) -> LineString:
    coords = list(line.coords)
    if coords[0] > coords[-1]:
        coords.reverse()
    return LineString(coords)


@staticmethod
def check_lines(
    partition_list: list[LineString], new_lines: list[LineString]
) -> list[LineString]:
    result = []
    for new_line in new_lines:
        if not any(new_line.within(line) for line in partition_list):
            result.append(new_line)
    return result


if __name__ == "__main__":
    # Create the polygon instance
    polygon1 = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
    polygon2 = Polygon([(1, 5), (1, 4), (3, 4), (3, 2), (5, 2), (5, 1), (8, 1), (8, 5)])
    polygon3 = Polygon([(0, 4), (2, 4), (2, 0), (5, 0), (5, 4), (7, 4), (7, 5), (0, 5)])
    
    
    polygon4 = Polygon(
        [
            (1, 5),
            (1, 4),
            (3, 4),
            (3, 3),
            (2, 3),
            (2, 1),
            (5, 1),
            (5, 2),
            (8, 2),
            (8, 1),
            (9, 1),
            (9, 4),
            (8, 4),
            (8, 5),
        ]
    )

    # Create a RectilinearPolygon instance
    rectilinear_polygon = RectilinearPolygon(polygon1)

    # Get the partition result``
    partition_result = rectilinear_polygon.partition()

    if partition_result:
        # Process the partition result
        print("Partition result:", partition_result)
    else:
        print("Partition could not be found.")

