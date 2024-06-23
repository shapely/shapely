"""
An implementation of the algorithms in:
"Minimum edge length partitioning of rectilinear polygons", by Lingas, Pinter, Rivest and Shamir from 1982
https://people.csail.mit.edu/rivest/pubs/LPRS82.pdf
Programmer: Dvir Borochov
Date: 10/6/24 
"""

import doctest
import logging
import heapq
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import pyplot as plt
from classes import PriorityQueueItem, ComperablePolygon
from shapely.ops import unary_union
from shapely.geometry import Polygon, LineString, Point
"""

TODO:
- change the PriorityQueueItem strucure to be : (priority, partition_list, [(figure, candidate_point)]  
"""

logger = logging.getLogger("polygon_partitioning")


def partition_polygon(polygon: Polygon):
    """
    The main function that partitions the given rectilinear polygon into rectangles of minimum total edge length.

    """
    rectilinear_polygon = RectilinearPolygon(polygon)
    if not rectilinear_polygon.is_rectilinear():
        return None

    initial_convex_point = rectilinear_polygon.find_convex_points()
  #  rectilinear_polygon.grid_points = rectilinear_polygon.get_grid_points()
    rectilinear_polygon.iterative_partition(initial_convex_point, [])
    return rectilinear_polygon.best_partition


class RectilinearPolygon:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon  # the main rectilinear polygon
        self.min_partition_length = float("inf")
        self.best_partition = []
        self.grid_points = set(self.get_grid_points())

    def iterative_partition(
        self, candidate_point: Point, partition_list: list[LineString]
    ):
        """
        Iteratively partitions the given candidate points and updates the best partition.

        Args:
            initial_candidate_points (list): The initial list of candidate points to consider for partitioning.
            initial_partition_list (list): The initial partition list.

        Returns:
            None (just update the best partition).
        """

        initial_priority = float("inf")
        pq = []  # Priority queue
        partial_figures = []
        initial_item = PriorityQueueItem(  # TODO: change the stracture of the queue to be : (priority, partition_list, [(figure, candidate_point)]
            initial_priority, [], [(ComperablePolygon(self.polygon), candidate_point)]
        )

        heapq.heappush(pq, initial_item)
        while pq:
            item: PriorityQueueItem = heapq.heappop(pq)
            partition_list = item.partition_list
            while item.candidates_and_figures:
                figure_candidate = item.candidates_and_figures.pop()
                partial_figure = figure_candidate[0]
                candidate_point = figure_candidate[1]

                logger.info(f"Processing candidate point: {candidate_point}")

                matching_and_blocks = self.find_matching_point(
                    candidate_point, partial_figure
                )
                if not matching_and_blocks:
                    continue
                for matching_point, blocked_rect in matching_and_blocks:
                    costructed_lines = self.get_new_internal_edges(blocked_rect)

                    if not costructed_lines:
                        logger.warning("No new internal edges found.")
                        continue

                    partition_list += (
                        costructed_lines  # adding the new lines to the partition list
                    )

                    new_figure_candidate = self.split_polygon(
                        partial_figures, costructed_lines
                    )

                new_total_length = sum(line.length for line in partition_list)
                new_priority = self.check_priority(
                    new_total_length, new_figure_candidate
                )
                if new_total_length >= self.min_partition_length:
                    logger.info(
                        f"cutting the search -  the new partition is longer than the best partition"
                    )
                    continue

                new_item = PriorityQueueItem(
                    new_priority, partition_list, new_figure_candidate
                )
                heapq.heappush(pq, new_item)

            if self.is_partitioned_into_rectangles(partial_figure, partition_list):
                if new_total_length < self.min_partition_length:
                    self.min_partition_length = new_total_length
                    self.best_partition = [line for line in partition_list]
                    logger.debug(f"New best partition found: {self.best_partition}")

                else:
                    logger.warning(f"Not the best : {partition_list}")
                    continue

    def check_priority(self, new_total_length, new_figure_candidate):
        """
        Calculate the ratio between the area of the new figure candidate and the new total length.

        Args:
            new_total_length (float): The new total length.
            new_figure_candidate (Polygon): The new figure candidate.

        Returns:
            float: The ratio between the area and the length.

        """
        total_area = sum(polygon.area for polygon, _ in new_figure_candidate)
        return new_total_length / total_area if total_area > 0 else float('inf')




    def split_polygon(self, polygon: Polygon, lines: list[LineString]) -> list[tuple[Polygon, Point]]:
        """
        Split a polygon with multiple lines and return non-rectangular polygons with candidate points.

        Args:
            polygon (Polygon): The polygon to be split.
            lines (list[LineString]): A list of lines to split the polygon with.

        Returns:
            list[tuple[Polygon, Point]]: A list of tuples, each containing a non-rectangular polygon
                                        and a candidate point for further processing.
        """
        # Ensure polygon is a Polygon object
        if isinstance(polygon, list):
            polygon = Polygon(polygon)
        
        # Combine all lines into a single MultiLineString
        all_lines = unary_union(lines)

        # Split the polygon
        split_parts = polygon.difference(all_lines)

        # Convert to a list of polygons
        if split_parts.geom_type == "Polygon":
            split_polygons = [split_parts]
        elif split_parts.geom_type == "MultiPolygon":
            split_polygons = list(split_parts.geoms)
        else:
            split_polygons = []

        # Filter out rectangular polygons and find candidate points
        result = []
        for poly in split_polygons:
            if poly.is_empty:
                continue
            if not self.is_rectangle(poly):
                candidate_point = self.find_candidate_point_from_boundary(poly, lines)
                if candidate_point:
                    result.append((poly, candidate_point))
            else:
                # For rectangles, use the centroid as the candidate point
                result.append((poly, poly.centroid))

        return result
    def find_candidate_point_from_boundary(self, polygon, lines):
        """
        Find a candidate point based on the lines that form the boundary of the polygon.

        Args:
            polygon (Polygon): The polygon to find a candidate point for.
            lines (list[LineString]): The original lines used for splitting.

        Returns:
            Point: A suitable candidate point, or None if not found.
        """
        boundary_lines = []
        for line in lines:
            if polygon.boundary.intersects(line):
                intersection = polygon.boundary.intersection(line)
                if intersection.geom_type == "LineString":
                    boundary_lines.append(intersection)
                elif intersection.geom_type == "MultiLineString":
                    boundary_lines.extend(list(intersection.geoms))

        if len(boundary_lines) == 1:
            # If there's only one line, return one of its endpoints
            return Point(boundary_lines[0].coords[0])
        elif len(boundary_lines) >= 2:
            # If there are two or more lines, return their intersection
            intersection_point = boundary_lines[0].intersection(boundary_lines[1])
            if intersection_point.geom_type == "Point":
                return intersection_point
            else:
                # If lines don't intersect at a point, return an endpoint of the first line
                return Point(boundary_lines[0].coords[0])
        else:
            # If no boundary lines are found, return None
            return None

    def get_new_internal_edges(self, blocked_rect: Polygon) -> list[LineString]:
        """
        Get the new internal edges that were added to the main polygon by the given blocked rectangle.

        Args:
            blocked_rect (Polygon): The blocked rectangle within the main polygon.

        Returns:
            list[LineString]: A list of new internal edges (LineString instances) added by the blocked rectangle.

        >>> polygon = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> blocked_rect = Polygon([(2,4), (6,4), (6, 0), (2, 0)])
        >>> new_internal_edges = rect_polygon.get_new_internal_edges(blocked_rect)
        >>> new_internal_edges
        [<LINESTRING (2 4, 6 4)>]
        """
        # Create the four edges of the blocked rectangle
        edge1 = LineString(
            [blocked_rect.exterior.coords[0], blocked_rect.exterior.coords[1]]
        )
        edge2 = LineString(
            [blocked_rect.exterior.coords[1], blocked_rect.exterior.coords[2]]
        )
        edge3 = LineString(
            [blocked_rect.exterior.coords[2], blocked_rect.exterior.coords[3]]
        )
        edge4 = LineString(
            [blocked_rect.exterior.coords[3], blocked_rect.exterior.coords[0]]
        )

        # Find the segments of each edge that are not part of the main polygon boundary
        internal_edges = []
        for edge in [edge1, edge2, edge3, edge4]:
            difference = edge.difference(self.polygon.boundary)
            if not difference.is_empty:
                if difference.geom_type == "MultiLineString":
                    for line in difference.geoms:  # Iterate over MultiLineString
                        internal_edges.append(line)
                else:
                    internal_edges.append(difference)

        return internal_edges

    def is_rectilinear(self) -> bool:
        """
        Check if the polygon is rectilinear (each internal engle is 90 degrees or 270 degrees)


        Returns:
            bool: True if the polygon is rectilinear, False otherwise.
        >>> polygon = Polygon([(0, 0), (0, 2), (0, 4), (2, 4), (2, 2), (2, 0)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_rectilinear()
        True

        >>> polygon = Polygon([(0, 0), (0, 4), (4, 4), (4, 0), (2,2)])
        >>> partitions = [LineString([(0, 2), (4, 3)])]  # Non-rectangular partition
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_rectilinear()
        False


        """
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
        >>> convex_points



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
        """
        Calculate the cross product to determine if the vertex is concave

        Args:
            i (int): Index of the vertex in the list of coordinates (0-based index)
            coords (list): List of coordinate tuples representing the polygon vertices

        Returns:
            bool: True if the vertex is concave, False otherwise

        >>> polygon = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_concave_vertex(2, list(polygon.exterior.coords))
        True
        """
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        x3, y3 = coords[(i + 1) % len(coords)]

        cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        return cross_product < 0

    def extend_lines(self, point):
        """
        extends the horizontal and vertical lines from the given point to the polygon bounds.

        Args:
            point (Point): The point from which to extend the lines.

        Returns:
            tuple: A tuple containing the extended horizontal and vertical lines.
        """
        x, y = point.x, point.y
        min_x, min_y, max_x, max_y = self.polygon.bounds

        # Create extended horizontal and vertical lines
        horizontal_line = LineString([(min_x, y), (max_x, y)])
        vertical_line = LineString([(x, min_y), (x, max_y)])

        return horizontal_line, vertical_line

    def get_grid_points(self):
        """
        the method returns the grid points inside the polygon
        the grid points are the intersection points of the polygon sides with the extended lines from the concave vertices and the polgyn vertices itself.


        >>> polygon = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> grid_points = rect_polygon.get_grid_points()
        >>> grid_points
        [<POINT (2 4)>, <POINT (8 4)>, <POINT (0 4)>, <POINT (2 0)>, <POINT (6 4)>, <POINT (0 6)>, <POINT (2 6)>, <POINT (8 6)>, <POINT (6 0)>, <POINT (6 6)>]


        """
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

    def find_matching_point(self, candidate: Point, partial_figure: Polygon) -> list[tuple[Point, Polygon]]:
        """
        Finds matching points on the grid inside the polygon and kitty-corner to the candidate point
        within a blocked rectangle inside the polygon.

        Args:
            candidate (Point): The candidate point.

        Returns:
            list: List of tuples containing Points representing the matching points and their corresponding blocked rectangles.
        """
        matching_and_blocks = []
        relevant_grid_points = []
        boundary = partial_figure.boundary

        # Iterate over the exterior coordinates of the partial figure
        for coord in partial_figure.exterior.coords:
            point = Point(coord)
            # Check if the point is within the partial figure or on the boundary
            if point.within(partial_figure) or point.intersects(boundary):
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
                    matching_and_blocks.append((point, blocked_rect))

        return matching_and_blocks

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

    def is_partitioned_into_rectangles(
        self, partial_figure: Polygon, partitions: list[LineString]
    ) -> bool:
        """
        Checks if the polygon is partitioned into rectangles.

        Args:
            partitions (list[LineString]): A list of LineString objects representing the partitions.

        Returns:
            bool: True if the polygon is partitioned into rectangles, False otherwise.

        """
        if not partitions:
            logger.info("No partitions provided.")
            return self.is_rectangle(self.polygon)
        polygons = self.split_polygon(partial_figure, partitions)
        return all(self.is_rectangle(poly) for poly in polygons)



if __name__ == "__main__":

    # Run the doctests
    doctest.testmod()
    polygon1 = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
    polygon2 = Polygon([(1, 5), (1, 4), (3, 4), (3, 2), (5, 2), (5, 1), (8, 1), (8, 5)])
    polygon3 = Polygon([(0, 4), (2, 4), (2, 0), (5, 0), (5, 4), (7, 4), (7, 5), (0, 5)])
    """
    """
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

    partition_result = partition_polygon(polygon4)
    # partition_result_2 = partition_polygon(polygon2)
    # partition_result_3 = partition_polygon(polygon3)

    if partition_result:
        # Process the partition result
        print("Partition result:", partition_result)
    else:
        print("Partition could not be found.")

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Create a Polygon patch and add it to the plot
    polygon_patch = MplPolygon(
        list(polygon4.exterior.coords),
        closed=True,
        edgecolor="blue",
        facecolor="lightblue",
    )
    ax.add_patch(polygon_patch)

    # Plot the LineString objects in a different color
    for line in partition_result:
        x, y = line.xy
        ax.plot(x, y, color="red")

    # Set the limits of the plot
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)

    # Set the aspect of the plot to be equal
    ax.set_aspect("equal")

    # Show the plot
    plt.show()
