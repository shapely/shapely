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
import random
from typing import List, Tuple

from matplotlib import pyplot as plt
from classes import PriorityQueueItem, ComperablePolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry import box as shapely_box
from threading import Lock
from matplotlib.patches import Polygon as MplPolygon

import concurrent.futures



# Set up logging
logger = logging.getLogger("polygon_partitioning")


def partition_polygon(polygon: Polygon):
    """
    The main function that partitions the given rectilinear polygon into rectangles of minimum total edge length.

    """
    # Check if the polygon is already a rectangle
    if polygon.area == polygon.minimum_rotated_rectangle.area:
        logger.error("The polygon is already a rectangle.")
        return None

    rectilinear_polygon = RectilinearPolygon(polygon)
    if not rectilinear_polygon.is_rectilinear():
        logger.error("The polygon is not rectilinear.")
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
        self.lock = Lock()

    def iterative_partition(
        self, candidate_point: Point, partition_list: List[LineString]
    ):
        initial_priority = 0  # Start with 0 priority to explore promising paths first
        pq = []
        initial_item = PriorityQueueItem(
            initial_priority,
            partition_list,
            [(ComperablePolygon(self.polygon), candidate_point)],
            0,
        )
        
        heapq.heappush(pq, initial_item)
        
        while pq:
            item: PriorityQueueItem = heapq.heappop(pq)
            
            if not item.candidates_and_figures:
                continue
            
            if item.priority >= self.min_partition_length:
                continue  # Prune branches that can't improve the current best
            
            figure_candidate = item.candidates_and_figures.pop()
            new_items = self.process_candidate(item, figure_candidate)
            
            for new_item in new_items:
                if new_item.splited_area == self.polygon.area:
                    # We found a complete partition
                    with self.lock:
                        if new_item.priority < self.min_partition_length:
                            self.min_partition_length = new_item.priority
                            self.best_partition = new_item.partition_list
                else:
                    heapq.heappush(pq, new_item)

    def process_candidate(
        self, item: PriorityQueueItem, figure_candidate: Tuple[ComperablePolygon, Point]
    ):
        partial_figure, candidate_point = figure_candidate
        
        matching_and_blocks = self.find_matching_point(candidate_point, partial_figure)
        new_items = []
        
        for matching_point, blocked_rect in matching_and_blocks:
            new_internal_edges = self.get_new_internal_edges(blocked_rect)
            if not new_internal_edges:
                continue
            
            splited_area, new_figures = self.split_polygon(
                partial_figure, blocked_rect, new_internal_edges
            )
            
            current_area = splited_area + item.splited_area
            new_partition_list = item.partition_list + new_internal_edges
            new_total_length = sum(line.length for line in new_partition_list)
            
            # Improved lower bound estimation
            lower_bound = new_total_length + self.estimate_remaining_length(new_figures)
            
            if lower_bound >= self.min_partition_length:
                continue
            
            new_priority = self.check_priority(new_total_length, new_figures)
            
            new_candidates_and_figures = item.candidates_and_figures + [
                (ComperablePolygon(fig), point) for fig, point in new_figures
            ]
            new_item = PriorityQueueItem(
                new_priority,
                new_partition_list,
                new_candidates_and_figures,
                current_area,
            )
            new_items.append(new_item)
        
        return new_items

    def estimate_remaining_length(self, figures):
        # Implement a function to estimate the minimum additional length
        # needed to complete the partition based on the remaining figures
        total_perimeter = sum(fig.length for fig, _ in figures)
        return total_perimeter / 2  # A simple heuristic, can be improved

    def check_priority(self, new_total_length, new_figures):
        total_area = sum(polygon.area for polygon, _ in new_figures)
        total_perimeter = sum(polygon.length for polygon, _ in new_figures)
        return (new_total_length + total_perimeter / 4) / total_area if total_area > 0 else float("inf")

    def split_polygon(
        self, polygon: Polygon, blocked_rect: Polygon, lines: list[LineString]
    ) -> list[tuple[Polygon, Point]]:
        """
        Split a polygon with multiple lines and return polygons with candidate points.

        Args:
            polygon (Polygon): The polygon to be split.
            lines (list[LineString]): A list of lines to split the polygon with.

        Returns:
            list[tuple[Polygon, Point]]: A list of tuples, each containing a polygon
                                        and a candidate point for further processing.
        """

        splited_area = blocked_rect.area
        split_result = polygon.difference(blocked_rect)
        logger.debug(f"Split result: {split_result}")

        # Handle different possible types of split_result
        if isinstance(split_result, Polygon):
            split_polygons = [split_result]
        elif isinstance(split_result, MultiPolygon):
            split_polygons = list(split_result.geoms)
        elif isinstance(split_result, GeometryCollection):
            split_polygons = [
                geom for geom in split_result.geoms if isinstance(geom, Polygon)
            ]
        else:
            split_polygons = []

        # Process the split polygons and find candidate points
        result = []
        for poly in split_polygons:
            if poly.is_empty or not poly.is_valid:
                continue
            if self.is_rectangle(poly):
                splited_area += poly.area
            else:
                candidate_point = self.find_candidate_point_from_boundary(poly, lines)
                if candidate_point:
                    result.append((poly, candidate_point))
                else:
                    logger.debug(f"No candidate point found for polygon: {poly}")

        logger.debug(f"Result: {result}")
        return splited_area, result

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
            return Point(boundary_lines[0].coords[1])
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

    def find_matching_point(
        self, candidate: Point, partial_figure: Polygon
    ) -> list[tuple[Point, Polygon]]:
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

        for point in self.grid_points:
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


@staticmethod
def plot_and_partition(polygon: Polygon):
    """
    Plot the given polygon and its partition.

    Args:
        polygon (Polygon): The polygon to plot and partition.
    """
    partition_result = partition_polygon(polygon)

    if not partition_result:
        # Process the partition result
        logger.error("Partition could not be found.")
    else:
        logger.debug("Partition result:", partition_result)

    #plot_poly.plotting(polygon, partition_result)
    
    fig, ax = plt.subplots()

    # Create a Polygon patch and add it to the plot
    polygon_patch = MplPolygon(
        list(polygon.exterior.coords),
        closed=True,
        edgecolor="blue",
        facecolor="lightblue",
    )
    ax.add_patch(polygon_patch)

    # Plot the LineString objects in a different color
    for line in partition_result:
        x, y = line.xy
        ax.plot(x, y, color="red")

    # Calculate the bounds of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Add a margin to the bounds to ensure the shape is not cut off
    margin = 1  # Adjust the margin as needed
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)

    # Set the aspect of the plot to be equal
    ax.set_aspect("equal")

    # Center the plot
    ax.set_xlim((min_x + max_x) / 2 - (max_x - min_x) / 2 - margin, (min_x + max_x) / 2 + (max_x - min_x) / 2 + margin)
    ax.set_ylim((min_y + max_y) / 2 - (max_y - min_y) / 2 - margin, (min_y + max_y) / 2 + (max_y - min_y) / 2 + margin)

    # Show the plot
    plt.show()
    


def generate_rectilinear_polygon(
    canvas_width: int = 20,
    canvas_height: int = 20,
    min_rectangles: int = 5,
    max_rectangles: int = 15,
    min_size: int = 1,
    max_size: int = 5,
    increments: int = 100,
) -> Polygon:
    """
    Generate a rectilinear polygon composed of touching rectangles.

    :param canvas_width: Width of the canvas
    :param canvas_height: Height of the canvas
    :param min_rectangles: Minimum number of rectangles
    :param max_rectangles: Maximum number of rectangles
    :param min_size: Minimum size of a rectangle
    :param max_size: Maximum size of a rectangle
    :param increments: Size increment for positioning and sizing
    :return: Shapely Polygon representing the rectilinear polygon
    source: https://github.com/stepmat/rectilinear_polygon_generator/blob/master/generate_polygon.py
    """
    num_rect = random.randint(min_rectangles, max_rectangles)
    boxes = []

    for _ in range(num_rect):
        valid = False
        while not valid:
            width = random.randint(min_size, max_size) * increments
            height = random.randint(min_size, max_size) * increments
            pos_x = (
                random.randint(max_size + 1, canvas_width - max_size - 1) * increments
            )
            pos_y = (
                random.randint(max_size + 1, canvas_height - max_size - 1) * increments
            )

            touching = False
            for box in boxes:
                if abs(pos_x - box[0]) * 2 < (width + box[2]) and abs(
                    pos_y - box[1]
                ) * 2 < (height + box[3]):
                    touching = True
                    break

            if touching or not boxes:
                valid = True

        boxes.append((pos_x, pos_y, width, height))

    # Extend rectangles to the ground
    bottom_ground = max(box[1] + box[3] // 2 for box in boxes)

    for i, box in enumerate(boxes):
        x, y, width, height = box
        bottom = y + height // 2

        if bottom < bottom_ground:
            intersects = any(
                x - width // 2 < other[0] + other[2] // 2
                and x + width // 2 > other[0] - other[2] // 2
                and bottom < other[1] + other[3] // 2
                for j, other in enumerate(boxes)
                if i != j
            )

            if not intersects:
                new_height = height + (bottom_ground - bottom) * 2
                new_y = y + (bottom_ground - bottom)
                boxes[i] = (x, new_y, width, new_height)

    # Convert rectangles to Shapely Polygons and union them
    shapely_polygons = [
        shapely_box(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y, w, h in boxes
    ]
    union_polygon = shapely_polygons[0]
    for polygon in shapely_polygons[1:]:
        union_polygon = union_polygon.union(polygon)

    # Simplify the polygon to remove unnecessary points
    simplified_polygon = union_polygon.simplify(1, preserve_topology=True)

    return simplified_polygon


if __name__ == "__main__":

    # Run the doctests
    doctest.testmod()
    polygon1 = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
    polygon2 = Polygon([(1, 5), (1, 4), (3, 4), (3, 2), (5, 2), (5, 1), (8, 1), (8, 5)])
    polygon3 = Polygon([(0, 4), (2, 4), (2, 0), (5, 0), (5, 4), (7, 4), (7, 5), (0, 5)])
    polygon4 = Polygon([(1, 5), (1, 4), (3, 4), (3, 2), (5, 2), (5, 1), (8, 1), (8, 5)])
    polygon5 = Polygon(
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
            (9, 0),
            (10, 0),
            (10, 5),
            (9, 5),
            (9, 4),
            (8, 4),
            (8, 5),
        ]
    )  
    polygon6 = Polygon([(1, 1), (1, 9), (9, 9), (9, 1)])  # Rectangle
    polygon7 = Polygon([(1, 1), (1, 9), (9, 9), (9, 7)])  # not rectlinear polygno

    #poly = generate_rectilinear_polygon()
    plot_and_partition(polygon5)
