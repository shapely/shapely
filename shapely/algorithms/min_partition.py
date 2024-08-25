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
import time

from matplotlib import pyplot as plt
import numpy as np
from classes import PriorityQueueItem, ComperablePolygon
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon, LineString, Point
from plot_poly import plotting
from rand_rect_poly import generate_rectilinear_polygon
from numba import jit

# Set up logging
logger = logging.getLogger("polygon_partitioning")
logger.setLevel(logging.DEBUG) # this should allow all messages to be displayed



@jit(nopython=True)
def is_rectilinear(coords):
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        if not (x1 == x2 or y1 == y2):
            return False
    return True

@jit(nopython=True)
def check_priority(new_total_length, total_area):
    """
    Calculate the ratio between the area of the new figure candidate and the new total length.

    Args:
        new_total_length (float): The new total length.
        new_figure_candidate (Polygon): The new figure candidate.

    Returns:
        float: The ratio between the area and the length.

    """
    return new_total_length / total_area if total_area > 0 else float("inf")


@staticmethod
def partition_polygon(polygon: Polygon):
    """
    The main function that partitions the given rectilinear polygon into rectangles of minimum total edge length.

    """
    # Check if the polygon is already a rectangle
    if polygon.area == polygon.minimum_rotated_rectangle.area:
        logger.error("The polygon is already a rectangle.")
        return None
    
    rectilinear_polygon = RectilinearPolygon(polygon)
    cords = np.array(rectilinear_polygon.polygon.exterior.coords)
    if not is_rectilinear(cords):
        logger.error("The polygon is not rectilinear.")
        return None

    initial_convex_point = rectilinear_polygon.find_convex_points()
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
        initial_item = PriorityQueueItem(
            initial_priority,
            partial_figures,
            [(ComperablePolygon(self.polygon), candidate_point)],
            0,  # splited area
        )

        heapq.heappush(pq, initial_item)
        while pq:
            item: PriorityQueueItem = heapq.heappop(pq)
            partition_list = item.partition_list
            candidates_and_figures = item.candidates_and_figures
            if not candidates_and_figures:
                logger.debug("No candidates and figures found.")
                continue
            figure_candidate = candidates_and_figures.pop()
            partial_figure = figure_candidate[0]
            candidate_point = figure_candidate[1]

            logger.info(f"Processing candidate point: {candidate_point}")

            matching_and_blocks = self.find_matching_point(
                candidate_point, partial_figure
            )
            new_partition_list = []
            for matching_point, blocked_rect in matching_and_blocks:
                new_internal_edges = self.get_new_internal_edges(blocked_rect)
                if not new_internal_edges:
                    logger.warning("No new internal edges found.")
                    continue

                splited_area, new_figures = self.split_polygon(
                    partial_figure, blocked_rect, new_internal_edges
                )
                logger.debug(f"new figures: {new_figures}")

                current_area = splited_area + item.splited_area
                logger.debug(f"current area: {current_area}")
                new_partition_list = (
                    partition_list + new_internal_edges
                )  # Add the new internal edges to the partition list
                new_total_length = sum(line.length for line in new_partition_list)
                new_priority = self.check_priority(new_total_length, new_figures)

                if new_total_length >= self.min_partition_length:
                    logger.debug(
                        "Cutting the search - new partition is longer than the best partition"
                    )
                    continue

                if self.polygon.area == current_area:
                    new_total_length = sum(line.length for line in new_partition_list)
                    if new_total_length < self.min_partition_length:
                        self.min_partition_length = new_total_length
                        self.best_partition = new_partition_list
                        logger.debug(f"New best partition found: {self.best_partition}")

                new_candidates_and_figures = candidates_and_figures + [
                    (ComperablePolygon(fig), point) for fig, point in new_figures
                ]
                new_item = PriorityQueueItem(
                    new_priority,
                    new_partition_list,
                    new_candidates_and_figures,
                    current_area,
                )
                heapq.heappush(pq, new_item)
                logger.warning(f"New item pushed to the priority queue")



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
        return new_total_length / total_area if total_area > 0 else float("inf")

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
        
        splited_area  = blocked_rect.area
        split_result = polygon.difference(blocked_rect)
        logger.debug(f"Split result: {split_result}")

        # Handle different possible types of split_result
        if isinstance(split_result, Polygon):
            split_polygons = [split_result]
        elif isinstance(split_result, MultiPolygon):
            split_polygons = list(split_result.geoms)
        elif isinstance(split_result, GeometryCollection):
            split_polygons = [geom for geom in split_result.geoms if isinstance(geom, Polygon)]
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
        Find matching points for a given candidate point within a polygon. The matching points
        are determined based on whether the candidate point is convex or concave.

        Args:
            candidate (Point): The candidate point.
            partial_figure (Polygon): The partial figure polygon.

        Returns:
            list[tuple[Point, Polygon]]: A list of tuples, each containing a matching point and its 
                                        corresponding blocked rectangle. 
                                        Returns an empty list if no matching points are found.
        """

        # Step 1: Filter relevant grid points within or on the boundary of the partial figure
        relevant_grid_points = [
            point for point in self.grid_points 
            if point.within(partial_figure) or point.intersects(partial_figure.boundary)
        ]
        
        # Step 2: Find incident lines of the candidate point
        constructed_lines = self.find_incident_lines(partial_figure, candidate)
        
        # Step 3: Initialize the list to store matching points
        matching_points = []
        
        # Step 4: Handle the case where the candidate point is convex

        if self.is_convex_point(partial_figure, candidate):
            blocked_rect = self.find_blocked_rectangle(partial_figure, candidate, constructed_lines[0], constructed_lines[1])
            
            # Filter grid points that are not within the blocked rectangle and are not the candidate itself
            relevant_matching_points = [
                point for point in relevant_grid_points 
                if not point.within(blocked_rect) and point != candidate
            ]
        
        # Step 5: Handle the case where the candidate point is concave
        else:
            relevant_matching_points = [
                point for point in relevant_grid_points 
                if point != candidate and (
                    self.has_projection(point, constructed_lines[0]) or
                    self.has_projection(point, constructed_lines[1])
                )
            ]
        
        # Step 6: For each relevant matching point, check if it forms a valid blocked rectangle
        for point in relevant_matching_points:
            min_x = min(candidate.x, point.x)
            max_x = max(candidate.x, point.x)
            min_y = min(candidate.y, point.y)
            max_y = max(candidate.y, point.y)

            blocked_rect = Polygon([
                (min_x, min_y), 
                (min_x, max_y), 
                (max_x, max_y), 
                (max_x, min_y)
            ])

            # Only add the point if the blocked rectangle is within the original polygon
            if blocked_rect.within(self.polygon):
                matching_points.append((point, blocked_rect))
        
        return matching_points
                    
                    
    def has_projection(self, point: Point, line: LineString) -> bool:
        """
        Check if there exists a straight line (either horizontal or vertical)
        that passes through both the given point and the provided line segment.

        Args:
            point (Point): The point to check.
            line (LineString): The line segment to check against.

        Returns:
            bool: True if such a projection exists, False otherwise.
        """
        px, py = point.x, point.y
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[1]

        # Check if a vertical line through the point intersects the line segment
        if x1 == x2 and min(y1, y2) <= py <= max(y1, y2):
            return True

        # Check if a horizontal line through the point intersects the line segment
        if y1 == y2 and min(x1, x2) <= px <= max(x1, x2):
            return True

        return False
        
        
    def is_convex_point(self, partial_figure: Polygon, point: Point) -> bool:
        """
        Determines if a given point is a convex point in the polygon.

        Args:
            point (Point): The point to check.

        Returns:
            bool: True if the point is convex, False if it is concave.
        """
        coords = list(partial_figure.exterior.coords[:-1])  # Exclude the last point, which is the same as the first

        if point not in [Point(coord) for coord in coords]:
            raise ValueError("The given point is not a vertex of the partial_figure.")
        
        point_index = coords.index((point.x, point.y))
        prev_point = Point(coords[point_index - 1])
        next_point = Point(coords[(point_index + 1) % len(coords)])

        # Vector from prev_point to point
        vec1 = (point.x - prev_point.x, point.y - prev_point.y)
        # Vector from point to next_point
        vec2 = (next_point.x - point.x, point.y - next_point.y)

        # Cross product of vec1 and vec2
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

        # Determine the orientation of the polygon (clockwise or counterclockwise)
        # Shapely's `signed_area` can be used to determine the winding direction
        orientation = partial_figure.exterior.is_ccw

        # Adjust convexity check based on the polygon's orientation
        if orientation:  # Counterclockwise orientation
            return True
        else:  # Clockwise orientation -  negative cross-product indicates a convex point.
            return False
        
    def find_blocked_rectangle(self,partial_figure: Polygon, candidate: Point, line1: LineString, line2: LineString) -> Polygon:
            """
            Find the blocked rectangle formed using two adjacent linestrings and the rest of the polygon.

            Args:
                candidate (Point): The common point of the two linestrings.
                line1 (LineString): The first linestring.
                line2 (LineString): The second linestring.

            Returns:
                Polygon: The blocked rectangle inside the given polygon.
            """
            # Ensure the candidate point is part of both linestrings
            if not (candidate.equals(Point(line1.coords[0])) or candidate.equals(Point(line1.coords[1]))):
                raise ValueError("The candidate point must be part of the first linestring")
            if not (candidate.equals(Point(line2.coords[0])) or candidate.equals(Point(line2.coords[1]))):
                raise ValueError("The candidate point must be part of the second linestring")

            # Get the endpoints of the linestrings excluding the candidate point
            other_points = []
            for line in [line1, line2]:
                for coord in line.coords:
                    if not candidate.equals(Point(coord)):
                        other_points.append(Point(coord))

            # Ensure there are exactly two other points
            if len(other_points) != 2:
                raise ValueError("The linestrings must form a valid rectangle")

            # Determine which point forms the vertical line and which forms the horizontal line
            if other_points[0].x == candidate.x:
                vertical_point = other_points[0]
                horizontal_point = other_points[1]
            else:
                vertical_point = other_points[1]
                horizontal_point = other_points[0]

            # Find the kitty-corner point
            kitty_corner = Point(horizontal_point.x, vertical_point.y)

            # Ensure the kitty-corner point is within the polygon
            if not partial_figure.intersects(kitty_corner) or not partial_figure.touches(kitty_corner):
                raise ValueError("The resulting rectangle is not within the original polygon")

            # Create the blocked rectangle
            blocked_rect_coords = [
                (candidate.x, candidate.y),
                (vertical_point.x, vertical_point.y),
                (kitty_corner.x, kitty_corner.y),
                (horizontal_point.x, horizontal_point.y),
                (candidate.x, candidate.y)
            ]

            blocked_rect = Polygon(blocked_rect_coords)

            # Ensure the blocked rectangle is within the original polygon
            if not partial_figure.contains(blocked_rect):
                # ValueError("The resulting rectangle is not within the original polygon")
                logger.debug(f"The resulting rectangle is not within the original polygon")
            return blocked_rect

    
    def find_incident_lines(self, partial_figure: Polygon,  point: Point) -> list[LineString]:
        """
        Find the two LineString objects incident to a given point in the polygon.

        Args:
            point (Point): The point to check.

        Returns:
            list[LineString]: A list of two LineString objects incident to the point.
        """
        if not partial_figure.contains(point) and not partial_figure.touches(point):
            raise ValueError("The given point is not a vertex or on the boundary of the polygon")


        coords = list(partial_figure.exterior.coords)
        incident_lines = []

        # Iterate through polygon edges
        for i in range(len(coords) - 1):  
            line = LineString([coords[i], coords[i + 1]])
            if line.intersects(point):
                incident_lines.append(line)
                    
        if len(incident_lines) != 2:
            raise ValueError("The given point is not a vertex or does not have exactly two incident lines")

        return incident_lines                
        


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

    # plotting(polygon, partition_result)

          


if __name__ == "__main__":

    times = []
    max_rectangles_list = []
    max_rectangles = 8
    partition_time = -1

    while partition_time < 120:  # Continue until partitioning time exceeds 120 seconds
        
        poly = generate_rectilinear_polygon(max_rectangles)
        plotting(poly, [])
        
        start = time.time()
        plot_and_partition(poly)
        end = time.time()
        
        partition_time = end - start
        logger.warning(f"Partitioning took {partition_time:.4f} seconds.")
        
        times.append(partition_time)
        max_rectangles_list.append(max_rectangles)
        
        max_rectangles += 1

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(max_rectangles_list, times, marker='o', linestyle='-', color='b')
    plt.title('Partition Time vs Number of Rectangles')
    plt.xlabel('Number of Rectangles (max_rectangles)')
    plt.ylabel('Partition Time (seconds)')
    plt.grid(True)
    plt.show()
        
        
           
    
    
