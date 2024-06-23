            # for matching_point in matching_points:
            #     new_partial_figure, new_lines = self.find_blocked_rectangle(
            #         candidate_point, matching_point
            #     )
            #     if new_lines is None:
            #         logger.info("No new lines found.")
            #         continue
            #     new_figures = partial_figures + [
            #         ComperablePolygon(new_partial_figure)
            #     ]  

            #     normalized_partition_list = {
            #         normalize_line(ComparableLineString(line)) for line in new_lines
            #     }
            #     new_partition_list = [
            #         ComparableLineString(line)
            #         for line in normalized_partition_list.union(partition_list)
            #     ]
            #     logger.info(f"New partition list: {new_partition_list}")

            #     new_total_length = sum(line.length for line in new_partition_list)
            #     new_priority = (
            #         new_total_length if new_total_length != 0 else float("inf")
            #     )

            #     if new_priority >= self.min_partition_length:
            #         logger.info(
            #             f"cutting the search -  the new partition is longer than the best partition"
            #         )
            #         continue

            #     if self.is_partitioned_into_rectangles(new_partition_list):
            #         if new_total_length < self.min_partition_length:
            #             self.min_partition_length = new_total_length
            #             self.best_partition = [line for line in new_partition_list]
            #             logger.debug(f"New best partition found: {self.best_partition}")
            #         else:
            #             logger.warning(f"Not the best : {new_partition_list}")
            #             continue

            #     new_candidate_point = self.find_candidate_point(new_partition_list)
            #     if new_candidate_point is None or new_candidate_point.is_empty:
            #         continue

            #     # Convert the candidate point to a tuple of coordinates
            #     new_candidate_point_tuple = tuple(new_candidate_point.coords[0])

            #     new_item = PriorityQueueItem(
            #         new_priority,
            #         new_candidate_point_tuple,
            #         new_partition_list,
            #         new_figures,
            #     )
            #     heapq.heappush(pq, new_item)
            #     logger.info(f"new item pushed : {new_item}")
            
        #         def find_blocked_rectangle(self, candidate: Point, matching: Point):
        # """
        # find the blocked rectangle between the candidate and the matching point.

        # Args:
        #     candidate (Point): The candidate point.
        #     matching (Point): The matching point.

        # Returns:
        # is exist:
        #     tuple: A tuple containing the blocked rectangle and the internal edges of the blocked rectangle.
        # else:
        #     None


        # >>> polygon = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
        # >>> rect_polygon = RectilinearPolygon(polygon)
        # >>> candidate = Point(2, 4)
        # >>> matching = Point(6, 6)
        # >>> rect_polygon.find_blocked_rectangle(candidate, matching)
        # (<POLYGON ((2 4, 2 6, 6 6, 6 4, 2 4))>, [<LINESTRING (2 4, 2 6)>, <LINESTRING (6 6, 6 4)>, <LINESTRING (6 4, 2 4)>])

        # """
        # # Create the four edges of the potential blocked rectangle
        # edge1 = LineString([candidate, Point(candidate.x, matching.y)])
        # edge2 = LineString([Point(candidate.x, matching.y), matching])
        # edge3 = LineString([matching, Point(matching.x, candidate.y)])
        # edge4 = LineString([Point(matching.x, candidate.y), candidate])

        # # Collect all edges
        # all_edges = [edge1, edge2, edge3, edge4]

        # # Convert each LineString to a list of coordinate tuples
        # coords_list = [list(edge.coords) for edge in all_edges]

        # # Concatenate the coordinate lists, removing duplicates
        # coords = list(dict.fromkeys(sum(coords_list, [])))  # TODO: fix and rerfactor

        # # Construct the partial polygon
        # polygon = Polygon(coords)

        # # Find the segments of each edge that are not part of the polygon boundary
        # internal_edges = []
        # for edge in all_edges:
        #     difference = edge.difference(self.polygon.boundary)  # TODO: explain this..
        #     logger.debug(f"Difference: {difference}")
        #     if not difference.is_empty:
        #         if difference.geom_type == "MultiLineString":
        #             for (
        #                 line
        #             ) in difference.geoms:  # Use .geoms to iterate over MultiLineString
        #                 internal_edges.append(line)
        #         else:
        #             internal_edges.append(difference)