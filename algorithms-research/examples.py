from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import split
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RectilinearPolygon:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def is_partitioned_into_rectangles(
        self, partial_figure: Polygon, partitions: list[LineString]
    ) -> bool:
        """
        Checks if the polygon is partitioned into rectangles.

        Args:
            partial_figure (Polygon): The partial figure to be checked.
            partitions (list[LineString]): A list of LineString objects representing the partitions.

        Returns:
            bool: True if the polygon is partitioned into rectangles, False otherwise.
        """
        logger.debug(f"Starting with partial figure: {partial_figure}")
        polygons = [partial_figure]
        
        for i, partition in enumerate(partitions):
            logger.debug(f"Processing partition {i}: {partition}")
            new_polygons = []
            for poly in polygons:
                if partition.intersects(poly):
                    logger.debug(f"Partition intersects polygon: {poly}")
                    split_result = split(poly, partition)
                    logger.debug(f"Split result: {split_result}")
                    if isinstance(split_result, MultiPolygon):
                        new_polygons.extend(list(split_result.geoms))
                    elif isinstance(split_result, Polygon):
                        new_polygons.append(split_result)
                    else:
                        new_polygons.append(poly)
                        logger.warning(f"Unexpected split result type: {type(split_result)}")
                else:
                    new_polygons.append(poly)
            polygons = new_polygons
            logger.debug(f"Polygons after partition {i}: {polygons}")
        
        logger.debug(f"Final number of partitioned polygons: {len(polygons)}")
        
        for i, polygon in enumerate(polygons):
            logger.debug(f"Checking if polygon {i} is a rectangle: {polygon}")
            if not self.is_rectangle(polygon):
                logger.debug(f"Polygon {i} is not a rectangle")
                return False
            else:
                logger.debug(f"Polygon {i} is a rectangle")
        
        reconstructed_figure = MultiPolygon(polygons)
        logger.debug(f"Reconstructed figure: {reconstructed_figure}")
        logger.debug(f"Original partial figure: {partial_figure}")
        if not self.geometries_almost_equal(reconstructed_figure, partial_figure):
            logger.debug("Reconstructed figure does not match the original partial figure")
            return False
        
        logger.debug(f"Polygon is successfully partitioned into {len(polygons)} rectangles")
        return True

    def is_rectangle(self, polygon: Polygon, tolerance: float = 1e-6) -> bool:
        coords = list(polygon.exterior.coords)
        logger.debug(f"Checking rectangle: coords = {coords}")
        if len(coords) != 5:
            logger.debug(f"Not a rectangle: {len(coords)} coords instead of 5")
            return False
        
        for i in range(4):
            p1, p2, p3 = coords[i], coords[(i+1) % 4], coords[(i+2) % 4]
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            dot_product = np.dot(v1, v2)
            logger.debug(f"Angle {i}: dot product = {dot_product}")
            if abs(dot_product) > tolerance:
                logger.debug(f"Not a rectangle: angle {i} is not 90 degrees")
                return False
        
        return True

    def geometries_almost_equal(self, geom1, geom2, tolerance: float = 1e-6) -> bool:
        """Check if two geometries are almost equal, accounting for floating point precision."""
        diff = geom1.symmetric_difference(geom2)
        logger.debug(f"Difference area: {diff.area}")
        return diff.area < tolerance

# Example usage
polygon4 = Polygon([(1, 5), (1, 4), (3, 4), (3,2), (5,2), (5, 1),(8,1), (8,5)])
partitions = [
    LineString([(5,5), (5,4)]),
    LineString([(5,4), (3,4)]),
    LineString([(5,4), (5,2)])
]

rect_polygon = RectilinearPolygon(polygon4)
result = rect_polygon.is_partitioned_into_rectangles(polygon4, partitions)
print(f"Is the polygon partitioned into rectangles? {result}")