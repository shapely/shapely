import logging
import random
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box


logger = logging.getLogger(__name__)
def generate_rectilinear_polygon(
    max_rectangles,
    canvas_width: int = 20,
    canvas_height: int = 20,
    min_rectangles: int = 4,
    min_size: int = 1,
    max_size: int = 4,
    increments: int = 5
) -> Polygon:
    """
    Generate a rectilinear polygon composed of touching rectangles.
    
    :param max_rectangles: Maximum number of rectangles
    :param canvas_width: Width of the canvas
    :param canvas_height: Height of the canvas
    :param min_rectangles: Minimum number of rectangles
    :param min_size: Minimum size of a rectangle
    :param max_size: Maximum size of a rectangle
    :param increments: Size increment for positioning and sizing
    :return: Shapely Polygon representing the rectilinear polygon
    """
    # num_rect = random.randint(min_rectangles, max_rectangles)
    # logger.warning(f"Generating {num_rect} rectangles")
    boxes = []

    for _ in range(max_rectangles):
        valid = False
        while not valid:
            width = random.randint(min_size, max_size) * increments
            height = random.randint(min_size, max_size) * increments
            pos_x = random.randint(max_size + 1, canvas_width - max_size - 1) * increments
            pos_y = random.randint(max_size + 1, canvas_height - max_size - 1) * increments

            touching = False
            for box in boxes:
                if (
                    abs(pos_x - box[0]) * 2 < (width + box[2]) and
                    abs(pos_y - box[1]) * 2 < (height + box[3])
                ):
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
                x - width // 2 < other[0] + other[2] // 2 and
                x + width // 2 > other[0] - other[2] // 2 and
                bottom < other[1] + other[3] // 2
                for j, other in enumerate(boxes) if i != j
            )
            
            if not intersects:
                new_height = height + (bottom_ground - bottom) * 2
                new_y = y + (bottom_ground - bottom)
                boxes[i] = (x, new_y, width, new_height)

    # Convert rectangles to Shapely Polygons and union them
    shapely_polygons = [shapely_box(x - w/2, y - h/2, x + w/2, y + h/2) for x, y, w, h in boxes]
    union_polygon = shapely_polygons[0]
    for polygon in shapely_polygons[1:]:
        union_polygon = union_polygon.union(polygon)

    # Simplify the polygon to remove unnecessary points
    simplified_polygon = union_polygon.simplify(1, preserve_topology=True)

    return simplified_polygon

