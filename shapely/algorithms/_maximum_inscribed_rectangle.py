import numpy as np
from shapely.geometry import Polygon, box, LineString
from shapely import contains_xy

def _compute_mask_fast_vectorized(polygon, grid_x, grid_y):
    xx, yy = np.meshgrid(grid_x, grid_y)
    mask_flat = contains_xy(polygon, xx.ravel(), yy.ravel())
    return mask_flat.reshape((len(grid_y), len(grid_x)))

def _max_histogram_area(heights):
    stack = []
    max_area = left = right = height = 0
    n = len(heights)
    for i in range(n + 1):
        h = heights[i] if i < n else 0
        while stack and h < heights[stack[-1]]:
            top = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            area = heights[top] * width
            if area > max_area:
                max_area = area
                left = stack[-1] + 1 if stack else 0
                right = i - 1
                height = heights[top]
        stack.append(i)
    return max_area, left, right, height
 
def _binary_search(polygon, x1, x2, coord_start, step, max_up=True, vertical=False):
    """
    Binary search to find maximum shift so the entire edge lies inside polygon.
    If vertical=False: checks horizontal edge (y constant).
    If vertical=True: checks vertical edge (x constant).
    """
    low, high = 0, step * 1000
    best_shift = 0

    while high - low > step:
        mid = (low + high) / 2

        if max_up:
            coord_edge = coord_start + mid
        else:
            coord_edge = coord_start - mid

        if vertical:
            edge_line = LineString([(coord_edge, x1), (coord_edge, x2)])
        else:
            edge_line = LineString([(x1, coord_edge), (x2, coord_edge)])

        inside = polygon.covers(edge_line)

        if inside:
            best_shift = mid
            low = mid
        else:
            high = mid

    return best_shift

def adjust_rectangle(polygon, x1, x2, y1_start, y2_start, step):
    # Shift vertically
    shift_up = _binary_search(polygon, x1, x2, y2_start, step, max_up=True, vertical=False)
    y2_shift = y2_start + shift_up
    y1_shift = y1_start + shift_up

    shift_down = _binary_search(polygon, x1, x2, y1_shift, step, max_up=False, vertical=False)
    y1_shift = y1_shift - shift_down

    # Shift horizontally
    shift_right = _binary_search(polygon, y1_shift, y2_shift, x2, step, max_up=True, vertical=True)
    x2_shift = x2 + shift_right
    x1_shift = x1 + shift_right

    shift_left = _binary_search(polygon, y1_shift, y2_shift, x1_shift, step, max_up=False, vertical=True)
    x1_shift = x1_shift - shift_left

    # Final check
    candidate_rect = box(x1_shift, y1_shift, x2_shift, y2_shift)
    if candidate_rect.is_valid and candidate_rect.area > 0 and polygon.covers(candidate_rect):
        return candidate_rect, candidate_rect.area

    return None, 0

def maximum_inscribed_rectangle(polygon: Polygon, resolution=20):
    minx, miny, maxx, maxy = polygon.bounds

    grid_x = np.linspace(minx, maxx, resolution)
    grid_y = np.linspace(miny, maxy, resolution)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]

    mask = _compute_mask_fast_vectorized(polygon, grid_x, grid_y)

    height_arr = np.zeros_like(mask, dtype=int)
    height_arr[0] = mask[0]
    for i in range(1, resolution):
        height_arr[i] = (height_arr[i - 1] + 1) * mask[i]

    max_area = 0
    best_rect = None

    for i in range(resolution):
        area, left, right, height = _max_histogram_area(height_arr[i])
        if area > 0:
            x1 = grid_x[left]
            x2 = grid_x[right]
            y2 = grid_y[i]
            y1 = y2 - height * dy

            candidate_rect, area = adjust_rectangle(
                polygon,
                x1, x2,
                y1, y2,
                dy * 0.01
            )

            if area > max_area:
                max_area = area
                best_rect = candidate_rect

    return best_rect
