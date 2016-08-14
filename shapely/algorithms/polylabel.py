from ..geometry import Point, LineString
from ..geos import TopologicalError
from heapq import heappush, heappop


class Cell(object):
    def __init__(self, x, y, h, polygon):
        self.centroid = Point(x, y)  # cell centroid
        self.h = h  # half of cell size

        # distance from cell centroid to polygon exterior
        self.distance = self._dist(polygon)

        # max distance to polygon within a cell
        self.max_distance = self.distance + h * 1.4142135623730951  # sqrt(2)

    # rich comparison operators for sorting in minimum priority queue
    def __lt__(self, other):
        return self.max_distance > other.max_distance

    def __le__(self, other):
        return self.max_distance >= other.max_distance

    def __eq__(self, other):
        return self.max_distance == other.max_distance

    def __ne__(self, other):
        return self.max_distance != other.max_distance

    def __gt__(self, other):
        return self.max_distance < other.max_distance

    def __ge__(self, other):
        return self.max_distance <= other.max_distance

    def _dist(self, polygon):
        """
        Signed distance from point to polygon outline
        (negative if point is outside)

        """
        inside = polygon.contains(self.centroid)
        distance = self.centroid.distance(LineString(polygon.exterior.coords))
        if inside:
            return distance
        return -distance


def polylabel(polygon, precision=1.0):
    """
    Finds pole of inaccessibility for a polygon. Based on
    https://github.com/mapbox/polylabel

    """
    if not polygon.is_valid:
        raise TopologicalError('Invalid polygon')
    minx, miny, maxx, maxy = polygon.bounds
    cell_size = min(maxx - minx, maxy - miny)
    h = cell_size / 2.0
    cell_queue = []

    # First best cell approximation is one constructed from the centroid
    # of the polygon
    x, y = polygon.centroid.coords[0]
    best_cell = Cell(x, y, 0, polygon)

    # build a regular square grid covering the polygon
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            heappush(cell_queue, Cell(x + h, y + h, h, polygon))
            y += cell_size
        x += cell_size

    # minimum priority queue
    while cell_queue:
        cell = heappop(cell_queue)

        # update the best cell if we find a better one
        if cell.distance > best_cell.distance:
            best_cell = cell

        # continue to the next iteration if we cant find a better solution
        # based on precision
        if cell.max_distance - best_cell.distance <= precision:
            continue

        # split the cell into quadrants
        h = cell.h / 2.0
        heappush(cell_queue,
                 Cell(cell.centroid.x - h, cell.centroid.y - h, h, polygon))
        heappush(cell_queue,
                 Cell(cell.centroid.x + h, cell.centroid.y - h, h, polygon))
        heappush(cell_queue,
                 Cell(cell.centroid.x - h, cell.centroid.y + h, h, polygon))
        heappush(cell_queue,
                 Cell(cell.centroid.x + h, cell.centroid.y + h, h, polygon))

    return best_cell.centroid
