from shapely.geometry import (
    Polygon,
    LineString
)

class PriorityQueueItem:
    def __init__(self, priority, candidate_point_tuple, partition_list, partial_figures):
        self.priority = priority
        self.candidate_point_tuple = candidate_point_tuple
        self.partition_list = partition_list
        self.partial_figures = partial_figures

    def __lt__(self, other):
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented
        return self.priority < other.priority

    def __repr__(self):
        return f"PriorityQueueItem(priority={self.priority}, candidate_point_tuple={self.candidate_point_tuple}, partition_list={self.partition_list}, partial_figures={self.partial_figures})"              
                
class ComparableLineString(LineString):
    def __lt__(self, other):
        if not isinstance(other, ComparableLineString):
            return NotImplemented
        return tuple(self.coords) < tuple(other.coords)

class Comperablepolygon(Polygon):
    def __lt__(self, other):
        if not isinstance(other, Comperablepolygon):
            return NotImplemented
        return tuple(self.exterior.coords) < tuple(other.exterior.coords)