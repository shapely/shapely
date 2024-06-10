'''
This file contains all the classes that are used in the algorithms module of shapely
Each class is a itrrable object that can be compared to other objects of the same class 
Programmer: Dvir Borochov
Date: 10/6/24 
'''

from shapely.geometry import (
    Polygon,
    LineString
)

class PriorityQueueItem:
    """
        Initialize a MyClass object.

        Args:
            priority (int): The priority of the object (sum of the partition linestring lengths)
            candidate_point_tuple (tuple): A tuple representing a candidate point.
            partition_list (list): A list of partitions.
            partial_figures (list): A list of partial figures.

        Returns:
            None
    """
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

class ComperablePolygon(Polygon):
    def __lt__(self, other):
        if not isinstance(other, ComperablePolygon):
            return NotImplemented
        return tuple(self.exterior.coords) < tuple(other.exterior.coords)