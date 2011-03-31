"""Geometry classes and factories
"""

from geo import box, shape, asShape, mapping
from point import Point, asPoint
from linestring import LineString, asLineString
from polygon import Polygon, asPolygon
from multipoint import MultiPoint, asMultiPoint
from multilinestring import MultiLineString, asMultiLineString
from multipolygon import MultiPolygon, asMultiPolygon
from collection import GeometryCollection

__all__ = [
    'box', 'shape', 'asShape', 'Point', 'asPoint', 'LineString', 'asLineString',
    'Polygon', 'asPolygon', 'MultiPoint', 'asMultiPoint',
    'MultiLineString', 'asMultiLineString', 'MultiPolygon', 'asMultiPolygon',
    'GeometryCollection', 'mapping'
    ]


