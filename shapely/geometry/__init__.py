"""Geometry classes and factories
"""

from .base import CAP_STYLE, JOIN_STYLE
from .geo import box, shape, mapping
from .point import Point
from .linestring import LineString
from .polygon import Polygon, LinearRing
from .multipoint import MultiPoint
from .multilinestring import MultiLineString
from .multipolygon import MultiPolygon
from .collection import GeometryCollection

__all__ = [
    'box', 'shape', 'mapping', 'Point', 'LineString', 'Polygon', 'MultiPoint',
    'MultiLineString', 'MultiPolygon', 'GeometryCollection', 'LinearRing',
    'CAP_STYLE', 'JOIN_STYLE',
]

# This needs to be called here to avoid circular references
import shapely.speedups
