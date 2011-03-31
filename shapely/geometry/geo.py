"""
Geometry factories based on the geo interface
"""

from point import Point, asPoint
from linestring import LineString, asLineString
from polygon import Polygon, asPolygon
from multipoint import MultiPoint, asMultiPoint
from multilinestring import MultiLineString, asMultiLineString
from multipolygon import MultiPolygon, MultiPolygonAdapter


def box(minx, miny, maxx, maxy, ccw=True):
    """Return a rectangular polygon with configurable normal vector"""
    coords = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if not ccw:
        coords = coords[::-1]
    return Polygon(coords)

def shape(context):
    """Return a new, independent geometry with coordinates *copied* from the
    context.
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context
    geom_type = ob.get("type").lower()
    if geom_type == "point":
        return Point(ob["coordinates"])
    elif geom_type == "linestring":
        return LineString(ob["coordinates"])
    elif geom_type == "polygon":
        return Polygon(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return MultiPoint(ob["coordinates"])
    elif geom_type == "multilinestring":
        return MultiLineString(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygon(ob["coordinates"], context_type='geojson')
    else:
        raise ValueError("Unknown geometry type: %s" % geom_type)

def asShape(context):
    """Adapts the context to a geometry interface. The coordinates remain
    stored in the context.
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context

    try:
        geom_type = ob.get("type").lower()
    except AttributeError:
        raise ValueError("Context does not provide geo interface")

    if geom_type == "point":
        return asPoint(ob["coordinates"])
    elif geom_type == "linestring":
        return asLineString(ob["coordinates"])
    elif geom_type == "polygon":
        return asPolygon(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return asMultiPoint(ob["coordinates"])
    elif geom_type == "multilinestring":
        return asMultiLineString(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygonAdapter(ob["coordinates"], context_type='geojson')
    else:
        raise ValueError("Unknown geometry type: %s" % geom_type)

def mapping(ob):
    """Returns a GeoJSON-like mapping"""
    return ob.__geo_interface__
