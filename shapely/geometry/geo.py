"""
Geometry factories based on the geo interface
"""

from .point import Point, asPoint
from .linestring import LineString, asLineString
from .polygon import Polygon, asPolygon
from .multipoint import MultiPoint, asMultiPoint
from .multilinestring import MultiLineString, asMultiLineString
from .multipolygon import MultiPolygon, MultiPolygonAdapter
from .collection import GeometryCollection


def box(minx, miny, maxx, maxy, ccw=True):
    """Returns a rectangular polygon with configurable normal vector"""
    coords = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if not ccw:
        coords = coords[::-1]
    return Polygon(coords)

def anchored_box(anchor='nw',x,y,width,height,ccw=True):
    """Return a rectangular polygon extruded from the specified anchor point.
    The anchor point can be a corner or the centre of an edge in relation to the centre point, or the centre itself.
    Anchor options from left to right, top to bottom are: nw,n,ne,w,c,e,sw,s,se . (Defaults to North-West)
    """
    if anchor =='nw':
        return box(x,y-height,x+width,y,ccw)
    elif anchor =='w':
        return box(x,y-height/2.0,x+width,y+height/2.0,ccw)
    elif anchor =='sw':
        return box (x,y,x+width,y+height,ccw)
    elif anchor =='n':
        return box(x-width/2.0,y-height,x+width/2.0,y,ccw)
    elif anchor =='c':
        return box(x-width/2.0,y-height/2.0,x+width/2.0,y+height/2.0,ccw)
    elif anchor == 's':
        return box(x-width/2.0,y,x+width/2.0,y+height,ccw)
    elif anchor == 'ne':
        return box(x-width,y-height,x,y,ccw)
    elif anchor == 'e':
        return box(x-width,y-height/2.0,x,y+height/2.0,ccw)
    elif anchor =='se':
        return box(x-width,y,x,y+height,ccw)
    else:
        raise ValueError('Unknown anchor point')
    

def shape(context):
    """Returns a new, independent geometry with coordinates *copied* from the
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
        if not ob["coordinates"]:
            return Polygon()
        else:
            return Polygon(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return MultiPoint(ob["coordinates"])
    elif geom_type == "multilinestring":
        return MultiLineString(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygon(ob["coordinates"], context_type='geojson')
    elif geom_type == "geometrycollection":
        geoms = [shape(g) for g in ob.get("geometries", [])]
        return GeometryCollection(geoms)
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
    elif geom_type == "geometrycollection":
        geoms = [asShape(g) for g in ob.get("geometries", [])]
        return GeometryCollection(geoms)
    else:
        raise ValueError("Unknown geometry type: %s" % geom_type)

def mapping(ob):
    """Returns a GeoJSON-like mapping"""
    return ob.__geo_interface__
