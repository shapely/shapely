
from point import PointAdapter
from linestring import LineStringAdapter
from polygon import PolygonAdapter
from multipoint import MultiPointAdapter
from multilinestring import MultiLineStringAdapter
from multipolygon import MultiPolygonAdapter


def asShape(context):
    """Adapts the context to a geometry interface. The coordinates remain
    stored in the context.
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context

    geom_type = ob.get("type").lower()

    if geom_type == "point":
        return PointAdapter(ob["coordinates"])
    elif geom_type == "linestring":
        return LineStringAdapter(ob["coordinates"])
    elif geom_type == "polygon":
        return PolygonAdapter(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return MultiPointAdapter(ob["coordinates"])
    elif geom_type == "multilinestring":
        return MultiLineStringAdapter(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygonAdapter(ob["coordinates"])
    else:
        raise ValueError, "Unknown geometry type: %s" % geom_type

