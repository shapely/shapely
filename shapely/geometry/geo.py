"""
Geometry factories based on the geo interface
"""
from shapely.geometry.point import Point, asPoint
from shapely.geometry.linestring import LineString, asLineString
from shapely.geometry.polygon import Polygon, asPolygon
from shapely.geometry.multipoint import MultiPoint, asMultiPoint
from shapely.geometry.multilinestring import MultiLineString, asMultiLineString
from shapely.geometry.multipolygon import MultiPolygon, MultiPolygonAdapter
from shapely.geometry.collection import GeometryCollection


def _get_ob_from_context(context):
    """
    Attempt to convert the context to an object shapely can understand.

    Parameters
    ----------
    context: dict like

    Returns
    -------
    tuple
        ob, geom_type
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context
    try:
        geom_type = ob.get("type").lower()
    except AttributeError:
        raise ValueError("Context does not provide geo interface")
    return ob, geom_type


def _extract_crs(ob):
    """
    Pull out the crs key/attribute from the object.

    Parameters
    ----------
    ob: object

    Returns
    -------
    object
        Returns the thing at ob['crs'] or ob.crs or None

    """
    crs = None
    try:
        crs = ob['crs']
    except (TypeError, KeyError):
        pass
    if crs is None and hasattr(ob, 'crs'):
        crs = ob.crs
    return crs


def box(minx, miny, maxx, maxy, ccw=True):
    """
    Returns a rectangular polygon with configurable normal vector

    Parameters
    ----------
    minx: number
    miny: number
    maxx: number
    maxy: number
    ccw: bool
        order of coordinates (counter-clock-wise)

    Returns
    -------
    Polygon
        Rectangular polygon with configurable normal vector

    """
    coords = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if not ccw:
        coords = coords[::-1]
    return Polygon(coords)


def shape(context):
    """
     Returns a new, independent geometry with coordinates *copied* from the
     context.


    Parameters
    ----------
    context: dict like

    Returns
    -------
    shapely object
        Returns an appropriate shapely geometry class depending on context passed in.

    Raises
    ______
    ValueError
        When the context contains a 'type' that doesn't exist.

    """
    ob, geom_type = _get_ob_from_context(context)
    try:
        shape_obj = TYPE_TO_SHAPE_MAP[geom_type](ob)
    except KeyError:
        raise ValueError("Unknown geometry type: %s" % geom_type)
    shape_obj.crs = _extract_crs(ob)
    return shape_obj


def asShape(context):
    """
    Adapts the context to a geometry interface. The coordinates remain
    stored in the context.

    Parameters
    ----------
    context: dict like

    Returns
    -------
    shapely object
        Returns an appropriate shapely geometry class depending on context passed in.

    Raises
    ______

    ValueError
        When the context contains a 'type' that doesn't exist.
    """
    ob, geom_type = _get_ob_from_context(context)
    try:
        shape_obj = TYPE_TO_SHAPE_ADAPTER_MAP[geom_type](ob)
    except KeyError:
        raise ValueError("Unknown geometry type: %s" % geom_type)
    shape_obj.crs = _extract_crs(ob)
    return shape_obj


def mapping(ob):
    """
    Returns a GeoJSON-like mapping

    Parameters
    ----------
    ob

    Returns
    -------
    dict
        a dict following the `__geo_interface__` standard.

    """
    return ob.__geo_interface__


def make_point(ob):
    return Point(ob['coordinates'])


def make_linestring(ob):
    return LineString(ob['coordinates'])


def make_polygon(ob):
    return Polygon(ob['coordinates'][0], ob['coordinates'][1:])


def make_multipoint(ob):
    return MultiPoint(ob['coordinates'])


def make_multilinestring(ob):
    return MultiLineString(ob['coordinates'])


def make_multipolygon(ob):
    return MultiPolygon(ob['coordinates'], context_type='geojson')


def make_geometrycollection(ob):
    geoms = [shape(g) for g in ob.get("geometries", [])]
    return GeometryCollection(geoms)


def make_point_adapter(ob):
    return asPoint(ob['coordinates'])


def make_linestring_adapter(ob):
    return asLineString(ob['coordinates'])


def make_polygon_adapter(ob):
    return asPolygon(ob['coordinates'][0], ob['coordinates'][1:])


def make_multipoint_adapter(ob):
    return asMultiPoint(ob['coordinates'])


def make_multilinestring_adapter(ob):
    return asMultiLineString(ob['coordinates'])


def make_multipolygon_adapter(ob):
    return MultiPolygonAdapter(ob['coordinates'], context_type='geojson')


def make_geometrycollection_adapter(ob):
    geoms = [asShape(g) for g in ob.get("geometries", [])]
    return GeometryCollection(geoms)


TYPE_TO_SHAPE_MAP = {
    'point': make_point,
    'linestring': make_linestring,
    'polygon': make_polygon,
    'multipoint': make_multipoint,
    'multilinestring': make_multilinestring,
    'multipolygon': make_multipolygon,
    'geometrycollection': make_geometrycollection,
}

TYPE_TO_SHAPE_ADAPTER_MAP = {
    'point': make_point_adapter,
    'linestring': make_linestring_adapter,
    'polygon': make_polygon_adapter,
    'multipoint': make_multipoint_adapter,
    'multilinestring': make_multilinestring_adapter,
    'multipolygon': make_multipolygon_adapter,
    'geometrycollection': make_geometrycollection_adapter,
}
