from .._geometry import get_point
from ..constructive import maximum_inscribed_circle


def polylabel(polygon, tolerance=1.0):
    """Finds pole of inaccessibility for a given polygon. Based on
    Vladimir Agafonkin's https://github.com/mapbox/polylabel

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    tolerance : int or float, optional
                `tolerance` represents the highest resolution in units of the
                input geometry that will be considered for a solution. (default
                value is 1.0).

    Returns
    -------
    shapely.geometry.Point
        A point representing the pole of inaccessibility for the given input
        polygon.

    Raises
    ------
    shapely.errors.TopologicalError
        If the input polygon is not a valid geometry.

    Example
    -------
    >>> from shapely import wkt
    >>> polygon = LineString([(0, 0), (50, 200), (100, 100), (20, 50),
    ... (-100, -20), (-150, -200)]).buffer(100)
    >>> label = polylabel(polygon, tolerance=10)
    >>> wkt.dumps(label, rounding_precision=12)
    'POINT (59.356155563646 121.839196297464)'
    """
    line = maximum_inscribed_circle(polygon, tolerance)
    return get_point(line, 0)
