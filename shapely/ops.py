"""Support for various GEOS geometry operations."""

from warnings import warn

import shapely
from shapely import lib
from shapely.algorithms.polylabel import polylabel  # noqa
from shapely.errors import GeometryTypeError, ShapelyDeprecationWarning
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    shape,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.polygon import orient as orient_
from shapely.prepared import prep

__all__ = [
    "cascaded_union",
    "clip_by_rect",
    "linemerge",
    "nearest_points",
    "operator",
    "orient",
    "polygonize",
    "polygonize_full",
    "shared_paths",
    "snap",
    "split",
    "substring",
    "transform",
    "triangulate",
    "unary_union",
    "validate",
    "voronoi_diagram",
]


class CollectionOperator:
    def shapeup(self, ob):
        if isinstance(ob, BaseGeometry):
            return ob
        else:
            try:
                return shape(ob)
            except (ValueError, AttributeError):
                return LineString(ob)

    def polygonize(self, lines):
        """Create polygons from a source of lines.

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.
        """
        source = getattr(lines, "geoms", None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(line) for line in source]
        collection = shapely.polygonize(obs)
        return collection.geoms

    def polygonize_full(self, lines):
        """Create polygons from a source of lines.

        The polygons and leftover geometries are returned as well.

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.

        Returns a tuple of objects: (polygons, cut edges, dangles, invalid ring
        lines). Each are a geometry collection.

        Dangles are edges which have one or both ends which are not incident on
        another edge endpoint. Cut edges are connected at both ends but do not
        form part of polygon. Invalid ring lines form rings which are invalid
        (bowties, etc).
        """
        source = getattr(lines, "geoms", None) or lines
        try:
            source = iter(source)
        except TypeError:
            source = [source]
        finally:
            obs = [self.shapeup(line) for line in source]
        return shapely.polygonize_full(obs)

    def linemerge(self, lines, directed=False):
        """Merge all connected lines from a source.

        The source may be a MultiLineString, a sequence of LineString objects,
        or a sequence of objects than can be adapted to LineStrings.  Returns a
        LineString or MultiLineString when lines are not contiguous.
        """
        source = None
        if getattr(lines, "geom_type", None) == "MultiLineString":
            source = lines
        elif hasattr(lines, "geoms"):
            # other Multi geometries
            source = MultiLineString([ls.coords for ls in lines.geoms])
        elif hasattr(lines, "__iter__"):
            try:
                source = MultiLineString([ls.coords for ls in lines])
            except AttributeError:
                source = MultiLineString(lines)
        if source is None:
            raise ValueError(f"Cannot linemerge {lines}")
        return shapely.line_merge(source, directed=directed)

    def cascaded_union(self, geoms):
        """Return the union of a sequence of geometries.

        .. deprecated:: 1.8
            This function was superseded by :meth:`unary_union`.
        """
        warn(
            "The 'cascaded_union()' function is deprecated. "
            "Use 'unary_union()' instead.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return shapely.union_all(geoms, axis=None)

    def unary_union(self, geoms):
        """Return the union of a sequence of geometries.

        Usually used to convert a collection into the smallest set of polygons
        that cover the same area.
        """
        return shapely.union_all(geoms, axis=None)


operator = CollectionOperator()
polygonize = operator.polygonize
polygonize_full = operator.polygonize_full
linemerge = operator.linemerge
cascaded_union = operator.cascaded_union
unary_union = operator.unary_union


def triangulate(geom, tolerance=0.0, edges=False):
    """Create the Delaunay triangulation and return a list of geometries.

    The source may be any geometry type. All vertices of the geometry will be
    used as the points of the triangulation.

    From the GEOS documentation:
    tolerance is the snapping tolerance used to improve the robustness of
    the triangulation computation. A tolerance of 0.0 specifies that no
    snapping will take place.

    If edges is False, a list of Polygons (triangles) will be returned.
    Otherwise the list of LineString edges is returned.

    """
    collection = shapely.delaunay_triangles(geom, tolerance=tolerance, only_edges=edges)
    return list(collection.geoms)


def voronoi_diagram(geom, envelope=None, tolerance=0.0, edges=False):
    """Construct a Voronoi Diagram [1] from the given geometry.

    Returns a list of geometries.

    Parameters
    ----------
    geom: geometry
        the input geometry whose vertices will be used to calculate
        the final diagram.
    envelope: geometry, None
        clipping envelope for the returned diagram, automatically
        determined if None. The diagram will be clipped to the larger
        of this envelope or an envelope surrounding the sites.
    tolerance: float, 0.0
        sets the snapping tolerance used to improve the robustness
        of the computation. A tolerance of 0.0 specifies that no
        snapping will take place.
    edges: bool, False
        If False, return regions as polygons. Else, return only
        edges e.g. LineStrings.

    GEOS documentation can be found at [2]

    Returns
    -------
    GeometryCollection
        geometries representing the Voronoi regions.

    Notes
    -----
    The tolerance `argument` can be finicky and is known to cause the
    algorithm to fail in several cases. If you're using `tolerance`
    and getting a failure, try removing it. The test cases in
    tests/test_voronoi_diagram.py show more details.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Voronoi_diagram
    [2] https://geos.osgeo.org/doxygen/geos__c_8h_source.html  (line 730)

    """
    try:
        result = shapely.voronoi_polygons(
            geom, tolerance=tolerance, extend_to=envelope, only_edges=edges
        )
    except shapely.GEOSException as err:
        errstr = "Could not create Voronoi Diagram with the specified inputs "
        errstr += f"({err!s})."
        if tolerance:
            errstr += " Try running again with default tolerance value."
        raise ValueError(errstr) from err

    if result.geom_type != "GeometryCollection":
        return GeometryCollection([result])
    return result


def validate(geom):
    """Return True if the geometry is valid."""
    return shapely.is_valid_reason(geom)


def transform(func, geom):
    """Apply `func` to all coordinates of `geom`.

    Returns a new geometry of the same type from the transformed coordinates.

    `func` maps x, y, and optionally z to output xp, yp, zp. The input
    parameters may iterable types like lists or arrays or single values.
    The output shall be of the same type. Scalars in, scalars out.
    Lists in, lists out.

    For example, here is an identity function applicable to both types
    of input.

      def id_func(x, y, z=None):
          return tuple(filter(None, [x, y, z]))

      g2 = transform(id_func, g1)

    Using pyproj >= 2.1, this example will accurately project Shapely geometries:

      import pyproj

      wgs84 = pyproj.CRS('EPSG:4326')
      utm = pyproj.CRS('EPSG:32618')

      project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

      g2 = transform(project, g1)

    Note that the always_xy kwarg is required here as Shapely geometries only support
    X,Y coordinate ordering.

    Lambda expressions such as the one in

      g2 = transform(lambda x, y, z=None: (x+1.0, y+1.0), g1)

    also satisfy the requirements for `func`.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type in ("Point", "LineString", "LinearRing", "Polygon"):
        # First we try to apply func to x, y, z sequences. When func is
        # optimized for sequences, this is the fastest, though zipping
        # the results up to go back into the geometry constructors adds
        # extra cost.
        try:
            if geom.geom_type in ("Point", "LineString", "LinearRing"):
                return type(geom)(zip(*func(*zip(*geom.coords))))
            elif geom.geom_type == "Polygon":
                shell = type(geom.exterior)(zip(*func(*zip(*geom.exterior.coords))))
                holes = [
                    type(ring)(zip(*func(*zip(*ring.coords))))
                    for ring in geom.interiors
                ]
                return type(geom)(shell, holes)

        # A func that assumes x, y, z are single values will likely raise a
        # TypeError, in which case we'll try again.
        except TypeError:
            if geom.geom_type in ("Point", "LineString", "LinearRing"):
                return type(geom)([func(*c) for c in geom.coords])
            elif geom.geom_type == "Polygon":
                shell = type(geom.exterior)([func(*c) for c in geom.exterior.coords])
                holes = [
                    type(ring)([func(*c) for c in ring.coords])
                    for ring in geom.interiors
                ]
                return type(geom)(shell, holes)

    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        return type(geom)([transform(func, part) for part in geom.geoms])
    else:
        raise GeometryTypeError(f"Type {geom.geom_type!r} not recognized")


def nearest_points(g1, g2):
    """Return the calculated nearest points in the input geometries.

    The points are returned in the same order as the input geometries.
    """
    seq = shapely.shortest_line(g1, g2)
    if seq is None:
        if g1.is_empty:
            raise ValueError("The first input geometry is empty")
        else:
            raise ValueError("The second input geometry is empty")

    p1 = shapely.get_point(seq, 0)
    p2 = shapely.get_point(seq, 1)
    return (p1, p2)


def snap(g1, g2, tolerance):
    """Snaps an input geometry (g1) to reference (g2) geometry's vertices.

    Parameters
    ----------
    g1 : geometry
        The first geometry
    g2 : geometry
        The second geometry
    tolerance : float
        The snapping tolerance

    Refer to :func:`shapely.snap` for full documentation.

    """
    return shapely.snap(g1, g2, tolerance)


def shared_paths(g1, g2):
    """Find paths shared between the two given lineal geometries.

    Returns a GeometryCollection with two elements:
     - First element is a MultiLineString containing shared paths with the
       same direction for both inputs.
     - Second element is a MultiLineString containing shared paths with the
       opposite direction for the two inputs.

    Parameters
    ----------
    g1 : geometry
        The first geometry
    g2 : geometry
        The second geometry

    """
    if not isinstance(g1, LineString):
        raise GeometryTypeError("First geometry must be a LineString")
    if not isinstance(g2, LineString):
        raise GeometryTypeError("Second geometry must be a LineString")
    return shapely.shared_paths(g1, g2)


class SplitOp:
    @staticmethod
    def _split_polygon_with_line(poly, splitter):
        """Split a Polygon with a LineString."""
        if not isinstance(poly, Polygon):
            raise GeometryTypeError("First argument must be a Polygon")
        if not isinstance(splitter, (LineString, MultiLineString)):
            raise GeometryTypeError("Second argument must be a (Multi)LineString")

        union = poly.boundary.union(splitter)

        # greatly improves split performance for big geometries with many
        # holes (the following contains checks) with minimal overhead
        # for common cases
        poly = prep(poly)

        # some polygonized geometries may be holes, we do not want them
        # that's why we test if the original polygon (poly) contains
        # an inner point of polygonized geometry (pg)
        return [
            pg for pg in polygonize(union) if poly.contains(pg.representative_point())
        ]

    @staticmethod
    def _split_line_with_line(line, splitter):
        """Split a LineString with another (Multi)LineString or (Multi)Polygon."""
        # if splitter is a polygon, pick it's boundary
        if splitter.geom_type in ("Polygon", "MultiPolygon"):
            splitter = splitter.boundary

        if not isinstance(line, LineString):
            raise GeometryTypeError("First argument must be a LineString")
        if not isinstance(splitter, LineString) and not isinstance(
            splitter, MultiLineString
        ):
            raise GeometryTypeError(
                "Second argument must be either a LineString or a MultiLineString"
            )

        # |    s\l   | Interior | Boundary | Exterior |
        # |----------|----------|----------|----------|
        # | Interior |  0 or F  |    *     |    *     |   At least one of these two must be 0  # noqa: E501
        # | Boundary |  0 or F  |    *     |    *     |   So either '0********' or '[0F]**0*****'  # noqa: E501
        # | Exterior |    *     |    *     |    *     |   No overlapping interiors ('1********')  # noqa: E501
        relation = splitter.relate(line)
        if relation[0] == "1":
            # The lines overlap at some segment (linear intersection of interiors)
            raise ValueError("Input geometry segment overlaps with the splitter.")
        elif relation[0] == "0" or relation[3] == "0":
            # The splitter crosses or touches the line's interior
            # --> return multilinestring from the split
            return line.difference(splitter)
        else:
            # The splitter does not cross or touch the line's interior
            # --> return collection with identity line
            return [line]

    @staticmethod
    def _split_line_with_point(line, splitter):
        """Split a LineString with a Point."""
        if not isinstance(line, LineString):
            raise GeometryTypeError("First argument must be a LineString")
        if not isinstance(splitter, Point):
            raise GeometryTypeError("Second argument must be a Point")

        # check if point is in the interior of the line
        if not line.relate_pattern(splitter, "0********"):
            # point not on line interior --> return collection with single identity line
            # (REASONING: Returning a list with the input line reference and creating a
            # GeometryCollection at the general split function prevents unnecessary
            # copying of linestrings in multipoint splitting function)
            return [line]
        elif line.coords[0] == splitter.coords[0]:
            # if line is a closed ring the previous test doesn't behave as desired
            return [line]

        # point is on line, get the distance from the first point on line
        distance_on_line = line.project(splitter)
        coords = list(line.coords)
        # split the line at the point and create two new lines
        current_position = 0.0
        for i in range(len(coords) - 1):
            point1 = coords[i]
            point2 = coords[i + 1]
            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]
            segment_length = (dx**2 + dy**2) ** 0.5
            current_position += segment_length
            if distance_on_line == current_position:
                # splitter is exactly on a vertex
                return [LineString(coords[: i + 2]), LineString(coords[i + 1 :])]
            elif distance_on_line < current_position:
                # splitter is between two vertices
                return [
                    LineString(coords[: i + 1] + [splitter.coords[0]]),
                    LineString([splitter.coords[0]] + coords[i + 1 :]),
                ]
        return [line]

    @staticmethod
    def _split_line_with_multipoint(line, splitter):
        """Split a LineString with a MultiPoint."""
        if not isinstance(line, LineString):
            raise GeometryTypeError("First argument must be a LineString")
        if not isinstance(splitter, MultiPoint):
            raise GeometryTypeError("Second argument must be a MultiPoint")

        chunks = [line]
        for pt in splitter.geoms:
            new_chunks = []
            for chunk in filter(lambda x: not x.is_empty, chunks):
                # add the newly split 2 lines or the same line if not split
                new_chunks.extend(SplitOp._split_line_with_point(chunk, pt))
            chunks = new_chunks

        return chunks

    @staticmethod
    def split(geom, splitter):
        """Split a geometry by another geometry and return a collection of geometries.

        This function is the theoretical opposite of the union of
        the split geometry parts. If the splitter does not split the geometry, a
        collection with a single geometry equal to the input geometry is
        returned.

        The function supports:
          - Splitting a (Multi)LineString by a (Multi)Point or (Multi)LineString
            or (Multi)Polygon
          - Splitting a (Multi)Polygon by a LineString

        It may be convenient to snap the splitter with low tolerance to the
        geometry. For example in the case of splitting a line by a point, the
        point must be exactly on the line, for the line to be correctly split.
        When splitting a line by a polygon, the boundary of the polygon is used
        for the operation. When splitting a line by another line, a ValueError
        is raised if the two overlap at some segment.

        Parameters
        ----------
        geom : geometry
            The geometry to be split
        splitter : geometry
            The geometry that will split the input geom

        Examples
        --------
        >>> import shapely.ops
        >>> from shapely import Point, LineString
        >>> pt = Point((1, 1))
        >>> line = LineString([(0,0), (2,2)])
        >>> result = shapely.ops.split(line, pt)
        >>> result.wkt
        'GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1), LINESTRING (1 1, 2 2))'

        """
        if geom.geom_type in ("MultiLineString", "MultiPolygon"):
            return GeometryCollection(
                [i for part in geom.geoms for i in SplitOp.split(part, splitter).geoms]
            )

        elif geom.geom_type == "LineString":
            if splitter.geom_type in (
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ):
                split_func = SplitOp._split_line_with_line
            elif splitter.geom_type == "Point":
                split_func = SplitOp._split_line_with_point
            elif splitter.geom_type == "MultiPoint":
                split_func = SplitOp._split_line_with_multipoint
            else:
                raise GeometryTypeError(
                    f"Splitting a LineString with a {splitter.geom_type} is "
                    "not supported"
                )

        elif geom.geom_type == "Polygon":
            if splitter.geom_type in ("LineString", "MultiLineString"):
                split_func = SplitOp._split_polygon_with_line
            else:
                raise GeometryTypeError(
                    f"Splitting a Polygon with a {splitter.geom_type} is not supported"
                )

        else:
            raise GeometryTypeError(
                f"Splitting {geom.geom_type} geometry is not supported"
            )

        return GeometryCollection(split_func(geom, splitter))


split = SplitOp.split


def substring(geom, start_dist, end_dist, normalized=False):
    """Return a line segment between specified distances along a LineString.

    Negative distance values are taken as measured in the reverse
    direction from the end of the geometry. Out-of-range index
    values are handled by clamping them to the valid range of values.

    If the start distance equals the end distance, a Point is returned.

    If the start distance is actually beyond the end distance, then the
    reversed substring is returned such that the start distance is
    at the first coordinate.

    Parameters
    ----------
    geom : LineString
        The geometry to get a substring of.
    start_dist : float
        The distance along `geom` of the start of the substring.
    end_dist : float
        The distance along `geom` of the end of the substring.
    normalized : bool, False
        Whether the distance parameters are interpreted as a
        fraction of the geometry's length.

    Returns
    -------
    Union[Point, LineString]
        The substring between `start_dist` and `end_dist` or a Point
        if they are at the same location.

    Raises
    ------
    TypeError
        If `geom` is not a LineString.

    Examples
    --------
    >>> from shapely.geometry import LineString
    >>> from shapely.ops import substring
    >>> ls = LineString((i, 0) for i in range(6))
    >>> ls.wkt
    'LINESTRING (0 0, 1 0, 2 0, 3 0, 4 0, 5 0)'
    >>> substring(ls, start_dist=1, end_dist=3).wkt
    'LINESTRING (1 0, 2 0, 3 0)'
    >>> substring(ls, start_dist=3, end_dist=1).wkt
    'LINESTRING (3 0, 2 0, 1 0)'
    >>> substring(ls, start_dist=1, end_dist=-3).wkt
    'LINESTRING (1 0, 2 0)'
    >>> substring(ls, start_dist=0.2, end_dist=-0.6, normalized=True).wkt
    'LINESTRING (1 0, 2 0)'

    Returning a `Point` when `start_dist` and `end_dist` are at the
    same location.

    >>> substring(ls, 2.5, -2.5).wkt
    'POINT (2.5 0)'

    """
    if not isinstance(geom, LineString):
        raise GeometryTypeError(
            "Can only calculate a substring of LineString geometries. "
            f"A {geom.geom_type} was provided."
        )

    # Filter out cases in which to return a point
    if start_dist == end_dist:
        return geom.interpolate(start_dist, normalized)
    elif not normalized and start_dist >= geom.length and end_dist >= geom.length:
        return geom.interpolate(geom.length, normalized)
    elif not normalized and -start_dist >= geom.length and -end_dist >= geom.length:
        return geom.interpolate(0, normalized)
    elif normalized and start_dist >= 1 and end_dist >= 1:
        return geom.interpolate(1, normalized)
    elif normalized and -start_dist >= 1 and -end_dist >= 1:
        return geom.interpolate(0, normalized)

    if normalized:
        start_dist *= geom.length
        end_dist *= geom.length

    # Filter out cases where distances meet at a middle point from opposite ends.
    if start_dist < 0 < end_dist and abs(start_dist) + end_dist == geom.length:
        return geom.interpolate(end_dist)
    elif end_dist < 0 < start_dist and abs(end_dist) + start_dist == geom.length:
        return geom.interpolate(start_dist)

    start_point = geom.interpolate(start_dist)
    end_point = geom.interpolate(end_dist)

    if start_dist < 0:
        start_dist = geom.length + start_dist  # Values may still be negative,
    if end_dist < 0:  # but only in the out-of-range
        end_dist = geom.length + end_dist  # sense, not the wrap-around sense.

    reverse = start_dist > end_dist
    if reverse:
        start_dist, end_dist = end_dist, start_dist

    start_dist = max(start_dist, 0)  # to avoid duplicating the first vertex

    if reverse:
        vertex_list = [tuple(*end_point.coords)]
    else:
        vertex_list = [tuple(*start_point.coords)]

    coords = list(geom.coords)
    current_distance = 0
    for p1, p2 in zip(coords, coords[1:]):  # noqa
        if start_dist < current_distance < end_dist:
            vertex_list.append(p1)
        elif current_distance >= end_dist:
            break

        current_distance += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    if reverse:
        vertex_list.append(tuple(*start_point.coords))
        # reverse direction result
        vertex_list = reversed(vertex_list)
    else:
        vertex_list.append(tuple(*end_point.coords))

    return LineString(vertex_list)


def clip_by_rect(geom, xmin, ymin, xmax, ymax):
    """Return the portion of a geometry within a rectangle.

    The geometry is clipped in a fast but possibly dirty way. The output is
    not guaranteed to be valid. No exceptions will be raised for topological
    errors.

    Parameters
    ----------
    geom : geometry
        The geometry to be clipped
    xmin : float
        Minimum x value of the rectangle
    ymin : float
        Minimum y value of the rectangle
    xmax : float
        Maximum x value of the rectangle
    ymax : float
        Maximum y value of the rectangle

    Notes
    -----
    New in 1.7.

    """
    if geom.is_empty:
        return geom
    return shapely.clip_by_rect(geom, xmin, ymin, xmax, ymax)


def orient(geom, sign=1.0):
    """Return a properly oriented copy of the given geometry.

    The signed area of the result will have the given sign. A sign of
    1.0 means that the coordinates of the product's exterior rings will
    be oriented counter-clockwise.

    Parameters
    ----------
    geom : Geometry
        The original geometry. May be a Polygon, MultiPolygon, or
        GeometryCollection.
    sign : float, optional.
        The sign of the result's signed area.

    Returns
    -------
    Geometry

    """
    if lib.geos_version >= (3, 12, 0):
        return shapely.orient_polygons(geom, exterior_cw=sign < 0)
    if isinstance(geom, BaseMultipartGeometry):
        return geom.__class__([orient(geom, sign) for geom in geom.geoms])
    if isinstance(geom, (Polygon,)):
        return orient_(geom, sign)
    return geom
