"""Base geometry class and utilities

Note: a third, z, coordinate value may be used when constructing
geometry objects, but has no effect on geometric analysis. All
operations are performed in the x-y plane. Thus, geometries with
different z values may intersect or be equal.
"""
import re
from typing import Optional, Tuple, TYPE_CHECKING, Union
from warnings import warn

import numpy as np

import shapely
from shapely import Geometry
from shapely._geometry_helpers import _geom_factory
from shapely.constructive import BufferCapStyle, BufferJoinStyle
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, GEOSException, ShapelyDeprecationWarning
from shapely.shapely_typing import (
    MaybeArray,
    MaybeArrayN,
    MaybeArrayNLike,
    MaybeGeometryArrayN,
    MaybeGeometryArrayNLike,
    NumpyArray,
    T,
)

if TYPE_CHECKING:
    from shapely import Point

GEOMETRY_TYPES = [
    "Point",
    "LineString",
    "LinearRing",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
]


def geom_factory(g, parent=None):
    """
    Creates a Shapely geometry instance from a pointer to a GEOS geometry.

    .. warning::
        The GEOS library used to create the GEOS geometry pointer
        and the GEOS library used by Shapely must be exactly the same, or
        unexpected results or segfaults may occur.

    .. deprecated:: 2.0
        Deprecated in Shapely 2.0, and will be removed in a future version.
    """
    warn(
        "The 'geom_factory' function is deprecated in Shapely 2.0, and will be "
        "removed in a future version",
        DeprecationWarning,
        stacklevel=2,
    )
    return _geom_factory(g)


def dump_coords(geom: "BaseGeometry"):
    """Dump coordinates of a geometry in the same order as data packing"""
    if not isinstance(geom, BaseGeometry):
        raise ValueError(
            "Must be instance of a geometry class; found " + geom.__class__.__name__
        )
    elif geom.geom_type in ("Point", "LineString", "LinearRing"):
        return geom.coords[:]
    elif geom.geom_type == "Polygon":
        return geom.exterior.coords[:] + [i.coords[:] for i in geom.interiors]
    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        # Recursive call
        return [dump_coords(part) for part in geom.geoms]
    else:
        raise GeometryTypeError("Unhandled geometry type: " + repr(geom.geom_type))


def _maybe_unpack(result: NumpyArray[T]) -> MaybeArray[T]:
    if result.ndim == 0:
        # convert numpy 0-d array / scalar to python scalar
        return result.item()
    else:
        # >=1 dim array
        return result


class CAP_STYLE:
    round = BufferCapStyle.round
    flat = BufferCapStyle.flat
    square = BufferCapStyle.square


class JOIN_STYLE:
    round = BufferJoinStyle.round
    mitre = BufferJoinStyle.mitre
    bevel = BufferJoinStyle.bevel


class BaseGeometry(Geometry):
    """
    Provides GEOS spatial predicates and topological operations.

    """

    __slots__ = []

    def __new__(cls):
        warn(
            "Directly calling the base class 'BaseGeometry()' is deprecated, and "
            "will raise an error in the future. To create an empty geometry, "
            "use one of the subclasses instead, for example 'GeometryCollection()'.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return shapely.from_wkt("GEOMETRYCOLLECTION EMPTY")

    @property
    def _ndim(self) -> int:
        return shapely.get_coordinate_dimension(self)

    def __bool__(self) -> bool:
        return self.is_empty is False

    def __nonzero__(self) -> bool:
        return self.__bool__()

    def __format__(self, format_spec: str) -> str:
        """Format a geometry using a format specification."""
        # bypass regexp for simple cases
        if format_spec == "":
            return shapely.to_wkt(self, rounding_precision=-1)
        elif format_spec == "x":
            return shapely.to_wkb(self, hex=True).lower()
        elif format_spec == "X":
            return shapely.to_wkb(self, hex=True)

        # fmt: off
        format_spec_regexp = (
            "(?:0?\\.(?P<prec>[0-9]+))?"
            "(?P<fmt_code>[fFgGxX]?)"
        )
        # fmt: on
        match = re.fullmatch(format_spec_regexp, format_spec)
        if match is None:
            raise ValueError(f"invalid format specifier: {format_spec}")

        prec, fmt_code = match.groups()

        if prec:
            prec = int(prec)
        else:
            # GEOS has a default rounding_precision -1
            prec = -1

        if not fmt_code:
            fmt_code = "g"

        if fmt_code in ("g", "G"):
            res = shapely.to_wkt(self, rounding_precision=prec, trim=True)
        elif fmt_code in ("f", "F"):
            res = shapely.to_wkt(self, rounding_precision=prec, trim=False)
        elif fmt_code in ("x", "X"):
            raise ValueError("hex representation does not specify precision")
        else:
            raise NotImplementedError(f"unhandled fmt_code: {fmt_code}")

        if fmt_code.isupper():
            return res.upper()
        else:
            return res

    def __repr__(self) -> str:
        try:
            wkt = super().__str__()
        except (GEOSException, ValueError):
            # we never want a repr() to fail; that can be very confusing
            return "<shapely.{} Exception in WKT writer>".format(
                self.__class__.__name__
            )

        # the total length is limited to 80 characters including brackets
        max_length = 78
        if len(wkt) > max_length:
            return f"<{wkt[: max_length - 3]}...>"

        return f"<{wkt}>"

    def __str__(self) -> str:
        return self.wkt

    def __reduce__(self):
        return (shapely.from_wkb, (shapely.to_wkb(self, include_srid=True),))

    # Operators
    # ---------

    def __and__(self, other: "BaseGeometry") -> "BaseGeometry":
        return self.intersection(other)

    def __or__(self, other: "BaseGeometry") -> "BaseGeometry":
        return self.union(other)

    def __sub__(self, other: "BaseGeometry") -> "BaseGeometry":
        return self.difference(other)

    def __xor__(self, other: "BaseGeometry") -> "BaseGeometry":
        return self.symmetric_difference(other)

    # Coordinate access
    # -----------------

    @property
    def coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        coords_array = shapely.get_coordinates(self, include_z=self.has_z)
        return CoordinateSequence(coords_array)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values"""
        raise NotImplementedError

    # Python feature protocol

    @property
    def __geo_interface__(self):
        """Dictionary representation of the geometry"""
        raise NotImplementedError

    # Type of geometry and its representations
    # ----------------------------------------

    def geometryType(self):
        warn(
            "The 'GeometryType()' method is deprecated, and will be removed in "
            "the future. You can use the 'geom_type' attribute instead.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return self.geom_type

    @property
    def type(self):
        warn(
            "The 'type' attribute is deprecated, and will be removed in "
            "the future. You can use the 'geom_type' attribute instead.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return self.geom_type

    @property
    def wkt(self):
        """WKT representation of the geometry"""
        # TODO(shapely-2.0) keep default of not trimming?
        return shapely.to_wkt(self, rounding_precision=-1)

    @property
    def wkb(self):
        """WKB representation of the geometry"""
        return shapely.to_wkb(self)

    @property
    def wkb_hex(self):
        """WKB hex representation of the geometry"""
        return shapely.to_wkb(self, hex=True)

    def svg(self, scale_factor=1.0, **kwargs):
        """Raises NotImplementedError"""
        raise NotImplementedError

    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        svg_top = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
        )
        if self.is_empty:
            return svg_top + "/>"
        else:
            # Establish SVG canvas that will fit all the data + small space
            xmin, ymin, xmax, ymax = self.bounds
            if xmin == xmax and ymin == ymax:
                # This is a point; buffer using an arbitrary size
                xmin, ymin, xmax, ymax = self.buffer(1).bounds
            else:
                # Expand bounds by a fraction of the data ranges
                expand = 0.04  # or 4%, same as R plots
                widest_part = max([xmax - xmin, ymax - ymin])
                expand_amount = widest_part * expand
                xmin -= expand_amount
                ymin -= expand_amount
                xmax += expand_amount
                ymax += expand_amount
            dx = xmax - xmin
            dy = ymax - ymin
            width = min([max([100.0, dx]), 300])
            height = min([max([100.0, dy]), 300])
            try:
                scale_factor = max([dx, dy]) / max([width, height])
            except ZeroDivisionError:
                scale_factor = 1.0
            view_box = f"{xmin} {ymin} {dx} {dy}"
            transform = f"matrix(1,0,0,-1,0,{ymax + ymin})"
            return svg_top + (
                'width="{1}" height="{2}" viewBox="{0}" '
                'preserveAspectRatio="xMinYMin meet">'
                '<g transform="{3}">{4}</g></svg>'
            ).format(view_box, width, height, transform, self.svg(scale_factor))

    @property
    def geom_type(self):
        """Name of the geometry's type, such as 'Point'"""
        return GEOMETRY_TYPES[shapely.get_type_id(self)]

    # Real-valued properties and methods
    # ----------------------------------

    @property
    def area(self) -> float:
        """Unitless area of the geometry"""
        return float(shapely.area(self))

    def distance(self, other: MaybeGeometryArrayNLike) -> MaybeArrayN[float]:
        """Unitless Cartesian distance to other geometry"""
        return _maybe_unpack(shapely.distance(self, other))

    def hausdorff_distance(self, other: MaybeGeometryArrayNLike) -> MaybeArrayN[float]:
        """Unitless hausdorff distance to other geometry"""
        return _maybe_unpack(shapely.hausdorff_distance(self, other))

    @property
    def length(self) -> float:
        """Unitless length of the geometry"""
        return float(shapely.length(self))

    @property
    def minimum_clearance(self) -> float:
        """Unitless distance by which a node could be moved to produce an invalid geometry"""
        return float(shapely.minimum_clearance(self))

    # Topological properties
    # ----------------------

    @property
    def boundary(self) -> "BaseGeometry":
        """Returns the topological boundary of a geometry
        (lower dimension geometry that bounds the object).

        Refer to `shapely.boundary` for full documentation."""
        return shapely.boundary(self)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        return tuple(shapely.bounds(self).tolist())

    @property
    def centroid(self):
        """Computes the geometric center (center-of-mass) of a geometry.

        Refer to `shapely.centroid` for full documentation."""
        return shapely.centroid(self)

    def point_on_surface(self):
        """Returns a point that intersects (guaranteed to be within) an input geometry, cheaply.

        Alias of `representative_point`.
        Refer to `shapely.point_on_surface` for full documentation."""
        return shapely.point_on_surface(self)

    def representative_point(self):
        """Returns a point that intersects (guaranteed to be within) an input geometry, cheaply.

        Alias of `point_on_surface`.
        Refer to `shapely.point_on_surface` for full documentation."""
        return shapely.point_on_surface(self)

    @property
    def convex_hull(self):
        """Computes the minimum convex geometry that encloses an input geometry.
        Imagine an elastic band stretched around the geometry:
        that's a convex hull, more or less.

        Refer to `shapely.convex_hull` for full documentation."""
        return shapely.convex_hull(self)

    @property
    def envelope(self):
        """Computes the minimum bounding box that encloses an input geometry.

        Refer to `shapely.envelope` for full documentation."""
        return shapely.envelope(self)

    @property
    def oriented_envelope(self):
        """Computes the oriented envelope (minimum rotated rectangle)
        that encloses an input geometry.

        Alias of `minimum_rotated_rectangle`.
        Refer to `shapely.oriented_envelope` for full documentation."""
        return shapely.oriented_envelope(self)

    @property
    def minimum_rotated_rectangle(self):
        """Computes the oriented envelope (minimum rotated rectangle)
        that encloses an input geometry.

        Alias of `oriented_envelope`.
        Refer to `shapely.oriented_envelope` for full documentation."""
        return shapely.oriented_envelope(self)

    def buffer(
        self,
        distance: float,
        quad_segs: int = 16,
        cap_style: Union[BufferCapStyle, str] = "round",
        join_style: Union[BufferJoinStyle, str] = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
        **kwargs,
    ):
        """Computes the buffer of a geometry for positive and negative buffer distance,
        A geometry that represents all points within a distance of this geometry.

        This function calls `shapely.buffer()` but also accepts the following aliases
        for `quad_segs` parameter:
        `quadsegs` (deprecated) and `resolution` (will be deprecated in Shapely 2.1)
        Refer to `shapely.buffer` for full documentation."""
        quadsegs = kwargs.pop("quadsegs", None)
        if quadsegs is not None:
            warn(
                "The `quadsegs` argument is deprecated. Use `quad_segs` instead.",
                FutureWarning,
            )
            quad_segs = quadsegs

        # TODO deprecate `resolution` keyword for shapely 2.1
        resolution = kwargs.pop("resolution", None)
        if resolution is not None:
            quad_segs = resolution
        if kwargs:
            kwarg = list(kwargs.keys())[0]  # noqa
            raise TypeError(f"buffer() got an unexpected keyword argument '{kwarg}'")

        if mitre_limit == 0.0:
            raise ValueError("Cannot compute offset from zero-length line segment")
        elif not np.isfinite(distance).all():
            raise ValueError("buffer distance must be finite")

        return shapely.buffer(
            self,
            distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
        )

    def simplify(self, tolerance, preserve_topology=True):
        """Returns a simplified version of an input geometry using the
        Douglas-Peucker algorithm.

        Refer to `shapely.simplify` for full documentation."""
        return shapely.simplify(self, tolerance, preserve_topology=preserve_topology)

    def normalize(self):
        """Converts geometry to normal form (or canonical form).

        Refer to `shapely.normalize` for full documentation."""
        return shapely.normalize(self)

    # Overlay operations
    # ---------------------------

    def difference(
        self, other: MaybeGeometryArrayNLike, grid_size: Optional[float] = None
    ) -> MaybeGeometryArrayN:
        """Returns the difference of the geometries.

        Refer to `shapely.difference` for full documentation."""
        return shapely.difference(self, other, grid_size=grid_size)

    def intersection(
        self, other: MaybeGeometryArrayNLike, grid_size: Optional[float] = None
    ) -> MaybeGeometryArrayN:
        """Returns the intersection of the geometries.

        Refer to `shapely.intersection` for full documentation."""
        return shapely.intersection(self, other, grid_size=grid_size)

    def symmetric_difference(
        self, other: MaybeGeometryArrayNLike, grid_size: Optional[float] = None
    ) -> MaybeGeometryArrayN:
        """Returns the symmetric difference of the geometries.

        Refer to `shapely.symmetric_difference` for full documentation."""
        return shapely.symmetric_difference(self, other, grid_size=grid_size)

    def union(
        self, other: MaybeGeometryArrayNLike, grid_size: Optional[float] = None
    ) -> MaybeGeometryArrayN:
        """Returns the union of the geometries.

        Refer to `shapely.union` for full documentation."""
        return shapely.union(self, other, grid_size=grid_size)

    # Unary predicates
    # ----------------

    @property
    def has_z(self) -> bool:
        """True if the geometry's coordinate sequence(s) have z values (are
        3-dimensional)"""
        return bool(shapely.has_z(self))

    @property
    def is_empty(self) -> bool:
        """True if the set of points in this geometry is empty, else False"""
        return bool(shapely.is_empty(self))

    @property
    def is_ring(self) -> bool:
        """True if the geometry is a closed ring, else False"""
        return bool(shapely.is_ring(self))

    @property
    def is_closed(self) -> bool:
        """True if the geometry is closed, else False

        Applicable only to 1-D geometries."""
        if self.geom_type == "LinearRing":
            return True
        return bool(shapely.is_closed(self))

    @property
    def is_simple(self) -> bool:
        """True if the geometry is simple, meaning that any self-intersections
        are only at boundary points, else False"""
        return bool(shapely.is_simple(self))

    @property
    def is_valid(self) -> bool:
        """True if the geometry is valid (definition depends on sub-class),
        else False"""
        return bool(shapely.is_valid(self))

    # Binary predicates
    # -----------------

    def relate(self, other: MaybeGeometryArrayN) -> MaybeArrayN[str]:
        """Returns the DE-9IM intersection matrix for the two geometries
        (string)"""
        return shapely.relate(self, other)

    def covers(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the geometry covers the other, else False"""
        return _maybe_unpack(shapely.covers(self, other))

    def covered_by(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the geometry is covered by the other, else False"""
        return _maybe_unpack(shapely.covered_by(self, other))

    def contains(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the geometry contains the other, else False"""
        return _maybe_unpack(shapely.contains(self, other))

    def contains_properly(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the geometry completely contains the other, with no
        common boundary points, else False

        Refer to `shapely.contains_properly` for full documentation."""
        return _maybe_unpack(shapely.contains_properly(self, other))

    def crosses(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the geometries cross, else False"""
        return _maybe_unpack(shapely.crosses(self, other))

    def disjoint(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if geometries are disjoint, else False"""
        return _maybe_unpack(shapely.disjoint(self, other))

    def equals(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if geometries are spatially equal.

        Refer to `shapely.equals` for full documentation."""
        return _maybe_unpack(shapely.equals(self, other))

    def intersects(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if geometries share any portion of space.

        Refer to `shapely.intersects` for full documentation."""
        return _maybe_unpack(shapely.intersects(self, other))

    def overlaps(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if geometries spatially overlap

        Refer to `shapely.overlaps` for full documentation."""
        return _maybe_unpack(shapely.overlaps(self, other))

    def touches(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if the only points shared between the geometries
        are on ir boundary.

        Refer to `shapely.touches` for full documentation."""
        return _maybe_unpack(shapely.touches(self, other))

    def within(self, other: MaybeGeometryArrayN) -> MaybeArrayN[bool]:
        """Returns True if geometry is completely inside the other

        Refer to `shapely.within` for full documentation."""
        return _maybe_unpack(shapely.within(self, other))

    def dwithin(self, other: MaybeGeometryArrayN, distance: float) -> MaybeArrayN[bool]:
        """Returns True if geometry is within a given distance from the other.

        Refer to `shapely.dwithin` for full documentation."""
        return _maybe_unpack(shapely.dwithin(self, other, distance))

    def equals_exact(
        self,
        other: MaybeGeometryArrayN,
        tolerance: float,
    ) -> MaybeArrayN[bool]:
        """Returns True if A and B are structurally equal, within a specified tolerance.

        Refer to `shapely.equals_exact` for full documentation."""
        return _maybe_unpack(shapely.equals_exact(self, other, tolerance))

    def almost_equals(
        self, other: MaybeGeometryArrayN, decimal: int = 6
    ) -> MaybeArrayN[bool]:
        """True if geometries are equal at all coordinates to a
        specified decimal place.

        .. deprecated:: 1.8.0
            The 'almost_equals()' method is deprecated
            and will be removed in Shapely 2.1 because the name is
            confusing. The 'equals_exact()' method should be used
            instead.

        Refers to approximate coordinate equality, which requires
        coordinates to be approximately equal and in the same order for
        all components of a geometry.

        Because of this it is possible for "equals()" to be True for two
        geometries and "almost_equals()" to be False.

        Examples
        --------
        >>> LineString(
        ...     [(0, 0), (2, 2)]
        ... ).equals_exact(
        ...     LineString([(0, 0), (1, 1), (2, 2)]),
        ...     1e-6
        ... )
        False

        Returns
        -------
        bool

        """
        warn(
            "The 'almost_equals()' method is deprecated and will be "
            "removed in Shapely 2.1; use 'equals_exact()' instead",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return self.equals_exact(other, 0.5 * 10 ** (-decimal))

    def relate_pattern(
        self, other: MaybeGeometryArrayN, pattern: str
    ) -> MaybeArrayN[str]:
        """
        Returns True if the DE-9IM string code for the relationship between
        the geometries satisfies the pattern, else False

        Refer to `shapely.relate_pattern` for full documentation."""
        return _maybe_unpack(shapely.relate_pattern(self, other, pattern))

    # Linear referencing
    # ------------------

    def line_locate_point(
        self, other: MaybeGeometryArrayN, normalized: bool = False
    ) -> MaybeArrayN[float]:
        """Returns the distance to the line origin of given point.
        (distance along this geometry to a point nearest the given point)

        Alias of `project`.
        Refer to `shapely.line_locate_point` for full documentation."""
        return shapely.line_locate_point(self, other, normalized=normalized)

    def project(
        self, other: MaybeGeometryArrayN, normalized: bool = False
    ) -> MaybeArrayN[float]:
        """Returns the distance to the line origin of given point.
        (distance along this geometry to a point nearest the given point)

        Alias of `line_locate_point`.
        Refer to `shapely.line_locate_point` for full documentation."""
        return shapely.line_locate_point(self, other, normalized=normalized)

    def line_interpolate_point(
        self, distance: MaybeArrayNLike[float], normalized: bool = False
    ) -> MaybeArrayN["Point"]:
        """Returns a point interpolated at given distance on a line.
        (a point at the specified distance along a linear geometry)

        Alias of `interpolate`.
        Refer to `shapely.line_interpolate_point` for full documentation."""
        return shapely.line_interpolate_point(self, distance, normalized=normalized)

    def interpolate(
        self, distance: MaybeArrayNLike[float], normalized: bool = False
    ) -> MaybeArrayN["Point"]:
        """Returns a point interpolated at given distance on a line.
        (a point at the specified distance along a linear geometry)

        Alias of `line_interpolate_point`.
        Refer to `shapely.line_interpolate_point` for full documentation."""
        return shapely.line_interpolate_point(self, distance, normalized=normalized)

    def segmentize(
        self, max_segment_length: MaybeArrayNLike[float]
    ) -> MaybeGeometryArrayN:
        """Adds vertices to line segments based on maximum segment length.

        Refer to `shapely.segmentize` for full documentation."""
        return shapely.segmentize(self, max_segment_length)

    def reverse(self) -> "BaseGeometry":
        """Returns a copy of this geometry with the order of coordinates reversed.

        Refer to `shapely.reverse` for full documentation."""
        return shapely.reverse(self)


class BaseMultipartGeometry(BaseGeometry):

    __slots__ = []

    @property
    def coords(self):
        raise NotImplementedError(
            "Sub-geometries may have coordinate sequences, "
            "but multi-part geometries do not"
        )

    @property
    def geoms(self) -> "GeometrySequence":
        return GeometrySequence(self)

    def __bool__(self) -> bool:
        return self.is_empty is False

    def svg(self, scale_factor: float = 1.0, color: Optional[str] = None) -> str:
        """Returns a group of SVG elements for the multipart geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        color : str, optional
            Hex string for stroke or fill color. Default is to use "#66cc99"
            if geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return "<g />"
        if color is None:
            color = "#66cc99" if self.is_valid else "#ff3333"
        return "<g>" + "".join(p.svg(scale_factor, color) for p in self.geoms) + "</g>"


class GeometrySequence:
    """
    Iterative access to members of a homogeneous multipart geometry.
    """

    # Attributes
    # ----------
    # _parent : object
    #     Parent (Shapely) geometry
    _parent = None

    def __init__(self, parent):
        self._parent = parent

    def _get_geom_item(self, i):
        return shapely.get_geometry(self._parent, i)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self._get_geom_item(i)

    def __len__(self) -> int:
        return shapely.get_num_geometries(self._parent)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, (int, np.integer)):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_geom_item(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_geom_item(i))
            return type(self._parent)(res or None)
        else:
            raise TypeError("key must be an index or slice")


class EmptyGeometry(BaseGeometry):
    def __new__(cls):
        """Create an empty geometry."""
        warn(
            "The 'EmptyGeometry()' constructor to create an empty geometry is "
            "deprecated, and will raise an error in the future. Use one of the "
            "geometry subclasses instead, for example 'GeometryCollection()'.",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return shapely.from_wkt("GEOMETRYCOLLECTION EMPTY")
