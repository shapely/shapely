from array import ArrayType
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union

T = TypeVar("T")
MaybeSequence = Union[T, Sequence[T]]

try:
    import numpy.typing as npt

    # The numpy typing module was introduced in numpy 1.20
    NumpyArray = npt.NDArray[T]
except ImportError:
    import numpy as np

    # This is a fallback workaround for numpy < 1.20 (which is no longer supported since Jun 21, 2022)
    NumpyArray = Union[np.ndarray, Sequence[T]]

if TYPE_CHECKING:
    from shapely import (  # NOQA
        GeometryCollection,
        LinearRing,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )
    from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry  # NOQA

# todo: specify shapes once resolved: Numpy arrays with specific shapes: https://github.com/numpy/numpy/issues/16544

NumpyArrayN = NumpyArray  # shape = (N,)
NumpyArray2 = NumpyArray  # shape = (2,)
NumpyArray3 = NumpyArray  # shape = (3,)
NumpyArray4 = NumpyArray  # shape = (4,)
NumpyArrayN2 = NumpyArray  # shape = (N,2)
NumpyArrayN3 = NumpyArray  # shape = (N,3)
NumpyArrayN4 = NumpyArray  # shape = (N,4)
MaybeArrayN2 = Union[NumpyArray2, NumpyArrayN2]  # shape = (2,) | (N,2)
MaybeArrayN3 = Union[NumpyArray3, NumpyArrayN3]  # shape = (3,) | (N,3)
MaybeArrayN4 = Union[NumpyArray4, NumpyArrayN4]  # shape = (4,) | (N,4)

MaybeArray = Union[T, NumpyArray[T]]
NumpyArrayLike = Union[Sequence[T], NumpyArray[T]]
MaybeArrayLike = Union[T, NumpyArrayLike[T]]

MaybeArrayN = Union[T, NumpyArrayN[T]]
NumpyArrayNLike = Union[Sequence[T], NumpyArrayN[T]]
MaybeArrayNLike = Union[T, NumpyArrayNLike[T]]

NumpyArrayN2Like = Union[NumpyArrayN2[T], Sequence[Sequence[T]], Sequence[Tuple[T, T]]]
NumpyArrayN3Like = Union[
    NumpyArrayN3[T], Sequence[Sequence[T]], Sequence[Tuple[T, T, T]]
]

Tuple2 = Tuple[T, T]
Tuple3 = Tuple[T, T, T]
Tuple4 = Tuple[T, T, T, T]
Tuple2or3 = Union[Tuple2, Tuple3]

NumpyArray2Like = Union[NumpyArray2[T], Sequence[T], Tuple2[T]]
NumpyArray3Like = Union[NumpyArray3[T], Sequence[T], Tuple3[T]]
NumpyArray2or3Like = Union[NumpyArray2Like, NumpyArray3Like]

NumpyArrayN2orN3 = Union[NumpyArrayN2, NumpyArrayN3]
NumpyArrayN2orN3Like = Union[NumpyArrayN2Like, NumpyArrayN3Like]

XYArrayTuple = Tuple[ArrayType, ArrayType]

GeometryArrayN = NumpyArrayN["BaseGeometry"]
GeometryArrayNLike = NumpyArrayNLike["BaseGeometry"]
MaybeGeometryArrayNLike = MaybeArrayNLike["BaseGeometry"]
MaybeGeometryArrayN = MaybeArrayN["BaseGeometry"]

GeometryArray = NumpyArray["BaseGeometry"]
GeometryArrayLike = NumpyArrayLike["BaseGeometry"]

GeoJSONlikeDict = Dict[str, Any]

PointsLike = NumpyArrayN2orN3Like[float]
Points2DLike = NumpyArrayN2Like[float]
Points3DLike = NumpyArrayN3Like[float]

Tuple2Floats = Tuple2[float]
Tuple3Floats = Tuple3[float]
Tuple2or3Floats = Tuple2or3[float]

PointLike = NumpyArray2or3Like[float]
LineStringsLike = PointsLike
LinearStringsLike = PointsLike
PolygonsLike = Union[NumpyArrayNLike["LinearRing"], PointsLike]
MultiPointsLike = Union[NumpyArrayNLike["Point"], NumpyArrayNLike[PointsLike]]
MultiLineStringsLike = Union[NumpyArrayNLike["LineString"], PointsLike]
MultiPolygonsLike = Union[NumpyArrayNLike["Polygon"], PointsLike]

LineStringLike = Union[LineStringsLike, "LineString", NumpyArrayNLike["Point"]]
LinearRingLike = Union[LineStringLike, "LinearRing"]
LinearRingHolesLike = Optional[NumpyArrayNLike[LinearRingLike]]
PolygonLike = Optional[Union["Polygon", LinearRingLike]]
MultiLineStringLike = Union[
    MultiLineStringsLike,
    "MultiLineString",
    NumpyArrayNLike[LineStringLike],
    "BaseMultipartGeometry",
]
MultiPointLike = Union[
    MultiPointsLike, "MultiPoint", NumpyArrayNLike[PointLike], "BaseMultipartGeometry"
]
MultiPolygonLike = Union[
    "BaseMultipartGeometry", "MultiPolygon", NumpyArrayNLike[PolygonLike]
]
LineStringsLikeSource = Union[
    MaybeGeometryArrayNLike,
    "MultiLineString",
    "GeometryCollection",
    Sequence[GeoJSONlikeDict],
]
