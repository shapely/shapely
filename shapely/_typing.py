from typing import Sequence, TYPE_CHECKING, TypeVar, Union

T = TypeVar("T")

try:
    import numpy.typing as npt

    # The numpy typing module was introduced in numpy 1.20
    NumpyArray = npt.NDArray[T]
    ArrayLike = Union[npt.ArrayLike, NumpyArray[T], Sequence[T]]

except ImportError:
    import numpy as np

    # This is a fallback workaround for numpy < 1.20 (which is no longer supported since Jun 21, 2022)
    NumpyArray = Union[np.ndarray, Sequence[T]]
    ArrayLike = Union[NumpyArray[T], Sequence[T]]

if TYPE_CHECKING:
    from shapely import Geometry  # NOQA

MaybeArray = Union[T, NumpyArray[T]]
MaybeArrayLike = Union[T, ArrayLike[T]]

GeometryMaybeArray = MaybeArray["Geometry"]
GeometryMaybeArrayLike = MaybeArrayLike["Geometry"]
