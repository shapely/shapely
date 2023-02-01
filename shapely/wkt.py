"""Load/dump geometries using the well-known text (WKT) format

Also provides pickle-like convenience functions.
"""
from typing import TYPE_CHECKING

import shapely

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def loads(data: str) -> "BaseGeometry":
    """
    Creates geometry from the Well-Known Text (WKT) representation.

    Refer to `shapely.from_wkt` for full documentation.
    """
    return shapely.from_wkt(data)


def load(fp) -> "BaseGeometry":
    """
    Load a geometry from an open file.

    Parameters
    ----------
    fp :
        A file-like object which implements a `read` method.

    Returns
    -------
    Shapely geometry object
    """
    data = fp.read()
    return loads(data)


def dumps(
    ob: "BaseGeometry", trim: bool = False, rounding_precision: int = -1, **kwargs
):
    """Returns the Well-Known Text (WKT) representation of a Geometry as a string.

    Default behavior returns full precision, without trimming trailing zeros.
    Refer to `shapely.to_wkt` for full documentation.
    """
    return shapely.to_wkt(
        ob, trim=trim, rounding_precision=rounding_precision, **kwargs
    )


def dump(ob, fp, trim: bool = False, rounding_precision: int = -1, **kwargs):
    """
    Dump a geometry to an open file.

    This function writes the output of `shapely.to_wkt()` to a file.
    For full parameter details see `shapely.to_wkt()`.
    """
    fp.write(
        shapely.to_wkt(ob, trim=trim, rounding_precision=rounding_precision, **kwargs)
    )
