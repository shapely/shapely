"""Load/dump geometries using the well-known binary (WKB) format.

Also provides pickle-like convenience functions.
"""
from typing import Optional, TYPE_CHECKING, Union

import shapely

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def loads(data: Union[bytes, str], hex: bool = False) -> "BaseGeometry":
    """Load a geometry from a WKB byte string, or hex-encoded string

    ``data`` is interpreted according to its type (ignoring `hex` parameter).

    Parameters
    ----------
    data: bytes or str
        WKB byte string, or hex-encoded string
    hex: bool
        This parameter is not required anymore (As of Shapely 2).


    Raises
    ------
    GEOSException, UnicodeDecodeError
        If ``data`` contains an invalid geometry.
    """
    return shapely.from_wkb(data)


def load(fp, hex: bool = False) -> "BaseGeometry":
    """Load a geometry from an open file.

    The data from the input file is interpreted according to the
    opening mode of the file, (ignoring `hex` parameter):
    - Open file in "rb" mode for binary WKB.
    - Open file in "r" mode for hex encoded WKB.

    Parameters
    ----------
    fp
        File opened for reading
    hex: bool
        This parameter is not required anymore (As of Shapely 2).

    Raises
    ------
    GEOSException, UnicodeDecodeError
        If the given file contains an invalid geometry.
    """
    data = fp.read()
    return shapely.from_wkb(data)


def dumps(
    ob: "BaseGeometry", hex: bool = False, srid: Optional[int] = None, **kwargs
) -> Union[bytes, str]:
    """Converts to the Well-Known Binary (WKB) representation of a Geometry.
    Output is to a byte string, or a hex-encoded string if ``hex=True``.

    Parameters
    ----------
    ob : geometry
        The geometry to export to well-known binary (WKB) representation
    hex : bool
        If true, export the WKB as a hexadecimal string. The default is to
        return a binary string/bytes object.
    srid : int
        Spatial reference system ID to include in the output. The default value
        means no SRID is included.
    **kwargs : kwargs, optional
        Keyword output options passed to :func:`~shapely.to_wkb`.
    """
    if srid is not None:
        # clone the object and set the SRID before dumping
        ob = shapely.set_srid(ob, srid)
        kwargs["include_srid"] = True
    if "big_endian" in kwargs:
        # translate big_endian=True/False into byte_order=0/1
        # but if not specified, keep the default of byte_order=-1 (native)
        big_endian = kwargs.pop("big_endian")
        byte_order = 0 if big_endian else 1
        kwargs.update(byte_order=byte_order)
    return shapely.to_wkb(ob, hex=hex, **kwargs)


def dump(ob: "BaseGeometry", fp, hex: bool = False, **kwargs):
    """Dump a geometry to an open file."""
    fp.write(dumps(ob, hex=hex, **kwargs))
