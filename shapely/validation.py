# TODO: allow for implementations using other than GEOS
from typing import TYPE_CHECKING

import shapely

__all__ = ["explain_validity", "make_valid"]

from shapely.shapely_typing import MaybeArrayN, MaybeGeometryArrayNLike

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def explain_validity(ob: MaybeGeometryArrayNLike) -> MaybeArrayN[str]:
    """Returns a string stating if a geometry is valid and if not, why.
    The explanation might include a location if there is a self-intersection or a
    ring self-intersection.

    Refer to `shapely.is_valid_reason` for full documentation.
    """
    return shapely.is_valid_reason(ob)


def make_valid(ob: "BaseGeometry") -> "BaseGeometry":
    """Returns repaired geometry according to the GEOS MakeValid algorithm.

    Refer to `shapely.make_valid` for full documentation.
    """
    if ob.is_valid:
        return ob
    return shapely.make_valid(ob)
