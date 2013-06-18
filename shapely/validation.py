# TODO: allow for implementations using other than GEOS

import sys

from shapely.geos import lgeos

def explain_validity(ob):
    reason = lgeos.GEOSisValidReason(ob._geom)
    if sys.version_info[0] < 3:
        return reason
    else:
        return reason.decode('ascii')

