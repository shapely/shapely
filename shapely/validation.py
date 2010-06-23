#

from shapely.geos import lgeos

def explain_validity(ob):
    return lgeos.GEOSisValidReason(ob._geom)

