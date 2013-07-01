import ogr
import pylab
from numpy import asarray

from shapely.wkb import loads

source = ogr.Open("/var/gis/data/world/world_borders.shp")
borders = source.GetLayerByName("world_borders")

fig = pylab.figure(1, figsize=(4,2), dpi=300)

while 1:
    feature = borders.GetNextFeature()
    if not feature:
        break
    
    geom = loads(feature.GetGeometryRef().ExportToWkb())
    a = asarray(geom)
    pylab.plot(a[:,0], a[:,1])

pylab.show()
