# -*- coding: utf-8 -*-

import shapely
import shapely.libgeos
from shapely.geometry import Point

print('Shapely version: {}'.format(shapely.__version__))
print('GEOS version: {}'.format(shapely.libgeos.geos_version_string))

p = Point(1, 2)
print('Example WKT: {}'.format(p.wkt))
print('Pi (approx): {}'.format(p.buffer(1, 10000).area))
