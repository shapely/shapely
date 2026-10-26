import urllib.request

import numpy as np
import matplotlib.pyplot as plt

import shapely
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE

## Downloading and preprocessing data

# download countries geojson from https://datahub.io/core/geo-countries
with urllib.request.urlopen("https://datahub.io/core/geo-countries/_r/-/data/countries.geojson") as f:
    geojson = f.read().decode("utf-8")

geoms = np.asarray(shapely.from_geojson(geojson).geoms)

# select countries of Africa
clip_polygon = shapely.from_wkt("POLYGON ((-23.714863 14.983714, 0.434636 -51.130505, 52.292267 -50.120681, 60.184885 -22.43548, 57.448957 14.358282, 44.596307 11.524365, 33.72577 34.648769, 7.226985 39.73104, -16.429284 33.649053, -23.714863 14.983714))")
geoms_africa = geoms[shapely.within(geoms, clip_polygon)]

# remove small islands for nicer illustration of coverage
temp = geoms_africa[shapely.area(geoms_africa) > 0.1]
parts, indices = shapely.get_parts(temp, return_index=True)
mask = shapely.area(parts) > 0.08
temp = shapely.multipolygons(parts[mask], indices=indices[mask])

# set precision to improve coverage validity
geoms_africa2 = shapely.set_precision(temp, 0.001)

## Coverage simplify

geoms_africa_simplified = shapely.coverage_simplify(geoms_africa2, tolerance=5.0)

# plot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=SIZE, dpi=90)

for geom in geoms_africa2:
    plot_polygon(geom, ax=ax1, add_points=False, color=BLUE)

ax1.set_title('a) original data')
ax1.axis("off")
ax1.set_aspect("equal")

for geom in geoms_africa_simplified:
    plot_polygon(geom, ax=ax2, add_points=False, color=BLUE)

ax2.set_title('b) coverage simplified')
ax2.axis("off")
ax2.set_aspect("equal")

plt.show()
