from matplotlib import pyplot
from shapely.geometry import MultiPolygon
from descartes.patch import PolygonPatch

from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
    
fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1: valid multi-polygon
ax = fig.add_subplot(121)

a = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
b = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]

multi1 = MultiPolygon([[a, []], [b, []]])

for polygon in multi1:
    plot_coords(ax, polygon.exterior)
    patch = PolygonPatch(polygon, facecolor=color_isvalid(multi1), edgecolor=color_isvalid(multi1, valid=BLUE), alpha=0.5, zorder=2)
    ax.add_patch(patch)

ax.set_title('a) valid')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

c = [(0, 0), (0, 1.5), (1, 1.5), (1, 0), (0, 0)]
d = [(1, 0.5), (1, 2), (2, 2), (2, 0.5), (1, 0.5)]

multi2 = MultiPolygon([[c, []], [d, []]])

for polygon in multi2:
    plot_coords(ax, polygon.exterior)
    patch = PolygonPatch(polygon, facecolor=color_isvalid(multi2), edgecolor=color_isvalid(multi2, valid=BLUE), alpha=0.5, zorder=2)
    ax.add_patch(patch)

ax.set_title('b) invalid')

set_limits(ax, -1, 3, -1, 3)

pyplot.show()

