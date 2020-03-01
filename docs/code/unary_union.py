from matplotlib import pyplot
from shapely.geometry import Point
from shapely.ops import unary_union
from descartes import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits

polygons = [Point(i, 0).buffer(0.7) for i in range(5)]

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

for ob in polygons:
    p = PolygonPatch(ob, fc=GRAY, ec=GRAY, alpha=0.5, zorder=1)
    ax.add_patch(p)

ax.set_title('a) polygons')

set_limits(ax, -2, 6, -2, 2)

#2
ax = fig.add_subplot(122)

u = unary_union(polygons)
patch2b = PolygonPatch(u, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch2b)

ax.set_title('b) union')

set_limits(ax, -2, 6, -2, 2)

pyplot.show()
