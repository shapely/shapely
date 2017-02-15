from matplotlib import pyplot
from shapely.geometry import MultiPoint, Point
from descartes.patch import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits

fig = pyplot.figure(1, figsize=SIZE, dpi=90) #1, figsize=SIZE, dpi=90)

p = Point(1, 1).buffer(1.5)

# 1
ax = fig.add_subplot(121)

q = p.simplify(0.2)

patch1a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch1a)

patch1b = PolygonPatch(q, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch1b)

ax.set_title('a) tolerance 0.2')

set_limits(ax, -1, 3, -1, 3)

#2
ax = fig.add_subplot(122)

r = p.simplify(0.5)

patch2a = PolygonPatch(p, facecolor=GRAY, edgecolor=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch2a)

patch2b = PolygonPatch(r, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch2b)

ax.set_title('b) tolerance 0.5')

set_limits(ax, -1, 3, -1, 3)

pyplot.show()


