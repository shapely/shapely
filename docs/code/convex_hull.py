from matplotlib import pyplot
from shapely.geometry import MultiPoint

from descartes.patch import PolygonPatch

from figures import GRAY, BLUE, SIZE, set_limits, plot_line

fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

points2 = MultiPoint([(0, 0), (2, 2)])
for p in points2:
    ax.plot(p.x, p.y, 'o', color=GRAY)
hull2 = points2.convex_hull
plot_line(ax, hull2, color=BLUE, alpha=0.5, zorder=2)

ax.set_title('a) N = 2')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

points1 = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

for p in points1:
    ax.plot(p.x, p.y, 'o', color=GRAY)
hull1 = points1.convex_hull
patch1 = PolygonPatch(hull1, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch1)

ax.set_title('b) N > 2')

set_limits(ax, -1, 4, -1, 3)

pyplot.show()


