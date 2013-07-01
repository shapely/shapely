from matplotlib import pyplot
from shapely.geometry import MultiPoint

from descartes.patch import PolygonPatch

from figures import SIZE

fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

points2 = MultiPoint([(0, 0), (2, 2)])
for p in points2:
    ax.plot(p.x, p.y, 'o', color='#999999')
hull2 = points2.convex_hull
x, y = hull2.xy
ax.plot(x, y, color='#6699cc', linewidth=3, alpha=0.5, zorder=2)

ax.set_title('a) N = 2')

xrange = [-1, 4]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

#2
ax = fig.add_subplot(122)

points1 = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

for p in points1:
    ax.plot(p.x, p.y, 'o', color='#999999')
hull1 = points1.convex_hull
patch1 = PolygonPatch(hull1, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
ax.add_patch(patch1)

ax.set_title('b) N > 2')

xrange = [-1, 4]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()


