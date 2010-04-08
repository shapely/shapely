from matplotlib import pyplot
from shapely.geometry import MultiPoint

from descartes.patch import PolygonPatch

fig = pyplot.figure(1, figsize=(7.5, 3), dpi=180)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

points1 = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
hull1 = points1.convex_hull

patch1 = PolygonPatch(hull1, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch1)

for p in points1:
    ax.plot(p.x, p.y, 'o', color='#999999')

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_aspect(1)
ax.set_axis_off()

#2
ax = fig.add_subplot(122)

points2 = MultiPoint([(0, 0), (2, 2)])
hull2 = points2.convex_hull

x, y = hull2.xy
ax.plot(x, y, color='#6699cc')

for p in points2:
    ax.plot(p.x, p.y, 'o', color='#999999')

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_aspect(1)
ax.set_axis_off()

fig.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.1)
fig.savefig('convex-hull.png')


