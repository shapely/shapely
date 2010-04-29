from matplotlib import pyplot
from shapely.geometry import MultiPoint, Point
from descartes.patch import PolygonPatch

from figures import SIZE

fig = pyplot.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)

p = Point(0, 0).buffer(1.0)

# 1
ax = fig.add_subplot(121)

q = p.simplify(0.1)

patch1a = PolygonPatch(p, facecolor='#cccccc', edgecolor='#999999')
ax.add_patch(patch1a)

patch1b = PolygonPatch(q, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch1b)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect(1)
ax.set_axis_off()

#2
ax = fig.add_subplot(122)

r = p.simplify(0.5)

patch2a = PolygonPatch(p, facecolor='#cccccc', edgecolor='#999999')
ax.add_patch(patch2a)

patch2b = PolygonPatch(r, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch2b)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect(1)
ax.set_axis_off()

fig.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.1)
fig.savefig('simplify.png')


