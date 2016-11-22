from matplotlib import pyplot
from shapely.geometry import Point
from descartes import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

a = Point(1, 1).buffer(1.5)
b = Point(2, 1).buffer(1.5)

# 1
ax = fig.add_subplot(121)

patch1 = PolygonPatch(a, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch1)
patch2 = PolygonPatch(b, fc=GRAY, ec=GRAY, alpha=0.2, zorder=1)
ax.add_patch(patch2)
c = a.union(b)
patchc = PolygonPatch(c, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patchc)

ax.set_title('a.union(b)')

set_limits(ax, -1, 4, -1, 3)

def plot_line(ax, ob, color=GRAY):
    x, y = ob.xy
    ax.plot(x, y, color, linewidth=3, solid_capstyle='round', zorder=1)

#2
ax = fig.add_subplot(122)

plot_line(ax, a.exterior)
plot_line(ax, b.exterior)

u = a.exterior.union(b.exterior)
if u.geom_type in ['LineString', 'LinearRing', 'Point']:
    plot_line(ax, u, color=BLUE)
elif u.geom_type is 'MultiLineString':
    for p in u:
        plot_line(ax, p, color=BLUE)

ax.set_title('a.boundary.union(b.boundary)')

set_limits(ax, -1, 4, -1, 3)

pyplot.show()

