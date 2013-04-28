from matplotlib import pyplot
from shapely.geometry import Polygon
from shapely import affinity
from descartes.patch import PolygonPatch

from figures import SIZE, BLUE, GRAY


def add_origin(ax, geom, origin):
    x, y = xy = affinity.interpret_origin(geom, origin, 2)
    ax.plot(x, y, 'o', color=GRAY, zorder=1)
    ax.annotate(str(xy), xy=xy, ha='center',
                textcoords='offset points', xytext=(0, 8))

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

triangle = Polygon([(1, 1), (2, 3), (3, 1)])

xrange = [0, 5]
yrange = [0, 4]

# 1
ax = fig.add_subplot(121)

patch = PolygonPatch(triangle, facecolor=GRAY, edgecolor=GRAY,
                     alpha=0.5, zorder=1)
triangle_a = affinity.scale(triangle, xfact=1.5, yfact=-1)
patch_a = PolygonPatch(triangle_a, facecolor=BLUE, edgecolor=BLUE,
                       alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.add_patch(patch_a)

add_origin(ax, triangle, 'center')

ax.set_title("a) xfact=1.5, yfact=-1")

ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

# 2
ax = fig.add_subplot(122)

patch = PolygonPatch(triangle, facecolor=GRAY, edgecolor=GRAY,
                     alpha=0.5, zorder=1)
triangle_b = affinity.scale(triangle, xfact=2, origin=(1, 1))
patch_b = PolygonPatch(triangle_b, facecolor=BLUE, edgecolor=BLUE,
                       alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.add_patch(patch_b)

add_origin(ax, triangle, (1, 1))

ax.set_title("b) xfact=2, origin=(1, 1)")

ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()
