from matplotlib import pyplot
from shapely.geometry import LineString
from shapely import affinity

from figures import SIZE, BLUE, GRAY


def add_origin(ax, geom, origin):
    x, y = xy = affinity.interpret_origin(geom, origin, 2)
    ax.plot(x, y, 'o', color=GRAY, zorder=1)
    ax.annotate(str(xy), xy=xy, ha='center',
                textcoords='offset points', xytext=(0, 8))


def plot_line(ax, ob, color):
    x, y = ob.xy
    ax.plot(x, y, color=color, alpha=0.7, linewidth=3,
            solid_capstyle='round', zorder=2)

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

line = LineString([(1, 3), (1, 1), (4, 1)])

xrange = [0, 5]
yrange = [0, 4]

# 1
ax = fig.add_subplot(121)

plot_line(ax, line, GRAY)
plot_line(ax, affinity.rotate(line, 90, 'center'), BLUE)
add_origin(ax, line, 'center')

ax.set_title(u"90\N{DEGREE SIGN}, default origin (center)")

ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

# 2
ax = fig.add_subplot(122)

plot_line(ax, line, GRAY)
plot_line(ax, affinity.rotate(line, 90, 'centroid'), BLUE)
add_origin(ax, line, 'centroid')

ax.set_title(u"90\N{DEGREE SIGN}, origin='centroid'")

ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()
