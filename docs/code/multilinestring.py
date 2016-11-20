from matplotlib import pyplot
from shapely.geometry import MultiLineString

from figures import SIZE, set_limits, plot_line_issimple, plot_bounds
from figures import plot_coords as _plot_coords

COLOR = {
    True:  '#6699cc',
    False: '#ffcc33'
    }

def v_color(ob):
    return COLOR[ob.is_simple]

def plot_coords(ax, ob):
    for line in ob:
        _plot_coords(ax, line, zorder=1)

def plot_lines(ax, ob):
    for line in ob:
        plot_line_issimple(ax, line, alpha=0.7, zorder=2)

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1: disconnected multilinestring
ax = fig.add_subplot(121)

mline1 = MultiLineString([((0, 0), (1, 1)), ((0, 2),  (1, 1.5), (1.5, 1), (2, 0))])

plot_coords(ax, mline1)
plot_bounds(ax, mline1)
plot_lines(ax, mline1)

ax.set_title('a) simple')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

mline2 = MultiLineString([((0, 0), (1, 1), (1.5, 1)), ((0, 2), (1, 1.5), (1.5, 1), (2, 0))])

plot_coords(ax, mline2)
plot_bounds(ax, mline2)
plot_lines(ax, mline2)

ax.set_title('b) complex')

set_limits(ax, -1, 3, -1, 3)

pyplot.show()

