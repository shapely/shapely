from matplotlib import pyplot
from shapely.geometry import LineString
from descartes import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits

def plot_coords(ax, x, y, color='#999999', zorder=1):
    ax.plot(x, y, 'o', color=color, zorder=zorder)

def plot_line(ax, ob, color=GRAY):
    parts = hasattr(ob, 'geoms') and ob or [ob]
    for part in parts:
        x, y = part.xy
        ax.plot(x, y, color=color, linewidth=3, solid_capstyle='round', zorder=1)

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
line_bounds = line.bounds
ax_range = [int(line_bounds[0] - 1.0), int(line_bounds[2] + 1.0)]
ay_range = [int(line_bounds[1] - 1.0), int(line_bounds[3] + 1.0)]

fig = pyplot.figure(1, figsize=(SIZE[0], 2 * SIZE[1]), dpi=90)

# 1
ax = fig.add_subplot(221)

plot_line(ax, line)
x, y = list(line.coords)[0]
plot_coords(ax, x, y)
offset = line.parallel_offset(0.5, 'left', join_style=1)
plot_line(ax, offset, color=BLUE)

ax.set_title('a) left, round')
set_limits(ax, ax_range[0], ax_range[1], ay_range[0], ay_range[1])

#2
ax = fig.add_subplot(222)

plot_line(ax, line)
x, y = list(line.coords)[0]
plot_coords(ax, x, y)

offset = line.parallel_offset(0.5, 'left', join_style=2)
plot_line(ax, offset, color=BLUE)

ax.set_title('b) left, mitred')
set_limits(ax, ax_range[0], ax_range[1], ay_range[0], ay_range[1])

#3
ax = fig.add_subplot(223)

plot_line(ax, line)
x, y = list(line.coords)[0]
plot_coords(ax, x, y)
offset = line.parallel_offset(0.5, 'left', join_style=3)
plot_line(ax, offset, color=BLUE)

ax.set_title('c) left, beveled')
set_limits(ax, ax_range[0], ax_range[1], ay_range[0], ay_range[1])

#4
ax = fig.add_subplot(224)

plot_line(ax, line)
x, y = list(line.coords)[0]
plot_coords(ax, x, y)
offset = line.parallel_offset(0.5, 'right', join_style=1)
plot_line(ax, offset, color=BLUE)

ax.set_title('d) right, round')
set_limits(ax, ax_range[0], ax_range[1], ay_range[0], ay_range[1])

pyplot.show()

