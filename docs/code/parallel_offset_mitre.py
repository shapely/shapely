import matplotlib.pyplot as plt
from shapely import LineString, get_point
from shapely.plotting import plot_line, plot_points

from figures import SIZE, BLUE, GRAY, set_limits

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
line_bounds = line.bounds
ax_range = [int(line_bounds[0] - 1.0), int(line_bounds[2] + 1.0)]
ay_range = [int(line_bounds[1] - 1.0), int(line_bounds[3] + 1.0)]

fig = plt.figure(1, figsize=(SIZE[0], 1.5 * SIZE[1]), dpi=90)

# 1
ax = fig.add_subplot(221)

plot_line(line, ax, add_points=False, color=GRAY)
plot_points(get_point(line, 0), ax=ax, color=GRAY)
offset = line.parallel_offset(0.5, 'left', join_style=2, mitre_limit=0.1)
plot_line(offset, ax=ax, add_points=False, color=BLUE)

ax.set_title('a) left, limit=0.1')
set_limits(ax, -2, 4, -1, 3)

#2
ax = fig.add_subplot(222)

plot_line(line, ax, add_points=False, color=GRAY)
plot_points(get_point(line, 0), ax=ax, color=GRAY)
offset = line.parallel_offset(0.5, 'left', join_style=2, mitre_limit=10.0)
plot_line(offset, ax=ax, add_points=False, color=BLUE)

ax.set_title('b) left, limit=10.0')
set_limits(ax, -2, 4, -1, 3)

#3
ax = fig.add_subplot(223)

plot_line(line, ax, add_points=False, color=GRAY)
plot_points(get_point(line, 0), ax=ax, color=GRAY)
offset = line.parallel_offset(0.5, 'right', join_style=2, mitre_limit=0.1)
plot_line(offset, ax=ax, add_points=False, color=BLUE)

ax.set_title('c) right, limit=0.1')
set_limits(ax, -2, 4, -1, 3)

#4
ax = fig.add_subplot(224)

plot_line(line, ax, add_points=False, color=GRAY)
plot_points(get_point(line, 0), ax=ax, color=GRAY)
offset = line.parallel_offset(0.5, 'right', join_style=2, mitre_limit=10.0)
plot_line(offset, ax=ax, add_points=False, color=BLUE)

ax.set_title('d) right, limit=10.0')
set_limits(ax, -2, 4, -1, 3)

plt.show()
