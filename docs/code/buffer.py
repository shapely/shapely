import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.plotting import plot_polygon, plot_line

from figures import SIZE, BLUE, GRAY, set_limits

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

plot_line(line, ax=ax, add_points=False, color=GRAY, linewidth=3)

dilated = line.buffer(0.5, cap_style=3)
plot_polygon(dilated, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('a) dilation, cap_style=3')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

plot_polygon(dilated, ax=ax, add_points=False, color=GRAY, alpha=0.5)

eroded = dilated.buffer(-0.3)
plot_polygon(eroded, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('b) erosion, join_style=1')

set_limits(ax, -1, 4, -1, 3)

plt.show()
