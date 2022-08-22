import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, LineString
from shapely.plotting import plot_polygon, plot_line, plot_points

from figures import DARKGRAY, GRAY, BLUE, SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

mp = MultiPoint([(0, 0), (0.5, 1.5), (1, 0.5), (0.5, 0.5)])
rect = mp.minimum_rotated_rectangle

plot_points(mp, ax=ax, color=GRAY)
plot_polygon(rect, ax=ax, add_points=False, color=BLUE, alpha=0.5, zorder=-1)
ax.set_title('a) MultiPoint')

set_limits(ax, -1, 2, -1, 2)

# 2
ax = fig.add_subplot(122)
ls = LineString([(-0.5, 1.2), (0.5, 0), (1, 1), (1.5, 0), (1.5, 0.5)])
rect = ls.minimum_rotated_rectangle

plot_line(ls, ax=ax, add_points=False, color=DARKGRAY, linewidth=3, alpha=0.5)
plot_polygon(rect, ax=ax, add_points=False, color=BLUE, alpha=0.5, zorder=-1)

set_limits(ax, -1, 2, -1, 2)

ax.set_title('b) LineString')

plt.show()
