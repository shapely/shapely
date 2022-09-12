import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.plotting import plot_polygon, plot_line, plot_points

from figures import GRAY, BLUE, SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

points2 = MultiPoint([(0, 0), (2, 2)])
plot_points(points2, ax=ax, color=GRAY)

hull2 = points2.convex_hull
plot_line(hull2, ax=ax, add_points=False, color=BLUE, zorder=3)

ax.set_title('a) N = 2')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

points1 = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
plot_points(points1, ax=ax, color=GRAY)

hull1 = points1.convex_hull
plot_polygon(hull1, ax=ax, add_points=False, color=BLUE, zorder=3, alpha=0.5)

ax.set_title('b) N > 2')

set_limits(ax, -1, 4, -1, 3)

plt.show()