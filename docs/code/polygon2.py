import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon, plot_points

from figures import GRAY, RED, SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 3: invalid polygon, ring touch along a line
ax = fig.add_subplot(121)

ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1), (0.5, 0)]
polygon = Polygon(ext, [int])

plot_polygon(polygon, ax=ax, add_points=False, color=RED)
plot_points(polygon, ax=ax, color=GRAY, alpha=0.7)

ax.set_title('c) invalid')

set_limits(ax, -1, 3, -1, 3)

#4: invalid self-touching ring
ax = fig.add_subplot(122)
ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int_1 = [(0.5, 0.25), (1.5, 0.25), (1.5, 1.25), (0.5, 1.25), (0.5, 0.25)]
int_2 = [(0.5, 1.25), (1, 1.25), (1, 1.75), (0.5, 1.75)]
polygon = Polygon(ext, [int_1, int_2])

plot_polygon(polygon, ax=ax, add_points=False, color=RED)
plot_points(polygon, ax=ax, color=GRAY, alpha=0.7)

ax.set_title('d) invalid')

set_limits(ax, -1, 3, -1, 3)

plt.show()
