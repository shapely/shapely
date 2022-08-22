import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from figures import SIZE, BLUE, RED, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: valid polygon
ax = fig.add_subplot(121)

ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
polygon = Polygon(ext, [int])

plot_polygon(polygon, ax=ax, color=BLUE)

ax.set_title('a) valid')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)
ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(1, 0), (0, 1), (0.5, 1.5), (1.5, 0.5), (1, 0)][::-1]
polygon = Polygon(ext, [int])

plot_polygon(polygon, ax=ax, color=RED)

ax.set_title('b) invalid')

set_limits(ax, -1, 3, -1, 3)

plt.show()
