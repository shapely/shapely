import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.plotting import plot_line, plot_points

from figures import BLUE, GRAY, YELLOW, GREEN, SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

a = LineString([(0, 0), (1, 1), (1,2), (2,2)])
b = LineString([(0, 0), (1, 1), (2,1), (2,2)])

# 1: disconnected multilinestring
ax = fig.add_subplot(121)

plot_line(a, ax, add_points=False, color=YELLOW, alpha=0.5)
plot_line(b, ax, add_points=False, color=GREEN, alpha=0.5)
plot_points(a, ax=ax, color=GRAY)
plot_points(b, ax=ax, color=GRAY)

ax.set_title('a) lines')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

x = a.intersection(b)

plot_line(a, ax=ax, color=GRAY, add_points=False)
plot_line(b, ax=ax, color=GRAY, add_points=False)

plot_line(x.geoms[0], ax=ax, color=BLUE)
plot_points(x.geoms[1], ax=ax, color=BLUE)

ax.set_title('b) collection')

set_limits(ax, -1, 3, -1, 3)

plt.show()
