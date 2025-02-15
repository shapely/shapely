import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString
from shapely.plotting import plot_line, plot_points

from figures import SIZE, BLACK, BLUE, GRAY, YELLOW, set_limits


fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: disconnected multilinestring
ax = fig.add_subplot(121)

mline1 = MultiLineString([((0, 0), (1, 1)), ((0, 2),  (1, 1.5), (1.5, 1), (2, 0))])

plot_line(mline1, ax=ax, color=BLUE)
plot_points(mline1, ax=ax, color=GRAY, alpha=0.7)
plot_points(mline1.boundary, ax=ax, color=BLACK)

ax.set_title('a) simple')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

mline2 = MultiLineString([((0, 0), (1, 1), (1.5, 1)), ((0, 2), (1, 1.5), (1.5, 1), (2, 0))])

plot_line(mline2, ax=ax, color=YELLOW)
plot_points(mline2, ax=ax, color=GRAY, alpha=0.7)
plot_points(mline2.boundary, ax=ax, color=BLACK)

ax.set_title('b) complex')

set_limits(ax, -1, 3, -1, 3)

plt.show()
