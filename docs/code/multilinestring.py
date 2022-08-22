import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString
from shapely.plotting import plot_line

from figures import SIZE, BLUE, YELLOW, set_limits


fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: disconnected multilinestring
ax = fig.add_subplot(121)

mline1 = MultiLineString([((0, 0), (1, 1)), ((0, 2),  (1, 1.5), (1.5, 1), (2, 0))])

for line in mline1.geoms:
    plot_line(line, ax=ax, color=BLUE)
#plot_coords(ax, mline1)
#plot_bounds(ax, mline1)
#plot_lines(ax, mline1)

ax.set_title('a) simple')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

mline2 = MultiLineString([((0, 0), (1, 1), (1.5, 1)), ((0, 2), (1, 1.5), (1.5, 1), (2, 0))])

for line in mline2.geoms:
    plot_line(line, ax=ax, color=YELLOW)
#plot_coords(ax, mline2)
#plot_bounds(ax, mline2)
#plot_lines(ax, mline2)

ax.set_title('b) complex')

set_limits(ax, -1, 3, -1, 3)

plt.show()
