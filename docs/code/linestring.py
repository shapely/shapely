import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.plotting import plot_line

from figures import SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1: simple line
ax = fig.add_subplot(121)
line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

plot_line(line, ax=ax, color='#6699cc', alpha=0.7)

ax.set_title('a) simple')

set_limits(ax, -1, 4, -1, 3)

# 2: complex line
ax = fig.add_subplot(122)
line2 = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (-1, 1), (1, 0)])

plot_line(line2, ax=ax, color='#ffcc33', alpha=0.7)

ax.set_title('b) complex')

set_limits(ax, -2, 3, -1, 3)

plt.show()
