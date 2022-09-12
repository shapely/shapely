import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.plotting import plot_polygon, plot_line

from figures import SIZE, BLUE, GRAY, set_limits

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

fig = plt.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

plot_line(line, ax=ax, add_points=False, color=GRAY, linewidth=3)

left_hand_side = line.buffer(0.5, single_sided=True)
plot_polygon(left_hand_side, ax=ax, add_points=False, color=BLUE, alpha=0.5)

ax.set_title('a) left hand buffer')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

plot_line(line, ax=ax, add_points=False, color=GRAY, linewidth=3)

right_hand_side = line.buffer(-0.3, single_sided=True)
plot_polygon(right_hand_side, ax=ax, add_points=False, color=GRAY, alpha=0.5)

ax.set_title('b) right hand buffer')

set_limits(ax, -1, 4, -1, 3)

plt.show()
