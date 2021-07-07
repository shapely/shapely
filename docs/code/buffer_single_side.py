from matplotlib import pyplot
from shapely.geometry import LineString
from descartes import PolygonPatch

from figures import SIZE, BLUE, GRAY, set_limits, plot_line

line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1
ax = fig.add_subplot(121)

plot_line(ax, line)

left_hand_side = line.buffer(0.5, single_sided=True)
patch1 = PolygonPatch(left_hand_side, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch1)

ax.set_title('a) left hand buffer')

set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

plot_line(ax, line)

right_hand_side = line.buffer(-0.3, single_sided=True)
patch2 = PolygonPatch(right_hand_side, fc=GRAY, ec=GRAY, alpha=0.5, zorder=1)
ax.add_patch(patch2)

ax.set_title('b) right hand buffer')

set_limits(ax, -1, 4, -1, 3)

pyplot.show()
