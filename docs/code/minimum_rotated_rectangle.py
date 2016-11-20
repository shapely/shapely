from shapely.geometry import MultiPoint, Polygon, LineString
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch

from figures import DARKGRAY, GRAY, BLUE, SIZE, set_limits

fig = plt.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

mp = MultiPoint([(0, 0), (0.5, 1.5), (1, 0.5), (0.5, 0.5)])
rect = mp.minimum_rotated_rectangle

for p in mp:
	ax.plot(p.x, p.y, 'o', color=GRAY)
patch = PolygonPatch(rect, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.set_title('a) MultiPoint')

set_limits(ax, -1, 2, -1, 2)

# 2
ax = fig.add_subplot(122)
ls = LineString([(-0.5, 1.2), (0.5, 0), (1, 1), (1.5, 0), (1.5, 0.5)])
rect = ls.minimum_rotated_rectangle

ax.plot(*ls.xy, color=DARKGRAY, linewidth=3, alpha=0.5, zorder=2)
patch = PolygonPatch(rect, facecolor=BLUE, edgecolor=BLUE, alpha=0.5, zorder=2)
ax.add_patch(patch)

set_limits(ax, -1, 2, -1, 2)

ax.set_title('b) LineString')

plt.show()
