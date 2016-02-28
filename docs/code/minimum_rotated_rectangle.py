from shapely.geometry import MultiPoint, Polygon, LineString
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch

from figures import SIZE

fig = plt.figure(1, figsize=SIZE, dpi=90)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

mp = MultiPoint([(0, 0), (0.5, 1.5), (1, 0.5), (0.5, 0.5)])
rect = mp.minimum_rotated_rectangle

for p in mp:
	ax.plot(p.x, p.y, 'o', color='#999999')
patch = PolygonPatch(rect, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
ax.add_patch(patch)
ax.set_title('a) MultiPoint')

xr = [-1, 2]
yr = [-1, 2]
ax.set_xlim(*xr)
ax.set_xticks(range(*xr) + [xr[-1]])
ax.set_ylim(*yr)
ax.set_yticks(range(*yr) + [yr[-1]])
ax.set_aspect(1)

# 2
ax = fig.add_subplot(122)
ls = LineString([(-0.5, 1.2), (0.5, 0), (1, 1), (1.5, 0), (1.5, 0.5)])
rect = ls.minimum_rotated_rectangle

ax.plot(*ls.xy, color='#333333', linewidth=3, alpha=0.5, zorder=2)
patch = PolygonPatch(rect, facecolor='#6699cc', edgecolor='#6699cc', alpha=0.5, zorder=2)
ax.add_patch(patch)

xr = [-1, 2]
yr = [-1, 2]
ax.set_xlim(*xr)
ax.set_xticks(range(*xr) + [xr[-1]])
ax.set_ylim(*yr)
ax.set_yticks(range(*yr) + [yr[-1]])
ax.set_aspect(1)

ax.set_title('b) LineString')

plt.show()
