from matplotlib import pyplot
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

from descartes.patch import PolygonPatch

fig = pyplot.figure(1, figsize=(7.5, 3), dpi=180)
fig.set_frameon(True)

# 1
ax = fig.add_subplot(121)

a = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

dilated = a.buffer(0.5)
patch1 = PolygonPatch(dilated, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch1)

x, y = a.xy
ax.plot(x, y, color='#999999')

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_axis_off()

#2
ax = fig.add_subplot(122)

patch2a = PolygonPatch(dilated, facecolor='#cccccc', edgecolor='#999999')
ax.add_patch(patch2a)

eroded = dilated.buffer(-0.3)

patch2b = PolygonPatch(eroded, facecolor='#99ccff', edgecolor='#6699cc')
ax.add_patch(patch2b)

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_axis_off()

fig.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.1)
fig.savefig('buffering.png')


