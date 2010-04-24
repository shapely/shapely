from matplotlib import pyplot
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

fig = pyplot.figure(1, figsize=(7.5, 3), dpi=180)

# 1: valid polygon
ax = fig.add_subplot(121)

ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
polygon = Polygon(ext, [int])

coords = list(polygon.interiors[0].coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

coords = list(polygon.exterior.coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

patch = PolygonPatch(polygon, facecolor=v_color(polygon), edgecolor=v_color(polygon), alpha=0.5)
ax.add_patch(patch)

ax.set_title('a) valid')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

#2: invalid self-touching ring
ax = fig.add_subplot(122)
ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
int = [(1, 0), (0, 1), (0.5, 1.5), (1.5, 0.5), (1, 0)][::-1]
polygon = Polygon(ext, [int])

coords = list(polygon.interiors[0].coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

coords = list(polygon.exterior.coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

patch = PolygonPatch(polygon, facecolor=v_color(polygon), edgecolor=v_color(polygon), alpha=0.5)
ax.add_patch(patch)

ax.set_title('b) invalid')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()

