from matplotlib import pyplot
from shapely.geometry import LineString

fig = pyplot.figure(1, figsize=(7.5, 3), dpi=180)

# 1: simple line
ax = fig.add_subplot(121)
line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])

coords = list(line.coords)
x, y = zip(*coords[1:-1])
ax.plot(x, y, 'o', color='#999999')

x, y = zip(*[coords[0], coords[-1]])
ax.plot(x, y, 'o', color='black')

x, y = line.xy
ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3.0)

ax.set_title('a) simple')

xrange = [-1, 4]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

#2: complex line
ax = fig.add_subplot(122)
line2 = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (-1, 1), (1, 0)])

coords = list(line2.coords)
x, y = zip(*coords[1:-1])
ax.plot(x, y, 'o', color='#999999')

x, y = zip(*[coords[0], coords[-1]])
ax.plot(x, y, 'o', color='#000000')

x, y = line2.xy
ax.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3.0)

ax.set_title('b) complex')

xrange = [-2, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()

