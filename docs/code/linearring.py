from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing

COLOR = {
    True:  '#6699cc',
    False: '#ff3333'
    }

def v_color(ob):
    return COLOR[ob.is_valid]

fig = pyplot.figure(1, figsize=(7.5, 3), dpi=180)

# 1: valid ring
ax = fig.add_subplot(121)
ring = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 0.8), (0, 0)])

coords = list(ring.coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

x, y = ring.xy
ax.plot(x, y, color=v_color(ring), alpha=0.7, linewidth=3.0)

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
ring2 = LinearRing([(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)])

coords = list(ring2.coords)
x, y = zip(*coords)
ax.plot(x, y, 'o', color='#999999')

x, y = ring2.xy
ax.plot(x, y, color=v_color(ring2), alpha=0.7, linewidth=3.0)

ax.set_title('b) invalid')

xrange = [-1, 3]
yrange = [-1, 3]
ax.set_xlim(*xrange)
ax.set_xticks(range(*xrange) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(range(*yrange) + [yrange[-1]])
ax.set_aspect(1)

pyplot.show()

