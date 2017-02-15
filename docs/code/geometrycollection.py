from matplotlib import pyplot
from shapely.geometry import LineString
from figures import BLUE, GRAY, YELLOW, GREEN, SIZE, set_limits, plot_coords

fig = pyplot.figure(1, figsize=SIZE, dpi=90) #1, figsize=(10, 4), dpi=180)

a = LineString([(0, 0), (1, 1), (1,2), (2,2)])
b = LineString([(0, 0), (1, 1), (2,1), (2,2)])

# 1: disconnected multilinestring
ax = fig.add_subplot(121)

plot_coords(ax, a)
plot_coords(ax, b)

x, y = a.xy
ax.plot(x, y, color=YELLOW, alpha=0.5, linewidth=3, solid_capstyle='round', zorder=2)

x, y = b.xy
ax.plot(x, y, color=GREEN, alpha=0.5, linewidth=3, solid_capstyle='round', zorder=2)

ax.set_title('a) lines')

set_limits(ax, -1, 3, -1, 3)

#2: invalid self-touching ring
ax = fig.add_subplot(122)

x, y = a.xy
ax.plot(x, y, color=GRAY, alpha=0.7, linewidth=1, solid_capstyle='round', zorder=1)
x, y = b.xy
ax.plot(x, y, color=GRAY, alpha=0.7, linewidth=1, solid_capstyle='round', zorder=1)

for ob in a.intersection(b):
    x, y = ob.xy
    if len(x) == 1:
        ax.plot(x, y, 'o', color=BLUE, zorder=2)
    else:
        ax.plot(x, y, color=BLUE, alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

ax.set_title('b) collection')

set_limits(ax, -1, 3, -1, 3)

pyplot.show()

