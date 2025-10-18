import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, LineString

def partition_polygon(polygon):
    # Dummy implementation for demonstration
    return [
        LineString([(2, 0), (2, 4)]),
        LineString([(6, 0), (6, 4)]),
        LineString([(0, 4), (8, 4)]),
        LineString([(8, 4), (8, 6)]),
    ]

if __name__ == "__main__":
    polygon1 = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])
    partition_result_1 = partition_polygon(polygon1)

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Create a Polygon patch and add it to the plot
    polygon_patch = MplPolygon(list(polygon1.exterior.coords), closed=True, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(polygon_patch)

    # Plot the LineString objects in a different color
    for line in partition_result_1:
        x, y = line.xy
        ax.plot(x, y, color='red')

    # Set the limits of the plot
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Show the plot
    plt.show()
