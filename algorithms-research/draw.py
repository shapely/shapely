import matplotlib.pyplot as plt

def draw_polygon(points):
    # Extract x and y coordinates from the list of tuples
    x_coords, y_coords = zip(*points)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the edges between vertices
    for i in range(len(points)):
        x1, y1 = points[i]
        if i == len(points) - 1:
            x2, y2 = points[0]  # Connect the last point to the first point
        else:
            x2, y2 = points[i + 1]
        ax.plot([x1, x2], [y1, y2], 'b-')

    # Set aspect ratio and axis limits
    ax.set_aspect('equal')
    ax.autoscale()

    # Show the plot
    plt.show()

# Example usage
points = [(2, 0), (6, 0), (4, 6), (4, 8), (6, 8), (0, 6), (0, 4), (4, 2)]
draw_polygon(points)