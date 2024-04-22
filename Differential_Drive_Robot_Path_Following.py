import numpy as np
import matplotlib.pyplot as plt
import Rapidly_exploring_Random_Trees


def plot_path_without_obstacles(xy_begin, xy_finish, iterations, number_of_samples):
    valid, path_to_go = Rapidly_exploring_Random_Trees.RRT(xy_begin, xy_finish, iterations, number_of_samples)

    for i in range(len(path_to_go) - 1):
        point = path_to_go[i]
        next_point = path_to_go[i + 1]
        plt.plot([point[1], next_point[1]], [point[2], next_point[2]], color='b')

        rect_outer = plt.Rectangle((point[1] - 0.1, point[2] - 0.1), 0.2, 0.2, color='b', fill=False)
        rect_inner = plt.Rectangle((point[1] - 0.05, point[2] - 0.05), 0.1, 0.1, color='b', fill=True)
        plt.gca().add_patch(rect_outer)
        plt.gca().add_patch(rect_inner)

    plt.plot(xy_begin[1], xy_begin[2], 'ro', label='Start')
    plt.plot(xy_finish[1], xy_finish[2], 'go', label='Goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path without Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_path_with_obstacles(xy_begin, xy_finish, iterations, number_of_samples, obst):
    valid, path_to_go = Rapidly_exploring_Random_Trees.RRT_with_obstacles(xy_begin, xy_finish, iterations,
                                                                          number_of_samples, obst)

    # Plot the path
    for i in range(len(path_to_go) - 1):
        point = path_to_go[i]
        next_point = path_to_go[i + 1]
        plt.plot([point[1], next_point[1]], [point[2], next_point[2]], color='b')

        rect_outer = plt.Rectangle((point[1] - 0.1, point[2] - 0.1), 0.2, 0.2, color='b', fill=False)
        rect_inner = plt.Rectangle((point[1] - 0.05, point[2] - 0.05), 0.1, 0.1, color='b', fill=True)
        plt.gca().add_patch(rect_outer)
        plt.gca().add_patch(rect_inner)

    # Plot obstacles
    for obstacle in obst:
        radius, x_center, y_center = obstacle
        circle = plt.Circle((x_center, y_center), radius, color='r', fill=False)
        plt.gca().add_artist(circle)

    plt.plot(xy_begin[1], xy_begin[2], 'ro', label='Start')
    plt.plot(xy_finish[1], xy_finish[2], 'go', label='Goal')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path with Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Initializing
x_begin = np.array([[np.pi / 2], [1], [4]])
x_finish = np.array([[-np.pi / 2], [6], [8]])
obstacles = [(0.2, 1.7, 6), (0.2, 2, 12)]

# plot_path_without_obstacles(x_begin, x_finish, 2000, 500)
plot_path_with_obstacles(x_begin, x_finish, 3000, 300, obstacles)
