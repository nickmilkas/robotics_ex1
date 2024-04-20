import numpy as np
import matplotlib.pyplot as plt
import Modeling

# Initial position is (1,4)
x_init = Modeling.x_values
x_goal = np.array([[0., 7., 6.]]).T


def state_to_tuple(xx):
    x = xx.reshape((-1, 1))
    return tuple(x[i, 0] for i in range(x.shape[0]))


def tuple_to_state(t):
    return np.array([[t[i]] for i in range(len(t))])


def nearest(x, tree_var):
    dist = np.inf
    near = None

    for t in tree_var:
        s = tuple_to_state(t)

        d = np.linalg.norm(s[1:] - x[1:])
        if d < dist:
            dist = d
            near = s
    return near


def cost_function(node, x_target):
    distance = np.linalg.norm(node[1:] - x_target[1:])

    return distance


def optimal_trajectory(x_start, x_target, k_samples):
    min_ul, max_ul = -0.5, 0.5
    min_ur, max_ur = -0.5, 0.5

    best_trajectory = None
    best_cost = np.inf
    for _ in range(k_samples):
        ul = np.random.uniform(min_ul, max_ul)
        ur = np.random.uniform(min_ur, max_ur)

        u_val = np.array([[ul], [ur]])
        # Forward simulation
        trajectory = Modeling.f_rk4_one_step(x_start, u_val)

        # Evaluate trajectory
        cost = cost_function(trajectory, x_target)

        if cost < best_cost:
            best_cost = cost
            best_trajectory = trajectory

        if np.linalg.norm(trajectory[1:] - x_target[1:]) < 0.1:
            return trajectory

    return best_trajectory


optimal_trajectory(x_init, x_goal, 5)


def RRT(x_start, x_goal, max_iters):
    tree = {state_to_tuple(x_init): []}
    best_path = []
    path_to_go = [state_to_tuple(x_start)]
    nodes = x_start

    for i in range(max_iters):
        x_new = optimal_trajectory(nodes, x_goal, 500)

        tree[state_to_tuple(x_start)] = []
        nodes = x_new
        path_to_go.append(state_to_tuple(x_new))
        if np.linalg.norm(x_goal[1:] - x_new[1:]) < 0.1:

            path_to_go.append(state_to_tuple(x_goal))

            return True, tree, path_to_go
    return False, tree, path_to_go


valid, tree, path_to_go = RRT(x_init, x_goal, 3000)


for point in path_to_go:
    plt.plot(point[1], point[2], marker='o', color='b')

plt.plot(x_init[1], x_init[2], 'ro', label='Start')
plt.plot(x_goal[1], x_goal[2], 'go', label='Goal')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Path to go')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

