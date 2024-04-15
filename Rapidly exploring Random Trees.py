import numpy as np
import matplotlib.pyplot as plt
import Modeling

x_init = Modeling.x_values
x_goal = np.array([[0., 3., 3.]]).T


# Let's make a helper for that
def state_to_tuple(xx):
    x = xx.reshape((-1, 1))
    return tuple(x[i, 0] for i in range(x.shape[0]))


def tuple_to_state(t):
    return np.array([[t[i]] for i in range(len(t))])


def sample_state():
    return np.random.rand(3, 1) * 10. - 5.


# Now we need to find the nearest node
# FIX THIS FOR 3X1 x
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


def cost_function(trajectory, x_target):
    return np.linalg.norm(trajectory[-1] - x_target)


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

    return best_trajectory



def valid_state(x):
    if (np.abs(x) > 5.).any():
        return False
    return True


def RRT(x_start, goal, max_iters):
    tree = {state_to_tuple(x_start): []}
    best_path = []

    for i in range(max_iters):
        x_sample = sample_state()

        x_nearest = nearest(x_sample, tree)

        # Both are 2x1 with (x,y) values
        if x_nearest is None:
            continue

        x_new = optimal_trajectory(x_nearest, x_sample, 1000)

        if not valid_state(x_new):
            continue

        tree[state_to_tuple(x_nearest)].append(x_new)
        tree[state_to_tuple(x_new)] = []

        if np.linalg.norm(goal[1:] - x_new[1:]) < 0.1:
            # If the goal is reached, backtrack to build the path
            best_path = [goal]
            parent = x_new

            while np.linalg.norm(parent - x_start[1:]) > 0.1:
                for node, children in tree.items():
                    if parent.tolist() in [child.tolist() for child in children]:
                        best_path.append(tuple_to_state(node))
                        parent = tuple_to_state(node)
                        break
            best_path.append(x_start)
            best_path.reverse()
            return True, tree, best_path

    return False, tree, best_path


valid, tree, path = RRT(x_init, x_goal, 1000)

print(valid, len(tree), len(path))

if valid:
    print("Path found!")
else:
    print("No path found within the maximum iterations.")

fig = plt.figure()
ax = fig.add_subplot(111)

# Plot nodes
for state_tuple in tree:
    state = np.array(state_tuple).reshape(-1, 1)
    ax.plot(state[0], state[1], 'ro', zorder=2)

    # Plot connections
    for child_state in tree[state_tuple]:
        ax.scatter(state[0], state[1], color='black', zorder=1)

plt.plot(x_init[0], x_init[1], 'ro', label='Start')
plt.plot(x_goal[0], x_goal[1], 'go', label='Goal')

plt.ylim(-5., 5.)
plt.xlim(-5., 5.)

plt.legend()
plt.show()

