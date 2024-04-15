import numpy as np
import matplotlib.pyplot as plt
import Modeling

x_init = np.array([[Modeling.x_values[1, 0]], [Modeling.x_values[2, 0]]])
x_goal = np.array([[3., 3.]]).T  # Goal position (2, 2)


# Let's make a helper for that
def state_to_tuple(xx):
    x = xx.reshape((-1, 1))
    return x[0, 0], x[1, 0]


# And for the inverse
def tuple_to_state(t):
    return np.array([[t[0], t[1]]]).T


def sample_state():
    return np.random.rand(2, 1) * 10. - 5.


# Now we need to find the nearest node
def nearest(x, tree_var):
    dist = np.inf
    near = None
    # Simplest checking! Scales badly in nodes O(n)
    for t in tree_var:
        s = tuple_to_state(t)
        d = np.linalg.norm(s - x)
        if d < dist:
            dist = d
            near = s
    return near


# Finally we need to connect two states
def connect(x_start, x_target):
    # Here we can check for collision free paths, max moving distance, and in general we should apply a local planner
    max_dist = 0.2
    return (x_target - x_start) / np.linalg.norm(x_target - x_start) * max_dist + x_start


def valid_state(x):
    if (np.abs(x) > 5.).any():
        return False
    return True


def RRT(x_start, x_goal, max_iters):
    tree = {state_to_tuple(x_start): []}
    best_path = []

    for i in range(max_iters):
        x_sample = sample_state()
        x_nearest = nearest(x_sample, tree)

        if x_nearest is None:
            continue

        x_new = connect(x_nearest, x_sample)

        if not valid_state(x_new):
            continue

        tree[state_to_tuple(x_nearest)].append(x_new)
        tree[state_to_tuple(x_new)] = []

        if np.linalg.norm(x_goal - x_new) < 0.1:
            # If the goal is reached, backtrack to build the path
            best_path = [x_goal]
            parent = x_new
            while np.linalg.norm(parent - x_start) > 0.1:
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
    for node in path:
        print(node)
else:
    print("No path found within the maximum iterations.")

fig = plt.figure()
ax = fig.add_subplot(111)

for s in tree:
    ax.plot(s[0], s[1], '.', zorder=2)

    for c in tree[s]:
        ax.plot([s[0], c[0, 0]], [s[1], c[1, 0]], zorder=1)

plt.ylim(-5., 5.)
plt.xlim(-5., 5.)
plt.title('RRT Path Planning')
plt.plot(x_goal[0, 0], x_goal[1, 0], 'kx', markersize=12)
plt.grid(True)
plt.show()
