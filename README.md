# University exercise in Robotics

def cost_function(node,x_target):
    dist = distance_function(node,x_target)
    angle = angle_difference(node,x_target)
    cost = 5 * dist + angle
    return cost


def optimal_trajectory(x_start, x_target, k_samples):
    min_ul, max_ul = -0.5, 0.5
    min_ur, max_ur = -0.5, 0.5

    all_trajectories = []
    best_trajectory = None
    best_cost = np.inf

    for _ in range(k_samples):
        ul = np.random.uniform(min_ul, max_ul)
        ur = np.random.uniform(min_ur, max_ur)

        u_val = np.array([[ul], [ur]])
        # Forward simulation
        trajectory = Modeling.f_rk4_one_step(x_start, u_val)

        cost = cost_function(trajectory,x_target)

        if cost < best_cost:
            best_cost = cost
            best_trajectory = trajectory

        if np.linalg.norm(trajectory[1:] - x_target[1:]) < 0.1:
            return trajectory

    return best_trajectory
