import numpy as np
import Modeling


def state_to_tuple(xx):
    x = xx.reshape((-1, 1))
    return tuple(x[i, 0] for i in range(x.shape[0]))


def tuple_to_state(t):
    return np.array([[t[i]] for i in range(len(t))])


def distance_function(node, x_target):
    distance = np.linalg.norm(node[1:] - x_target[1:])
    return distance


def angle_difference(node, x_target):
    angle_diff = np.abs(np.arctan2(x_target[2] - node[2], x_target[1] - node[1]) - node[0])

    return angle_diff


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

        distance = distance_function(trajectory, x_target)
        all_trajectories.append((trajectory, distance))

        if distance < best_cost:
            best_cost = distance
            best_trajectory = trajectory

        if np.linalg.norm(trajectory[1:] - x_target[1:]) < 0.1:
            return trajectory
    all_trajectories.sort(key=lambda x: x[1])

    smallest_angle_diff = np.inf
    count = 1
    for elements in all_trajectories:
        angle = angle_difference(elements[0], x_target)
        if angle < smallest_angle_diff:
            best_trajectory = elements[0]
        count += 1
        if count == 3:
            break

    return best_trajectory


def optimal_trajectory_with_obstacles(x_start, x_target, k_samples, obs_positions=None):
    min_ul, max_ul = -0.5, 0.5
    min_ur, max_ur = -0.5, 0.5

    best_trajectory = None
    best_cost = np.inf

    for _ in range(k_samples):
        ul = np.random.uniform(min_ul, max_ul)
        ur = np.random.uniform(min_ur, max_ur)

        u_val = np.array([[ul], [ur]])

        trajectory = Modeling.f_rk4_one_step(x_start, u_val)
        cost = distance_function(trajectory, x_target)

        avoid_obstacle = True
        if obs_positions:
            for obs in obs_positions:
                distance = (trajectory[1] - obs[1]) ** 2 + (trajectory[2] - obs[2]) ** 2
                if distance < 0.2 + obs[0] ** 2:
                    avoid_obstacle = False
                    angle_to_obstacle = np.arctan2(trajectory[2] - obs[2], trajectory[1] - obs[1])
                    angle_to_target = np.arctan2(x_target[2] - trajectory[2], x_target[1] - trajectory[1])
                    diff_angle = angle_to_target - angle_to_obstacle
                    ul += np.sin(diff_angle) * 0.1
                    ur -= np.sin(diff_angle) * 0.1

        if avoid_obstacle and cost < best_cost:
            best_cost = cost
            best_trajectory = trajectory

        if np.linalg.norm(trajectory[1:] - x_target[1:]) < 0.1:
            return trajectory

    return best_trajectory


def RRT(xy_start, xy_goal, max_iters, number_of_samples):
    tree = {state_to_tuple(xy_start): []}
    path_to_go = [state_to_tuple(xy_start)]
    nodes = xy_start

    for i in range(max_iters):
        x_new = optimal_trajectory(nodes, xy_goal, number_of_samples)

        tree[state_to_tuple(xy_start)] = []
        nodes = x_new
        path_to_go.append(state_to_tuple(x_new))
        if np.linalg.norm(xy_goal[1:] - x_new[1:]) < 0.1:
            path_to_go.append(state_to_tuple(xy_goal))

            return True, path_to_go
    return False, path_to_go


def RRT_with_obstacles(xy_start, xy_goal, max_iters, number_of_samples, obs_positions=None):
    tree = {state_to_tuple(xy_start): []}
    path_to_go = [state_to_tuple(xy_start)]
    nodes = xy_start

    for i in range(max_iters):
        x_new = optimal_trajectory_with_obstacles(nodes, xy_goal, number_of_samples, obs_positions)

        tree[state_to_tuple(xy_start)] = []
        nodes = x_new
        path_to_go.append(state_to_tuple(x_new))
        if np.linalg.norm(xy_goal[1:] - x_new[1:]) < 0.1:
            path_to_go.append(state_to_tuple(xy_goal))

            return True, path_to_go

    return False, path_to_go
