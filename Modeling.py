import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Initializing
u_values = np.array([[0.5], [0.45]])
x_values = np.array([[np.pi / 2], [1], [4]])


# Question A
def diff_kinematics(x: np.ndarray, u: np.ndarray):
    r = 0.1
    d = 0.25
    stand = r / (2 * d)
    omega = (u[1, 0] - u[0, 0]) * stand
    vel = (u[0, 0] + u[1, 0]) * stand * d
    if u[1, 0] > 0.5 or u[0, 0] > 0.5:
        message = "Give smaller u_values values. Must be no more than 0.5!"
        return message
    else:
        x_dot = np.array([[omega],
                          [vel * np.cos(x[0, 0])],
                          [vel * np.sin(x[0, 0])]])

        return x_dot


# Question B
def visualize_transitions(pos: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25

    def rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    for i in range(0, len(pos), 40):
        position = pos[i]
        bpb = np.array([[-d, -d]]).T
        bpw = rot(position[0, 0]) @ bpb + position[1:, :]

        # Add the initial x and y coordinates to the position
        bpw[0, 0] += x_values[1, 0]
        bpw[1, 0] += x_values[2, 0]

        rect = Rectangle((float(bpw[0, 0]), float(bpw[1, 0])), 3 * d, 2 * d, edgecolor='black', fill=False,
                         angle=float(position[0, 0] * 180. / np.pi))
        # add rectangle to plot
        ax.add_patch(rect)

    plt.xlim(-1, 12)
    plt.ylim(-1, 12)
    plt.show()


def find_pos(u_val):
    dt = 0.1
    p = np.zeros((3, 1))
    all_poses = [p]
    for index in range(1200):
        v = diff_kinematics(p, u_val)
        p = p + v * dt

        all_poses += [p]

    return np.array(all_poses)


# visualize_transitions(find_pos(u_values))


# Question 3
def f_rk4_many_steps(x0, u0, steps):
    dt = 0.1
    states_of_algorithm = [x0]
    x = np.copy(x0)
    for _ in range(steps):
        f1 = diff_kinematics(x, u0)
        f2 = diff_kinematics(x + 0.5 * dt * f1, u0)
        f3 = diff_kinematics(x + 0.5 * dt * f2, u0)
        f4 = diff_kinematics(x + dt * f3, u0)
        x = x + dt / 6. * (f1 + 2. * f2 + 2. * f3 + f4)
        states_of_algorithm.append(np.copy(x))

    return np.array(states_of_algorithm)


def f_rk4_one_step(x0, u0):
    dt = 0.1

    f1 = diff_kinematics(x0, u0)
    f2 = diff_kinematics(x0 + 0.5 * dt * f1, u0)
    f3 = diff_kinematics(x0 + 0.5 * dt * f2, u0)
    f4 = diff_kinematics(x0 + dt * f3, u0)
    x_pos = x0 + dt / 6. * (f1 + 2. * f2 + 2. * f3 + f4)
    return x_pos


def visualize_transitions_with_rk4(x0, steps):
    states = f_rk4_many_steps(x_values, u_values, steps)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25

    def rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    for i in range(len(states)):
        position = states[i]
        bpb = np.array([[-d, -d], [d, -d], [d, d], [-d, d], [-d, -d]]).T
        bpw = rot(position[0, 0]) @ bpb + position[1:3, :]

        rect = plt.Polygon(bpw.T, edgecolor='black', fill=False)
        ax.add_patch(rect)

    plt.xlim(-1, 12)
    plt.ylim(-1, 12)
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    plt.grid(True)
    plt.show()


#visualize_transitions_with_rk4(x_values, 4100)
