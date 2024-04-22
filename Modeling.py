import numpy as np
import matplotlib.pyplot as plt

# Initializing
u_values = np.array([[0.42], [0.43]])
x_values = np.array([[np.pi / 12], [1], [7]])


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

def visualize_transitions_with_euler(u_val, xy_val, steps):
    pos = find_pos(u_val, xy_val, steps)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25

    def rot(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    for i in range(len(pos)):
        position = pos[i]
        bpb = np.array([[-d, -d], [d, -d], [d, d], [-d, d], [-d, -d]]).T
        bpw = rot(position[0, 0]) @ bpb + position[1:3, :]

        rect = plt.Polygon(bpw.T, edgecolor='black', fill=False)
        ax.add_patch(rect)

    plt.xlim(-1, 12)
    plt.ylim(-1, 12)
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
    plt.title('Euler Method Transitions')
    plt.grid(True)
    plt.show()


def find_pos(u_val, x_val, steps):
    dt = 0.1
    p = x_val
    all_poses = [p]
    for index in range(steps):
        v = diff_kinematics(p, u_val)
        p = p + v * dt
        all_poses += [p]
    return np.array(all_poses)


# Question 3
def f_rk4_one_step(x0, u0):
    dt = 0.1
    f1 = diff_kinematics(x0, u0)
    f2 = diff_kinematics(x0 + 0.5 * dt * f1, u0)
    f3 = diff_kinematics(x0 + 0.5 * dt * f2, u0)
    f4 = diff_kinematics(x0 + dt * f3, u0)
    x_pos = x0 + dt * (f1 + 2. * f2 + 2. * f3 + f4) / 6.
    return x_pos


def visualize_transitions_with_rk4(u_val, xy_val, steps):
    states = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = 0.25
    position = xy_val
    for i in range(steps):
        pos = f_rk4_one_step(position, u_val)
        states.append(pos)
        position = pos
    states = np.array(states)

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
    plt.title('RK4 Method Transitions')
    plt.grid(True)
    plt.show()

# visualize_transitions_with_euler(u_values, x_values, 2000)

# visualize_transitions_with_rk4(u_values, x_values, 2000)
