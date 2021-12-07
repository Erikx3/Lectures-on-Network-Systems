import numpy as np
import matplotlib as mpl

# Bug example specific
def plot_circle_and_bugs(ax, radius, bugs):
    # Set xlim, ylim and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Add a circle
    circle = mpl.patches.Circle((0, 0), radius=radius, fill=False)
    ax.add_patch(circle)
    np.random.seed(4)
    for i in range(0, bugs.shape[0]):
        # Add a bug
        bug = mpl.patches.Circle((bugs[i, 1], bugs[i, 2]), radius=radius * 0.08, color=np.random.rand(3,))
        ax.add_patch(bug)


def init_bugs(radius, n_bugs, n_steps):
    np.random.seed()

    # For each bug store its theta position and the resulting (x, y) position on the plane
    bugs = np.zeros((n_bugs, 3))

    # Each state contains the position of all bugs
    states_bugs = np.zeros((n_steps, n_bugs, 3))

    # Set initial positions and state
    for i in range(0, n_bugs):
        theta_i = (i * np.pi/n_bugs + np.random.uniform(-np.pi/(n_bugs*3), np.pi/(n_bugs*3))) % (2*np.pi)
        bugs[i] = [theta_i, radius * np.cos(theta_i), radius * np.sin(theta_i)]

    # Save the first state
    states_bugs[0] = bugs
    return bugs, states_bugs


def simulate_bugs_cyclic_pursuit(radius, n_bugs, n_steps, gain, bugs, states_bugs):
    for k in range(1, n_steps):

        old_bugs = bugs.copy()
        # Update bugs's positions
        for i in range(0, n_bugs):
            dist_cc = (old_bugs[(i+1) % n_bugs,0] - old_bugs[i,0]) % (2*np.pi)
            u = gain * dist_cc
            theta_new = (old_bugs[i,0] + u) % (2*np.pi)
            bugs[i] = [theta_new, radius * np.cos(theta_new), radius * np.sin(theta_new)]

        states_bugs[k] = bugs
    return states_bugs


def simulate_bugs_cyclic_balancing(radius, n_bugs, n_steps, gain, bugs, states_bugs):
    for k in range(1, n_steps):
        old_bugs = bugs.copy()
        # Update bugs's positions
        for i in range(0, n_bugs):
            dist_cc = (old_bugs[(i+1) % n_bugs,0] - old_bugs[i,0]) % (2*np.pi)
            dist_c = (old_bugs[i,0] - old_bugs[(i-1) % n_bugs,0]) % (2*np.pi)
            u = gain * dist_cc - gain*dist_c
            theta_new = (old_bugs[i,0] + u) % (2*np.pi)
            bugs[i] = [theta_new, radius * np.cos(theta_new), radius * np.sin(theta_new)]

        states_bugs[k] = bugs
    return states_bugs
