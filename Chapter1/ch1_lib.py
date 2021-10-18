import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets

def init_network_sim_plot(G, states, fig, vbound=None, pos=None):
    """
    Function to initialize Network:
        - Setting up the figure with colorbar
        - Determine position of Nodes and boundaries for values

    Parameters
    ----------
    G : nx.graph
        Graph that will be displayed
    states: np.array
        Values of nodes through the simulation
    ax: plt.axes
        Previously defined axis handle
    vbound:
        Value boundaries for the coloring
    pos : np.array
        Position of nodes if already provided for plot

    Returns
    -------
    fig, ax
        figure and axes where the graph can be plotted on
    vbound
        Boundary values that should be used from now on
    pos
        position of the node for the graph plot
    """

    if pos is None:
        pos = nx.drawing.layout.spring_layout(G)
    if vbound is None:
        vmax_s = np.amax(states)
        vmin_s = np.amin(states)
    else:
        vmax_s = vbound[1]
        vmin_s = vbound[0]
    norm = mpl.colors.Normalize(vmin=vmin_s, vmax=vmax_s)
    cmap = plt.get_cmap('viridis')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm)
    return fig, [vmin_s, vmax_s], pos


def update_network(i, G, states_m, ax, vbound, pos):
    """
    Function to plot and visualize the state values in a graph

    Parameters
    ----------
    i: int
        timestep
    G : nx.graph
        Graph that will be displayed
    states: np.array
        Values of nodes through the simulation
    ax: plt.axes
        Previously defined axis handle
    vbound:
        Value boundaries for the coloring
    pos : np.array
        Position of nodes if already provided for plot

    Returns
    -------
    None
    """
    ax.clear()  # Clear axes
    colors = states_m[i]
    cmap = plt.get_cmap('viridis')
    # draw with new color map
    nx.draw_networkx(G, pos, node_size=200, cmap=cmap,
                     node_color=colors, vmin=vbound[0], vmax=vbound[1], ax=ax)

    # From here: Add actual state value as label
    pos_higher = {}
    x_off = 0.00  # offset on the x axis
    y_off = 0.03  # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0] + x_off, v[1] + y_off)
    labels = ["%.4g" % num for num in states_m[i]]  # Casting node value to string
    labels = dict(zip(list(range(1, len(labels) + 1)), labels))  # Matching node value to node number in a dict
    nx.draw_networkx_labels(G, pos_higher, labels, ax=ax)

def simulate_network(A, x_0, t):
    """
    Function to simulate the network and save the state
    """
    states = np.zeros((t,A.shape[0]))
    states[0,:]=x_0
    for i in range(1,t):
        states[i,:] = A @ states[i-1,:]
    return states

def plot_node_val_2D(states, x_0, t, ax, legend=True):
    """
    Function to plot the states on a 2d-axis
    """
    x_axis = np.arange(t)
    for i in range(states.shape[1]):
        ax.plot(x_axis, states[:,i], label=str(i+1))
    average = np.ones(t) * np.sum(x_0)/states.shape[1]
    ax.plot(x_axis, average, '--', label='average')
    if legend:
        ax.legend()


def create_widgets_play_slider(fnc, minv, maxv, step=1, play_speed=200):
    """
    Create an integer slider and a playbutton for interactive plots
    Parameters
    ----------
    fnc: function
        function where the widget will be applied on
    minv : int
        minimum value for the slider
    maxv: int
        maximum value for the slider
    step: int
        step for the slider
    play_speed: int
        Speed for incrementing the integer for the play button in ms

    Returns
    -------
    widgets
    """
    slider = widgets.IntSlider(min=minv, max=maxv, step=step, continuous_update=True)
    play = widgets.Play(min=minv,max=maxv, step=step, interval=play_speed)

    slider.observe(fnc, 'value')
    widgets.jslink((play, 'value'), (slider, 'value'))
    widgets.VBox([play, slider])
    return widgets.VBox([play, slider])

# Bug example specific

def plot_circle_and_bugs(ax, radius, bugs):
    # Set xlim, ylim and aspect ratio
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Add a circle
    circle = mpl.patches.Circle((0, 0), radius=radius, fill=False)
    ax.add_patch(circle)

    for i in range(0, bugs.shape[0]):
        # Add a bug
        bug = mpl.patches.Circle((bugs[i, 1], bugs[i, 2]), radius=radius * 0.08)
        ax.add_patch(bug)


def init_bugs(radius, n_bugs, n_steps):
    np.random.seed()

    # For each bug store its theta position and the resulting (x, y) position on the plane
    bugs = np.zeros((n_bugs, 3))

    # Each state contains the position of all bugs
    states_bugs = np.zeros((n_steps, n_bugs, 3))

    # Set initial positions and state
    for i in range(0, n_bugs):
        theta_i = (i * np.pi/6 + np.random.uniform(-np.pi/16, np.pi/16)) % (2*np.pi)
        bugs[i] = [theta_i, radius * np.cos(theta_i), radius * np.sin(theta_i)]

    # Save the first state
    states_bugs[0] = bugs
    return bugs, states_bugs


def simulate_bugs_cyclic_pursuit(radius, n_bugs, n_steps, gain, bugs, states_bugs):
    for k in range(1, n_steps):

        # Update bugs's positions
        for i in range(0, n_bugs):
            dist_cc = (bugs[(i+1) % n_bugs,0] - bugs[i,0]) % (2*np.pi)
            u = gain * dist_cc
            theta_new = (bugs[i,0] + u) % (2*np.pi)
            bugs[i] = [theta_new, radius * np.cos(theta_new), radius * np.sin(theta_new)]

        states_bugs[k] = bugs
    return states_bugs


def simulate_bugs_cyclic_balancing(radius, n_bugs, n_steps, gain, bugs, states_bugs):
    for k in range(1, n_steps):

        # Update bugs's positions
        for i in range(0, n_bugs):
            dist_cc = (bugs[(i+1) % n_bugs,0] - bugs[i,0]) % (2*np.pi)
            dist_c = (bugs[i,0] - bugs[(i-1) % n_bugs,0]) % (2*np.pi)
            u = gain * dist_cc - gain*dist_c
            theta_new = (bugs[i,0] + u) % (2*np.pi)
            bugs[i] = [theta_new, radius * np.cos(theta_new), radius * np.sin(theta_new)]

        states_bugs[k] = bugs
    return states_bugs
