import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
import scipy.linalg as spla


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


def update_network(i, G, states_m, ax, vbound, pos, labels=True):
    """
    Function to plot and visualize the state values in a graph
    Requirements:
        - Graph label of nodes must go from 0...n
        - states is a timesteps*n np.array

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
    if labels:
        # From here: Add actual state value as label
        pos_higher = {}
        x_off = 0.00  # offset on the x axis
        y_off = 0.03  # offset on the y axis
        for k, v in pos.items():
            pos_higher[k] = (v[0] + x_off, v[1] + y_off)
        labels = ["%.4g" % num for num in states_m[i]]  # Casting node value to string
        labels = dict(zip(list(range(0, len(labels))), labels))  # Matching node value to node number in a dict
        nx.draw_networkx_labels(G, pos_higher, labels, ax=ax)
        # If there is a failure due to key error, it is required for the labels to be numbered from 0 .. n!!


def simulate_network(A, x_0, t):
    """
    Function to simulate the network and save the state
    """
    states = np.zeros((t,A.shape[0]))
    states[0,:]=x_0
    for i in range(1,t):
        states[i,:] = A @ states[i-1,:]
    return states


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


# Putting some simulations parts into custom functions
def interactive_network_plot(G, states, pos, t, fig, ax):
    fig, v_bound, pos = init_network_sim_plot(G, states, fig, pos=pos)

    def inter(timestep):
        update_network(timestep['new'], G=G, states_m=states, ax=ax, vbound=v_bound, pos=pos)
        return None

    # Plot initial configuration
    update_network(0, G=G, states_m=states, ax=ax, vbound=v_bound, pos=pos)

    widget = create_widgets_play_slider(fnc=inter, minv=0, maxv=t-1, step=1, play_speed=1000)

    return widget


# Linear Algebra related functions

def matprint(mat, fmt="g"):
    """
    Own defined function to beautiful print matrices in the output.
    """
    # Handling 1d-arrays
    if np.ndim(mat) == 1:
        mat = mat.reshape(mat.shape[0], 1)
    # Handling column spaces
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]

    for x in mat:
        for i, y in enumerate(x):
            if i == 0:
                print(("|  {:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
            else:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("|")


def plot_spectrum(M, ax):
    """
    Scatter plot of eigs in complex plane overlayed by radius circle
    """
    eigvals = spla.eig(M)[0]  # Extract only eigenvalues

    ax.scatter(eigvals.real, eigvals.imag)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    r = max(abs(eigvals))  # Note: abs() automatically calculated the magnitude of complex values
    circle = mpl.patches.Circle((0, 0), radius=r, alpha=.2, ec='black')
    ax.add_patch(circle)
    ax.axhline(0, color='black', ls='--', linewidth=1)
    ax.axvline(0, color='black', ls='--', linewidth=1)
    return eigvals, ax


def plot_gersgorin_disks(M, ax):
    """
    Scatter plot of eigenvalues overlayed with gersgorin disks (marked with green edges)
    """
    eigvals, ax = plot_spectrum(M, ax)

    row_sums = M.sum(axis=1)
    for i in range(M.shape[0]):
        ax.add_patch(mpl.patches.Circle((M[i, i], 0), radius=row_sums[i] - M[i, i],
                                        alpha=.6 / M.shape[0], ec='green'))