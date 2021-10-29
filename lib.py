import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets


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


def update_network(i, G, states_m, ax, vbound, pos):
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
