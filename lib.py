import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
import scipy.linalg as spla
from matplotlib.colors import ListedColormap
from math import gcd
from functools import reduce


def is_periodic(G):
    """
    https://stackoverflow.com/questions/54030163/periodic-and-aperiodic-directed-graphs
    Own function to test, whether a given Graph is aperiodic:
    """
    if not nx.is_strongly_connected(G):
        print("G is not strongly connected, periodicity not defined.")
        return False
    cycles = list(nx.algorithms.cycles.simple_cycles(G))
    cycles_sizes = [len(c) for c in cycles]  # Find all cycle sizes
    cycles_gcd = reduce(gcd, cycles_sizes)  # Find greatest common divisor of all cycle sizes
    is_periodic = cycles_gcd > 1
    return is_periodic


def plot_node_val_2D(states, x_0, t, ax, legend=True, avg=True):
    """
    Function to plot the states on a 2d-axis
    """
    x_axis = np.arange(t)
    for i in range(states.shape[1]):
        ax.plot(x_axis, states[:,i], label=str(i))
    if avg:
        average = np.ones(t) * np.sum(x_0)/states.shape[1]
        ax.plot(x_axis, average, '--', label='mean(x_0)')
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
        try:
            nx.draw_networkx_labels(G, pos_higher, labels, ax=ax)
        except KeyError as e:
            raise Exception('Make sure ur Nodes are labeled from 0 ... n-1') from e
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


def plot_spectrum(M, ax, patch=True):
    """
    Scatter plot of eigs in complex plane overlayed by radius circle
    """
    eigvals = spla.eig(M)[0]  # Extract only eigenvalues

    ax.scatter(eigvals.real, eigvals.imag)
    if max(abs(eigvals)) < 1:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

    r = max(abs(eigvals))  # Note: abs() automatically calculated the magnitude of complex values
    circle = mpl.patches.Circle((0, 0), radius=r, alpha=.2, ec='black')
    if patch:
        ax.add_patch(circle)
    ax.axhline(0, color='black', ls='--', linewidth=1)
    ax.axvline(0, color='black', ls='--', linewidth=1)
    ax.axis('equal')
    return eigvals, ax


def plot_gersgorin_disks(M, ax, patch_spectrum=True):
    """
    Scatter plot of eigenvalues overlayed with gersgorin disks (marked with green edges)
    """
    eigvals, ax = plot_spectrum(M, ax, patch_spectrum)

    row_sums = M.sum(axis=1)
    for i in range(M.shape[0]):
        radius = row_sums[i] - M[i, i]
        ax.add_patch(mpl.patches.Circle((M[i, i], 0), radius=radius,
                                        alpha=.6 / M.shape[0], ec='green'))
    ax.autoscale_view()


def sum_of_powers(M, n):
    """Returns the sum of the [0,n) powers of M"""
    result = np.zeros(M.shape)
    for i in range(n):
        result += np.linalg.matrix_power(M, i)
    return result


def is_irreducible(M):
    """Returns whether or not given square, positive matrix is irreducible"""
    Mk = sum_of_powers(M, M.shape[0])
    return not np.any(Mk == 0)


def is_node_globally_reachable(M, i):
    """Returns whether or not given node in given square, positive matrix is globally reachable"""
    power_sum = sum_of_powers(M, M.shape[0])
    return not np.any(power_sum[:, i] == 0)


def is_primitive(M):
    """
    Returns whether of not a given square is primitive

    Corollary 8.5.8 in Horn & Johnson, Matrix Analysis:
    Let A be an n×n non-negative matrix. Then A is primitive if and only if A^(n2−2n+2) has only positive entries.
    """
    n = M.shape[0]
    return not np.any(np.linalg.matrix_power(M, n ** 2 - 2 * n + 2) == 0)


def create_G_from_adj_matrix(A):
    """

    :param A: Adjacency Matrix
    :return: Graph with node 0 ... n based on adjacency matrix
    """
    G = nx.DiGraph()
    for i in range(len(A)):
        for j in range(len(A[i])):
            if (A[i, j]):
                G.add_edge(i, j, weight=A[i, j])
    return G


def draw_adj_matrix(A, ax):
    """
    Draw network based on given adjacency matrix
    """
    G = create_G_from_adj_matrix(A)
    pos = nx.drawing.layout.spring_layout(G, seed=4)
    nx.draw_networkx(G, pos=pos, node_size=100, ax=ax, connectionstyle='arc3, rad = 0.1')
    return G


def plot_matrix_binary(M, ax, name=''):
    """
    Drawing binary plot of adjacency matrix
    """
    blue_map = ListedColormap(["blue", "white"])
    zeros = M == 0
    im = ax.imshow(zeros, cmap=blue_map)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_condensated_graph(G, axs3, pos=None):
    """

    :param G: Graph which condensation should be drawn
    :param axs3: 1 dim axis with at least 3 subplots locations
    """

    # Visualization of initial Graph
    if pos is None:
        pos_rand = nx.spring_layout(G)  #setting the positions with respect to G, not k.
    else:
        pos_rand = pos
    nx.draw_networkx(G, pos=pos_rand, node_size=40, ax=axs3[0], connectionstyle='arc3, rad = 0.2', with_labels=False)

    # Algorithm to find the condensed graph:
    G_conden = nx.algorithms.components.condensation(G)

    all_col = []
    # We do the following for coloring scheme and saving that coloring scheme for the condensated graph
    for u, node in G_conden.nodes(data=True):
        sg = node['members']  # This contains a set of nodes from previous graph, that belongs to the condensated node
        co = np.random.rand(1,3)
        all_col.append(co)
        nx.draw_networkx_nodes(G.subgraph(sg), pos=pos_rand, node_size=40, node_color=co, ax=axs3[1])
        nx.draw_networkx_edges(G, pos=pos_rand, edgelist=G.edges(sg), edge_color=co, ax=axs3[1], connectionstyle='arc3, rad = 0.2')

    nx.draw_networkx(G_conden, node_size=40, ax=axs3[2], node_color=all_col, connectionstyle='arc3, rad = 0.2', with_labels=False)
    axs3[0].set_xlabel("Original Digraph")
    axs3[1].set_xlabel("Strongly connected components")
    axs3[2].set_xlabel("Condensation");