"""Program to simulate some results for the biophysics project."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, BoundaryNorm


# Customization of the figures parameters
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["font.size"] = 16
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["savefig.format"] = "png"
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["figure.figsize"] = (12.8, 7.2)


def interaction(cross=1):
    """
    Returns the matrix containing the interaction energies between the blocs.

    cross: gives which interactions are taken into account. See p40.
           0, 1, 2 correspond to the small, medium, large barrier respectively.
    """
    # TODO maybe use something to distinguish links that are specific to the barrier (to use a different color when plotting)
    J = np.zeros((13, 13))
    for i in range(13):
        for j in range(13):
            # Interactions for all barriers
            if j == 0 or i == 0:  # 0 means empty, so it does not interact with anything
                continue
            if j == i+1:
                J[i, j] = 1
                J[j, i] = 1  # Makes the matrix symmetric
            elif (i-1, j-1) in [(1, 5), (5, 4), (4, 3), (3, 2), (2, 6)]:  # -1 because the indexes start from 0
                J[i, j] = 1
                J[j, i] = 1
            elif j == i+6:  # The paper indicates i+5, but I think it is i+6 (e.g. 2 should link with 8, and not 7).
                J[i, j] = 1
                J[j, i] = 1

            # Interactions that depend on the barrier
            if cross == 0 and (i-1, j-1) in [(5, 8), (4, 9)]:
                J[i, j] = 1
                J[j, i] = 1
            elif cross == 1 and (i-1, j-1) in [(5, 8), (3, 10)]:
                J[i, j] = 1
                J[j, i] = 1
    return J


def plot_interaction(cross=1):
    """Plots the interaction matrix. cross: indicates which barrier to use."""
    if cross != 0 and cross != 1:  # Set the argument to 2 if it is different than 0 or 1
        cross = 2
    J = interaction(cross)
    title_list = ["small", "medium", "large"]
    title = f"Matrix for {title_list[cross]} barrier"

    fig, ax = plt.subplots()
    ax.pcolor(J)
    ax.set(title=title, xlabel="i", ylabel="j")
    plt.show()


def plot_interactions():
    """Plots all of the interaction matrixes."""
    fig, ax = plt.subplots(1, 3, sharey=True)
    title_list = ["small", "medium", "large"]
    for i in range(3):
        J = interaction(i)
        title = f"Matrix for {title_list[i]} barrier"
        ax[i].pcolor(J)
        if i == 0:  # Only put the label on y axis on the 1st figure.
            ax[i].set(title=title, xlabel="i", ylabel="j")
        else:
            ax[i].set(title=title, xlabel="i")
    plt.show()


def create():
    """
    Creates the assembly.

    The assembly is represented as an array of size (6, N).
    The elements are integers from 0 to 12, where 0 means empty, and 1 to 12 a tile of corresponding number.
    """
    return np.array([[1, 2, 3, 4, 5, 6]], dtype=int)


def add_line(initial, line):
    """
    Adds a new line to the assembly.

    The inputs are the assembly, and an array or a list of size 6 to append to the assembly.
    """
    # Checks that the new line has the right format
    if not all([isinstance(elem, (int, np.int32)) and elem >= 0 and elem <= 12 for elem in line]):
        raise ValueError(f"Invalid elements in added line: {line}.")
    if len(line) != 6:
        raise ValueError(f"The added line {line} must be of length 6 (and not {len(line)}).")

    new_line = np.array(line, dtype=int)
    return np.block([[initial], [new_line]])


def remove_line(initial):
    """Remove the last line of the assembly."""
    return initial[:-1, :]


def energy(assembly):
    """Compute the energy of the assembly."""
    J = interaction()  # TODO: see how to get the interaction matrix to avoir recomputing it each time: add an argument ?
    e = 0
    a, b = assembly.shape

    # This adds 0 below and on the right of the assembly matrix.
    # They will give no interaction with the rest of the assembly
    # but allow to handle the boundaries more easily
    low = np.zeros(a+1, dtype=int)
    right = np.zeros(b, dtype=int)
    A2 = np.block([[assembly], [right]])
    A = np.column_stack([A2, low])

    # Computation of the energy
    for i in range(a):  # Loops on the shape of the original assembly
        for j in range(b):
            # A has a shape of a+1, b+1, so no out-of-index errors
            val = A[i, j]  # Type of the block at i, j
            val_b = A[i+1, j]  # Value of the block below (or to the right in the article's figures)
            val_r = A[i, j+1]  # Value right (or below in the figure)
            e += J[val, val_r]  # Interaction on the right
            e += J[val, val_b]  # Interaction below
    return e


def evolve(Nstep=100):
    """
    Create and simulate the growth of an assembly.

    Nstep is the number of steps.
    Returns the assembly, and the energy at each step.
    """
    A = create()
    E = np.zeros(Nstep)

    for n in range(Nstep):  # TODO implement the evolution here, for now it just adds random lines
        line = np.random.randint(0, 13, 6, dtype=int)
        A = add_line(A, line)
        E[n] = energy(A)

    return A, E


def plot_energy(Nstep=100):
    """Plot the energy as a function of the step."""
    A, E = evolve(Nstep)
    fig, ax = plt.subplots()
    ax.plot(E)
    ax.set(title="Energy of the assembly", xlabel="Step", ylabel="Energy")
    plt.show()


def plot_assembly(Nstep=10):  # TODO add interactions ?
    """Plot the assembly with a colormap."""
    A, E = evolve(Nstep)

    cmap = mpl.colormaps.get_cmap("viridis")
    cmap.set_under("w")  # Sets the 0 values to white color.
    ticks = np.arange(1, 13)  # Renormalize to [1, 12], so 0 are below the min
    norm = BoundaryNorm(ticks, cmap.N)  # Renormalize and discretize the colormap
    fig, ax = plt.subplots()
    img = ax.pcolor(A.T, cmap=cmap, norm=norm)  # Transposed, so that it is plotted like in the paper.
    fig.colorbar(img, ticks=ticks, norm=norm)
    ax.set(title="Assembly", xlabel="Line", ylabel="Column")
    plt.show()



# TODO if needed, define a function to compute energy only on the lasts lines, and add to the previous result (to avoid computing the same things, so it is faster)
# TODO Implement Monte Carlo and evolution of the assembly
