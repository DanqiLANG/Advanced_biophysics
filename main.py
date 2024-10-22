"""Program to simulate some results for the biophysics project."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
    J = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            # Interactions for all barriers
            if j == i+1:
                J[i, j] = 1
            elif (i, j) in [(1, 5), (5, 4), (4, 3), (3, 2), (2, 6)]:
                J[i, j] = 1
            elif j == i+6:  # The paper indicates i+5, but I think it is i+6 (e.g. 1 should link with 7)
                J[i, j] = 1

            # Interactions that depend on the barrier
            if cross == 0 and (i, j) in [(5, 8), (4, 9)]:
                J[i, j] = 1
            elif cross == 1 and (i, j) in [(5, 8), (3, 10)]:
                J[i, j] = 1
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


# TODO define the assembly (use a list of size 6, n that contains the value (from 1 to 12)) ?
# TODO code a funtion to compute the total energy of the assembly (to reproduce fig b of p40).
