#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Tudat import
from tudatpy import constants

# General import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

#######################################################################################################################
### Generate figures of merit #########################################################################################
#######################################################################################################################

degrees_to_consider = [2]

# Set output directory
output_directory = "./output/tidal_forcing_analysis"

fontsize = 12

for degree in degrees_to_consider:
    output_path = os.path.join(output_directory, f"degree_{degree}")

    # Load tidal forcing history
    tidal_forcing_history = np.loadtxt(os.path.join(output_path, "tidal_forcing_history.dat"))
    tidal_forcing_history_cosine = tidal_forcing_history[:, 1]
    tidal_forcing_history_sine = tidal_forcing_history[:, 2]

    # Load distance history
    distance_history = np.loadtxt(os.path.join(output_path, "distance_history.dat"))

    # Load kepler elements history
    kepler_elements_history = np.loadtxt(os.path.join(output_path, "kepler_elements_history.dat"))

    # Compute mean value of tidal forcing
    mean_tidal_forcing = [np.mean(tidal_forcing_history_cosine), np.mean(tidal_forcing_history_sine)]
    np.savetxt(os.path.join(output_path, "mean_tidal_forcing.txt"), mean_tidal_forcing)

    # Compute variation range of tidal forcing
    variation_range_tidal_forcing = np.array([
        [min(tidal_forcing_history_cosine), max(tidal_forcing_history_cosine)],
        [min(tidal_forcing_history_sine), max(tidal_forcing_history_sine)],
    ])
    np.savetxt(os.path.join(output_path, "variation_range_tidal_forcing.txt"), variation_range_tidal_forcing)

    # Plot tidal forcing history, cosine component
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax = fig.add_subplot(4, 1, 1)
    ax.plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
            abs(tidal_forcing_history_cosine),
            color="blue")
    ax.hlines(abs(mean_tidal_forcing[0]),
              xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
              xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    ax.set_ylabel(r"$\Delta C_{2,0}$  [-]", fontsize=fontsize)
    mean_tidal_forcing_handle = mlines.Line2D([], [],
                                              color="black",
                                              linestyle="--",
                                              linewidth=3,
                                              label="Mean")
    ax.legend(handles=[mean_tidal_forcing_handle], loc="upper right", fontsize=fontsize)
    ax.text(0.0, 2.841e-5, f"Mean:  {abs(mean_tidal_forcing[0])}", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylim(top=2.85e-5)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    # Plot tidal forcing history, sine component
    ax = fig.add_subplot(4, 1, 2)
    ax.plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
            abs(tidal_forcing_history_sine),
            color="blue")
    ax.set_ylabel(r"$\Delta S_{2,0}$  [-]", fontsize=fontsize)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    # Plot distance history
    ax = fig.add_subplot(4, 1, 3)
    ax.plot(distance_history[:, 0] / constants.JULIAN_DAY,
             distance_history[:, 1],
             color="blue")
    ax.set_ylabel("Distance Enceladus-Saturn  [m]", fontsize=fontsize)
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    # Plot eccentricity history
    ax = fig.add_subplot(4, 1, 4)
    ax.plot(kepler_elements_history[:, 0] / constants.JULIAN_DAY,
             kepler_elements_history[:, 2],
             color="blue")
    ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax.set_ylabel("Eccentricity  [-]", fontsize=fontsize)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")

    fig.savefig(os.path.join(output_path, "tidal_forcing.pdf"))
    plt.close(fig)
