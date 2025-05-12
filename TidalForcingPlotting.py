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

    # Load distance history
    distance_history = np.loadtxt(os.path.join(output_path, "distance_history.dat"))

    # Load kepler elements history
    kepler_elements_history = np.loadtxt(os.path.join(output_path, "kepler_elements_history.dat"))

    # Compute mean value of tidal forcing
    mean_tidal_forcing = np.mean(tidal_forcing_history[:, 1])
    np.savetxt(os.path.join(output_path, "mean_tidal_forcing.txt"), [mean_tidal_forcing])

    # Compute variation range of tidal forcing
    variation_range_tidal_forcing = [min(tidal_forcing_history[:, 1]), max(tidal_forcing_history[:, 1])]
    np.savetxt(os.path.join(output_path, "variation_range_tidal_forcing.txt"), variation_range_tidal_forcing)

    # Plot tidal forcing history
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
            abs(tidal_forcing_history[:, 1]),
            color="blue")
    ax1.hlines(abs(mean_tidal_forcing),
              xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
              xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    ax1.set_ylabel(f"Degree {degree} tidal forcing  [-]", fontsize=fontsize)
    mean_tidal_forcing_handle = mlines.Line2D([], [],
                                              color="black",
                                              linestyle="--",
                                              linewidth=3,
                                              label="Mean")
    ax1.legend(handles=[mean_tidal_forcing_handle], loc="upper right", fontsize=fontsize)
    ax1.set_yscale("log")
    ax1.grid(True, which="both")
    ax1.tick_params(labelsize=fontsize, which="both")
    # Plot distance history
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(distance_history[:, 0] / constants.JULIAN_DAY,
             distance_history[:, 1],
             color="blue")
    ax2.set_ylabel("Distance Enceladus-Saturn  [m]", fontsize=fontsize)
    ax2.set_yscale("log")
    ax2.grid(True, which="both")
    ax2.tick_params(labelsize=fontsize, which="both")
    # Plot eccentricity history
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(kepler_elements_history[:, 0] / constants.JULIAN_DAY,
             kepler_elements_history[:, 2],
             color="blue")
    ax3.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax3.set_ylabel("Eccentricity  [-]", fontsize=fontsize)
    ax3.grid(True, which="both")
    ax3.tick_params(labelsize=fontsize, which="both")

    fig.savefig(os.path.join(output_path, "tidal_forcing.pdf"))
    plt.close(fig)
