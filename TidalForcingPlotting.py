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

for degree in degrees_to_consider:
    output_path = os.path.join(output_directory, f"degree_{degree}")

    # Load tidal forcing history
    tidal_forcing_history = np.loadtxt(os.path.join(output_path, "tidal_forcing_history.dat"))

    # Compute mean value of tidal forcing
    mean_tidal_forcing = np.mean(tidal_forcing_history[:, 1])
    np.savetxt(os.path.join(output_path, "mean_tidal_forcing.txt"), [mean_tidal_forcing])

    # Generate plot of tidal forcing history
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
            abs(tidal_forcing_history[:, 1]),
            color="blue")
    ax.hlines(abs(mean_tidal_forcing),
              xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
              xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    ax.set_xlabel(r"$t - t_{0}$  [days]")
    ax.set_ylabel(f"Degree {degree} tidal forcing  [-]")
    ax.set_yscale("log")
    ax.grid(True, which="both")

    mean_tidal_forcing_handle = mlines.Line2D([], [],
                                              color="black",
                                              linestyle="--",
                                              linewidth=3,
                                              label="Mean")
    ax.legend(handles=[mean_tidal_forcing_handle], loc="upper right")

    fig.savefig(os.path.join(output_path, "tidal_forcing.pdf"))
    plt.close(fig)
