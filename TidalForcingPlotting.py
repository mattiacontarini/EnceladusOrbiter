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

    # Load Enceladus-fixed spherical coordinates of Saturn
    saturn_spherical_coordinates_history = np.loadtxt(os.path.join(output_path, "saturn_spherical_coordinates_history.dat"))

    # Compute mean value of tidal forcing
    mean_tidal_forcing = [np.mean(tidal_forcing_history_cosine), np.mean(tidal_forcing_history_sine)]
    np.savetxt(os.path.join(output_path, "mean_tidal_forcing.txt"), mean_tidal_forcing)

    # Compute mean value of Enceladus-fixed latitude and longitude
    mean_latitude = np.mean(saturn_spherical_coordinates_history[:, 2])
    mean_longitude = np.mean(saturn_spherical_coordinates_history[:, 3])

    # Compute variation range of tidal forcing
    variation_range_tidal_forcing = np.array([
        [min(tidal_forcing_history_cosine), max(tidal_forcing_history_cosine)],
        [min(tidal_forcing_history_sine), max(tidal_forcing_history_sine)],
    ])
    np.savetxt(os.path.join(output_path, "variation_range_tidal_forcing.txt"), variation_range_tidal_forcing)

    # Plot tidal forcing history, cosine component
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    ax = fig.add_subplot(5, 1, 1)
    ax.plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
            tidal_forcing_history_cosine,
            color="blue")
    ax.hlines(mean_tidal_forcing[0],
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
    ax.text(0.0, -0.001375, f"Mean:  {mean_tidal_forcing[0]}", fontsize=fontsize)
    ax.set_ylim(top=-0.00137)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    # Plot distance history
    ax = fig.add_subplot(5, 1, 2)
    ax.plot(distance_history[:, 0] / constants.JULIAN_DAY,
             distance_history[:, 1],
             color="blue")
    ax.set_ylabel("Distance [m]", fontsize=fontsize)
    # ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")

    # Plot Enceladus-fixed latitude of Saturn
    ax = fig.add_subplot(5, 1, 3)
    ax.plot(saturn_spherical_coordinates_history[:, 0] / constants.JULIAN_DAY,
            np.rad2deg(saturn_spherical_coordinates_history[:, 2]),
            color="blue",
            label="Latitude")
    ax.hlines(np.rad2deg(mean_latitude),
              xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
              xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    mean_latitude_handle = mlines.Line2D([], [],
                                              color="black",
                                              linestyle="--",
                                              linewidth=3,
                                              label="Mean")
    ax.set_ylabel(r"Saturn latitude  [deg]", fontsize=fontsize)
    ax.set_ylim(top=0.0006)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    ax.legend(handles=[mean_latitude_handle], loc="upper right", fontsize=fontsize)
    ax.text(0.0, 0.0005, f"Mean: {np.rad2deg(mean_latitude)} deg", fontsize=fontsize)

    # Plot Enceladus-fixed longitude of Saturn
    ax = fig.add_subplot(5, 1, 4)
    ax.plot(saturn_spherical_coordinates_history[:, 0] / constants.JULIAN_DAY,
            np.rad2deg(saturn_spherical_coordinates_history[:, 3]),
            color="blue",
            label="Longitude")
    ax.hlines(np.rad2deg(mean_longitude),
              xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
              xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    mean_longitude_handle = mlines.Line2D([], [],
                                         color="black",
                                         linestyle="--",
                                         linewidth=3,
                                         label="Mean")
    ax.set_ylabel(r"Saturn longitude  [deg]", fontsize=fontsize)
    ax.set_ylim(top=-5.15)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")
    ax.legend(handles=[mean_longitude_handle], loc="upper right", fontsize=fontsize)
    ax.text(0.0, -5.33, f"Mean: {np.rad2deg(mean_longitude)} deg", fontsize=fontsize)

    # Plot eccentricity history
    ax = fig.add_subplot(5, 1, 5)
    ax.plot(kepler_elements_history[:, 0] / constants.JULIAN_DAY,
             kepler_elements_history[:, 2],
             color="blue")
    ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax.set_ylabel("Eccentricity  [-]", fontsize=fontsize)
    ax.grid(True, which="both")
    ax.tick_params(labelsize=fontsize, which="both")

    fig.savefig(os.path.join(output_path, "tidal_forcing.pdf"))
    plt.close(fig)
