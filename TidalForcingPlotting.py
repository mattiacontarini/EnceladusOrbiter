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

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig

def perform_tidal_forcing_analysis_plotting(output_directory,
                                   degrees_to_consider,
                                   fontsize=12):

    #######################################################################################################################
    ### Generate figures of merit #########################################################################################
    #######################################################################################################################

    for degree in degrees_to_consider:
        degree_output_path = os.path.join(output_directory, f"degree_{degree}")

        nb_orders = len(range(degree + 1))
        fig, axes = plt.subplots(nb_orders, 2, constrained_layout=True, figsize=(10, 10))
        for order in range(degree + 1):

            # Define input path
            input_path = os.path.join(degree_output_path, f"order_{order}")

            # Load tidal forcing history
            tidal_forcing_history = np.loadtxt(os.path.join(input_path, "tidal_forcing_history.dat"))
            tidal_forcing_history_cosine = tidal_forcing_history[:, 1]
            tidal_forcing_history_sine = tidal_forcing_history[:, 2]

            # Load mean tidal forcing
            mean_tidal_forcing_cosine = np.loadtxt(os.path.join(input_path, "mean_tidal_forcing_cosine.txt"))
            mean_tidal_forcing_sine = np.loadtxt(os.path.join(input_path, "mean_tidal_forcing_sine.txt"))

            # Plot cosine component of tidal forcing history
            axes[order, 0].plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
                                tidal_forcing_history_cosine,
                                color="tab:blue")
            axes[order, 1].plot(tidal_forcing_history[:, 0] / constants.JULIAN_DAY,
                                tidal_forcing_history_sine,
                                color="tab:blue")

            # Add average line
            axes[order, 0].hlines(mean_tidal_forcing_cosine,
                                  xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
                                  xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
                                  linestyles="--",
                                  color="black",
                                  linewidth=3)
            axes[order, 1].hlines(mean_tidal_forcing_sine,
                                  xmin=tidal_forcing_history[0, 0] / constants.JULIAN_DAY,
                                  xmax=tidal_forcing_history[-1, 0] / constants.JULIAN_DAY,
                                  linestyles="--",
                                  color="black",
                                  linewidth=3)

            # Print average value
            axes[order, 0].set_title(f"Mean:  {mean_tidal_forcing_cosine}",
                                     loc="right",
                                     fontsize=fontsize)
            axes[order, 1].set_title(f"Mean:  {mean_tidal_forcing_sine}",
                                     loc="right",
                                     fontsize=fontsize)

            # Print axes labels
            axes[order, 0].set_ylabel(f"Tidal forcing ({degree}, {order}), cosine  [-]",
                                      fontsize=fontsize)
            axes[order, 1].set_ylabel(f"Tidal forcing ({degree}, {order}), sine  [-]",
                                      fontsize=fontsize)

        mean_value_handle = mlines.Line2D([], [],
                                              color="black",
                                              linestyle="--",
                                              linewidth=3,
                                              label="Mean")
        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            ax.legend(handles=[mean_value_handle], fontsize=fontsize, loc="upper right")


        axes[order, 0].set_xlabel(r"$t - t_{0}$  [days]",
                                  fontsize=fontsize)
        axes[order, 1].set_xlabel(r"$t - t_{0}$  [days]",
                                  fontsize=fontsize)

        plt.delaxes(axes[0, 1])
        plt.savefig(os.path.join(degree_output_path, f"tidal_forcing_degree_{degree}.pdf"))
        plt.close(fig)


    # Load kepler elements history of Enceladus
    enceladus_keplerian_state_history = np.loadtxt(os.path.join(output_directory,
                                                                "enceladus_keplerian_state_history.dat"))

    # Load Enceladus-fixed spherical coordinates of Saturn
    saturn_spherical_coordinates_history = np.loadtxt(os.path.join(output_directory,
                                                                   "saturn_spherical_coordinates_history.dat"))

    # Compute mean value of Enceladus-fixed latitude and longitude
    mean_latitude = np.mean(saturn_spherical_coordinates_history[:, 2])
    mean_longitude = np.mean(saturn_spherical_coordinates_history[:, 3])

    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 6.5))
    # Plot distance history
    axes[0, 0].plot(saturn_spherical_coordinates_history[:, 0] / constants.JULIAN_DAY,
                    saturn_spherical_coordinates_history[:, 1],
                    color="tab:blue")
    axes[0, 0].set_ylabel("Distance [m]", fontsize=fontsize)

    # Plot eccentricity history
    axes[0, 1].plot(enceladus_keplerian_state_history[:, 0] / constants.JULIAN_DAY,
                    enceladus_keplerian_state_history[:, 2],
                    color="tab:blue")
    axes[0, 1].hlines(np.mean(enceladus_keplerian_state_history[:, 2]),
                      xmin = enceladus_keplerian_state_history[0, 0] / constants.JULIAN_DAY,
                      xmax = enceladus_keplerian_state_history[-1, 0] / constants.JULIAN_DAY,
                      linestyles="--",
                      color="black",
                      linewidth=3)
    axes[0, 1].set_ylabel("Eccentricity  [-]", fontsize=fontsize)
    axes[0, 1].set_title(f"Mean:  {np.mean(enceladus_keplerian_state_history[:, 2])}", fontsize=fontsize)
    mean_eccentricity_handle = mlines.Line2D([], [],
                                             color="black",
                                             linestyle="--",
                                             linewidth=3,
                                             label="Mean")
    axes[0, 1].legend(handles=[mean_eccentricity_handle], fontsize=fontsize, loc="upper right")

    # Plot Enceladus-fixed latitude of Saturn
    axes[1, 0].plot(saturn_spherical_coordinates_history[:, 0] / constants.JULIAN_DAY,
                    np.rad2deg(saturn_spherical_coordinates_history[:, 2]),
                    color="tab:blue")
    axes[1, 0].hlines(np.rad2deg(mean_latitude),
              xmin=saturn_spherical_coordinates_history[0, 0] / constants.JULIAN_DAY,
              xmax=saturn_spherical_coordinates_history[-1, 0] / constants.JULIAN_DAY,
              linestyles="--",
              color="black",
              linewidth=3)
    mean_latitude_handle = mlines.Line2D([], [],
                                         color="black",
                                         linestyle="--",
                                         linewidth=3,
                                         label="Mean")
    axes[1, 0].set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    axes[1, 0].set_ylabel(r"Saturn latitude  [deg]", fontsize=fontsize)
    axes[1, 0].set_title(f"Mean: {np.rad2deg(mean_latitude)} deg", fontsize=fontsize)
    axes[1, 0].legend(handles=[mean_latitude_handle], loc="upper right", fontsize=fontsize)

    # Plot Enceladus-fixed longitude of Saturn
    axes[1, 1].plot(saturn_spherical_coordinates_history[:, 0] / constants.JULIAN_DAY,
                    np.rad2deg(saturn_spherical_coordinates_history[:, 3]),
                    color="tab:blue")
    axes[1, 1].hlines(np.rad2deg(mean_longitude),
                      xmin=saturn_spherical_coordinates_history[0, 0] / constants.JULIAN_DAY,
                      xmax=saturn_spherical_coordinates_history[-1, 0] / constants.JULIAN_DAY,
                      linestyles="--",
                      color="black",
                      linewidth=3)
    mean_longitude_handle = mlines.Line2D([], [],
                                          color="black",
                                          linestyle="--",
                                          linewidth=3,
                                          label="Mean")
    axes[1, 1].set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    axes[1, 1].set_ylabel(r"Saturn longitude  [deg]", fontsize=fontsize)
    axes[1, 1].set_title(f"Mean: {np.rad2deg(mean_longitude)} deg", fontsize=fontsize)
    axes[1, 1].legend(handles=[mean_longitude_handle], loc="upper right", fontsize=fontsize)

    for ax in axes.flat:
        ax.grid(True, which="both")
        ax.tick_params(labelsize=fontsize, which="both")

    plt.savefig(os.path.join(output_directory, "orbit_parameters_history.pdf"))
    plt.close(fig)


def perform_tidal_correction_verification_plotting(output_directory,
                                                   degree_order,
                                                   fontsize=12):

    ###################################################################################################################
    ### Generate figures of merit #####################################################################################
    ###################################################################################################################

    # Load cosine and sine gravity coefficient variation
    cosine_coefficient_variation = np.loadtxt(os.path.join(output_directory, "enceladus_cosine_coefficient_variation.dat"))
    sine_coefficient_variation = np.loadtxt(os.path.join(output_directory, "enceladus_sine_coefficient_variation.dat"))

    # Plot cosine and sine coefficient variation
    average_cosine_coefficient_variation_list = []
    average_sine_coefficient_variation_list = []
    mean_value_handle = mlines.Line2D([], [],
                                      color="black",
                                      linestyle="--",
                                      linewidth=3,
                                      label="Mean")

    fig, axes = plt.subplots(len(degree_order), 2, constrained_layout=True, figsize=(9, 9))
    for i in range(len(degree_order)):
        axes[i, 0].plot(cosine_coefficient_variation[:, 0] / constants.JULIAN_DAY,
                              cosine_coefficient_variation[:, i + 1])
        axes[i, 1].plot(sine_coefficient_variation[:, 0] / constants.JULIAN_DAY,
                              sine_coefficient_variation[:, i + 1])

        # Compute mean of cosine and sine coefficient variation
        average_cosine_coefficient_variation = np.mean(cosine_coefficient_variation[:, i + 1])
        average_sine_coefficient_variation = np.mean(sine_coefficient_variation[:, i + 1])
        average_cosine_coefficient_variation_list.append(average_cosine_coefficient_variation)
        average_sine_coefficient_variation_list.append(average_sine_coefficient_variation)

        axes[i, 0].hlines(average_cosine_coefficient_variation,
                          xmin=cosine_coefficient_variation[0, 0] / constants.JULIAN_DAY,
                          xmax=cosine_coefficient_variation[-1, 0] / constants.JULIAN_DAY,
                          linestyles="--",
                          color="black",
                          linewidth=3)
        axes[i, 1].hlines(average_sine_coefficient_variation,
                          xmin=sine_coefficient_variation[0, 0] / constants.JULIAN_DAY,
                          xmax=sine_coefficient_variation[-1, 0] / constants.JULIAN_DAY,
                          linestyles="--",
                          color="black",
                          linewidth=3)
        axes[i, 0].set_title(f"Mean: {average_cosine_coefficient_variation}", fontsize=fontsize, loc="right")
        axes[i, 1].set_title(f"Mean: {average_sine_coefficient_variation}", fontsize=fontsize, loc="right")

        axes[i, 0].legend(handles=[mean_value_handle], loc="upper right", fontsize=fontsize)
        axes[i, 1].legend(handles=[mean_value_handle], loc="upper right", fontsize=fontsize)

        degree = degree_order[i][0]
        order = degree_order[i][1]
        axes[i, 0].set_ylabel(r"$\Delta \bar{C}$" + f"  ({degree},{order})  [-]", fontsize=fontsize)
        axes[i, 1].set_ylabel(r"$\Delta \bar{S}$" + f"  ({degree},{order})  [-]", fontsize=fontsize)

    axes[-1, 0].set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    axes[-1, 1].set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)

    for ax in axes.flat:
        ax.grid(True, which="both")
        ax.tick_params(labelsize=fontsize)
    plt.delaxes(axes[0, 1])

    plt.savefig(os.path.join(output_directory, "gravity_coefficient_variation_correction.pdf"))
    plt.close(fig)

    np.savetxt(os.path.join(output_directory, "average_cosine_coefficient_variation.txt"),
               average_cosine_coefficient_variation_list)
    np.savetxt(os.path.join(output_directory, "average_sine_coefficient_variation.txt"),
               average_sine_coefficient_variation_list)


def perform_h2_partials_analysis_plotting(output_directory, fontsize=12):

    # Load drL_dh2 partials
    #fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 12))
    for lander_name in CovAnalysisConfig.lander_names:
        drL_dh2 = np.loadtxt(os.path.join(output_directory, f"drL_dh2_{lander_name}_lander.dat"))
        drL_dh2_average = np.loadtxt(os.path.join(output_directory, f"drL_dh2_average_{lander_name}_lander.dat"))

        fig, ax = plt.subplots()
        ax.plot(drL_dh2[:, 0] / constants.JULIAN_DAY, drL_dh2[:, 1] - drL_dh2_average[0], label="x", marker=".")
        ax.plot(drL_dh2[:, 0] / constants.JULIAN_DAY, drL_dh2[:, 2] - drL_dh2_average[1], label="y", marker=".")
        ax.plot(drL_dh2[:, 0] / constants.JULIAN_DAY, drL_dh2[:, 3] - drL_dh2_average[2], label="z", marker=".")
        ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
        ax.set_ylabel(r"$\Delta \mathbf{r}_{L}$  [m]", fontsize=fontsize)
        ax.set_title(f"Lander {lander_name}", fontsize=fontsize)
        ax.grid(True)
        ax.tick_params(labelsize=fontsize)
        fig.tight_layout()
        fig.savefig(os.path.join(output_directory, f"h2_partials_analysis_{lander_name}.pdf"))
        plt.close(fig)

def main():

    # Set output directory
    output_directory = "./output/tidal_forcing_analysis"

    perform_tidal_forcing_analysis_flag = False
    if perform_tidal_forcing_analysis_flag:
        output_directory_analysis = os.path.join(output_directory, "2025.06.03.16.31.28/tidal_forcing_computation")
        perform_tidal_forcing_analysis_plotting(output_directory_analysis,
                                                [2],
                                                14)

    perform_tidal_correction_verification_flag = False
    if perform_tidal_correction_verification_flag:
        output_directory_correction = os.path.join(output_directory, "2025.05.28.11.45.48/tidal_forcing_correction")
        perform_tidal_correction_verification_plotting(output_directory_correction,
                                                       [(2, 0), (2, 1), (2, 2)],
                                                       14)

    perform_h2_partials_analysis_plotting_flag = True
    if perform_h2_partials_analysis_plotting_flag:
        output_directory_partials = os.path.join(output_directory, "2025.06.20.16.11.35/h2_partials")
        perform_h2_partials_analysis_plotting(output_directory_partials, 14)

if __name__ == "__main__":
    main()
