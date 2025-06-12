#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Tudat import
from tudatpy import constants

# General import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os


#######################################################################################################################
### Functions #########################################################################################################
#######################################################################################################################

def plot_tuning_parameters_analysis(input_path,
                                    fontsize=12):

    # Set initial states to consider
    initial_state_indices = [1, 2, 3]

    # Set list of simulation durations to consider
    simulation_durations = [28.0 * constants.JULIAN_DAY,
                            60.0 * constants.JULIAN_DAY,
                            90.0 * constants.JULIAN_DAY,]

    # Set list of arc durations to consider
    arc_durations = [1.0 * constants.JULIAN_DAY,
                     2.0 * constants.JULIAN_DAY,
                     7.0 * constants.JULIAN_DAY]

    # Set list of values for the Kaula multiplier for a priori constraint on standard deviation
    kaula_constraint_multipliers = [1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of values for the a priori constraint on the empirical accelerations
    a_priori_empirical_accelerations = [1e-9, 1e-8, 1e-7, 1e-6]

    # Set list of values for the a priori constraint on the landers position
    a_priori_lander_position = [1e2, 1e3]

    # Include range observable flag
    include_lander_range_observable_flag = [False, True]

    # Set list of values for the duration of the arc-wise empirical accelerations
    empirical_accelerations_arc_duration = [0.5 * constants.JULIAN_DAY, 1.0 * constants.JULIAN_DAY]

    # Set list of values for the cadence of the data
    tracking_arc_duration = [4.0 * 3600.0, 6.0 * 3600.0, 8.0 * 3600.0]

    # Set list of values for the a priori constraint on the position of the rotation pole
    a_priori_rotation_pole_position = np.deg2rad([[np.infty, np.infty], [0.1, 0.1], [1e-2, 1e-2]])

    # Set list of values for the a priori constraint on the position rate of the rotation pole
    a_priori_rotation_pole_rate = np.deg2rad([[np.infty, np.infty], [0.1, 0.1], [1e-2, 1e-2]])

    # Set list of values for the a priori constraint on the radiation pressure coefficient
    a_priori_radiation_pressure_coefficient  = [np.infty, 0.1, 1e-10]

    # Set list of number of landers to include in the simulation
    lander_to_include = [[None],
                         ["L1"],
                         ["L1", "L2"],
                         ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]]

    parameters_to_tune = {
        "initial_state_index": initial_state_indices,
        "arc_duration": arc_durations,
        "simulation_duration": simulation_durations,
        "kaula_constraint_multiplier": kaula_constraint_multipliers,
        "a_priori_empirical_acceleration": a_priori_empirical_accelerations,
        "a_priori_lander_position": a_priori_lander_position,
        "include_lander_range_observable_flag": include_lander_range_observable_flag,
        "empirical_accelerations_arc_duration": empirical_accelerations_arc_duration,
        "tracking_arc_duration": tracking_arc_duration,
        "a_priori_rotation_pole_position": a_priori_rotation_pole_position,
        "a_priori_rotation_pole_rate": a_priori_rotation_pole_rate,
        "a_priori_radiation_pressure_coefficient": a_priori_radiation_pressure_coefficient
    }

    colors_rsw_interval = ["blue", "green", "red"]
    markers_rsw_interval = ["o", "*", "X"]

    parameters_to_tune_axis_label = {
        "initial_state_index": "K- orbit index  [-]",
        "arc_duration": "Arc duration  [days]",
        "simulation_duration": "Simulation duration  [days]",
        "kaula_constraint_multiplier": "Kaula constraint multiplier  [-]",
        "a_priori_empirical_acceleration": r"A priori $\sigma$ empirical acc.  [m/s$^2$]",
        "a_priori_lander_position": r"A priori $\sigma$ lander position  [m]",
        "include_lander_range_observable_flag": "Lander range observable inclusion flag  [-]",
        "empirical_accelerations_arc_duration": "Empirical acc. arc duration  [hours]",
        "tracking_arc_duration": "Tracking arc duration  [hours]",
        "a_priori_rotation_pole_position": r"A priori $\sigma$ rotation pole position  [deg]",
        "a_priori_rotation_pole_rate": r"A priori $\sigma$ rotation pole rate  [deg/s]",
        "a_priori_radiation_pressure_coefficient": r"A priori $\sigma$ radiation pressure coefficient  [-]",
    }

    parameters_to_tune_label = {
        "initial_state_index": "K- orbit index",
        "arc_duration": "Arc duration",
        "simulation_duration": "Simulation duration",
        "kaula_constraint_multiplier": "Kaula constraint multiplier",
        "a_priori_empirical_acceleration": r"A priori $\sigma$ empirical accelerations",
        "a_priori_lander_position": r"A priori $\sigma$ lander position",
        "include_lander_range_observable_flag": "Lander range observable inclusion flag",
        "empirical_accelerations_arc_duration": "Empirical accelerations arc duration",
        "tracking_arc_duration": "Tracking arc duration",
        "a_priori_rotation_pole_position": r"A priori $\sigma$ rotation pole position",
        "a_priori_rotation_pole_rate": r"A priori $\sigma$ rotation pole rate",
        "a_priori_radiation_pressure_coefficient": r"A priori $\sigma$ radiation pressure coefficient",
    }

    rsw_interval_min_value_handle = mpatches.Patch(
        color=colors_rsw_interval[0],
        label="min",
    )
    rsw_interval_median_value_handle = mpatches.Patch(
        color=colors_rsw_interval[1],
        label="median",
    )
    rsw_interval_max_value_handle = mpatches.Patch(
        color=colors_rsw_interval[2],
        label="max",
    )
    rsw_interval_radial_direction_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker=markers_rsw_interval[0],
        linestyle="None",
        label="radial"
    )
    rsw_interval_along_track_direction_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker=markers_rsw_interval[1],
        linestyle="None",
        label="along-track"
    )
    rsw_interval_cross_track_direction_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker=markers_rsw_interval[2],
        linestyle="None",
        label="cross-track"
    )

    real_part_handle = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="P",
        linestyle="None",
        label="Re()"
    )

    imaginary_part_handle = mlines.Line2D(
        [],
        [],
        color="midnightblue",
        marker="P",
        linestyle="None",
        label="Im()"
    )

    RA_handle = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="P",
        linestyle="None",
        label="RA"
    )

    DE_handle = mlines.Line2D(
        [],
        [],
        color="midnightblue",
        marker="P",
        linestyle="None",
        label="DE"
    )

    RA_rate_handle = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="P",
        linestyle="None",
        label="RA rate"
    )

    DE_rate_handle = mlines.Line2D(
        [],
        [],
        color="midnightblue",
        marker="P",
        linestyle="None",
        label="DE rate"
    )

    for lander in lander_to_include:
        input_path_lander = os.path.join(input_path, f"lander_to_include_case_{lander_to_include.index(lander)}")

        for parameter_key in list(parameters_to_tune.keys()):
            input_path_parameter = os.path.join(input_path_lander, parameter_key)

            # Create plot of figures of merit for current considered parameter
            fig, axes = plt.subplots(5, 2, constrained_layout=True, figsize=(10, 14))
            for parameter_value_index in range(len(parameters_to_tune[parameter_key])):
                parameter_value = parameters_to_tune[parameter_key][parameter_value_index]
                input_path_configuration = os.path.join(input_path_parameter, f"configuration_{parameter_value_index}")
                input_path_covariance_results = os.path.join(input_path_configuration, "covariance_results")

                # Load results
                condition_number_covariance_matrix = np.loadtxt(
                    os.path.join(input_path_covariance_results, "condition_number_covariance_matrix.dat")
                )
                max_estimatable_degree_gravity_field = np.loadtxt(
                    os.path.join(input_path_covariance_results, "max_estimatable_degree_gravity_field.dat")
                )
                formal_error_initial_position_rsw_interval = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_initial_position_rsw_interval.dat")
                )
                formal_error_empirical_accelerations_rsw_interval = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_empirical_accelerations_rsw_interval.dat")
                )
                formal_error_love_number = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_love_number.dat")
                )
                formal_error_radial_love_number = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_radial_love_number.dat")
                )
                formal_error_libration_amplitude = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_libration_amplitude.dat")
                )
                formal_error_pole_position = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_pole_position.dat")
                )
                formal_error_pole_rate = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_pole_rate.dat")
                )
                nb_observations_ratio = np.loadtxt(
                     os.path.join(input_path_covariance_results, "nb_observations_ratio.dat")
                )

                if parameter_key == "simulation_duration":
                    parameter_value = parameter_value / constants.JULIAN_DAY
                elif parameter_key == "arc_duration":
                    parameter_value = parameter_value / constants.JULIAN_DAY
                elif parameter_key == "empirical_accelerations_arc_duration":
                    parameter_value = parameter_value / 3600.0
                elif parameter_key == "tracking_arc_duration":
                    parameter_value = parameter_value / 3600.0
                elif parameter_key == "lander_to_include":
                    parameter_value = len(parameter_value)
                elif parameter_key == "a_priori_rotation_pole_position":
                    parameter_value = str(np.rad2deg(parameter_value))
                elif parameter_key == "a_priori_rotation_pole_rate":
                    parameter_value = str(np.rad2deg(parameter_value))
                elif parameter_key == "a_priori_radiation_pressure_coefficient":
                    parameter_value = str(parameter_value)

                # Plot results
                axes[0, 0].scatter(parameter_value, condition_number_covariance_matrix, color="black", marker="P")
                axes[0, 1].scatter(parameter_value, max_estimatable_degree_gravity_field, color="black", marker="P")
                for i in range(len(colors_rsw_interval)):
                    for j in range(len(markers_rsw_interval)):
                        axes[1, 0].scatter(parameter_value,
                                           formal_error_initial_position_rsw_interval[i, j],
                                           color=colors_rsw_interval[i],
                                           marker=markers_rsw_interval[j])
                        axes[1, 1].scatter(parameter_value,
                                           formal_error_empirical_accelerations_rsw_interval[i, j],
                                           color=colors_rsw_interval[i],
                                           marker=markers_rsw_interval[j])
                axes[2, 0].scatter(parameter_value, nb_observations_ratio, color="black", marker="P")
                axes[2, 1].scatter(parameter_value, formal_error_love_number[0], color="orange", marker="P")
                axes[2, 1].scatter(parameter_value, formal_error_love_number[1], color="midnightblue", marker="P")
                axes[3, 0].scatter(parameter_value, np.rad2deg(formal_error_libration_amplitude), color="black", marker="P")
                axes[3, 1].scatter(parameter_value, np.rad2deg(formal_error_pole_position[0]), color="orange", marker="P")
                axes[3, 1].scatter(parameter_value, np.rad2deg(formal_error_pole_position[1]), color="midnightblue", marker="P")
                axes[4, 0].scatter(parameter_value, np.rad2deg(formal_error_pole_rate[0]), color="orange", marker="P")
                axes[4, 0].scatter(parameter_value, formal_error_pole_rate[1], color="midnightblue", marker="P")
                axes[4, 1].scatter(parameter_value, formal_error_radial_love_number, color="black", marker="P")

            axes[0, 0].set_ylabel("Condition number cov. matrix  [-]", fontsize=fontsize)
            axes[0, 0].set_yscale("log")
            axes[0, 1].set_ylabel("Max. degree gravity field [-]", fontsize=fontsize)
            axes[1, 0].set_ylabel(r"$\sigma$ initial position  [m]", fontsize=fontsize)
            axes[1, 1].set_ylabel(r"$\sigma$ empirical acc.  [m/s$^{2}$]", fontsize=fontsize)
            axes[2, 0].set_ylabel("no. lander data / no. GS data  [-]", fontsize=fontsize)
            axes[2, 1].set_ylabel(r"$\sigma$ $k_2$ Love number  [-]", fontsize=fontsize)
            axes[2, 1].legend(handles=[real_part_handle, imaginary_part_handle], fontsize=fontsize)
            axes[3, 0].set_ylabel(r"$\sigma$ libration amplitude  [deg]", fontsize=fontsize)
            axes[3, 1].set_ylabel(f"$\sigma$ pole position  [deg]", fontsize=fontsize)
            axes[3, 1].legend(handles=[RA_handle, DE_handle], fontsize=fontsize)
            axes[4, 0].set_xlabel(parameters_to_tune_axis_label[parameter_key], fontsize=fontsize)
            axes[4, 0].set_ylabel(f"$\sigma$ pole rate  [deg/s]", fontsize=fontsize)
            axes[4, 0].legend(handles=[RA_rate_handle, DE_rate_handle], fontsize=fontsize)
            axes[4, 1].set_xlabel(parameters_to_tune_axis_label[parameter_key], fontsize=fontsize)
            axes[4, 1].set_ylabel(f"$\sigma$ $h_2$ Love number  [-]", fontsize=fontsize)


            for ax in axes.flat:
                ax.tick_params(labelsize=fontsize)
                ax.grid(True, which="both")
            if parameter_key == "kaula_constraint_multiplier" or parameter_key == "a_priori_empirical_acceleration":
                for ax in axes.flat:
                    ax.set_xscale("log")
            elif parameter_key == "simulation_duration":
                for ax in axes.flat:
                    ax.set_xlim(left=20)
            elif parameter_key == "a_priori_lander_position":
                for ax in axes.flat:
                    ax.set_xlim(left=0)

            fig.legend(handles=[rsw_interval_min_value_handle,
                                rsw_interval_median_value_handle,
                                rsw_interval_max_value_handle,
                                rsw_interval_radial_direction_handle,
                                rsw_interval_along_track_direction_handle,
                                rsw_interval_cross_track_direction_handle],
                       fontsize=fontsize,
                       bbox_to_anchor=(0.85, 0.2),)
            fig.suptitle(f"Parameter: {parameters_to_tune_label[parameter_key]}", fontsize=fontsize)
            fig.savefig(os.path.join(input_path_parameter, "figures_of_merit.pdf"))
            plt.close(fig)


def summarise_tuning_parameters_analysis(input_path,
                                              fontsize=12):

    # Set initial states to consider
    initial_state_indices = [1, 2, 3]

    # Set list of simulation durations to consider
    simulation_durations = [28.0 * constants.JULIAN_DAY,
                            60.0 * constants.JULIAN_DAY,
                            90.0 * constants.JULIAN_DAY, ]

    # Set list of arc durations to consider
    arc_durations = [1.0 * constants.JULIAN_DAY,
                     2.0 * constants.JULIAN_DAY,
                     7.0 * constants.JULIAN_DAY]

    # Set list of values for the Kaula multiplier for a priori constraint on standard deviation
    kaula_constraint_multipliers = [1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of values for the a priori constraint on the empirical accelerations
    a_priori_empirical_accelerations = [1e-9, 1e-8, 1e-7, 1e-6]

    # Set list of values for the a priori constraint on the landers position
    a_priori_lander_position = [1e2, 1e3]

    # Include range observable flag
    include_lander_range_observable_flag = [False, True]

    # Set list of values for the duration of the arc-wise empirical accelerations
    empirical_accelerations_arc_duration = [0.5 * constants.JULIAN_DAY, 1.0 * constants.JULIAN_DAY]

    # Set list of values for the cadence of the data
    tracking_arc_duration = [4.0 * 3600.0, 6.0 * 3600.0, 8.0 * 3600.0]

    # Set list of values for the a priori constraint on the position of the rotation pole
    a_priori_rotation_pole_position = np.deg2rad([[np.infty, np.infty], [0.1, 0.1], [1e-2, 1e-2]])

    # Set list of values for the a priori constraint on the position rate of the rotation pole
    a_priori_rotation_pole_rate = np.deg2rad([[np.infty, np.infty], [0.1, 0.1], [1e-2, 1e-2]])

    # Set list of values for the a priori constraint on the radiation pressure coefficient
    a_priori_radiation_pressure_coefficient = [np.infty, 0.1, 1e-10]

    # Set list of number of landers to include in the simulation
    lander_to_include = [[None],
                         ["L1"],
                         ["L1", "L2"],
                         ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]]

    parameters_to_tune = {
        "initial_state_index": initial_state_indices,
        "arc_duration": arc_durations,
        "simulation_duration": simulation_durations,
        "kaula_constraint_multiplier": kaula_constraint_multipliers,
        "a_priori_empirical_acceleration": a_priori_empirical_accelerations,
        "a_priori_lander_position": a_priori_lander_position,
        "include_lander_range_observable_flag": include_lander_range_observable_flag,
        "empirical_accelerations_arc_duration": empirical_accelerations_arc_duration,
        "tracking_arc_duration": tracking_arc_duration,
        "a_priori_rotation_pole_position": a_priori_rotation_pole_position,
        "a_priori_rotation_pole_rate": a_priori_rotation_pole_rate,
        "a_priori_radiation_pressure_coefficient": a_priori_radiation_pressure_coefficient
    }

    configurations_list = []
    configurations_counter = 0

    parameters_of_interest = dict(
        max_estimatable_degree_gravity_field= [],
        formal_error_love_number = [],
        formal_error_libration_amplitude = [],
        formal_error_pole_position = [],
    )
    parameters_of_interest_axis_labels = dict(
        max_estimatable_degree_gravity_field="Max. degree gravity field  [-]",
        formal_error_love_number = r"$\sigma$ $k_2$ Love number  [-]",
        formal_error_libration_amplitude = r"$\sigma$ libration amplitude  [deg]",
        formal_error_pole_position = r"$\sigma$ pole position  [deg]",
    )

    for lander_index in range(len(lander_to_include)):
        input_path_lander = os.path.join(input_path, f"lander_to_include_case_{lander_index}")
        lander_configuration = int(lander_index)

        for parameter_key in list(parameters_to_tune.keys()):
            parameter_configuration = int(list(parameters_to_tune.keys()).index(parameter_key))
            input_path_parameter = os.path.join(input_path_lander, parameter_key)

            for parameter_value_index in range(len(parameters_to_tune[parameter_key])):
                parameter_value_configuration = int(parameter_value_index)
                input_path_configuration = os.path.join(input_path_parameter, f"configuration_{parameter_value_index}")
                input_path_covariance_results = os.path.join(input_path_configuration, "covariance_results")

                # Load results
                max_estimatable_degree_gravity_field = np.loadtxt(
                    os.path.join(input_path_covariance_results, "max_estimatable_degree_gravity_field.dat")
                )
                formal_error_love_number = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_love_number.dat")
                )
                formal_error_libration_amplitude = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_libration_amplitude.dat")
                )
                formal_error_pole_position = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_pole_position.dat")
                )

                parameters_of_interest["max_estimatable_degree_gravity_field"].append(max_estimatable_degree_gravity_field)
                parameters_of_interest["formal_error_love_number"].append(formal_error_love_number)
                parameters_of_interest["formal_error_libration_amplitude"].append(np.rad2deg(formal_error_libration_amplitude))
                parameters_of_interest["formal_error_pole_position"].append(np.rad2deg(formal_error_pole_position))

                configurations_list.append(f"{lander_configuration}.{parameter_configuration}.{parameter_value_configuration}")
                configurations_counter += 1

    nb_parameters_of_interest = len(parameters_of_interest.keys())
    configurations = np.arange(1, configurations_counter + 1, 1, dtype=int)

    RA_handle = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        label="RA"
    )

    DE_handle = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        label="DE"
    )

    real_part_handle = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        label="Re()"
    )

    imaginary_part_handle = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        label="Im()"
    )

    for i in range(nb_parameters_of_interest):
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(1, 1, 1)
        parameter_key = list(parameters_of_interest.keys())[i]
        if parameter_key == "formal_error_love_number" or parameter_key == "formal_error_pole_position":
            for j in range(len(configurations_list)):
                ax.scatter(configurations_list[j], parameters_of_interest[parameter_key][j][0], color="blue")
                ax.scatter(configurations_list[j], parameters_of_interest[parameter_key][j][1], color="red")
                #ax.scatter(parameters_of_interest[parameter_key][i][0], configurations_list[i])
        else:
            ax.scatter(configurations_list, parameters_of_interest[parameter_key], color="black")
            #ax.scatter(parameters_of_interest[parameter_key], configurations_list)
        if parameter_key == "formal_error_pole_position":
            ax.legend(handles=[RA_handle, DE_handle], fontsize=fontsize)
        elif parameter_key == "formal_error_love_number":
            ax.legend(handles=[real_part_handle, imaginary_part_handle], fontsize=fontsize)
        ax.set_xlabel("Configuration  [-]", fontsize=fontsize)
        ax.set_ylabel(parameters_of_interest_axis_labels[parameter_key], fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=10, rotation=90)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.grid(True, which="both")
        fig.tight_layout()
        if parameter_key == "formal_error_libration_amplitude" or parameter_key == "formal_error_pole_position" or parameter_key == "formal_error_love_number":
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(input_path, f"summary_{parameter_key}.pdf"))
        plt.close(fig)


#######################################################################################################################
### Generate figures of merit #########################################################################################
#######################################################################################################################
def main():

    # Analyse parameters of interest varying one at a time
    plot_tuning_parameters_analysis_flag = False
    if plot_tuning_parameters_analysis_flag:
        input_directory = "output/covariance_analysis/tuning_parameters_analysis"
        time_stamp_folder = "2025.06.02.09.40.15"
        input_path = os.path.join(input_directory, time_stamp_folder)
        plot_tuning_parameters_analysis(input_path)

    summarise_tuning_parameters_analysis_flag = True
    if summarise_tuning_parameters_analysis_flag:
        input_directory = "output/covariance_analysis/tuning_parameters_analysis"
        time_stamp_folder = "2025.06.02.09.40.15"
        input_path = os.path.join(input_directory, time_stamp_folder)
        summarise_tuning_parameters_analysis(input_path)


if __name__ == "__main__":
    main()
