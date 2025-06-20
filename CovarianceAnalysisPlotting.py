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
    configurations_lander_position_list = []
    configurations_counter = 0

    parameters_of_interest = dict(
        max_estimatable_degree_gravity_field= [],
        formal_error_love_number = [],
        formal_error_libration_amplitude = [],
        formal_error_pole_position = [],
        formal_error_pole_rate = [],
        rms_formal_error_lander_position = []
    )
    parameters_of_interest_axis_labels = dict(
        max_estimatable_degree_gravity_field="Max. degree gravity field  [-]",
        formal_error_love_number = r"$\sigma$ $k_2$ Love number  [-]",
        formal_error_libration_amplitude = r"$\sigma$ libration amplitude  [deg]",
        formal_error_pole_position = r"$\sigma$ pole position  [deg]",
        formal_error_pole_rate = r"$\sigma$ pole rate  [deg s$^{-1}$]",
        rms_formal_error_lander_position = "RMS formal error lander position  [m]",
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
                formal_error_pole_rate = np.loadtxt(
                    os.path.join(input_path_covariance_results, "formal_error_pole_rate.dat")
                )
                indices_estimation_parameters = np.loadtxt(
                    os.path.join(input_path_covariance_results, "indices_estimation_parameters.dat")
                )

                # Retrieve formal error of lander position
                if lander_index != 0:
                    formal_errors = np.loadtxt(
                        os.path.join(input_path_covariance_results, "formal_errors.dat")
                    )

                    formal_error_lander_position = []

                    for k in range(indices_estimation_parameters.shape[0]):
                        if indices_estimation_parameters[k, 1] == 3.0:
                            index_start_lander_position = int(indices_estimation_parameters[k, 0])
                            break

                    length_lander_position = 3
                    if lander_index == 1:
                        nb_landers = 1
                    elif lander_index == 2:
                        nb_landers = 2
                    elif lander_index == 3:
                        nb_landers = 9

                    for l in range(nb_landers):
                        formal_error_lander_position.append(
                            list(formal_errors[index_start_lander_position + length_lander_position * l:
                                               index_start_lander_position + length_lander_position * l + length_lander_position]))

                    rms_position = np.zeros((4,))
                    for l in range(nb_landers):
                        rms_position[0] += formal_error_lander_position[l][0] ** 2
                        rms_position[1] += formal_error_lander_position[l][1] ** 2
                        rms_position[2] += formal_error_lander_position[l][2] ** 2
                    rms_position[0] = np.sqrt(rms_position[0] / nb_landers)
                    rms_position[1] = np.sqrt(rms_position[1] / nb_landers)
                    rms_position[2] = np.sqrt(rms_position[2] / nb_landers)
                    rms_position[3] = np.mean(rms_position[:3])

                    parameters_of_interest["rms_formal_error_lander_position"].append(rms_position)
                    configurations_lander_position_list.append(
                        f"{lander_configuration}.{parameter_configuration}.{parameter_value_configuration}")


                parameters_of_interest["max_estimatable_degree_gravity_field"].append(max_estimatable_degree_gravity_field)
                parameters_of_interest["formal_error_love_number"].append(formal_error_love_number)
                parameters_of_interest["formal_error_libration_amplitude"].append(np.rad2deg(formal_error_libration_amplitude))
                parameters_of_interest["formal_error_pole_position"].append(np.rad2deg(formal_error_pole_position))
                parameters_of_interest["formal_error_pole_rate"].append(np.rad2deg(formal_error_pole_rate))

                configurations_list.append(f"{lander_configuration}.{parameter_configuration}.{parameter_value_configuration}")
                configurations_counter += 1

    nb_parameters_of_interest = len(parameters_of_interest.keys())

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

    position_x_handle = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        label="x"
    )

    position_y_handle = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        label="y"
    )

    position_z_handle = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        label="z"
    )

    position_average_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        label="Average"
    )

    for i in range(nb_parameters_of_interest):
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(1, 1, 1)
        parameter_key = list(parameters_of_interest.keys())[i]
        if parameter_key == "formal_error_love_number" or parameter_key == "formal_error_pole_position" or parameter_key == "formal_error_pole_rate":
            for j in range(len(configurations_list)):
                ax.scatter(configurations_list[j], parameters_of_interest[parameter_key][j][0], color="blue")
                ax.scatter(configurations_list[j], parameters_of_interest[parameter_key][j][1], color="red")

        elif parameter_key == "rms_formal_error_lander_position":
            for j in range(len(configurations_lander_position_list)):
                ax.scatter(configurations_lander_position_list[j], parameters_of_interest[parameter_key][j][0],
                           color="blue")
                ax.scatter(configurations_lander_position_list[j], parameters_of_interest[parameter_key][j][1],
                           color="red")
                ax.scatter(configurations_lander_position_list[j], parameters_of_interest[parameter_key][j][2],
                           color="green")
                ax.scatter(configurations_lander_position_list[j], parameters_of_interest[parameter_key][j][3],
                           color="black")
        else:
            ax.scatter(configurations_list, parameters_of_interest[parameter_key], color="black")

        if parameter_key == "formal_error_love_number":
            ax.set_ylim(bottom=1e-5, top=2e-2)
        elif parameter_key == "formal_error_pole_position":
            ax.set_ylim(bottom=1e-7, top=1e0)
        elif parameter_key == "formal_error_pole_rate":
            ax.set_ylim(bottom=1e-13, top=1e-7)
        elif parameter_key == "rms_formal_error_lander_position":
            ax.set_ylim(bottom=1e-3, top=1e2)
        elif parameter_key == "formal_error_libration_amplitude":
            ax.set_ylim(bottom=1e-7, top=1e-2)
        elif parameter_key == "max_estimatable_degree_gravity_field":
            ax.set_ylim(bottom=6)

        if parameter_key == "formal_error_pole_position" or parameter_key == "formal_error_pole_rate":
            ax.legend(handles=[RA_handle, DE_handle], fontsize=fontsize)
        elif parameter_key == "formal_error_love_number":
            ax.legend(handles=[real_part_handle, imaginary_part_handle], fontsize=fontsize)
        elif parameter_key == "rms_formal_error_lander_position":
            ax.legend(handles=[position_x_handle, position_y_handle, position_z_handle, position_average_handle], fontsize=fontsize)
        ax.set_xlabel("Configuration  [-]", fontsize=fontsize)
        ax.set_ylabel(parameters_of_interest_axis_labels[parameter_key], fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=10, rotation=90)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.grid(True, which="both")
        fig.tight_layout()
        if (parameter_key == "formal_error_libration_amplitude" or parameter_key == "formal_error_pole_position" or
                parameter_key == "formal_error_love_number" or parameter_key == "formal_error_pole_rate" or
                 parameter_key == "rms_formal_error_lander_position"):
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(input_path, f"summary_{parameter_key}.pdf"))
        plt.close(fig)


def plot_tuning_parameters_refinement_analysis(input_path,
                                               no_configurations,
                                               fontsize=12):

    parameters_of_interest = dict(
        max_estimatable_degree_gravity_field= [],
        formal_error_love_number = [],
        formal_error_libration_amplitude = [],
        formal_error_pole_position = [],
        formal_error_pole_rate = [],
        rms_formal_error_degree_2 = []
    )
    parameters_of_interest_axis_labels = dict(
        max_estimatable_degree_gravity_field="Max. degree gravity field  [-]",
        formal_error_love_number = r"$\sigma$ $k_2$ Love number  [-]",
        formal_error_libration_amplitude = r"$\sigma$ libration amplitude  [deg]",
        formal_error_pole_position = r"$\sigma$ pole position  [deg]",
        formal_error_pole_rate = r"$\sigma$ pole rate  [deg s$^{-1}$]",
        rms_formal_error_degree_2 = "RMS formal error degree 2 SH gravity  [-]",
    )

    for i in range(no_configurations):
        input_path_configuration = os.path.join(input_path, f"configuration_no_{i}")
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
        formal_error_pole_rate = np.loadtxt(
            os.path.join(input_path_covariance_results, "formal_error_pole_rate.dat")
        )
        rms_formal_error_degree_2 = np.loadtxt(
            os.path.join(input_path_covariance_results, "rms_formal_error_degree_2.dat")
        )

        parameters_of_interest["max_estimatable_degree_gravity_field"].append(max_estimatable_degree_gravity_field)
        parameters_of_interest["formal_error_love_number"].append(formal_error_love_number)
        parameters_of_interest["formal_error_libration_amplitude"].append(formal_error_libration_amplitude)
        parameters_of_interest["formal_error_pole_position"].append(formal_error_pole_position)
        parameters_of_interest["formal_error_pole_rate"].append(formal_error_pole_rate)
        parameters_of_interest["rms_formal_error_degree_2"].append(rms_formal_error_degree_2)

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

    nb_parameters_of_interest = len(parameters_of_interest.keys())

    for i in range(nb_parameters_of_interest):
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(1, 1, 1)
        parameter_key = list(parameters_of_interest.keys())[i]
        if parameter_key == "formal_error_love_number" or parameter_key == "formal_error_pole_position" or parameter_key == "formal_error_pole_rate":
            for j in range(no_configurations):
                ax.scatter(j, parameters_of_interest[parameter_key][j][0], color="blue")
                ax.scatter(j, parameters_of_interest[parameter_key][j][1], color="red")
        else:
            ax.scatter(np.arange(0, no_configurations, 1), parameters_of_interest[parameter_key], color="black")
        if parameter_key == "formal_error_pole_position" or parameter_key == "formal_error_pole_rate":
            ax.legend(handles=[RA_handle, DE_handle], fontsize=fontsize)
        elif parameter_key == "formal_error_love_number":
            ax.legend(handles=[real_part_handle, imaginary_part_handle], fontsize=fontsize)
        ax.set_xlabel("Configuration  [-]", fontsize=fontsize)
        ax.set_ylabel(parameters_of_interest_axis_labels[parameter_key], fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=10, rotation=90)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.grid(True, which="both")
        fig.tight_layout()
        if (parameter_key == "formal_error_libration_amplitude" or parameter_key == "formal_error_pole_position" or
                parameter_key == "formal_error_love_number" or parameter_key == "formal_error_pole_rate"):
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(input_path, f"summary_{parameter_key}.pdf"))
        plt.close(fig)


def plot_lander_location_analysis(input_path, fontsize=12):

    # Load latitudes and longitudes range
    latitudes_range = np.rad2deg(np.loadtxt(os.path.join(input_path, "latitudes_range.txt")))
    longitudes_range = np.rad2deg(np.loadtxt(os.path.join(input_path, "longitudes_range.txt")))

    # Prepare storage of results
    max_estimatable_degree_gravity_field_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    formal_error_love_number_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    formal_error_libration_amplitude_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    formal_error_pole_RA_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    formal_error_pole_DE_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    rms_formal_error_degree_2_cosine_store = np.zeros((len(latitudes_range), len(longitudes_range)))
    formal_error_radial_love_number_store = np.zeros((len(latitudes_range), len(longitudes_range)))

    parameters_of_interest_store = dict()
    parameters_of_interest_store_labels = dict(
        max_estimatable_degree_gravity_field="Max estimatable gravity degree",
        formal_error_love_number="Formal error love number",
        formal_error_libration_amplitude="Formal error libration amplitude",
        formal_error_pole_RA="Formal error pole RA",
        formal_error_pole_DE="Formal error pole DE",
        rms_formal_error_degree_2="RMS formal error gravity degree 2",
        formal_error_radial_love_number=r"Formal error $h_{2}$ love number",
    )

    for i in range(len(latitudes_range)):
        latitude_case_path = os.path.join(input_path, f"latitude_case_{i}")
        for j in range(len(longitudes_range)):
            longitude_case_path = os.path.join(latitude_case_path, f"longitude_case_{i}")
            covariance_results_path = os.path.join(longitude_case_path, "covariance_results")

            # Load results
            max_estimatable_degree_gravity_field = np.loadtxt(
                os.path.join(covariance_results_path, "max_estimatable_degree_gravity_field.dat")
            )
            formal_error_love_number = np.loadtxt(
                os.path.join(covariance_results_path, "formal_error_love_number.dat")
            )
            formal_error_libration_amplitude = np.loadtxt(
                os.path.join(covariance_results_path, "formal_error_libration_amplitude.dat")
            )
            formal_error_pole_position = np.loadtxt(
                os.path.join(covariance_results_path, "formal_error_pole_position.dat")
            )
            rms_formal_error_degree_2 = np.loadtxt(
                os.path.join(covariance_results_path, "rms_formal_error_degree_2.dat")
            )
            formal_error_radial_love_number = np.loadtxt(
                os.path.join(covariance_results_path, "formal_error_radial_love_number.dat")
            )

            max_estimatable_degree_gravity_field_store[i, j] = max_estimatable_degree_gravity_field
            formal_error_love_number_store[i, j] = np.sqrt(formal_error_love_number[0] ** 2 + formal_error_love_number[1] ** 2)
            formal_error_libration_amplitude_store[i, j] = formal_error_libration_amplitude
            formal_error_pole_RA_store[i, j] = formal_error_pole_position[0]
            formal_error_pole_DE_store[i, j] = formal_error_pole_position[1]
            rms_formal_error_degree_2_cosine_store[i, j] = rms_formal_error_degree_2[0]
            formal_error_radial_love_number_store[i, j] = formal_error_radial_love_number

    parameters_of_interest_store["max_estimatable_degree_gravity_field"]=(max_estimatable_degree_gravity_field_store)
    parameters_of_interest_store["formal_error_love_number"]=formal_error_love_number_store
    parameters_of_interest_store["formal_error_libration_amplitude"]=formal_error_libration_amplitude_store
    parameters_of_interest_store["formal_error_pole_RA"]=formal_error_pole_RA_store
    parameters_of_interest_store["formal_error_pole_DE"]=formal_error_pole_DE_store
    parameters_of_interest_store["rms_formal_error_degree_2"]=rms_formal_error_degree_2_cosine_store
    parameters_of_interest_store["formal_error_radial_love_number"]=formal_error_radial_love_number_store

    # Plot figures of merit
    parameters_keys = list(parameters_of_interest_store.keys())
    for i in range(len(parameters_keys)):
        parameter_key = parameters_keys[i]

        plt.imshow(parameters_of_interest_store[parameter_key], aspect='auto', interpolation='none')
        plt.colorbar()
        plt.yticks(latitudes_range, latitudes_range, fontsize=fontsize)
        plt.xticks(longitudes_range, longitudes_range, fontsize=fontsize)
        plt.xlabel("Latitude  [deg]", fontsize=fontsize)
        plt.ylabel("Longitude  [deg]", fontsize=fontsize)
        plt.title(parameters_of_interest_store_labels[parameter_key], fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(input_path, f"summary_{parameter_key}.pdf"))
        plt.close()


def plot_h2_partials_analysis(input_path, fontsize=12):

    covariance_results_path = os.path.join(input_path, "covariance_results")

    # Load results
    drL_dh2_partials = np.loadtxt(os.path.join(covariance_results_path, "drL_dh2_partials.dat"))
    dh_dh2_partials = np.loadtxt(os.path.join(covariance_results_path, "dh_dh2_partials.dat"))

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax1.scatter(dh_dh2_partials[:, 0] / constants.JULIAN_DAY, dh_dh2_partials[:, 1], marker=".")
    ax1.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax1.set_ylabel(r"$dh/dh_{2}$  [respective IS unit]", fontsize=fontsize)
    ax1.grid(True)
    ax2 = fig.add_subplot(122)
    ax2.scatter(drL_dh2_partials[:, 0] / constants.JULIAN_DAY, drL_dh2_partials[:, 1], label="x", marker=".")
    ax2.scatter(drL_dh2_partials[:, 0] / constants.JULIAN_DAY, drL_dh2_partials[:, 2], label="y", marker=".")
    ax2.scatter(drL_dh2_partials[:, 0] / constants.JULIAN_DAY, drL_dh2_partials[:, 3], label="z", marker=".")
    ax2.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax2.set_ylabel(r"$\Delta \mathbf{r}_{L} - mean(\Delta \mathbf{r}_{L})$", fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    ax2.grid(True)
    fig.savefig(os.path.join(input_path, "h2_partials.pdf"))




#######################################################################################################################
### Generate figures of merit #########################################################################################
#######################################################################################################################
def main():

    # Analyse parameters of interest varying one at a time
    plot_tuning_parameters_analysis_flag = False
    if plot_tuning_parameters_analysis_flag:
        input_directory = "./output/covariance_analysis/tuning_parameters_analysis"
        time_stamp_folder = "2025.06.02.09.40.15"
        input_path = os.path.join(input_directory, time_stamp_folder)
        plot_tuning_parameters_analysis(input_path)

    summarise_tuning_parameters_analysis_flag = False
    if summarise_tuning_parameters_analysis_flag:
        input_directory = "./output/covariance_analysis/tuning_parameters_analysis"
        time_stamp_folder = "2025.06.02.09.40.15"
        input_path = os.path.join(input_directory, time_stamp_folder)
        summarise_tuning_parameters_analysis(input_path, 14)

    plot_tuning_parameters_refinement_analysis_flag = False
    if plot_tuning_parameters_refinement_analysis_flag:
        input_directory = "./output/covariance_analysis/tuning_parameters_refinement_analysis"
        no_configurations = 36
        plot_tuning_parameters_refinement_analysis(input_directory,
                                                   no_configurations)

    plot_lander_location_analysis_flag = False
    if plot_lander_location_analysis_flag:
        input_directory = "./output/covariance_analysis/lander_location_analysis"
        time_stamp_folder = "2025.06.17.17.08.38"
        input_path = os.path.join(input_directory, time_stamp_folder)
        plot_lander_location_analysis(input_path)

    plot_h2_love_number_partials_flag = True
    if plot_h2_love_number_partials_flag:
        input_directory = "./output/covariance_analysis/single_case_analysis"
        time_stamp_folder = "2025.06.20.16.31.24"
        input_path = os.path.join(input_directory, time_stamp_folder)
        plot_h2_partials_analysis(input_path)


if __name__ == "__main__":
    main()
