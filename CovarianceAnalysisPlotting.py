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
                            60.0 * constants.JULIAN_DAY]

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
        "lander_to_include": lander_to_include,
    }

    colors_rsw_interval = ["blue", "green", "red"]
    markers_rsw_interval = ["o", "*", "X"]

    parameters_to_tune_label = {
        "initial_state_index": "K- orbit index  [-]",
        "arc_duration": "Arc duration  [days]",
        "simulation_duration": "Simulation duration  [days]",
        "kaula_constraint_multiplier": "Kaula constraint multiplier  [-]",
        "a_priori_empirical_acceleration": r"A priori $\sigma$ empirical acc.  [m/s$^2$]",
        "a_priori_lander_position": r"A priori $\sigma$ lander position  [m]",
        "include_lander_range_observable_flag": "Lander range observable inclusion flag  [-]",
        "empirical_accelerations_arc_duration": "Empirical acc. arc duration  [hours]",
        "tracking_arc_duration": "Tracking arc duration  [hours]",
        "lander_to_include": "Number of landers  [-]",
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
        color=colors_rsw_interval[0],
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


    for parameter_key in list(parameters_to_tune.keys()):
        input_path_parameter = os.path.join(input_path, parameter_key)

        # Create plot of figures of merit for current considered parameter
        fig, axes = plt.subplots(2, 2, constrained_layout=True)
        for parameter_value in parameters_to_tune[parameter_key]:
            parameter_value_index = parameters_to_tune[parameter_key].index(parameter_value)
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

            # Plot results
            axes[0, 0].scatter(parameter_value, condition_number_covariance_matrix, color="blue")
            axes[0, 1].scatter(parameter_value, max_estimatable_degree_gravity_field, color="blue")
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

        axes[0, 0].set_ylabel("Condition number cov. matrix  [-]", fontsize=fontsize)
        axes[0, 1].set_ylabel("Maximum estimatable degree gravity field [-]", fontsize=fontsize)
        axes[1, 0].set_ylabel("Formal error initial position  [m]", fontsize=fontsize)
        axes[1, 0].set_xlabel(parameters_to_tune_label[parameter_key], fontsize=fontsize)
        axes[1, 0].legend(handles=[rsw_interval_min_value_handle,
                                   rsw_interval_median_value_handle,
                                   rsw_interval_max_value_handle,
                                   rsw_interval_radial_direction_handle,
                                   rsw_interval_along_track_direction_handle,
                                   rsw_interval_cross_track_direction_handle],
                          fontsize=fontsize)
        axes[1, 1].set_ylabel("Formal error empirical accelerations  [m/s$^{2}$]", fontsize=fontsize)
        axes[1, 1].set_xlabel(parameters_to_tune_label[parameter_key], fontsize=fontsize)
        axes.grid(True, which="both")
        axes.tick_params(labelsize=fontsize)
        fig.savefig(os.path.join(input_path_parameter, "figures_of_merit.pdf"))
        plt.close(fig)



#######################################################################################################################
### Generate figures of merit #########################################################################################
#######################################################################################################################
def main():

    # Analyse parameters of interest varying one at a time
    plot_tuning_parameters_analysis_flag = False
    if plot_tuning_parameters_analysis_flag:
        input_directory = "output/covariance_analysis/tuning_parameters_analysis"
        time_stamp_folder = "2025.05.12.08.39.55"
        input_path = os.path.join(input_directory, time_stamp_folder)
        plot_tuning_parameters_analysis(input_path)



























