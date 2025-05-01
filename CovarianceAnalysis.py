#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################
# Files and variables import
from CovarianceAnalysisObject import CovarianceAnalysis

# Tudat import
from tudatpy.kernel.interface import spice
from tudatpy import constants

# Packages import
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import time

def full_parameters_spectrum_analysis(time_stamp,
                                      save_simulation_results_flag,
                                      save_covariance_results_flag,
                                      fontsize=12):

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp"
    ]
    spice.load_standard_kernels(kernels_to_load)

    # Set output path
    output_directory = "./output/covariance_analysis/parameters_spectrum_analysis"
    output_directory = os.path.join(output_directory, time_stamp)
    os.makedirs(output_directory, exist_ok=True)

    # Initialize covariance analysis object
    UDP = CovarianceAnalysis.from_config()

    # Set flag for saving results
    UDP.save_simulation_results_flag = save_simulation_results_flag
    UDP.save_covariance_results_flag = save_covariance_results_flag

    # Set initial states to consider
    initial_state_indices = [1, 2, 3]

    # Set list of simulation durations to consider
    simulation_durations = [28.0 * constants.JULIAN_DAY,
                            60.0 * constants.JULIAN_DAY,
                            180.0 * constants.JULIAN_DAY,
                            1.0 * constants.JULIAN_YEAR]

    # Set list of arc durations to consider
    arc_durations = [1.0 * constants.JULIAN_DAY,
                     2.0 * constants.JULIAN_DAY,
                     7.0 * constants.JULIAN_DAY]

    # Set list of values for the Kaula multiplier for a priori constraint on standard deviation
    kaula_constraint_multipliers = [1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of values for the a priori constraint on the empirical accelerations
    a_priori_empirical_accelerations = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of value for the a priori constraint on the landers position
    a_priori_lander_position = [1e2, 1e3]

    configuration_index = 0

    # Perform covariance analysis with all parameters
    for initial_state_index in initial_state_indices:
        UDP.initial_state_index = initial_state_index
        for simulation_duration in simulation_durations:
            UDP.simulation_duration = simulation_duration
            for arc_duration in arc_durations:
                UDP.arc_duration = arc_duration
                for kaula_constraint_multiplier in kaula_constraint_multipliers:
                    UDP.kaula_constraint_multiplier = kaula_constraint_multiplier
                    for a_priori_empirical_accelerations_current in a_priori_empirical_accelerations:
                        UDP.a_priori_empirical_accelerations = a_priori_empirical_accelerations_current
                        for a_priori_lander_position_current in a_priori_lander_position:
                            UDP.a_priori_lander_position = a_priori_lander_position_current

                            # Create output path for results of current problem configuration
                            output_path = os.path.join(output_directory, f"configuration_no_{configuration_index}")
                            os.makedirs(output_path, exist_ok=True)

                            # Perform covariance analysis
                            UDP.save_problem_configuration(output_path)
                            UDP.perform_covariance_analysis(output_path)

                            # Update configuration index
                            configuration_index += 1

    # Create plots for figures of merit
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    nb_configurations = configuration_index
    for configuration_index in range(nb_configurations):

        # Define input path
        input_path = os.path.join(output_directory, f"configuration_no_{configuration_index}")

        # Load figures of merit
        condition_number_covariance_matrix = np.loadtxt(os.path.join(input_path, "condition_number_covariance_matrix.dat"))
        max_estimatable_degree_gravity_field = np.loadtxt(
            os.path.join(input_path, "max_estimatable_degree_gravity_field.dat"))
        formal_error_initial_position_interval = np.loadtxt(
            os.path.join(input_path, "formal_error_initial_position_interval.dat"))

        # Make plots
        axes[0, 0].scatter(configuration_index, condition_number_covariance_matrix, color="black")
        axes[0, 1].scatter(configuration_index, max_estimatable_degree_gravity_field, color="black")
        axes[1, 0].scatter(configuration_index, formal_error_initial_position_interval[0], color="blue")
        axes[1, 0].scatter(configuration_index, formal_error_initial_position_interval[1], color="red")
    plt.delaxes(axes[1, 1])
    axes[1, 0].set_xlabel("Configuration idx  [-]", fontsize=fontsize)
    axes[0, 1].set_xlabel("Configuration idx  [-]", fontsize=fontsize)
    axes[0, 0].set_ylabel("Condition number cov. matrix  [-]", fontsize=fontsize)
    axes[0, 1].set_ylabel("Max estimatable degree gravity field  [-]", fontsize=fontsize)
    axes[1, 0].set_ylabel("Formal error initial position  [m]", fontsize=fontsize)
    axes[1, 0].legend(fontsize=fontsize)

    for ax in axes:
        ax.tick_params(labelsize=fontsize)

    figure_filepath = os.path.join(output_directory, "figures_of_merit.pdf")
    fig.savefig(fname=figure_filepath)
    plt.close(fig)


def perform_tuning_parameters_analysis(time_stamp,
                                       save_simulation_results_flag,
                                       save_covariance_results_flag,
                                       fontsize=12):

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp"
    ]
    spice.load_standard_kernels(kernels_to_load)

    # Set output path
    output_folder = "./output/covariance_analysis/tuning_parameters_analysis"
    output_folder = os.path.join(output_folder, time_stamp)
    os.makedirs(output_folder, exist_ok=True)

    # Set initial states to consider
    initial_state_indices = [1, 2, 3]

    # Set list of simulation durations to consider
    simulation_durations = [28.0 * constants.JULIAN_DAY,
                            60.0 * constants.JULIAN_DAY,
                            180.0 * constants.JULIAN_DAY,
                            1.0 * constants.JULIAN_YEAR]

    # Set list of arc durations to consider
    arc_durations = [1.0 * constants.JULIAN_DAY,
                     2.0 * constants.JULIAN_DAY,
                     7.0 * constants.JULIAN_DAY]

    # Set list of values for the Kaula multiplier for a priori constraint on standard deviation
    kaula_constraint_multipliers = [1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of values for the a priori constraint on the empirical accelerations
    a_priori_empirical_accelerations = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    # Set list of value for the a priori constraint on the landers position
    a_priori_lander_position = [1e2, 1e3]

    parameters_to_tune = {
        "arc_duration": arc_durations,
        "simulation_duration": simulation_durations,
        "kaula_constraint_multiplier": kaula_constraint_multipliers,
        "a_priori_empirical_acceleration": a_priori_empirical_accelerations,
        "a_priori_lander_position": a_priori_lander_position
    }

    parameters_to_tune_axis_labels = {
        "simulation_duration": "Simulation duration  [days]",
        "arc_duration": "Arc duration  [days]",
        "kaula_constraint_multiplier": "Kaula constraint multiplier  [-]",
        "a_priori_empirical_acceleration": r"A priori empirical acceleration [m s$^{-2}$]",
        "a_priori_lander_position": r"A priori lander position [m]",
    }

    # Perform covariance analysis varying one parameter singularly
    for parameter_key in list(parameters_to_tune.keys()):
        print(f"Analysing parameter {parameter_key}")

        output_path_parameter = os.path.join(output_folder, parameter_key)
        os.makedirs(output_path_parameter, exist_ok=True)

        for parameter_value in parameters_to_tune[parameter_key]:
            print(f"Running with parameter value = {parameter_value}")

            # Initialize covariance analysis object
            UDP = CovarianceAnalysis.from_config()

            # Set flag for saving results
            UDP.save_simulation_results_flag = save_simulation_results_flag
            UDP.save_covariance_results_flag = save_covariance_results_flag

            parameter_value_index = parameters_to_tune[parameter_key].index(parameter_value)
            output_path = os.path.join(output_path_parameter, f"configuration_{parameter_value_index}")
            os.makedirs(output_path, exist_ok=True)
            if parameter_key == "simulation_duration":
                UDP.simulation_duration = parameter_value
            elif parameter_key == "arc_duration":
                UDP.arc_duration = parameter_value
            elif parameter_key == "kaula_constraint_multiplier":
                UDP.kaula_constraint_multiplier = parameter_value
            elif parameter_key == "a_priori_empirical_acceleration":
                UDP.a_priori_empirical_accelerations = parameter_value
            elif parameter_key == "a_priori_lander_position":
                UDP.a_priori_lander_position = parameter_value
            else:
                raise Exception("Unknown key for parameters to tune.")

            UDP.save_problem_configuration(output_path)
            UDP.perform_covariance_analysis(output_path)

            # Pause execution for 3 seconds
            time.sleep(3)

    # Analyse figures of merit
    for parameter_key in list(parameters_to_tune.keys()):

        output_path_parameter = os.path.join(output_folder, parameter_key)
        os.makedirs(output_path_parameter, exist_ok=True)

        # Create figures of merit plot for current considered parameter
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        for parameter_value in parameters_to_tune[parameter_key]:

            # Load output data
            parameter_value_index = parameters_to_tune[parameter_key].index(parameter_value)
            input_directory = f"{output_folder}/{parameter_key}/configuration_{parameter_value_index}/covariance_results"
            condition_number_covariance_matrix = np.loadtxt(os.path.join(input_directory, "condition_number_covariance_matrix.dat"))
            max_estimatable_degree_gravity_field = np.loadtxt(os.path.join(input_directory, "max_estimatable_degree_gravity_field.dat"))
            formal_error_initial_position_interval = np.loadtxt(os.path.join(input_directory, "formal_error_initial_position_interval.dat"))

            if parameter_key == "simulation_duration":
                parameter_value = parameter_value / constants.JULIAN_DAY
            elif parameter_key == "arc_duration":
                parameter_value = parameter_value / constants.JULIAN_DAY

            # Make plots
            axes[0, 0].scatter(parameter_value, condition_number_covariance_matrix, color="black")
            axes[0, 1].scatter(parameter_value, max_estimatable_degree_gravity_field, color="black")
            axes[1, 0].scatter(parameter_value, formal_error_initial_position_interval[0], color="blue", label="Min value")
            axes[1, 0].scatter(parameter_value, formal_error_initial_position_interval[1], color="orange", label="Max value")
        axes[1, 0].set_xlabel(parameters_to_tune_axis_labels[parameter_key], fontsize=fontsize)
        plt.delaxes(axes[1, 1])
        axes[0, 1].set_xlabel(parameters_to_tune_axis_labels[parameter_key], fontsize=fontsize)

        axes[0, 0].set_ylabel("Condition number cov. matrix [-]", fontsize=fontsize)
        axes[0, 1].set_ylabel("Maximum estimatable degree gravity field  [-]", fontsize=fontsize)
        axes[1, 0].set_ylabel("Formal error initial position  [-]", fontsize=fontsize)
        axes[1, 0].legend(fontsize=fontsize)
        axes[0, 0].tick_params(labelsize=fontsize)
        axes[0, 1].tick_params(labelsize=fontsize)
        axes[1, 0].tick_params(labelsize=fontsize)

        figure_filename = os.path.join(output_path_parameter, "figures_of_merit.pdf")
        fig.savefig(fname=figure_filename)
        plt.close(fig)


def single_case_analysis(time_stamp,
                         save_simulation_results_flag,
                         save_covariance_results_flag):
    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp"
    ]
    spice.load_standard_kernels(kernels_to_load)

    # Set output path
    output_folder = "./output/covariance_analysis/single_case_analysis"
    output_path = os.path.join(output_folder, time_stamp)
    os.makedirs(output_path, exist_ok=True)

    # Initialize covariance analysis object
    UDP = CovarianceAnalysis.from_config()

    # Set flag for saving results
    UDP.save_simulation_results_flag = save_simulation_results_flag
    UDP.save_covariance_results_flag = save_covariance_results_flag

    # Perform covariance analysis
    UDP.perform_covariance_analysis(output_path)


def main():
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Set whether the results of the covariance analysis should be saved
    save_simulation_results_flag = False
    save_covariance_results_flag = False

    perform_full_parameters_spectrum_analysis_flag = False
    if perform_full_parameters_spectrum_analysis_flag:
        full_parameters_spectrum_analysis(time_stamp,
                                          save_simulation_results_flag,
                                          save_covariance_results_flag)

    perform_tuning_paramaters_analysis_flag = True
    if perform_tuning_paramaters_analysis_flag:
        perform_tuning_parameters_analysis(time_stamp,
                                           save_simulation_results_flag,
                                           save_covariance_results_flag)

    perform_single_case_analysis_flag = False
    if perform_single_case_analysis_flag:
        single_case_analysis(time_stamp,
                             save_simulation_results_flag,
                             save_covariance_results_flag)


if __name__ == "__main__":
    main()
