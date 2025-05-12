#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################
# Files and variables import
from CovarianceAnalysisObject import CovarianceAnalysis
from auxiliary.utilities import plotting_utilities as PlotUtil

# Tudat import
from tudatpy.kernel.interface import spice
from tudatpy import constants

# Packages import
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess

def full_parameters_spectrum_analysis(time_stamp,
                                      save_simulation_results_flag,
                                      save_covariance_results_flag):

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp",
        # "kernels/de440.bsp",
        # "kernels/sat441l.bsp"
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





def perform_tuning_parameters_analysis(time_stamp,
                                       save_simulation_results_flag,
                                       save_covariance_results_flag):

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp",
        # "kernels/de440.bsp",
        # "kernels/sat441l.bsp"
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
                            #180.0 * constants.JULIAN_DAY,
                            #1.0 * constants.JULIAN_YEAR
                            ]

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
    lander_to_include = [ [None],
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

    # Perform covariance analysis varying one parameter singularly
    for parameter_key in list(parameters_to_tune.keys()):

        output_path_parameter = os.path.join(output_folder, parameter_key)
        os.makedirs(output_path_parameter, exist_ok=True)

        for parameter_value in parameters_to_tune[parameter_key]:

            print(" ")
            print(f"Analysing parameter {parameter_key} with value = {parameter_value}")
            print(" ")

            # Initialize covariance analysis object
            UDP = CovarianceAnalysis.from_config()

            # Set flag for saving results
            UDP.save_simulation_results_flag = save_simulation_results_flag
            UDP.save_covariance_results_flag = save_covariance_results_flag

            parameter_value_index = parameters_to_tune[parameter_key].index(parameter_value)
            output_path = os.path.join(output_path_parameter, f"configuration_{parameter_value_index}")
            os.makedirs(output_path, exist_ok=True)
            if parameter_key == "initial_state_index":
                UDP.initial_state_index = parameter_value
            elif parameter_key == "simulation_duration":
                UDP.simulation_duration = parameter_value
            elif parameter_key == "arc_duration":
                UDP.arc_duration = parameter_value
            elif parameter_key == "kaula_constraint_multiplier":
                UDP.kaula_constraint_multiplier = parameter_value
            elif parameter_key == "a_priori_empirical_acceleration":
                UDP.a_priori_empirical_accelerations = parameter_value
            elif parameter_key == "a_priori_lander_position":
                UDP.a_priori_lander_position = parameter_value
            elif parameter_key == "include_lander_range_observable_flag":
                UDP.include_lander_range_observable_flag = parameter_value
            elif parameter_key == "empirical_accelerations_arc_duration":
                UDP.empirical_accelerations_arc_duration = parameter_value
            elif parameter_key == "tracking_arc_duration":
                UDP.tracking_arc_duration = parameter_value
            elif parameter_key == "lander_to_include":
                UDP.lander_to_include = parameter_value
            else:
                raise Exception("Unknown key for parameters to tune.")

            UDP.save_problem_configuration(output_path)
            UDP.perform_covariance_analysis(output_path)


def single_case_analysis(time_stamp,
                         save_simulation_results_flag,
                         save_covariance_results_flag):
    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        #    "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp",
        # "kernels/de440.bsp",
        # "kernels/sat441l.bsp"
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

    UDP.include_lander_range_observable_flag = False
    UDP.lander_to_include = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
    UDP.use_station_position_consider_parameter_flag = True

    # Perform covariance analysis
    UDP.save_problem_configuration(output_path)
    UDP.perform_covariance_analysis(output_path)


def main():
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Set whether the results of the covariance analysis should be saved
    save_simulation_results_flag = False
    save_covariance_results_flag = True

    # Analyse every combination of parameters of interest
    perform_full_parameters_spectrum_analysis_flag = False
    if perform_full_parameters_spectrum_analysis_flag:
        full_parameters_spectrum_analysis(time_stamp,
                                          save_simulation_results_flag,
                                          save_covariance_results_flag)

    # Analyse parameters of interest varying one at a time
    perform_tuning_parameters_analysis_flag = False
    if perform_tuning_parameters_analysis_flag:
        perform_tuning_parameters_analysis(time_stamp,
                                           save_simulation_results_flag,
                                           save_covariance_results_flag)

    # Perform the covariance analysis for only one base set
    perform_single_case_analysis_flag = True
    if perform_single_case_analysis_flag:
        single_case_analysis(time_stamp,
                             save_simulation_results_flag,
                             save_covariance_results_flag)

    # Plot the distribution of the landers
    plot_lander_distribution_flag = False
    if plot_lander_distribution_flag:
        output_path = "./output/covariance_analysis"
        PlotUtil.plot_lander_distribution(output_path, fontsize=14)


if __name__ == "__main__":
    main()
