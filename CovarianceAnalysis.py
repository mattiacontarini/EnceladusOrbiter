#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from CovarianceAnalysisObject import CovarianceAnalysis

# Tudat import
from tudatpy.kernel.interface import spice
from tudatpy import constants

# Packages import
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import sys


def tuning_parameters_spectrum_analysis():

    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Add path to compiled version of Tudat
    sys.path.insert(0,
                    "/Users/mattiacontarini/tudat-bundle/tudatpy/src/tudatpy/numerical_simulation/environment_setup/rotation_model/expose_rotation_model.cpp")

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
    output_folder = "./output/covariance_analysis/parameters_spectrum_analysis"
    output_path = os.path.join(output_folder, time_stamp)
    os.makedirs(output_path, exist_ok=True)

    # Initialize covariance analysis object
    UDP = CovarianceAnalysis.from_config()

    # Set flag for saving results
    save_results_flag = False
    UDP.save_results_flag = save_results_flag

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
                     3.0 * constants.JULIAN_DAY,
                     7.0 * constants.JULIAN_DAY]

    # Set list of values for the Kaula multiplier for a priori constraint on standard deviation
    kaula_constraint_multipliers = [1e-6, 1e-5, 1e-4, 1e-3]

    # Perform covariance analysis with all parameters
    for initial_state_index in initial_state_indices:
        UDP.initial_state_index = initial_state_index
        for simulation_duration in simulation_durations:
            UDP.simulation_duration = simulation_duration
            for arc_duration in arc_durations:
                UDP.arc_duration = arc_duration
                for kaula_constraint_multiplier in kaula_constraint_multipliers:
                    UDP.kaula_constraint_multiplier = kaula_constraint_multiplier

                    UDP.perform_covariance_analysis(output_path)


def main():

    perform_tuning_parameters_spectrum_analysis_flag = True
    if perform_tuning_parameters_spectrum_analysis_flag:
        tuning_parameters_spectrum_analysis()


if __name__ == "__main__":
    main()
