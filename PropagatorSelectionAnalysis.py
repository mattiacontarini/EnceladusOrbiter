"""
Given a stable orbit solution with the dynamical model used by Benedikter et al.,
the integrator and dynamical model are selected.
"""

# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import utilities as Util

# Tudat import
from tudatpy.astro import element_conversion
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy import constants

# Packages import
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os


def perform_single_propagation_example(initial_cartesian_state, orbit_ID):
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define output folder
    output_folder = "./output/propagator_selection"

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = ["./kernels/de438.bsp", "./kernels/sat427.bsp"]
    spice.load_standard_kernels(kernels_to_load)

    # Define propagator object
    UDP = OrbitPropagator.from_config()

    # Integrate orbit
    [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_cartesian_state)

    # Save propagation setup
    propagation_setup = Util.compile_propagation_setup()
    Util.save_propagation_setup(propagation_setup=propagation_setup,
                                output_folder=output_folder,
                                time_stamp=time_stamp)

    # Save results
    Util.save_results(state_history=state_history,
                      dependent_variables_history=dependent_variable_history,
                      output_folder=output_folder,
                      time_stamp=time_stamp)

    # Plot trajectory
    Util.plot_trajectory(state_history=state_history,
                         output_folder=output_folder,
                         time_stamp=time_stamp,
                         orbit_ID=orbit_ID,
                         color=["red"])


def perform_integrators_selection(initial_cartesian_state,
                                  fixed_step_sizes,
                                  fixed_step_integrator_coefficients,
                                  fixed_step_integrator_coefficients_name):

    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define output folder
    output_folder = f"./output/propagator_selection/{time_stamp}"

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = ["./kernels/de438.bsp", "./kernels/sat427.bsp"]
    spice.load_standard_kernels(kernels_to_load)


    for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:
        coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
        coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]

        for fixed_step_size in fixed_step_sizes:
            # Define propagator object
            UDP = OrbitPropagator.from_config()

            # Compute state history and dependent variable history for the two benchmarks
            benchmarks_state_dependent_variable_history = Util.generate_benchmarks(initial_cartesian_state,
                                                                                   fixed_step_size,
                                                                                   fixed_step_integrator_coefficient,
                                                                                   coefficient_set_name,
                                                                                   UDP,
                                                                                   output_folder
                                                                                   )
            first_benchmark_state_history = benchmarks_state_dependent_variable_history[0]
            second_benchmark_state_history = benchmarks_state_dependent_variable_history[1]

            # Compute integration error of first benchmark, assuming truncation error is dominant
            first_benchmark_state_history_difference = Util.compute_benchmarks_state_history_difference(
                first_benchmark_state_history,
                second_benchmark_state_history,
                fixed_step_size,
                coefficient_set_name,
                output_folder
            )

            first_benchmark_integration_error = Util.compute_integration_error(
                first_benchmark_state_history_difference,
                fixed_step_size,
                coefficient_set_name,
                output_folder)


#######################################################################################################################
###
#######################################################################################################################

def main():
    flag_perform_single_propagation_example = False
    if flag_perform_single_propagation_example:

        # Retrieve initial state
        initial_cartesian_state = Benedikter.K1_initial_cartesian_state

        if initial_cartesian_state.all() == Benedikter.K1_initial_cartesian_state.all():
            orbit_ID = "K1' Enceladus orbit"
        elif initial_cartesian_state.all() == Benedikter.K2_initial_cartesian_state.all():
            orbit_ID = "K2' Enceladus orbit"
        elif initial_cartesian_state.all() == Benedikter.K3_initial_cartesian_state.all():
            orbit_ID = "K3' Enceladus orbit"

        perform_single_propagation_example(initial_cartesian_state, orbit_ID)

    flag_perform_integrators_selection = True
    if flag_perform_integrators_selection:

        # Select whether the orbit solution ought to be propagated or not
        propagate_orbits_flag = True

        # Retrieve initial state
        initial_cartesian_state = Benedikter.K1_initial_cartesian_state

        # Select time steps for fixed step size integrator
        fixed_step_sizes = [2.5, 5, 10, 15, 20, 25, 30, 40]

        # Select coefficient sets for fixed step size integrator
        fixed_step_integrator_coefficients = [numerical_simulation.propagation_setup.integrator.CoefficientSets.rk_4,
                                              numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_45,
                                              numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56,
                                              numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_78,
                                              numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_89,
                                              numerical_simulation.propagation_setup.integrator.CoefficientSets.rkdp_87]

        # Names of coefficient sets
        fixed_step_integrator_coefficients_name = ["RK4",
                                                   "RKF4(5)",
                                                   "RKF5(6)",
                                                   "RKF7(8)",
                                                   "RKF8(9)",
                                                   "RKDP8(7)"]

        perform_integrators_selection(initial_cartesian_state=initial_cartesian_state,
                                      fixed_step_sizes=fixed_step_sizes,
                                      fixed_step_integrator_coefficients=fixed_step_integrator_coefficients,
                                      fixed_step_integrator_coefficients_name=fixed_step_integrator_coefficients_name,
                                      )


if __name__ == "__main__":
    main()
