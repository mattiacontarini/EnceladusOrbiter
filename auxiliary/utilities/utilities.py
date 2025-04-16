"""
File used to store utilities of general application.
"""

# Files and variables import
from auxiliary import OrbitPropagatorConfig as PropConfig

# Tudat import
from tudatpy.data import save2txt
from tudatpy import numerical_simulation
from tudatpy import constants

# Packages import
import os
import numpy as np
import matplotlib.pyplot as plt


def save_results(state_history,
                 dependent_variables_history,
                 output_folder,
                 time_stamp):
    os.makedirs(output_folder, exist_ok=True)

    results_folder = output_folder + "/" + time_stamp
    os.makedirs(results_folder, exist_ok=True)

    save2txt(state_history,
             "state_history.dat",
             results_folder)
    save2txt(dependent_variables_history,
             "dependent_variables_history.dat",
             results_folder)


def save_plot(plot_object,
              output_folder,
              time_stamp):
    os.makedirs(output_folder, exist_ok=True)

    results_folder = output_folder + "/" + time_stamp
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(results_folder + "/trajectory_3d.png")
    plt.close()


def compile_propagation_setup():
    propagation_setup = dict()
    propagation_setup["simulation_start_epoch"] = PropConfig.simulation_start_epoch
    propagation_setup["simulation_end_epoch"] = PropConfig.simulation_end_epoch
    propagation_setup["simulation_duration_days"] = PropConfig.simulation_duration / constants.JULIAN_DAY

    bodies_acceleration = PropConfig.acceleration_settings_on_vehicle.keys()
    for body in bodies_acceleration:
        propagation_setup[body] = PropConfig.acceleration_settings_on_vehicle[body]

    return propagation_setup


def save_propagation_setup(propagation_setup,
                           output_folder,
                           time_stamp):
    os.makedirs(output_folder, exist_ok=True)

    results_folder = output_folder + "/" + time_stamp
    os.makedirs(results_folder, exist_ok=True)
    save2txt(propagation_setup, "propagation_setup.dat", results_folder)


def save_population(population, index, output_path):
    IDs = np.atleast_2d(population.get_ID()).T
    individuals = population.get_x()
    fitness = population.get_f()

    population = np.hstack((IDs, individuals, fitness))
    file_path = output_path + "/populations/population_{index}.txt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    np.savetxt(file_path, population)


def generate_benchmarks(initial_state,
                        benchmark_step_size,
                        coefficient_set,
                        coefficient_set_name,
                        UDP,
                        output_path,
                        are_dependent_variables_present=True
                        ):

    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size
    second_benchmark_step_size = first_benchmark_step_size / 2

    # Create integrator settings for the first benchmark, using a fixed step size integrator
    first_benchmark_integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=first_benchmark_step_size,
        coefficient_set=coefficient_set
    )
    UDP.integrator_settings = first_benchmark_integrator_settings

    # Retrieve state and dependent variable history for the first benchmark
    [first_benchmark_state_history, first_benchmark_dependent_variable_history, first_benchmark_computational_time] = UDP.retrieve_history(initial_state)

    # Create integrator settings for the second benchmark in the same way
    second_benchmark_integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=second_benchmark_step_size,
        coefficient_set=coefficient_set
    )
    UDP.integrator_settings = second_benchmark_integrator_settings

    # Retrieve state and dependent variable history for the first benchmark
    [second_benchmark_state_history, second_benchmark_dependent_variable_history, second_benchmark_computational_time] = UDP.retrieve_history(initial_state)

    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_state_history,
                 'benchmark_1_fixed_step_' + str(first_benchmark_step_size) + "_coefficient_set_" +
                 coefficient_set_name + '_state_history.dat',
                 output_path)
        save2txt(second_benchmark_state_history,
                 'benchmark_2_fixed_step_' + str(second_benchmark_step_size) + "_coefficient_set_" +
                 coefficient_set_name + '_state_history.dat',
                 output_path)

    # Add items to be returned
    return_list = [first_benchmark_state_history, second_benchmark_state_history, first_benchmark_computational_time,
                   second_benchmark_computational_time]

    if are_dependent_variables_present:

        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable_history,
                     'benchmark_1_fixed_step_' + str(first_benchmark_step_size) + "_coefficient_set_" +
                     coefficient_set_name + '_dependent_variables.dat',
                     output_path)
            save2txt(second_benchmark_dependent_variable_history,
                     'benchmark_2_fixed_step_' + str(second_benchmark_step_size) + "_coefficient_set_" +
                     coefficient_set_name + '_dependent_variables.dat',
                     output_path)

        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable_history)
        return_list.append(second_benchmark_dependent_variable_history)

    return return_list


def compute_benchmarks_state_history_difference(first_benchmark_state_history,
                                                second_benchmark_state_history,
                                                filename,
                                                output_path):

    state_history_difference = dict()

    first_benchmark_epochs = list(first_benchmark_state_history.keys())

    for epoch in first_benchmark_epochs[:-1]:
        state_history_difference[epoch] = first_benchmark_state_history[epoch] - second_benchmark_state_history[epoch]

    save2txt(state_history_difference,
             filename,
             output_path)

    return state_history_difference


def compute_integration_error(state_history_difference,
                              first_benchmark_step_size,
                              coefficient_set_name,
                              output_path):

    epochs = list(state_history_difference.keys())
    integration_error = dict()

    for epoch in epochs:
        integration_error[epoch] = np.linalg.norm(state_history_difference[epoch][:3])

    save2txt(integration_error,
             "benchmark_fixed_step_" + str(first_benchmark_step_size) + "_coefficient_set_" +
             coefficient_set_name + '_integration_error.dat',
             output_path)

    return integration_error


def array2dict(array):
    dim = array.shape

    dictionary = dict()
    for i in range(dim[0]):
        key = array[i, 0]
        dictionary[key] = array[i, 1:]

    return dictionary
