"""
File used to store utilities of general application.
"""

# Files and variables import
from auxiliary import OrbitPropagatorConfig as PropConfig

# Tudat import
from tudatpy.data import save2txt
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import gravitation
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy import plotting

# Packages import
import os
import numpy as np
import matplotlib.pyplot as plt


def save_results(state_history,
                 dependent_variables_history,
                 output_folder):

    save2txt(state_history,
             "state_history.dat",
             output_folder)
    save2txt(dependent_variables_history,
             "dependent_variables_history.dat",
             output_folder)


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
                           output_folder):
    os.makedirs(output_folder, exist_ok=True)

    save2txt(propagation_setup, "propagation_setup.dat", output_folder)


def plot_trajectory(state_history,
                    output_folder,
                    orbit_ID,
                    color):
    fig, ax = plotting.trajectory_3d(state_history,
                                     ["Vehicle"],
                                     "Enceladus",
                                     [],
                                     "J2000",
                                     True,
                                     color
                                     )

    Enceladus_radius = spice.get_average_radius("Enceladus")

    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), Enceladus_radius * np.sin(v))
    y = np.outer(np.sin(u), Enceladus_radius * np.sin(v))
    z = np.outer(np.ones(np.size(u)), Enceladus_radius * np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)
    ax.set_title(orbit_ID)
    plt.tight_layout()

    plt.savefig(output_folder + f"/{orbit_ID}_trajectory_3d.pdf")
    plt.close()


def get_gravity_field_settings_enceladus_park():
    enceladus_gravitational_parameter = 7.210366688598896E+9

    enceladus_reference_radius = 256600.0  # m

    enceladus_unnormalized_cosine_coeffs = np.array([  # Park et al.
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [-5477.45E-06, 7.86E-06, 1517.9E-06, 0],
        [177.82E-06, 0, 0, 0]
    ])

    enceladus_unnormalized_sine_coeffs = np.array([  # Park et al.
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 7.6E-06, -275.31E-06, 0],  # 2 0 - 2 1 - 2 2
        [0, 0, 0, 0]
    ])

    enceladus_normalized_coeffs = gravitation.normalize_spherical_harmonic_coefficients(
        enceladus_unnormalized_cosine_coeffs, enceladus_unnormalized_sine_coeffs
    )

    enceladus_normalized_cosine_coeffs = enceladus_normalized_coeffs[0]
    enceladus_normalized_sine_coeffs = enceladus_normalized_coeffs[1]

    enceladus_associated_reference_frame = "IAU_Enceladus"

    enceladus_gravity_field_settings = numerical_simulation.environment_setup.gravity_field.spherical_harmonic(
        enceladus_gravitational_parameter,
        enceladus_reference_radius,
        enceladus_normalized_cosine_coeffs,
        enceladus_normalized_sine_coeffs,
        enceladus_associated_reference_frame
    )

    return enceladus_gravity_field_settings


def get_synodic_rotation_model_enceladus(simulation_initial_epoch):
    saturn_gravitational_parameter = 3.793120749865224E+16
    initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus",
                                                                      "Saturn",
                                                                      "J2000",
                                                                      "None",
                                                                      simulation_initial_epoch)
    keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus,
                                                                          saturn_gravitational_parameter)
    rotation_rate_enceladus = np.sqrt(saturn_gravitational_parameter / keplerian_state_enceladus[0] ** 3)

    return rotation_rate_enceladus


def get_gravity_field_settings_saturn_iess():
    saturn_gravitational_parameter = 3.793120749865224E+16
    saturn_reference_radius = 60330000.0  # From Iess et al. (2019)
    saturn_unnormalized_cosine_coeffs = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.629061510215236E-02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [-9.519974025353707E-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
        [9.351185734877162E-04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [5.984128286091720E-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [-8.676367491774778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [-4.808382695890572E-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
        [1.393087926846997E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
        [-8.921515415583946E-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
        [-5.425691388908470E-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10

    ])
    saturn_unnormalized_sine_coeffs = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    saturn_normalized_coeffs = gravitation.normalize_spherical_harmonic_coefficients(
        saturn_unnormalized_cosine_coeffs, saturn_unnormalized_sine_coeffs
    )

    saturn_normalized_cosine_coeffs = saturn_normalized_coeffs[0]
    saturn_normalized_sine_coeffs = saturn_normalized_coeffs[1]

    saturn_associated_reference_frame = "IAU_Saturn"

    saturn_gravity_field_settings = numerical_simulation.environment_setup.gravity_field.spherical_harmonic(
        saturn_gravitational_parameter,
        saturn_reference_radius,
        saturn_normalized_cosine_coeffs,
        saturn_normalized_sine_coeffs,
        saturn_associated_reference_frame
    )

    return saturn_gravity_field_settings


def save_population(population, index, output_path):
    IDs = np.atleast_2d(population.get_ID()).T
    individuals = population.get_x()
    fitness = population.get_f()

    population = np.hstack((IDs, individuals, fitness))
    file_path = output_path + "/populations/population_{index}.txt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    np.savetxt(file_path, population)


def compute_change_of_average_fitness(fitness_list):
    average_fitness_old = compute_average_fitness_multiple_populations(
        fitness_list[: len(fitness_list) - 1]
    )
    average_fitness_new = compute_average_fitness_of_population(
        fitness_list[len(fitness_list) - 1]
    )

    fitness_change = np.abs(
        (average_fitness_new - average_fitness_old) / average_fitness_old
    )

    return fitness_change


def compute_average_fitness_multiple_populations(fitness_list):
    number_of_populations = len(fitness_list)
    average_fitness_populations = []
    for i in range(number_of_populations):
        current_average_fitness = (
            compute_average_fitness_of_population(fitness_list[i])
        )
        average_fitness_populations.append(current_average_fitness)

    average_fitness = np.mean(average_fitness_populations)
    return average_fitness


def compute_average_fitness_of_population(fitness_points):
    # Create auxiliary variable with all points

    fitness_average = np.mean(fitness_points)

    return fitness_average


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
    second_benchmark_step_size = first_benchmark_step_size / 4

    # Create integrator settings for the first benchmark, using a fixed step size integrator
    first_benchmark_integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=first_benchmark_step_size,
        coefficient_set=coefficient_set
    )
    UDP.integrator_settings = first_benchmark_integrator_settings

    # Retrieve state and dependent variable history for the first benchmark
    [first_benchmark_state_history, first_benchmark_dependent_variable_history,
     first_benchmark_computational_time] = UDP.retrieve_history(initial_state)

    # Create integrator settings for the second benchmark in the same way
    second_benchmark_integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=second_benchmark_step_size,
        coefficient_set=coefficient_set
    )
    UDP.integrator_settings = second_benchmark_integrator_settings

    # Retrieve state and dependent variable history for the first benchmark
    [second_benchmark_state_history, second_benchmark_dependent_variable_history,
     second_benchmark_computational_time] = UDP.retrieve_history(initial_state)

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
            np.savetxt(output_path + "/benchmark_1_fixed_step_" + str(first_benchmark_step_size) + "_coefficient_set_" +
                       coefficient_set_name + "_computational_time.dat",
                       [first_benchmark_computational_time])

        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable_history)
        return_list.append(second_benchmark_dependent_variable_history)

    return return_list


def compute_benchmarks_state_history_difference(first_benchmark_state_history,
                                                second_benchmark_state_history,
                                                first_benchmark_step_size,
                                                coefficient_set_name,
                                                output_path):
    state_history_difference = dict()

    first_benchmark_epochs = list(first_benchmark_state_history.keys())

    for epoch in first_benchmark_epochs[:-1]:
        state_history_difference[epoch] = first_benchmark_state_history[epoch] - second_benchmark_state_history[epoch]

    save2txt(state_history_difference,
             "benchmark_fixed_step_" + str(first_benchmark_step_size) + "_coefficient_set_" +
             coefficient_set_name + '_state_history_difference.dat',
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
