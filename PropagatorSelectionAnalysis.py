"""
Given a stable orbit solution with the dynamical model used by Benedikter et al.,
the integrator and dynamical model are selected.
"""

# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import utilities as Util

# Tudat import
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.util import result2array
from tudatpy import constants

# Packages import
import datetime
import matplotlib.pyplot as plt
import os


def perform_single_propagation_example(initial_cartesian_state,
                                       orbit_ID,
                                       output_folder,
                                       fontsize=12):
    # Define propagator object
    UDP = OrbitPropagator.from_config()

    # Integrate orbit
    [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_cartesian_state)

    # Save results
    Util.save_results(state_history=state_history,
                      dependent_variables_history=dependent_variable_history,
                      output_folder=output_folder)

    # Plot trajectory
    Util.plot_trajectory(state_history=state_history,
                         output_folder=output_folder,
                         orbit_ID=orbit_ID,
                         color=["red", "blue"])

    # Plot altitude
    dependent_variable_history_array = result2array(dependent_variable_history)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(dependent_variable_history_array[:, 0] / constants.JULIAN_DAY,
            dependent_variable_history_array[:, 2])
    ax.set_ylabel("Altitude  [m]", fontsize=fontsize)
    ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax.set_title(orbit_ID, fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.5e5, top=2.8e5)
    ax.grid(which="both")
    ax.tick_params(which="both", labelsize=fontsize)
    output_path = os.path.join(output_folder, f"{orbit_ID}_altitude_history.pdf")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def perform_integrators_selection(initial_cartesian_state,
                                  fixed_step_sizes,
                                  fixed_step_integrator_coefficients,
                                  fixed_step_integrator_coefficients_name,
                                  output_folder):
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


def perform_integrator_refinement(initial_cartesian_state,
                                  fixed_step_sizes,
                                  fixed_step_integrator_coefficients,
                                  fixed_step_integrator_coefficients_name,
                                  additional_acceleration_settings_on_vehicle,
                                  additional_acceleration_labels,
                                  output_folder):
    additional_bodies_to_analyse = list(additional_acceleration_settings_on_vehicle.keys())
    for body in additional_bodies_to_analyse:
        for i in range(len(additional_acceleration_settings_on_vehicle[body])):
            additional_acceleration_setting = additional_acceleration_settings_on_vehicle[body][i]
            case = additional_acceleration_labels[body][i]

            output_directory = os.path.join(output_folder, f"{body}_{case}_case")
            os.makedirs(output_directory, exist_ok=True)

            for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:
                coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
                coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]

                for fixed_step_size in fixed_step_sizes:
                    # Define propagator object
                    UDP = OrbitPropagator.from_config()
                    UDP.acceleration_settings_on_vehicle[body] = [additional_acceleration_setting]

                    # Compute state history and dependent variable history for the two benchmarks
                    benchmarks_state_dependent_variable_history = Util.generate_benchmarks(initial_cartesian_state,
                                                                                           fixed_step_size,
                                                                                           fixed_step_integrator_coefficient,
                                                                                           coefficient_set_name,
                                                                                           UDP,
                                                                                           output_directory
                                                                                           )
                    first_benchmark_state_history = benchmarks_state_dependent_variable_history[0]
                    second_benchmark_state_history = benchmarks_state_dependent_variable_history[1]

                    # Compute integration error of first benchmark, assuming truncation error is dominant
                    first_benchmark_state_history_difference = Util.compute_benchmarks_state_history_difference(
                        first_benchmark_state_history,
                        second_benchmark_state_history,
                        fixed_step_size,
                        coefficient_set_name,
                        output_directory
                    )
                    first_benchmark_integration_error = Util.compute_integration_error(
                        first_benchmark_state_history_difference,
                        fixed_step_size,
                        coefficient_set_name,
                        output_directory)


def check_integrator_performance(initial_cartesian_states,
                                 additional_accelerations,
                                 fixed_step_sizes,
                                 fixed_step_integrator_coefficients,
                                 fixed_step_integrator_coefficients_name,
                                 output_folder):
    for j in range(len(initial_cartesian_states)):

        initial_cartesian_state = initial_cartesian_states[j]
        output_directory = os.path.join(output_folder, f"initial_state_{j + 1}")

        for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:
            coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
            coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]

            for fixed_step_size in fixed_step_sizes:
                # Define propagator object
                UDP = OrbitPropagator.from_config()

                # Update acceleration settings
                for body in list(additional_accelerations.keys()):
                    UDP.acceleration_settings_on_vehicle[body] = additional_accelerations[body]

                # Compute state history and dependent variable history for the two benchmarks
                benchmarks_state_dependent_variable_history = Util.generate_benchmarks(initial_cartesian_state,
                                                                                       fixed_step_size,
                                                                                       fixed_step_integrator_coefficient,
                                                                                       coefficient_set_name,
                                                                                       UDP,
                                                                                       output_directory
                                                                                       )
                first_benchmark_state_history = benchmarks_state_dependent_variable_history[0]
                second_benchmark_state_history = benchmarks_state_dependent_variable_history[1]

                # Compute integration error of first benchmark, assuming truncation error is dominant
                first_benchmark_state_history_difference = Util.compute_benchmarks_state_history_difference(
                    first_benchmark_state_history,
                    second_benchmark_state_history,
                    fixed_step_size,
                    coefficient_set_name,
                    output_directory
                )
                first_benchmark_integration_error = Util.compute_integration_error(
                    first_benchmark_state_history_difference,
                    fixed_step_size,
                    coefficient_set_name,
                    output_directory)


#######################################################################################################################
###
#######################################################################################################################

def main():
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define output folder
    output_folder = f"./output/propagator_selection/{time_stamp}"
    os.makedirs(output_folder, exist_ok=True)

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
                       "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp"]
    spice.load_standard_kernels(kernels_to_load)

    flag_perform_single_propagation_example = False
    if flag_perform_single_propagation_example:

        # Retrieve initial state
        initial_cartesian_state_index = 2
        if initial_cartesian_state_index == 1:
            orbit_ID = "K1"
            initial_cartesian_state = Benedikter.K1_initial_cartesian_state
        elif initial_cartesian_state_index == 2:
            orbit_ID = "K2"
            initial_cartesian_state = Benedikter.K2_initial_cartesian_state
        elif initial_cartesian_state_index == 3:
            orbit_ID = "K3"
            initial_cartesian_state = Benedikter.K3_initial_cartesian_state
        else:
            raise ValueError("Incorrect initial cartesian state index")

        perform_single_propagation_example(initial_cartesian_state, orbit_ID, output_folder, 14)

    flag_perform_integrators_selection = False
    if flag_perform_integrators_selection:
        output_directory = os.path.join(output_folder, "integrator_selection")
        os.makedirs(output_directory, exist_ok=True)

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
                                      output_folder=output_directory
                                      )

    flag_perform_integrator_refinement = True
    if flag_perform_integrator_refinement:
        # Output directory
        output_directory = os.path.join(output_folder, "integrator_refinement")
        os.makedirs(output_directory, exist_ok=True)

        # Select initial cartesian state
        initial_cartesian_state = Benedikter.K1_initial_cartesian_state

        # Select acceleration settings to assess
        additional_acceleration_settings_on_vehicle = dict(
            Sun=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity(),
                numerical_simulation.propagation_setup.acceleration.radiation_pressure()
            ],
            Mimas=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Tethys=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Dione=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Rhea=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Titan=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ]
        )

        additional_acceleration_labels = dict(
            Sun=["GM", "SRP"],
            Mimas=["GM"],
            Tethys=["GM"],
            Dione=["GM"],
            Rhea=["GM"],
            Titan=["GM"]
        )

        # Select time steps for fixed step size integrator
        fixed_step_sizes = [2.5, 5, 10, 15, 20, 25, 30, 40]

        # Select coefficient sets for fixed step size integrator
        fixed_step_integrator_coefficients = [numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56]

        # Names of coefficient sets
        fixed_step_integrator_coefficients_name = ["RKF5(6)"]

        perform_integrator_refinement(initial_cartesian_state=initial_cartesian_state,
                                      fixed_step_sizes=fixed_step_sizes,
                                      fixed_step_integrator_coefficients=fixed_step_integrator_coefficients,
                                      fixed_step_integrator_coefficients_name=fixed_step_integrator_coefficients_name,
                                      additional_acceleration_settings_on_vehicle=additional_acceleration_settings_on_vehicle,
                                      additional_acceleration_labels=additional_acceleration_labels,
                                      output_folder=output_directory)

    flag_check_integrator_performance = True
    if flag_check_integrator_performance:
        output_directory = os.path.join(output_folder, "check_integrator_performance")
        os.makedirs(output_directory, exist_ok=True)

        # Retrieve initial state
        initial_cartesian_states = [Benedikter.K1_initial_cartesian_state,
                                    Benedikter.K2_initial_cartesian_state,
                                    Benedikter.K3_initial_cartesian_state]

        # Select time steps for fixed step size integrator
        fixed_step_sizes = [15]

        # Select coefficient sets for fixed step size integrator
        fixed_step_integrator_coefficients = [numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56]

        # Names of coefficient sets
        fixed_step_integrator_coefficients_name = ["RKF5(6)"]

        # Additional perturbing accelerations
        # Select acceleration settings to assess
        additional_acceleration_settings_on_vehicle = dict(
            Sun=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity(),
                numerical_simulation.propagation_setup.acceleration.radiation_pressure()
            ],
            Mimas=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Tethys=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Dione=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Rhea=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ],
            Titan=[
                numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
            ]
        )

        check_integrator_performance(initial_cartesian_states=initial_cartesian_states,
                                     additional_accelerations=additional_acceleration_settings_on_vehicle,
                                     fixed_step_sizes=fixed_step_sizes,
                                     fixed_step_integrator_coefficients=fixed_step_integrator_coefficients,
                                     fixed_step_integrator_coefficients_name=fixed_step_integrator_coefficients_name,
                                     output_folder=output_directory
                                     )


if __name__ == "__main__":
    main()
