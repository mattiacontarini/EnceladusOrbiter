"""
Given a stable orbit solution with the dynamical model used by Benedikter et al.,
the integrator and dynamical model are selected.
"""

# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import utilities as Util
from auxiliary.OrbitPropagatorConfig import bodies_to_create

# Tudat import
from tudatpy.astro import element_conversion
from tudatpy.interface import spice
from tudatpy import numerical_simulation

# Packages import
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


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
                                  fixed_step_integrator_coefficients_name,
                                  generate_plot_flag,
                                  linestyles_coefficient_sets,
                                  coefficient_sets_legend_handles,
                                  colors_step_sizes,
                                  step_sizes_legend_handles, ):
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define output folder
    output_folder = f"./output/propagator_selection/{time_stamp}"

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()

    # Initialize plot
    if generate_plot_flag:
        fig, ax = plt.subplots(figsize=(8, 8))

    for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:
        coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
        coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]
        linestyle = linestyles_coefficient_sets[coefficient_set_index]

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
            first_benchmark_integration_error = Util.compute_benchmarks_state_history_difference(
                first_benchmark_state_history,
                second_benchmark_state_history,
                fixed_step_size,
                coefficient_set_name
            )

            color = colors_step_sizes[fixed_step_sizes.index(fixed_step_size)]
            # Plot integration error
            if generate_plot_flag:
                ax.plot(first_benchmark_integration_error, linestyle=linestyle, color=color)

    if generate_plot_flag:
        fig.legend(handles=[coefficient_sets_legend_handles, step_sizes_legend_handles], )
        plt.tight_layout()
        plt.show()


def main():

    flag_perform_single_propagation_example = True
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

    flag_perform_integrators_selection = False
    if flag_perform_integrators_selection:

        # Retrieve initial state
        initial_cartesian_state = Benedikter.K1_initial_cartesian_state

        # Select time steps for fixed step size integrator
        fixed_step_sizes = [2.5, 5, 10, 15, 20, 30]

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

        generate_plot_flag = True

        # Line-styles for coefficient sets
        linestyles_coefficient_sets = ["-",
                                       ":",
                                       "-.",
                                       "--",
                                       (0, (3, 1, 1, 1, 1, 1)),
                                       (0, (5, 1))]

        # Make handles for coefficients sets
        coefficient_sets_legend_handles = []
        for i in range(len(linestyles_coefficient_sets)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle=linestyles_coefficient_sets[i],
                                   color="black",
                                   label=fixed_step_integrator_coefficients_name[i])
            coefficient_sets_legend_handles.append(handle)

        # Colors for fixed time steps
        colors_step_sizes = ["red",
                             "blue",
                             "green",
                             "orange",
                             "purple",
                             "cyan"]

        # Maked handles for step sizes
        step_sizes_legend_handles = []
        for i in range(len(colors_step_sizes)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle="-",
                                   color=colors_step_sizes[i],
                                   label=str(fixed_step_sizes[i]))

            step_sizes_legend_handles.append(handle)

        perform_integrators_selection(initial_cartesian_state=initial_cartesian_state,
                                      fixed_step_sizes=fixed_step_sizes,
                                      fixed_step_integrator_coefficients=fixed_step_integrator_coefficients,
                                      fixed_step_integrator_coefficients_name=fixed_step_integrator_coefficients_name,
                                      generate_plot_flag=generate_plot_flag,
                                      linestyles_coefficient_sets=linestyles_coefficient_sets,
                                      coefficient_sets_legend_handles=coefficient_sets_legend_handles,
                                      colors_step_sizes=colors_step_sizes,
                                      step_sizes_legend_handles=step_sizes_legend_handles,
                                      )


if __name__ == "__main__":
    main()
