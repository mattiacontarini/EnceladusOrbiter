# Tudat import
from tudatpy import constants
from tudatpy import numerical_simulation

# General imports
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os


def main():
    flag_perform_integrator_selection = True
    if flag_perform_integrator_selection:

        # Select input folder
        input_folder = "./output/propagator_selection/2025.04.15.14.00.33/integrator_selection"

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

        # Line-styles for coefficient sets
        linestyles_coefficient_sets = ["-",
                                       ":",
                                       "-.",
                                       "--",
                                       (0, (3, 1, 1, 1, 1, 1)),
                                       (0, (5, 1))]

        # Colors for fixed time steps
        colors_step_sizes = ["red",
                             "blue",
                             "green",
                             "orange",
                             "purple",
                             "cyan",
                             "lime",
                             "fuchsia"]

        # Set fontsize
        fontsize = 12

        # Maked handles for step sizes
        step_sizes_legend_handles = []
        for i in range(len(colors_step_sizes)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle="-",
                                   color=colors_step_sizes[i],
                                   label=r"$\Delta t = $" + str(fixed_step_sizes[i]) + " s")

            step_sizes_legend_handles.append(handle)

        # Make handles for coefficients sets
        coefficient_sets_legend_handles = []
        for i in range(len(linestyles_coefficient_sets)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle=linestyles_coefficient_sets[i],
                                   color="black",
                                   label=fixed_step_integrator_coefficients_name[i])
            coefficient_sets_legend_handles.append(handle)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize=(8, 8), constrained_layout=True, nrows=3,
                                                                 ncols=2)
        for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:

            coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
            coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]
            linestyle = linestyles_coefficient_sets[coefficient_set_index]

            if coefficient_set_index == 0:
                ax = ax1
            elif coefficient_set_index == 1:
                ax = ax2
            elif coefficient_set_index == 2:
                ax = ax3
            elif coefficient_set_index == 3:
                ax = ax4
            elif coefficient_set_index == 4:
                ax = ax5
            elif coefficient_set_index == 5:
                ax = ax6

            for fixed_step_size in fixed_step_sizes:
                file_path = (input_folder + "/" + "benchmark_fixed_step_" + str(fixed_step_size) +
                             "_coefficient_set_" + coefficient_set_name + '_integration_error.dat')

                first_benchmark_integration_error_array = np.loadtxt(file_path)

                color = colors_step_sizes[fixed_step_sizes.index(fixed_step_size)]
                ax.plot(first_benchmark_integration_error_array[1:, 0] / constants.JULIAN_DAY,
                        first_benchmark_integration_error_array[1:, 1], linestyle="-", color=color)
                ax.grid(True)
                ax.set_yscale("log")
                ax.set_title(coefficient_set_name)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)

            if coefficient_set_index == 4 or coefficient_set_index == 5:
                ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
            if coefficient_set_index == 0 or coefficient_set_index == 2 or coefficient_set_index == 4:
                ax.set_ylabel(r"$\epsilon_{\mathbf{r}}$  [m]", fontsize=fontsize)

        ax6.legend(handles=step_sizes_legend_handles, loc="lower right", ncol=2, fontsize=fontsize)
        plt.savefig(input_folder + "/integration_error.pdf")
        plt.close()

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(figsize=(8, 8), constrained_layout=True, nrows=3,
                                                                 ncols=2)
        for fixed_step_integrator_coefficient in fixed_step_integrator_coefficients:

            coefficient_set_index = fixed_step_integrator_coefficients.index(fixed_step_integrator_coefficient)
            coefficient_set_name = fixed_step_integrator_coefficients_name[coefficient_set_index]
            linestyle = linestyles_coefficient_sets[coefficient_set_index]

            if coefficient_set_index == 0:
                ax = ax1
            elif coefficient_set_index == 1:
                ax = ax2
            elif coefficient_set_index == 2:
                ax = ax3
            elif coefficient_set_index == 3:
                ax = ax4
            elif coefficient_set_index == 4:
                ax = ax5
            elif coefficient_set_index == 5:
                ax = ax6

            for fixed_step_size in fixed_step_sizes:
                file_path = (input_folder + "/" + "benchmark_1_fixed_step_" + str(fixed_step_size) +
                             "_coefficient_set_" + coefficient_set_name + '_computational_time.dat')

                first_benchmark_computational_time = np.loadtxt(file_path)

                color = colors_step_sizes[fixed_step_sizes.index(fixed_step_size)]
                ax.plot(fixed_step_size,
                        first_benchmark_computational_time, linestyle=" ", marker="o", color=color)
                ax.grid(True)
                ax.set_title(coefficient_set_name)
                ax.set_ylim(bottom=0)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)

            if coefficient_set_index == 4 or coefficient_set_index == 5:
                ax.set_xlabel("Step size [s]", fontsize=fontsize)
            if coefficient_set_index == 0 or coefficient_set_index == 2 or coefficient_set_index == 4:
                ax.set_ylabel("Total computational time [s]", fontsize=fontsize)

        plt.savefig(input_folder + "/total_computational_time.pdf")
        plt.close()

    flag_perform_integrator_refinement = False
    if flag_perform_integrator_refinement:

        # Select input folder
        input_folder = "./output/propagator_selection/2025.04.15.14.00.33/integrator_refinement"

        # Select time steps for fixed step size integrator
        fixed_step_sizes = [2.5, 5, 10, 15, 20, 25, 30, 40]

        # Select coefficient sets for fixed step size integrator
        fixed_step_integrator_coefficients = [numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56]

        # Names of coefficient sets
        fixed_step_integrator_coefficients_name = ["RKF5(6)"]

        # Line-styles for coefficient sets
        linestyles_coefficient_sets = ["-"]

        # Colors for fixed time steps
        colors_step_sizes = ["red",
                             "blue",
                             "green",
                             "orange",
                             "purple",
                             "cyan",
                             "lime",
                             "fuchsia"]

        additional_acceleration_labels = dict(
            Sun=["GM", "SRP"],
            Mimas=["GM"],
            Tethys=["GM"],
            Dione=["GM"],
            Rhea=["GM"],
            Titan=["GM"]
        )

        # Maked handles for step sizes
        step_sizes_legend_handles = []
        for i in range(len(colors_step_sizes)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle="-",
                                   color=colors_step_sizes[i],
                                   label=r"$\Delta t = $" + str(fixed_step_sizes[i]) + " s")

            step_sizes_legend_handles.append(handle)

        # Make handles for coefficients sets
        coefficient_sets_legend_handles = []
        for i in range(len(linestyles_coefficient_sets)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle=linestyles_coefficient_sets[i],
                                   color="black",
                                   label=fixed_step_integrator_coefficients_name[i])
            coefficient_sets_legend_handles.append(handle)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(8, 8),
                                                                             constrained_layout=True)
        axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        counter = 0
        additional_bodies_labels = list(additional_acceleration_labels.keys())
        for body in additional_bodies_labels:
            for i in range(len(additional_acceleration_labels[body])):
                acceleration_label = additional_acceleration_labels[body][i]
                input_directory = os.path.join(input_folder, f"{body}_{acceleration_label}_case")

                ax = axes_list[counter]

                for coefficient_set_label in fixed_step_integrator_coefficients_name:
                    coefficient_set_index = fixed_step_integrator_coefficients_name.index(coefficient_set_label)
                    coefficient_set_linestyle = linestyles_coefficient_sets[coefficient_set_index]
                    for fixed_step_size in fixed_step_sizes:
                        step_size_index = fixed_step_sizes.index(fixed_step_size)
                        step_size_color = colors_step_sizes[step_size_index]

                        integration_error = np.loadtxt(
                            input_directory + f"/benchmark_fixed_step_{fixed_step_size}_coefficient_set_{coefficient_set_label}_integration_error.dat")
                        ax.plot(integration_error[1:, 0] / constants.JULIAN_DAY,
                                integration_error[1:, 1],
                                linestyle=coefficient_set_linestyle,
                                color=step_size_color)
                        ax.set_yscale("log")
                        ax.set_title(body + " " + acceleration_label + f"; {coefficient_set_label}")
                        ax.grid(True)

                        if counter % 2 == 0:
                            ax.set_ylabel(r"$\epsilon_{\mathbf{r}}$  [m]")
                        if counter == 5 or counter == 6:
                            ax.set_xlabel(r"$t - t_{0}$  [days]")

                counter += 1

        fig.delaxes(ax8)
        fig.legend(handles=step_sizes_legend_handles, loc="lower right", ncol=2, bbox_to_anchor=(0.95, 0.075), fontsize=12)
        plt.savefig(input_folder + "/integrator_refinement.pdf")
        plt.close()

    flag_check_integrator_performance = False
    if flag_check_integrator_performance:

        # Select input folder
        input_folder = "./output/propagator_selection/2025.04.16.11.02.45/check_integrator_performance"

        # Select time steps for fixed step size integrator
        fixed_step_size = 15

        # Names of coefficient sets
        fixed_step_integrator_coefficient_name = "RKF5(6)"

        # Initial states names and indices
        initial_cartesian_state_indices = [1, 2, 3]
        initial_cartesian_state_names = ["K1", "K2", "K3"]

        # Colors for initial state
        colors_initial_states = ["red", "blue", "orange"]

        # Linestyle for initial state
        linestyles_initial_state = ["-", "--", "-."]

        # Make handles for initial states
        initial_state_legend_handles = []
        for i in range(len(linestyles_initial_state)):
            handle = mlines.Line2D([],
                                   [],
                                   linestyle=linestyles_initial_state[i],
                                   color=colors_initial_states[i],
                                   label=initial_cartesian_state_names[i])
            initial_state_legend_handles.append(handle)


        fig, ax = plt.subplots(1, 1,)
        for i in initial_cartesian_state_indices:
            linestyle = linestyles_initial_state[i-1]
            color = colors_initial_states[i-1]

            input_directory = os.path.join(input_folder, f"initial_state_{i}")

            file_path = (input_directory + "/" + "benchmark_fixed_step_" + str(fixed_step_size) +
                             "_coefficient_set_" + fixed_step_integrator_coefficient_name + '_integration_error.dat')

            first_benchmark_integration_error_array = np.loadtxt(file_path)

            ax.plot(first_benchmark_integration_error_array[1:, 0] / constants.JULIAN_DAY,
                        first_benchmark_integration_error_array[1:, 1], linestyle=linestyle, color=color)
        ax.grid(True)
        ax.set_yscale("log")
        ax.set_title(fixed_step_integrator_coefficient_name + fr"; $\Delta t = {fixed_step_size}$ s")
        ax.set_ylabel(r"$\epsilon_{\mathbf{r}}$ [m]")
        ax.set_xlabel(r"$t - t_{0}$  [days]")

        ax.legend(handles=initial_state_legend_handles, loc="lower right", ncol=2)
        plt.savefig(input_folder + "/integration_performance.pdf")
        plt.close()



if __name__ == "__main__":
    main()
