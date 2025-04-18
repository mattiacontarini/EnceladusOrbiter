
# Tudat import
from tudatpy import constants

# General import
import numpy as np
import matplotlib.pyplot as plt
import os


def study_propagation_error(input_path,
                            output_path,
                            nb_arcs,
                            fontsize=12):

    # Plot state and acceleration difference over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)

    for i in range(nb_arcs):
        state_difference_history = np.loadtxt(input_path + f"/state_difference_history_arc_{i}.dat")
        acceleration_difference_history = np.loadtxt(input_path + f"/acceleration_difference_norm_history_arc_{i}.dat")
        ax1.plot(state_difference_history[1:, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(state_difference_history[1:, 1:4], axis=1))
        ax2.plot(acceleration_difference_history[:, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(acceleration_difference_history[:, 1:4], axis=1))

    ax1.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax1.set_ylabel(r"$|| \Delta \mathbf{r} (t)||$  [m]", fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax2.set_ylabel(r"$|| \Delta \mathbf{a} (t)||$  [m s$^{-2}$]", fontsize=fontsize)
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.set_yscale("log")
    fig.suptitle("Perturbed vs base case")
    plot_output_path = os.path.join(output_path, "arcs_position_acceleration_difference.pdf")
    plt.savefig(plot_output_path)
    plt.close()


def study_delta_v_correction(input_path,
                             output_path,
                             fontsize=12
                             ):

    # Plot initial velocity correction
    initial_velocity_correction_list = np.loadtxt(input_path + "/multi_arc_initial_velocity_correction.dat")
    nb_arcs = len(initial_velocity_correction_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(9, 5))
    ax1.plot(np.arange(1, nb_arcs + 1, 1), initial_velocity_correction_list, color="black", alpha=0.5)
    ax1.set_xlabel("Arc index  [-]", fontsize=fontsize)
    ax1.set_ylabel(r"Velocity correction norm  [m s$^{-1}$]", fontsize=fontsize)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-5)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.grid(True)

    # Plot position difference history after correction
    for i in range(nb_arcs):
        color = (np.random.random(), np.random.random(), np.random.random())

        ax1.scatter(i+1, initial_velocity_correction_list[i], marker='o', color=color)

        state_difference_history_current_arc = np.loadtxt(input_path + f"/state_difference_history_arc_{i}.dat")
        ax2.plot(state_difference_history_current_arc[1:, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(state_difference_history_current_arc[1:, 1:4], axis=1), color=color)
        ax2.scatter(state_difference_history_current_arc[-1, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(state_difference_history_current_arc[-1, 1:4]), marker='o', color=color)

    ax2.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax2.set_ylabel(r"$||\Delta \mathbf{r}(t)||$  [m]", fontsize=fontsize)
    ax2.set_yscale("log")
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.grid(True)

    plot_output_path = os.path.join(output_path, "initial_velocity_correction.pdf")
    plt.savefig(plot_output_path)
    plt.close()


def main():

    flag_study_propagation_error = False
    if flag_study_propagation_error:
        input_path = "./output/arc_wise_propagation_error/2025.04.18.11.35.31/arc_duration_1.0_days/simulation_results"
        output_path = input_path
        study_propagation_error(input_path,
                                output_path,
                                nb_arcs=28)

    flag_study_delta_v_correction = True
    if flag_study_delta_v_correction:
        input_path = "./output/arc_wise_propagation_error/2025.04.18.11.35.31/arc_duration_1.0_days/delta_v_correction"
        output_path = input_path
        study_delta_v_correction(input_path,
                                 output_path)


if __name__ == "__main__":
    main()