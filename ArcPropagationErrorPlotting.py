

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

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(1, nb_arcs + 1, 1), initial_velocity_correction_list, marker='o', color="blue")
    ax.set_xlabel("Arc index  [-]", fontsize=fontsize)
    ax.set_ylabel(r"Velocity correction norm  [m s$^{-1}$]", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid(which="both")

    plot_output_path = os.path.join(output_path, "velocity_correction_norm.pdf")
    plt.savefig(plot_output_path)
    plt.close()


def main():
    input_path = "./output/arc_wise_propagation_error/2025.04.18.11.09.11/arc_duration_1.0_days/delta_v_correction"
    output_path = input_path
    study_delta_v_correction(input_path,
                            output_path)
