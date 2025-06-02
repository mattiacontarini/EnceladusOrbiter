
# Files and variables import
from OrbitPropagator import OrbitPropagator
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import utilities as Util

# Tudat import
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy import constants

# Packages import
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


def compute_orbit_difference(simulation_duration,
                             output_folder):

    # Retrieve initial states
    initial_states = [Benedikter.K1_initial_cartesian_state,
                      Benedikter.K2_initial_cartesian_state,
                      Benedikter.K3_initial_cartesian_state]

    kernels_to_load = [
        ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
         "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp"],
        ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
         "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp"]
    ]

    # Set up orbit propagator object with nominal settings
    UDP = OrbitPropagator.from_config()

    # Generate nominal orbits with considered kernels
    for j in range(len(kernels_to_load)):
        additional_kernels = kernels_to_load[j]

        # Load SPICE kernels
        spice.clear_kernels()
        spice.load_standard_kernels()
        spice.load_standard_kernels(additional_kernels)

        output_path = os.path.join(output_folder, f"kernels_case_{j}")
        os.makedirs(output_path, exist_ok=True)

        UDP.simulation_end_epoch = UDP.simulation_start_epoch + simulation_duration

        for i in range(len(initial_states)):
            initial_state = initial_states[i]

            # Propagate orbit
            [state_history, dependent_variable_history, computational_time] = UDP.retrieve_history(initial_state)

            # Save orbit to file
            save2txt(state_history,
                    f"nominal_state_history_{i + 1}.dat",
                    output_path)

            save2txt(dependent_variable_history,
                    f"nominal_dependent_variable_history_{i + 1}.dat",
                     output_path)


    # Compute position difference between the propagated orbits
    for i in range(len(initial_states)):

        state_history_1_array = np.loadtxt(os.path.join(output_folder, f"kernels_case_0/nominal_state_history_{i + 1}.dat"))
        state_history_2_array = np.loadtxt(os.path.join(output_folder, f"kernels_case_1/nominal_state_history_{i + 1}.dat"))
        dependent_variable_history_1_array = np.loadtxt(
            os.path.join(output_folder, f"kernels_case_0/nominal_dependent_variable_history_{i + 1}.dat"))
        dependent_variable_history_2_array = np.loadtxt(
            os.path.join(output_folder, f"kernels_case_1/nominal_dependent_variable_history_{i + 1}.dat"))


        state_history_1 = Util.array2dict(state_history_1_array)
        state_history_2 = Util.array2dict(state_history_2_array)
        dependent_variable_history_1 = Util.array2dict(dependent_variable_history_1_array)
        dependent_variable_history_2 = Util.array2dict(dependent_variable_history_2_array)

        state_history_difference = dict()
        dependent_variable_history_difference = dict()
        for epoch in list(state_history_1.keys()):
            state_history_difference[epoch] = state_history_2[epoch] - state_history_1[epoch]
            dependent_variable_history_difference[epoch] = dependent_variable_history_2[epoch] - dependent_variable_history_1[epoch]

        save2txt(state_history_difference,
                 f"state_history_difference_{i + 1}.dat",
                 output_folder)
        save2txt(dependent_variable_history_difference,
                 f"dependent_variable_history_difference_{i + 1}.dat",
                 output_folder)


def plot_orbit_difference(output_folder,
                          fontsize=12):

    # Plot position difference between the propagated orbits
    fig = plt.figure()
    ax = plt.subplot(111)
    state_history_difference_1 = np.loadtxt(os.path.join(output_folder, "state_history_difference_1.dat"))
    state_history_difference_2 = np.loadtxt(os.path.join(output_folder, "state_history_difference_2.dat"))
    state_history_difference_3 = np.loadtxt(os.path.join(output_folder, "state_history_difference_3.dat"))
    ax.plot(state_history_difference_1[:, 0] / constants.JULIAN_DAY,
                    np.linalg.norm(state_history_difference_1[:, 1:4], axis=1),
                    color="blue",
                    label="K1",
                    linestyle="-")
    ax.plot(state_history_difference_2[:, 0] / constants.JULIAN_DAY,
                    np.linalg.norm(state_history_difference_2[:, 1:4], axis=1),
                    color="red",
                    label="K2",
                    linestyle="--")
    ax.plot(state_history_difference_3[:, 0] / constants.JULIAN_DAY,
                    np.linalg.norm(state_history_difference_3[:, 1:4], axis=1),
                    color="lime",
                    label="K3",
                    linestyle=(0, (5, 5)))
    ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax.set_ylabel(r"$||\Delta \mathbf{r}(t)||$  [m]", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(True)
    ax.set_yscale("log")
    fig.savefig(os.path.join(output_folder, "state_history_difference.pdf"))
    plt.close(fig)

    # Plot altitude history difference
    fig = plt.figure()
    ax = plt.subplot(111)
    dependent_variable_history_difference_1 = np.loadtxt(
        os.path.join(output_folder, "dependent_variable_history_difference_1.dat"))
    dependent_variable_history_difference_2 = np.loadtxt(
        os.path.join(output_folder, "dependent_variable_history_difference_2.dat"))
    dependent_variable_history_difference_3 = np.loadtxt(
        os.path.join(output_folder, "dependent_variable_history_difference_3.dat"))
    ax.plot(dependent_variable_history_difference_2[:, 0] / constants.JULIAN_DAY,
                 dependent_variable_history_difference_2[:, 2],
                 color="red",
                 label="K2",
                 linestyle="-")
    ax.plot(dependent_variable_history_difference_3[:, 0] / constants.JULIAN_DAY,
                 dependent_variable_history_difference_3[:, 2],
                 color="lime",
                 label="K3",
                 linestyle="-")
    ax.plot(dependent_variable_history_difference_1[:, 0] / constants.JULIAN_DAY,
                 dependent_variable_history_difference_1[:, 2],
                 color="blue",
                 label="K1",
                 linestyle="-")
    ax.set_xlabel(r"$t - t_{0}$  [days]", fontsize=fontsize)
    ax.set_ylabel(r"$\Delta h(t)$  [m]", fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(True)
    fig.savefig(os.path.join(output_folder, "altitude_history_difference.pdf"))
    plt.close(fig)


def main():

    # Define output file
    output_folder = "./output/ephemeris_model_difference"

    # # Retrieve current time stamp
    # time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    # output_folder = os.path.join(output_folder, time_stamp)
    os.makedirs(output_folder, exist_ok=True)

    # Compute orbit difference
    compute_orbit_difference_flag = True
    if compute_orbit_difference_flag:
        simulation_duration = 28.0 * constants.JULIAN_DAY
        compute_orbit_difference(simulation_duration,
                                 output_folder)

    # Plot orbit difference
    plot_orbit_difference_flag = True
    if plot_orbit_difference_flag:
        plot_orbit_difference(output_folder)


if __name__ == "__main__":
    main()