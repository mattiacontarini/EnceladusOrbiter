# Tudat import
from tudatpy import constants

# Packages import
import matplotlib.pyplot as plt
import numpy as np


#######################################################################################################################
###
#######################################################################################################################
def main():
    study_physical_model_error = True
    if study_physical_model_error:

        # Select input path
        input_path = "./output/accelerations_selection/2025.04.11.16.32.35"
        output_path = input_path

        # Bodies considered
        bodies_to_study = ["Sun",
                           "Mimas",
                           "Tethys",
                           "Dione",
                           "Rhea",
                           "Titan",
                           "Jupiter_barycenter",
                           "Uranus_barycenter",
                           "Neptune_barycenter",
                           "Mars_barycenter",
                           "Earth_barycenter",
                           "Venus",
                           "Mercury"]

        # Colors to use
        bodies_color_code = dict(
            Sun=["olive", "orange"],
            Enceladus=["blue"],
            Saturn=["red"],
            Mimas=["green"],
            Tethys=["pink"],
            Dione=["cyan"],
            Rhea=["gray"],
            Titan=["purple"],
            Jupiter_barycenter=["black"],
            Uranus_barycenter=["lime"],
            Neptune_barycenter=["magenta"],
            Mars_barycenter=["darkred"],
            Earth_barycenter=["slateblue"],
            Venus=["yellow"],
            Mercury=["gold"],
        )

        # Colors to use
        labels_code = dict(
            Sun="Sun",
            Enceladus="Enceladus",
            Saturn="Saturn",
            Mimas="Mimas",
            Tethys="Tethys",
            Dione="Dione",
            Rhea="Rhea",
            Titan="Titan",
            Jupiter_barycenter="Jupiter Bar.",
            Uranus_barycenter="Uranus Bar.",
            Neptune_barycenter="Neptune Bar.",
            Mars_barycenter="Mars Bar.",
            Earth_barycenter="Earth Bar.",
            Venus="Venus",
            Mercury="Mercury",
            SRP="SRP"
        )

        case_keys = dict(
            Sun=["GM", "SRP"],
            Mimas=["GM"],
            Tethys=["GM"],
            Dione=["GM"],
            Rhea=["GM"],
            Titan=["GM"],
            Jupiter_barycenter=["GM"],
            Uranus_barycenter=["GM"],
            Neptune_barycenter=["GM"],
            Mars_barycenter=["GM"],
            Earth_barycenter=["GM"],
            Venus=["GM"],
            Mercury=["GM"],
        )

        # Plot results
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        fig2, ax2 = plt.subplots(figsize=(7, 6))

        # Plot accelerations of Enceladus and Saturn
        benchmark_dependent_variable_history = np.loadtxt(input_path + f"/dependent_variable_history_benchmark.dat")
        ax1.plot(benchmark_dependent_variable_history[:, 0] / constants.JULIAN_DAY,
                 benchmark_dependent_variable_history[:, 2],
                 color=bodies_color_code["Enceladus"][0],
                 label=labels_code["Enceladus"] + " GM + SH", )
        ax1.plot(benchmark_dependent_variable_history[:, 0] / constants.JULIAN_DAY,
                 benchmark_dependent_variable_history[:, 3],
                 color=bodies_color_code["Saturn"][0],
                 label=labels_code["Saturn"] + " GM + SH", )

        for body in bodies_to_study:

            nb_accelerations_to_study = len(case_keys[body])
            for i in range(nb_accelerations_to_study):

                case = case_keys[body][i]

                dependent_variable_history = np.loadtxt(
                    input_path + f"/dependent_variable_history_{body}_case_{case}.dat")

                if body == "Sun" and i == 1:
                    srp_acceleration_epochs = []
                    srp_acceleration = []
                    for j in range(dependent_variable_history[:, 0].shape[0]):
                        if dependent_variable_history[j, 2] != 0:
                            srp_acceleration_epochs.append(dependent_variable_history[j, 0])
                            srp_acceleration.append(dependent_variable_history[j, 2])
                    ax1.plot(np.array(srp_acceleration_epochs) / constants.JULIAN_DAY,
                             srp_acceleration,
                             color=bodies_color_code[body][i],
                             label=labels_code[body] + " " + case, )
                else:
                    ax1.plot(dependent_variable_history[:, 0] / constants.JULIAN_DAY,
                             dependent_variable_history[:, 2],
                             color=bodies_color_code[body][i],
                             label=labels_code[body] + " " + case)

                state_history_difference = np.loadtxt(
                    input_path + f"/benchmark_state_history_difference_{body}_case_{case}.dat")
                ax2.plot(state_history_difference[:, 0] / constants.JULIAN_DAY,
                         np.linalg.norm(state_history_difference[:, 1:4], axis=1),
                         color=bodies_color_code[body][i],
                         label=labels_code[body] + " " + case)

        # Load state history difference for Saturn case
        state_history_difference = np.loadtxt(input_path + "/state_history_difference_study_Saturn.dat")

        ax2.plot(state_history_difference[:, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(state_history_difference[:, 1:4], axis=1),
                 color=bodies_color_code["Saturn"][0],
                 label=labels_code["Saturn"] + " SH")

        pos = ax1.get_position()
        ax1.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
        fig1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4, fancybox=True)
        ax1.set_xlabel(r"$t - t_{0}$  [days]")
        ax1.set_ylabel(r"$|| \mathbf{a}(t) ||$ [m s$^{-2}$]")
        ax1.set_yscale("log")
        ax1.grid(True)
        fig1.savefig(output_path + "/accelerations_magnitude.pdf")

        pos = ax2.get_position()
        ax2.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
        fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=4, fancybox=True)
        ax2.set_xlabel(r"$t - t_{0}$  [days]")
        ax2.set_ylabel(r"$||\Delta \mathbf{r} (t)||$ [m]")
        ax2.set_yscale("log")
        ax2.grid(True)
        fig2.savefig(output_path + "/position_difference_norm.pdf")
        plt.close()

        # Load acceleration history difference for Saturn case
        acceleration_history_difference = np.loadtxt(input_path + "/acceleration_history_difference_study_Saturn.dat")

        fig3, ax3 = plt.subplots()
        ax3.plot(acceleration_history_difference[:, 0] / constants.JULIAN_DAY,
                 np.linalg.norm(acceleration_history_difference[:, 1:4], axis=1),
                 color=bodies_color_code["Saturn"][0],
                 )
        ax3.set_title("Point mass vs SH case, Saturn")
        ax3.set_xlabel(r"$t - t_{0}$  [days]")
        ax3.set_ylabel(r"$|| \Delta \mathbf{a}(t)||$  [m s$^{-2}$]")
        ax3.grid(which="both")
        ax3.set_yscale("log")
        ax3.set_ylim(bottom=1e-6)
        fig3.savefig(output_path + "/acceleration_history_difference_study_Saturn.pdf")
        plt.close()


if __name__ == "__main__":
    main()
