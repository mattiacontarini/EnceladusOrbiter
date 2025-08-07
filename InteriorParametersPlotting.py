# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def plot_one_at_a_time_interior_parameters_analysis(input_path, fontsize=12):

    layers = ["core", "ocean", "shell"]
    interior_parameters = ["R0", "rho0", "mu0", "eta0", "Ks0"]
    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$R_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$R_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    k2_handle = mlines.Line2D([],
                              [],
                              color="blue",
                              linestyle="-",
                              marker=" ",
                              label=r"$k_2$")
    h2_handle = mlines.Line2D([],
                              [],
                              color="red",
                              linestyle="-",
                              marker=" ",
                              label=r"$h_2$")

    # Plot Love numbers
    for layer in layers:
        layer_input_path = os.path.join(input_path, layer)
        layer_index = layers.index(layer)

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            results = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = results[:, 0]
            k2_love_number_values = results[:, 1]
            h2_love_number_values = results[:, 2]

            if interior_parameter_index == 0:
                ax = axes[0, 0]
            elif interior_parameter_index == 1:
                ax = axes[0, 1]
            elif interior_parameter_index == 2:
                ax = axes[1, 0]
            elif interior_parameter_index == 3:
                ax = axes[1, 1]
            elif interior_parameter_index == 4:
                ax = axes[2, 0]

            ax.plot(interior_parameter_values, k2_love_number_values.real, label=r"$K_2$", color="blue")
            ax.plot(interior_parameter_values, h2_love_number_values.real, label=r"$h_2$", color="red")
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)

            if interior_parameter_index % 2 == 0:
                ax.set_ylabel("Love number  [-]", fontsize=fontsize)

        fig.legend(handles=[k2_handle, h2_handle], fontsize=fontsize, bbox_to_anchor=(0.85, 0.2))

        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            ax.set_yscale("log")
        plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurements: $k_2$ & $h_2$ Love numbers. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, "tidal_love_numbers.pdf"))
        plt.close(fig)

    # Plot libration amplitude
    for layer in layers:
        layer_input_path = os.path.join(input_path, layer)
        layer_index = layers.index(layer)

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            results = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = results[:, 0]
            libration_amplitude = np.rad2deg(results[:, 3])

            if interior_parameter_index == 0:
                ax = axes[0, 0]
            elif interior_parameter_index == 1:
                ax = axes[0, 1]
            elif interior_parameter_index == 2:
                ax = axes[1, 0]
            elif interior_parameter_index == 3:
                ax = axes[1, 1]
            elif interior_parameter_index == 4:
                ax = axes[2, 0]

            ax.plot(interior_parameter_values, libration_amplitude, color="blue")
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)
            if interior_parameter_index % 2 == 0:
                ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            # ax.set_yscale("log")
        plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurement: libration amplitude. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, "libration_amplitude.pdf"))
        plt.close(fig)


def plot_monte_carlo_interior_parameters_analysis(input_path,
                                                  nominal_observations_dissipative_shell,
                                                  std_observations_dissipative_shell,
                                                  filter_parameters_flag,
                                                  fontsize=12):

    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    layers = ["core", "ocean", "shell"]

    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$d_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$d_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    dissipative_shell = np.zeros((2, 3))
    dissipative_shell[0, :] = nominal_observations_dissipative_shell
    dissipative_shell[1, :] = std_observations_dissipative_shell

    # Load results
    if filter_parameters_flag:
        observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
        interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

        # Convert libration amplitude to deg
        observations[:, 0] = np.rad2deg(observations[:, 0])

        # Filter parameter and observations based on the feasibility of the interior model
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = filter_parameters(interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])
    else:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_feasibility.dat"))

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])


    for i in range(len(layers)):
        layer = layers[i]
        parameters = filtered_parameters_feasibility[:, 5*i:5*i+5]
        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        fig2, axes2 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        fig3, axes3 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        for j in range(parameters.shape[1]):

            if j == 0:
                ax = axes[0, 0]
                ax2 = axes2[0, 0]
                ax3 = axes3[0, 0]
            elif j == 1:
                ax = axes[0, 1]
                ax2 = axes2[0, 1]
                ax3 = axes3[0, 1]
            elif j == 2:
                ax = axes[1, 0]
                ax2 = axes2[1, 0]
                ax3 = axes3[1, 0]
            elif j == 3:
                ax = axes[1, 1]
                ax2 = axes2[1, 1]
                ax3 = axes3[1, 1]
            elif j == 4:
                ax = axes[2, 0]
                ax2 = axes2[2, 0]
                ax3 = axes3[2, 0]

            if j == 0:
                if layer == "ocean":
                    ax.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 2], color="black")
                elif layer == "shell":
                    ax.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 2], color="black")
                else:
                    ax.scatter(parameters[:, j], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(filtered_parameters_feasibility[:, j], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(filtered_parameters_feasibility[:, j], filtered_observations_feasibility[:, 2], color="black")
            else:
                ax.scatter(parameters[:, j], filtered_observations_feasibility[:, 0], color="black")
                ax2.scatter(parameters[:, j], filtered_observations_feasibility[:, 1], color="black")
                ax3.scatter(parameters[:, j], filtered_observations_feasibility[:, 2], color="black")


            ax.axhline(y=dissipative_shell[0, 0], color="lime")
            ax2.axhline(y=dissipative_shell[0, 1], color="lime")
            ax3.axhline(y=dissipative_shell[0, 2], color="lime")

            if j == 0:
                if layer == "ocean":
                    ax.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 0]), max(parameters[:, j] - filtered_parameters_feasibility[:, 0])],
                        dissipative_shell[0, 0] + dissipative_shell[1, 0],
                        dissipative_shell[0, 0] - dissipative_shell[1, 0],
                        color="lime",
                        alpha=0.7,
                    )
                    ax2.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 0]), max(parameters[:, j] - filtered_parameters_feasibility[:, 0])],
                        dissipative_shell[0, 1] + dissipative_shell[1, 1],
                        dissipative_shell[0, 1] - dissipative_shell[1, 1],
                        color="lime",
                        alpha=0.7,
                    )
                    ax3.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 0]), max(parameters[:, j] - filtered_parameters_feasibility[:, 0])],
                        dissipative_shell[0, 2] + dissipative_shell[1, 2],
                        dissipative_shell[0, 2] - dissipative_shell[1, 2],
                        color="lime",
                        alpha=0.7,
                    )
                elif layer == "shell":
                    ax.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 5]), max(parameters[:, j] - filtered_parameters_feasibility[:, 5])],
                        dissipative_shell[0, 0] + dissipative_shell[1, 0],
                        dissipative_shell[0, 0] - dissipative_shell[1, 0],
                        color="lime",
                        alpha=0.7,
                    )
                    ax2.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 5]), max(parameters[:, j] - filtered_parameters_feasibility[:, 5])],
                        dissipative_shell[0, 1] + dissipative_shell[1, 1],
                        dissipative_shell[0, 1] - dissipative_shell[1, 1],
                        color="lime",
                        alpha=0.7,
                    )
                    ax3.fill_between(
                        [min(parameters[:, j] - filtered_parameters_feasibility[:, 5]), max(parameters[:, j] - filtered_parameters_feasibility[:, 5])],
                        dissipative_shell[0, 2] + dissipative_shell[1, 2],
                        dissipative_shell[0, 2] - dissipative_shell[1, 2],
                        color="lime",
                        alpha=0.7,
                    )
                else:
                    ax.fill_between(
                        [min(parameters[:, j]), max(parameters[:, j])],
                        dissipative_shell[0, 0] + dissipative_shell[1, 0],
                        dissipative_shell[0, 0] - dissipative_shell[1, 0],
                        color="lime",
                        alpha=0.7,
                    )
                    ax2.fill_between(
                        [min(parameters[:, j]), max(parameters[:, j])],
                        dissipative_shell[0, 1] + dissipative_shell[1, 1],
                        dissipative_shell[0, 1] - dissipative_shell[1, 1],
                        color="lime",
                        alpha=0.7,
                    )
                    ax3.fill_between(
                        [min(parameters[:, j]), max(parameters[:, j])],
                        dissipative_shell[0, 2] + dissipative_shell[1, 2],
                        dissipative_shell[0, 2] - dissipative_shell[1, 2],
                        color="lime",
                        alpha=0.7,
                    )
            else:
                ax.fill_between(
                    [min(parameters[:, j]), max(parameters[:, j])],
                    dissipative_shell[0, 0] + dissipative_shell[1, 0],
                    dissipative_shell[0, 0] - dissipative_shell[1, 0],
                    color="lime",
                    alpha=0.7,
                )
                ax2.fill_between(
                    [min(parameters[:, j]), max(parameters[:, j])],
                    dissipative_shell[0, 1] + dissipative_shell[1, 1],
                    dissipative_shell[0, 1] - dissipative_shell[1, 1],
                    color="lime",
                    alpha=0.7,
                )
                ax3.fill_between(
                    [min(parameters[:, j]), max(parameters[:, j])],
                    dissipative_shell[0, 2] + dissipative_shell[1, 2],
                    dissipative_shell[0, 2] - dissipative_shell[1, 2],
                    color="lime",
                    alpha=0.7,
                )

            ax.set_xlabel(interior_parameters_labels[5*i + j], fontsize=fontsize)
            ax2.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            ax3.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            if j % 2 == 0:
                ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
                ax2.set_ylabel(r"$k_2$  [-]", fontsize=fontsize)
                ax3.set_ylabel(r"$h_2$  [-]", fontsize=fontsize)
            if j > 1:
                ax.set_xscale("log")
                ax2.set_xscale("log")
                ax3.set_xscale("log")
            ax.grid(True)
            ax2.grid(True)
            ax3.grid(True)
        fig.delaxes(axes[2, 1])
        fig2.delaxes(axes2[2, 1])
        fig3.delaxes(axes3[2, 1])
        fig.suptitle(f"Measurements: libration amplitude. Layer: {layer}", fontsize=fontsize)
        fig2.suptitle(f"Measurements: k2 Love number. Layer: {layer}", fontsize=fontsize)
        fig3.suptitle(f"Measurements: h2 Love number. Layer: {layer}", fontsize=fontsize)
        fig.savefig(os.path.join(plots_path, f"observations_trends_libration_amplitude_{layer}.pdf"))
        fig2.savefig(os.path.join(plots_path, f"observations_trends_k2_Love_number_{layer}.pdf"))
        fig3.savefig(os.path.join(plots_path, f"observations_trends_h2_Love_number_{layer}.pdf"))


def filter_parameters(parameters, observations):
    nb_simulations = parameters.shape[0]

    counter = 0
    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):
        core_density = parameters[i, 1]
        ocean_density = parameters[i, 6]

        if core_density <= 2000.0 or core_density >= 3000.0 or ocean_density <= 1000.0 or ocean_density >= 1300.0:
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    print("Nb. of invalid simulations: ", counter)
    return filtered_parameters, filtered_observations, nb_simulations-counter


def filter_observations(parameters, observations, nominal_observations):
    nb_simulations = parameters.shape[0]
    nb_observations = observations.shape[1]
    counter = 0

    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):

        delete_flag = False
        for j in range(nb_observations):
            if (observations[i, j] >= nominal_observations[0, j] + nominal_observations[1, j]):
                delete_flag = True
                break

        if delete_flag:
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    print("Nb. of invalid simulations: ", counter)
    return filtered_parameters, filtered_observations, nb_simulations-counter

def main():

    plot_one_at_a_time_interior_parameters_analysis_flag = False
    if plot_one_at_a_time_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/preliminary_sensitivity_analysis"
        plot_one_at_a_time_interior_parameters_analysis(input_path)

    plot_monte_carlo_interior_parameters_analysis_flag = True
    if plot_monte_carlo_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.06.08.16.27"
        input_path = os.path.join(input_path, time_stamp)
        nominal_observations_dissipative_shell = [0.091, 0.0317, 0.0848]  # Park et al. (2024), Bagheri et al. (2025)
        std_observations_dissipative_shell = [0.009, 0.0130, 0.0359]  # Park et al. (2024), Bagheri et al. (2025)
        plot_monte_carlo_interior_parameters_analysis(input_path,
                                                      nominal_observations_dissipative_shell,
                                                      std_observations_dissipative_shell,
                                                      filter_parameters_flag=False
                                                      )

if __name__ == "__main__":
    main()
