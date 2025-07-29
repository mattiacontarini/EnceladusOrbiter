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


def plot_monte_carlo_interior_parameters_analysis(input_path, fontsize=12):

    layers = ["core", "ocean", "shell"]

    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$R_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$R_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    # Load results
    observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
    interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

    filtered_parameters, filtered_observations = filter_parameters_and_observations(interior_models, observations)

    for i in range(len(layers)):
        layer = layers[i]
        parameters = filtered_parameters[:, 5*i:5*i+5]
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

            ax.scatter(filtered_parameters[:, j], filtered_observations[:, 0], color="black")
            ax2.scatter(filtered_parameters[:, j], filtered_observations[:, 2], color="black")
            ax3.scatter(filtered_parameters[:, j], filtered_observations[:, 1], color="black")
            ax.set_xlabel(interior_parameters_labels[5*i + j], fontsize=fontsize)
            ax2.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            ax3.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            if j % 2 == 0:
                ax.set_ylabel(r"$k_2$ Love number  [-]", fontsize=fontsize)
                ax2.set_ylabel(r"$\phi$  [-]", fontsize=fontsize)
                ax3.set_ylabel(r"$h_2$ Love number  [-]", fontsize=fontsize)
            ax.grid(True)
            ax2.grid(True)
            ax3.grid(True)
        fig.delaxes(axes[2, 1])
        fig2.delaxes(axes2[2, 1])
        fig3.delaxes(axes3[2, 1])
        fig.suptitle(f"Measurements: k2 Love number. Layer: {layer}", fontsize=fontsize)
        fig2.suptitle(f"Measurements: libration amplitude. Layer: {layer}", fontsize=fontsize)
        fig3.suptitle(f"Measurements: h2 Love number. Layer: {layer}", fontsize=fontsize)
        fig.savefig(os.path.join(input_path, f"observations_trends_k2_Love_number_{layer}.pdf"))
        fig2.savefig(os.path.join(input_path, f"observations_trends_libration_amplitude_{layer}.pdf"))
        fig3.savefig(os.path.join(input_path, f"observations_trends_h2_Love_number_{layer}.pdf"))


def filter_parameters_and_observations(parameters, observations):
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
    return filtered_parameters, filtered_observations


def main():

    plot_one_at_a_time_interior_parameters_analysis_flag = False
    if plot_one_at_a_time_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/preliminary_sensitivity_analysis"
        plot_one_at_a_time_interior_parameters_analysis(input_path)

    plot_monte_carlo_interior_parameters_analysis_flag = True
    if plot_monte_carlo_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.07.26.10.33.59"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis(input_path)



if __name__ == "__main__":
    main()
